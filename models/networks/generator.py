# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
#08.09 change pad

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from models.networks.base_network import BaseNetwork
from models.networks.normalization import get_nonspade_norm_layer, equal_lr
from models.networks.architecture import ResnetBlock as ResnetBlock
from models.networks.architecture import SPADEResnetBlock as SPADEResnetBlock
from models.networks.architecture import Attention
from models.networks.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d

class SPADEGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        return parser

    def __init__(self, opt):
        super().__init__()
        self.opt = opt
        nf = opt.ngf

        self.sw, self.sh = self.compute_latent_vector_size(opt)

        ic = 0 + (3 if 'warp' in self.opt.CBN_intype else 0) + (self.opt.semantic_nc if 'sketch' in self.opt.CBN_intype else 0)
        self.fc = nn.Conv2d(ic, 16 * nf, 3, padding=1)
        if opt.eqlr_sn:
            self.fc = equal_lr(self.fc)

        self.head_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.G_middle_0 = SPADEResnetBlock(16 * nf, 16 * nf, opt)
        self.G_middle_1 = SPADEResnetBlock(16 * nf, 16 * nf, opt)

        self.up_0 = SPADEResnetBlock(16 * nf, 8 * nf, opt)
        self.up_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        if opt.use_attention:
            self.attn = Attention(4 * nf, 'spectral' in opt.norm_G)
        self.up_2 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.up_3 = SPADEResnetBlock(2 * nf, 1 * nf, opt)

        final_nc = nf

        self.conv_img = nn.Conv2d(final_nc, 3, 3, padding=1)
        self.up = nn.Upsample(scale_factor=2)

        self.color_enhance = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=1)  # 保持输入输出通道一致
        )

        # SCFT模块
        self.scft = SCFTModule(in_channels=3, opt=opt)


    def compute_latent_vector_size(self, opt):
        num_up_layers = 5

        sw = opt.crop_size // (2**num_up_layers)
        sh = round(sw / opt.aspect_ratio)

        return sw, sh

    def forward(self, input, warp_out=None):
        # 适配输入通道
        #input = self.input_adapt(input)
        seg = input if warp_out is None else warp_out

        # we downsample segmap and run convolution
        x = F.interpolate(seg, size=(self.sh, self.sw))
        x = self.fc(x)
        x = self.head_0(x, seg)

        x = self.up(x)
        x = self.G_middle_0(x, seg)

        x = self.G_middle_1(x, seg)

        x = self.up(x)
        x = self.up_0(x, seg)
        x = self.up(x)
        x = self.up_1(x, seg)

        x = self.up(x)
        if self.opt.use_attention:
            x = self.attn(x)
        x = self.up_2(x, seg)
        x = self.up(x)
        x = self.up_3(x, seg)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = torch.tanh(x)
        # 添加值域保护
        x = torch.clamp(x, -0.999, 0.999)
        # SCFT处理
        if self.opt.use_scft:
            attention_map = self.scft(x)  # 生成注意力图
            x = x * attention_map  # 应用注意力加权
            
        # 颜色增强模块
        if self.opt.use_color_enhance:
            x = self.color_enhance(x)
            x = torch.clamp(x, -1, 1)

            
        return x

    def init_new_modules(self):
        if hasattr(self, 'color_enhance'):
            for m in self.color_enhance.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
        
        if hasattr(self, 'scft'):
            for m in self.scft.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

class AdaptiveFeatureGenerator(BaseNetwork):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.set_defaults(norm_G='spectralspadesyncbatch3x3')
        parser.add_argument('--num_upsampling_layers',
                            choices=('normal', 'more', 'most'), default='normal',
                            help="If 'more', adds upsampling layer between the two middle resnet blocks. If 'most', also add one more upsampling + resnet layer at the end of the generator")

        return parser

    def __init__(self, opt):
        # TODO: kernel=4, concat noise, or change architecture to vgg feature pyramid
        super().__init__()
        self.opt = opt
        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(opt.spade_ic, ndf, kw, stride=1, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
        if opt.warp_stride == 2:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=1, padding=pw))
        else:
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, opt.adaptor_kernel, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt
        
        nf = opt.ngf

        self.head_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        if opt.adaptor_nonlocal:
            self.attn = Attention(8 * nf, False)
        self.G_middle_0 = SPADEResnetBlock(8 * nf, 8 * nf, opt, use_se=opt.adaptor_se)
        self.G_middle_1 = SPADEResnetBlock(8 * nf, 4 * nf, opt, use_se=opt.adaptor_se)

        if opt.adaptor_res_deeper:
            self.deeper0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
            if opt.dilation_conv:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=2)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt, dilation=4)
                self.degridding0 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=2, dilation=2))
                self.degridding1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 3, stride=1, padding=1))
            else:
                self.deeper1 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
                self.deeper2 = SPADEResnetBlock(4 * nf, 4 * nf, opt)

    def forward(self, input, seg):
        x = self.layer1(input)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))

        x = self.head_0(x, seg)
        if self.opt.adaptor_nonlocal:
            x = self.attn(x)
        x = self.G_middle_0(x, seg)
        x = self.G_middle_1(x, seg)
        if self.opt.adaptor_res_deeper:
            x = self.deeper0(x, seg)
            x = self.deeper1(x, seg)
            x = self.deeper2(x, seg)
            if self.opt.dilation_conv:
                x = self.degridding0(x)
                x = self.degridding1(x)

        return x

class ReverseGenerator(BaseNetwork):
    def __init__(self, opt, ic, oc, size):
        super().__init__()
        self.opt = opt
        self.downsample = True if size == 256 else False
        nf = opt.ngf
        opt.spade_ic = ic
        if opt.warp_reverseG_s:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 4 * nf, opt)
        else:
            self.backbone_0 = SPADEResnetBlock(4 * nf, 8 * nf, opt)
            self.backbone_1 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_2 = SPADEResnetBlock(8 * nf, 8 * nf, opt)
            self.backbone_3 = SPADEResnetBlock(8 * nf, 4 * nf, opt)
        self.backbone_4 = SPADEResnetBlock(4 * nf, 2 * nf, opt)
        self.backbone_5 = SPADEResnetBlock(2 * nf, nf, opt)
        del opt.spade_ic
        if self.downsample:
            kw = 3
            pw = int(np.ceil((kw - 1.0) / 2))
            ndf = opt.ngf
            norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
            self.layer1 = norm_layer(nn.Conv2d(ic, ndf, kw, stride=1, padding=pw))
            self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, 4, stride=2, padding=pw))
            self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=1, padding=pw))
            self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, 4, stride=2, padding=pw))
            self.up = nn.Upsample(scale_factor=2)
        self.actvn = nn.LeakyReLU(0.2, False)
        self.conv_img = nn.Conv2d(nf, oc, 3, padding=1)

    def forward(self, x):
        input = x
        if self.downsample:
            x = self.layer1(input)
            x = self.layer2(self.actvn(x))
            x = self.layer3(self.actvn(x))
            x = self.layer4(self.actvn(x))
        x = self.backbone_0(x, input)
        if not self.opt.warp_reverseG_s:
            x = self.backbone_1(x, input)
            x = self.backbone_2(x, input)
            x = self.backbone_3(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_4(x, input)
        if self.downsample:
            x = self.up(x)
        x = self.backbone_5(x, input)
        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.tanh(x)
        return x

class DomainClassifier(BaseNetwork):
    def __init__(self, opt):
        super().__init__()
        nf = opt.ngf
        kw = 4 if opt.domain_rela else 3
        pw = int((kw - 1.0) / 2)
        self.feature = nn.Sequential(nn.Conv2d(4 * nf, 2 * nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(2 * nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(2 * nf, nf, kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(nf, affine=True),
                                nn.LeakyReLU(0.2, False),
                                nn.Conv2d(nf, int(nf // 2), kw, stride=2, padding=pw),
                                SynchronizedBatchNorm2d(int(nf // 2), affine=True),
                                nn.LeakyReLU(0.2, False))  #32*8*8
        model = [nn.Linear(int(nf // 2) * 8 * 8, 100),
                SynchronizedBatchNorm1d(100, affine=True),
                nn.ReLU()]
        if opt.domain_rela:
            model += [nn.Linear(100, 1)]
        else:
            model += [nn.Linear(100, 2),
                      nn.LogSoftmax(dim=1)]
        self.classifier = nn.Sequential(*model)

    def forward(self, x):
        x = self.feature(x)
        x = self.classifier(x.view(x.shape[0], -1))
        return x

class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class EMA():
    def __init__(self, mu):
        self.mu = mu
        self.shadow = {}
        self.original = {}

    def register(self, name, val):
        self.shadow[name] = val.clone()

    def __call__(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                decay = self.mu
                new_average = (1.0 - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def assign(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.original[name] = param.data.clone()
                param.data = self.shadow[name]
                
    def resume(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                param.data = self.original[name]

class SCFTModule(nn.Module):
    """SCFT模块实现"""
    def __init__(self, in_channels, opt):
        super().__init__()
        self.opt = opt
        
        # 修复通道数计算
        reduced_channels = max(1, in_channels // 4)  # 确保至少1个通道
        
        # 空间注意力分支
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )
        
        # 修改通道注意力分支
        self.channel_conv = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),  # 使用调整后的通道数
            nn.ReLU(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        
    def forward(self, x):   
        # 空间注意力分支
        spatial_att = self.spatial_conv(x)    
        # 通道注意力分支
        channel_att = self.channel_conv(x)
        reduced_channels = max(1, x.shape[1] // 4)
        
        # 自适应融合
        combined_att = spatial_att * channel_att
        
        
        return combined_att

# class SCFTModule(nn.Module):
#     """基于Transformer的SCFT模块实现"""
#     def __init__(self, in_channels, opt, dv=992):
#         super().__init__()
#         self.opt = opt
#         self.dv = dv
        
#         # 特征投影层，将输入特征映射到dv维度
#         self.feature_proj = nn.Linear(in_channels, dv)
        
#         # Transformer注意力层
#         self.w_q = nn.Linear(dv, dv)
#         self.w_k = nn.Linear(dv, dv)
#         self.w_v = nn.Linear(dv, dv)
        
#         # 输出投影层，将结果映射回原始通道数
#         self.output_proj = nn.Linear(dv, in_channels)
        
#     def forward(self, x):
#         """
#         Args:
#             x: 输入特征图 (B, C, H, W)
#         Returns:
#             attention_map: 注意力图 (B, C, H, W)
#         """
#         B, C, H, W = x.shape
        
#         # 将特征图转换为序列格式 (B, H*W, C)
#         x_flat = x.permute(0, 2, 3, 1).contiguous().view(B, H*W, C)
        
#         # 特征投影到dv维度
#         x_proj = self.feature_proj(x_flat)  # (B, H*W, dv)
        
#         # 自注意力计算（使用相同的特征作为query, key, value）
#         query = self.w_q(x_proj)  # (B, H*W, dv)
#         key = self.w_k(x_proj)    # (B, H*W, dv)
#         value = self.w_v(x_proj)  # (B, H*W, dv)
        
#         # 计算注意力权重
#         attention_map = self.scaled_dot_product(query, key, value)  # (B, H*W, dv)
        
#         # 残差连接
#         attention_map = attention_map + x_proj
        
#         # 投影回原始通道数
#         attention_map = self.output_proj(attention_map)  # (B, H*W, C)
        
#         # 转换回特征图格式 (B, C, H, W)
#         attention_map = attention_map.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        
#         # 应用Sigmoid激活，确保输出在[0,1]范围内
#         attention_map = torch.sigmoid(attention_map)
        
#         return attention_map
    
#     def scaled_dot_product(self, query, key, value, mask=None):
#         """计算缩放点积注意力"""
#         d_k = query.shape[-1]
        
#         # 计算注意力分数
#         scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        
#         # 应用mask（如果提供）
#         if mask is not None:
#             scores = scores.masked_fill(mask == 0, -1e9)
        
#         # 计算注意力权重
#         attention_weights = F.softmax(scores, dim=-1)
        
#         # 应用注意力到value
#         output = torch.matmul(attention_weights, value)
        
#         return output
