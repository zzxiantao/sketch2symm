"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

import sys
import argparse
import os
from util import util
import torch
import models
import data
import pickle


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        # 实验配置
        parser.add_argument('--name', type=str, default='sketch2img_warp_13', help='实验名称，决定存储样本和模型的位置')

        parser.add_argument('--gpu_ids', type=str, default='0', help='GPU ID列表，例如：0  0,1,2  0,2，使用-1表示CPU')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='模型保存路径')
        parser.add_argument('--model', type=str, default='pix2pix', help='选择使用的模型类型')
        parser.add_argument('--norm_G', type=str, default='spectralinstance', help='生成器的标准化方式：实例标准化或批标准化')
        parser.add_argument('--norm_D', type=str, default='spectralinstance', help='判别器的标准化方式：实例标准化或批标准化')
        parser.add_argument('--norm_E', type=str, default='spectralinstance', help='编码器的标准化方式：实例标准化或批标准化')
        parser.add_argument('--phase', type=str, default='train', help='运行阶段：train训练/val验证/test测试')

        # 输入输出尺寸
        parser.add_argument('--batchSize', type=int, default=1, help='输入批处理大小')
        parser.add_argument('--preprocess_mode', type=str, default='scale_width_and_crop', help='图像加载时的预处理方式', choices=("resize_and_crop", "crop", "scale_width", "scale_width_and_crop", "scale_shortside", "scale_shortside_and_crop", "fixed", "none"))
        parser.add_argument('--load_size', type=int, default=256, help='图像缩放尺寸，最终会裁剪到crop_size')
        parser.add_argument('--crop_size', type=int, default=256, help='最终裁剪尺寸（在缩放之后进行裁剪）')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='图像宽高比，最终高度为crop_size/aspect_ratio')
        parser.add_argument('--label_nc', type=int, default=182, help='输入标签类别数（不包含未知类）')
        parser.add_argument('--contain_dontcare_label', action='store_true', help='标签图中是否包含dontcare标签（255值）')
        parser.add_argument('--output_nc', type=int, default=3, help='输出图像通道数')

        # 数据设置
        parser.add_argument('--dataset_mode', type=str, default='celebahqedge', help='使用的数据集模式')
        parser.add_argument('--serial_batches', action='store_true', help='是否按顺序加载批次（否则随机）')
        parser.add_argument('--no_flip', action='store_true', help='是否禁用图像翻转数据增强')
        parser.add_argument('--nThreads', default=16, type=int, help='数据加载线程数')
        parser.add_argument('--max_dataset_size', type=int, default=sys.maxsize, help='最大加载样本数')
        parser.add_argument('--load_from_opt_file', action='store_true', help='从检查点加载配置作为默认')
        parser.add_argument('--cache_filelist_write', action='store_true', help='将文件列表缓存到文本文件加速加载')
        parser.add_argument('--cache_filelist_read', action='store_true', help='从缓存文件列表读取')

        # 显示设置
        parser.add_argument('--display_winsize', type=int, default=400, help='显示窗口尺寸')

        # 生成器配置
        parser.add_argument('--netG', type=str, default='spade', help='生成器架构选择（pix2pixhd | spade）')
        parser.add_argument('--ngf', type=int, default=64, help='生成器首层卷积滤波器数量')
        parser.add_argument('--init_type', type=str, default='xavier', help='网络初始化方式[normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_variance', type=float, default=0.02, help='初始化分布的方差')
        parser.add_argument('--z_dim', type=int, default=256, help='潜在向量z的维度')

        # 实例特征配置
        parser.add_argument('--CBN_intype', type=str, default='warp', help='CBN输入类型：warp形变/sketch草图/warp_mask混合')
        parser.add_argument('--maskmix', default=True, help='在对应网络中使用掩模')
        parser.add_argument('--use_attention', default=True, help='在G和D中使用非局部注意力块')
        parser.add_argument('--warp_mask_losstype', type=str, default='none', help='形变掩模损失类型：none无/direct直接/cycle循环')
        parser.add_argument('--show_warpmask', action='store_true', help='保存形变掩模可视化结果')
        parser.add_argument('--match_kernel', type=int, default=3, help='对应矩阵匹配核尺寸')
        parser.add_argument('--adaptor_kernel', type=int, default=3, help='域适配器的卷积核尺寸')
        parser.add_argument('--PONO', default=True, help='使用位置归一化')
        parser.add_argument('--PONO_C', default=True, help='在对应模块使用C通道归一化')
        parser.add_argument('--eqlr_sn', action='store_true', help='使用均衡学习率(否则使用谱归一化)')
        parser.add_argument('--vgg_normal_correct', default=True, help='修正VGG标准化并用ctx模型替代VGG特征匹配')
        parser.add_argument('--weight_domainC', type=float, default=0.0, help='域分类损失权重')
        parser.add_argument('--domain_rela', action='store_true', help='在域分类器中使用相对论损失')
        parser.add_argument('--use_ema', action='store_true', help='在生成器中使用指数移动平均')
        parser.add_argument('--ema_beta', type=float, default=0.999, help='EMA的beta参数')
        parser.add_argument('--warp_cycle_w', type=float, default=1, help='循环形变对齐损失权重')
        parser.add_argument('--two_cycle', action='store_true', help='输入->参考->输入的两次循环')
        parser.add_argument('--apex', action='store_true', help='使用NVIDIA APEX加速')
        parser.add_argument('--warp_bilinear', default=True, help='使用双线性上采样形变')
        parser.add_argument('--adaptor_res_deeper', action='store_true', help='在域适配器中使用6个残差块')
        parser.add_argument('--adaptor_nonlocal', action='store_true', help='在域适配器中添加非局部块')
        parser.add_argument('--adaptor_se', action='store_true', help='在域适配器中使用SE模块')
        parser.add_argument('--dilation_conv', action='store_true', help='深层适配器中使用空洞卷积')
        parser.add_argument('--use_coordconv', action='store_true', help='在CorrNet中使用坐标卷积')
        parser.add_argument('--warp_patch', action='store_true', help='使用4x4块级形变')
        parser.add_argument('--warp_stride', type=int, default=4, help='形变矩阵步长（256/warp_stride）')
        parser.add_argument('--mask_noise', action='store_true', help='在掩模中添加噪声')
        parser.add_argument('--noise_for_mask', action='store_true', help='用噪声替代掩模')
        parser.add_argument('--video_like', action='store_true', help='视频类数据处理（用于DeepFashion）')
        parser.add_argument('--lambda_color', type=float, default=0.0, help='颜色损失权重')
        parser.add_argument('--color_loss_type', type=str, default='histogram', choices=['histogram', 'moment', 'perceptual'])
        parser.add_argument('--use_scft', default=True, 
                          help='启用SCFT注意力模块')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, unknown = parser.parse_known_args()

        # modify model-related parser options
        model_name = opt.model
        model_option_setter = models.get_option_setter(model_name)
        parser = model_option_setter(parser, self.isTrain)

        # modify dataset-related parser options
        dataset_mode = opt.dataset_mode
        dataset_option_setter = data.get_option_setter(dataset_mode)
        parser = dataset_option_setter(parser, self.isTrain)

        opt, unknown = parser.parse_known_args()

        # if there is opt_file, load it.
        # The previous default options will be overwritten
        if opt.load_from_opt_file:
            parser = self.update_options_from_file(parser, opt)

        opt = parser.parse_args()
        self.parser = parser
        return opt

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

    def option_file_path(self, opt, makedir=False):
        expr_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if makedir:
            util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt')
        return file_name

    def save_options(self, opt):
        file_name = self.option_file_path(opt, makedir=True)
        with open(file_name + '.txt', 'wt') as opt_file:
            for k, v in sorted(vars(opt).items()):
                comment = ''
                default = self.parser.get_default(k)
                if v != default:
                    comment = '\t[default: %s]' % str(default)
                opt_file.write('{:>25}: {:<30}{}\n'.format(str(k), str(v), comment))

        with open(file_name + '.pkl', 'wb') as opt_file:
            pickle.dump(opt, opt_file)

    def update_options_from_file(self, parser, opt):
        new_opt = self.load_options(opt)
        for k, v in sorted(vars(opt).items()):
            if hasattr(new_opt, k) and v != getattr(new_opt, k):
                new_val = getattr(new_opt, k)
                parser.set_defaults(**{k: new_val})
        return parser

    def load_options(self, opt):
        file_name = self.option_file_path(opt, makedir=False)
        new_opt = pickle.load(open(file_name + '.pkl', 'rb'))
        return new_opt

    def parse(self, save=False):

        opt = self.gather_options()  #gather options from base, train, dataset, model
        opt.isTrain = self.isTrain   # train or test

        self.print_options(opt)
        if opt.isTrain:
            self.save_options(opt)

        # Set semantic_nc based on the option.
        # This will be convenient in many places
        opt.semantic_nc = opt.label_nc + \
            (1 if opt.contain_dontcare_label else 0)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        assert len(opt.gpu_ids) == 0 or opt.batchSize % len(opt.gpu_ids) == 0, \
            "Batch size %d is wrong. It must be a multiple of # GPUs %d." \
            % (opt.batchSize, len(opt.gpu_ids))

        self.opt = opt
        return self.opt
