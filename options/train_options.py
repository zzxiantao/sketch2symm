"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        # for displays
        parser.add_argument('--display_freq', type=int, default=2000, help='frequency of showing training results on screen')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=10, help='frequency of saving checkpoints at the end of epochs')

        # for training
        parser.add_argument('--dataroot', type=str, default='./imgs/train/imagenet', help='数据集根目录路径')
        parser.add_argument('--continue_train', action='store_true', help='继续训练：加载最新模型')
        parser.add_argument('--which_epoch', type=str, default='latest', help='加载哪个epoch的模型？设为latest使用最新缓存模型')
        parser.add_argument('--niter', type=int, default=30, help='初始学习率的迭代次数（非总epoch数，总epoch数=niter + niter_decay）')
        parser.add_argument('--niter_decay', type=int, default=470, help='线性衰减学习率到零的迭代次数')
        parser.add_argument('--optimizer', type=str, default='adam', help='优化器选择')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam优化器的beta1动量参数')
        parser.add_argument('--beta2', type=float, default=0.999, help='Adam优化器的beta2动量参数')
        parser.add_argument('--lr', type=float, default=0.0001, help='Adam优化器的初始学习率')
        parser.add_argument('--D_steps_per_G', type=int, default=1, help='每个生成器迭代对应的判别器迭代次数')

        # for discriminators
        parser.add_argument('--ndf', type=int, default=64, help='判别器第一卷积层的滤波器数量')
        parser.add_argument('--lambda_feat', type=float, default=10.0, help='特征匹配损失的权重')
        parser.add_argument('--lambda_vgg', type=float, default=15.0, help='VGG损失的权重')
        parser.add_argument('--no_ganFeat_loss', action='store_true', help='启用则不使用判别器特征匹配损失')
        parser.add_argument('--gan_mode', type=str, default='hinge', help='GAN损失模式选择(ls|original|hinge)')
        parser.add_argument('--netD', type=str, default='multiscale', help='判别器结构选择(n_layers|multiscale|image)')
        parser.add_argument('--no_TTUR', action='store_true', help='禁用TTUR训练方案')

        parser.add_argument('--which_perceptual', type=str, default='4_2', help='感知损失使用的VGG层(relu5_2 或 relu4_2)')#层数越深对于语义信息越丰富，但计算量越大，颜色损失越严重，所以这里选择3_2,并且增大lambda_vgg的权重增大感知损失
        parser.add_argument('--weight_perceptual', type=float, default=0.001, help='感知损失的总体权重')
        parser.add_argument('--weight_mask', type=float, default=0.0, help='形变掩模损失的权重，用于direct/cycle模式')
        parser.add_argument('--real_reference_probability', type=float, default=0.7, help='自监督训练概率')
        parser.add_argument('--hard_reference_probability', type=float, default=0.2, help='困难参考样本训练概率')
        parser.add_argument('--weight_gan', type=float, default=10.0, help='第一阶段所有GAN损失的权重')
        parser.add_argument('--novgg_featpair', type=float, default=10.0, help='无VGG时域适应中的特征对损失权重')
        parser.add_argument('--D_cam', type=float, default=0.0, help='判别器中CAM损失的权重')
        parser.add_argument('--warp_self_w', type=float, default=0.0, help='自形变对齐损失的权重')
        parser.add_argument('--fm_ratio', type=float, default=1.0, help='VGG特征匹配损失与上下文损失的比率')
        parser.add_argument('--use_22ctx', action='store_true', help='启用则在上下文损失中使用2-2层特征')
        parser.add_argument('--ctx_w', type=float, default=1.5, help='上下文损失的权重')
        parser.add_argument('--mask_epoch', type=int, default=-1, help='当启用noise_for_mask时，前mask_epoch使用真实掩模训练，后续使用噪声')

        parser.add_argument('--use_color_enhance',  default=True, help='启用颜色增强模块')
        self.isTrain = True
        return parser
