"""
Copyright (C) 2019 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""

from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        BaseOptions.initialize(self, parser)
        parser.add_argument('--dataroot', type=str, default='./imgs/imgtest', help='数据集根目录路径')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--which_epoch', type=str, default='380', help='which epoch to load? set to latest to use latest cached model')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='how many test images to run')
        parser.add_argument('--save_per_img', default=True, help='if specified, save per image')
        parser.add_argument('--show_corr', action='store_true', help='if specified, save bilinear upsample correspondence')
        parser.add_argument('--use_color_enhance', default=True, help='enable color enhancement')
        parser.add_argument('--which_perceptual', type=str, default='4_2', help='感知损失使用的VGG层(relu5_2 或 relu4_2)')
        parser.add_argument('--use_22ctx', action='store_true', help='启用则在上下文损失中使用2-2层特征')
        parser.set_defaults(preprocess_mode='scale_width_and_crop', crop_size=224, load_size=224, display_winsize=224)
        parser.set_defaults(serial_batches=True)
        parser.set_defaults(no_flip=True)
        parser.set_defaults(phase='test')
        self.isTrain = False
        return parser
