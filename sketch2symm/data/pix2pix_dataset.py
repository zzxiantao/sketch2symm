# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from data.base_dataset import BaseDataset, get_params, get_transform
import torch
import torchvision.transforms as transforms
from PIL import Image
import util.util as util
import os
import random
import re
#from scipy.ndimage.filters import gaussian_filter


class Pix2pixDataset(BaseDataset):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser.add_argument('--no_pairing_check', action='store_true',
                            help='If specified, skip sanity check of correct label-image file pairing')
        return parser

    def initialize(self, opt):
        self.opt = opt

        label_paths, image_paths = self.get_paths(opt)

        if opt.dataset_mode != 'celebahq' and opt.dataset_mode != 'deepfashion':
            util.natural_sort(label_paths)
            util.natural_sort(image_paths)

        label_paths = label_paths[:opt.max_dataset_size]
        image_paths = image_paths[:opt.max_dataset_size]

        if not opt.no_pairing_check:
            for path1, path2 in zip(label_paths, image_paths):
                assert self.paths_match(path1, path2), \
                    "The label-image pair (%s, %s) do not look like the right pair because the filenames are quite different. Are you sure about the pairing? Please see data/pix2pix_dataset.py to see what is going on, and use --no_pairing_check to bypass this." % (path1, path2)

        self.label_paths = label_paths
        self.image_paths = image_paths

        size = len(self.label_paths)
        self.dataset_size = size

        self.real_reference_probability = 1 if opt.phase == 'test' else opt.real_reference_probability
        self.hard_reference_probability = 0 if opt.phase == 'test' else opt.hard_reference_probability
        self.ref_dict, self.train_test_folder = self.get_ref(opt)

    def get_paths(self, opt):
        label_paths = []
        image_paths = []
        assert False, "A subclass of Pix2pixDataset must override self.get_paths(self, opt)"
        return label_paths, image_paths

    def paths_match(self, path1, path2):
        filename1_without_ext = os.path.splitext(os.path.basename(path1))[0]
        filename2_without_ext = os.path.splitext(os.path.basename(path2))[0]
        return filename1_without_ext == filename2_without_ext

    def get_label_tensor(self, path):
        label = Image.open(path)
        params1 = get_params(self.opt, label.size)
        transform_label = get_transform(self.opt, params1, method=Image.NEAREST, normalize=False)
        label_tensor = transform_label(label) * 255.0
        label_tensor[label_tensor == 255] = self.opt.label_nc  # 'unknown' is opt.label_nc
        return label_tensor, params1

    def __getitem__(self, index):
        # Label Image
        label_path = self.label_paths[index]
        label_tensor, params1 = self.get_label_tensor(label_path)

        # input image (real images)
        image_path = self.image_paths[index]
        #print(f"输入草图路径: {label_path}")
        #print(f"参考图像路径: {image_path}")
        if not self.opt.no_pairing_check:
            assert self.paths_match(label_path, image_path), \
                "The label_path %s and image_path %s don't match." % \
                (label_path, image_path)
        image = Image.open(image_path)
        image = image.convert('RGB')

        transform_image = get_transform(self.opt, params1)
        image_tensor = transform_image(image)

        ref_tensor = 0
        label_ref_tensor = 0

        random_p = random.random()
        if random_p < self.real_reference_probability or self.opt.phase == 'test':  # True
            if hasattr(self, 'get_ref_key'):
                key = self.get_ref_key(image_path)
            else:
                key = image_path.replace('\\', '/').split('DeepFashion/')[-1] if self.opt.dataset_mode == 'deepfashion' else os.path.basename(image_path)
            # print(key)
            
            # 安全获取参考值，如果找不到键则使用默认值
            if key in self.ref_dict:
                val = self.ref_dict[key]
            else:
                # 当找不到对应的键时，使用默认值
                print(f"警告：在参考字典中找不到键 '{key}'，使用默认参考")
                # 根据类别提供默认参考
                if "/ShapeNet/image/" in image_path:
                    pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/00\.png"
                    match = re.search(pattern, image_path)
                    if match:
                        category_id = match.group(1)
                        if category_id == "02691156":
                            val = ["02691156.png", "02691156.png"]
                        elif category_id == "02828884":
                            val = ["02828884.png", "02828884.png"]
                        elif category_id == "02933112":
                            val = ["02933112.png", "02933112.png"]
                        elif category_id == "02958343":
                            val = ["02958343.png", "02958343.png"]
                        elif category_id == "03001627":
                            val = ["03001627.png", "03001627.png"]
                        elif category_id == "03211117":
                            val = ["03211117.png", "03211117.png"]
                        elif category_id == "03636649":
                            val = ["03636649.png", "03636649.png"]
                        elif category_id == "03691459":
                            val = ["03691459.png", "03691459.png"]
                        elif category_id == "04090263":
                            val = ["04090263.png", "04090263.png"]
                        elif category_id == "04256520":
                            val = ["04256520.png", "04256520.png"]
                        elif category_id == "04379243":
                            val = ["04379243.png", "04379243.png"]
                        elif category_id == "04401088":
                            val = ["04401088.png", "04401088.png"]
                        elif category_id == "04530566":
                            val = ["04530566.png", "04530566.png"]
                        else:
                            val = ["1.png", "1.png"]
                    else:
                        val = ["1.png", "1.png"]
                else:
                    val = ["1.png", "1.png"]
            
            #控制自参考的概率，后面训练尝试保存val的两个图像
            if random_p < self.hard_reference_probability:
                path_ref = val[1]  #hard reference
            else:
                path_ref = val[0] #easy reference 自身图像
            
            # 使用固定的参考图片路径，无论原始val值是什么
            if "/ShapeNet/image/" in image_path:
            #     # 从图像路径中提取类别ID
            #     pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/00\.png"
            #     match = re.search(pattern, image_path)
            #     if match:
            #         category_id = match.group(1)
            #
            #         # 根据类别ID选择对应的固定参考图像
            #         if category_id == "02691156":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/02691156.png"
            #         elif category_id == "02828884":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/02828884.png"
            #         elif category_id == "02933112":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/02933112.png"
            #         elif category_id == "02958343":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/02958343.png"
            #         elif category_id == "03001627":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/03001627.png"
            #         elif category_id == "03211117":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/03211117.png"
            #         elif category_id == "03636649":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/03636649.png"
            #         elif category_id == "03691459":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/03691459.png"
            #         elif category_id == "04090263":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/04090263.png"
            #         elif category_id == "04256520":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/04256520.png"
            #         elif category_id == "04379243":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/04379243.png"
            #         elif category_id == "04401088":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/04401088.png"
            #         elif category_id == "04530566":
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/04530566.png"
            #         else:
            #             # 如果是其他类别，使用默认参考图像
            #             path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
            #     else:
            #         # 如果无法提取类别ID，使用默认参考图像
            #         path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
                
                #使用不固定的参考图片
                ref_parts = path_ref.split('/')
                if len(ref_parts) == 2:  # 确保格式为"类别ID/模型ID"
                    category_id, model_id = ref_parts
                    # 构建完整参考图像路径
                    path_ref = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{category_id}/{model_id}/easy/00.png"
            ref_parts = path_ref.split('/')
            if len(ref_parts) == 2:  # 确保格式为"类别ID/模型ID"
                category_id, model_id = ref_parts
                # 构建完整参考图像路径
                path_ref = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{category_id}/{model_id}/easy/00.png"
                
                # 检查参考图像是否存在，如果不存在则使用备用模型ID
                if not os.path.exists(path_ref):
                    # 定义每个类别的备用模型ID
                    fallback_models = {
                        '02691156': '1a04e3eab45ca15dd86060f189eb133',  # airplane
                        '02958343': '1a0bc9ab92c915167ae33d942430658c',  # car
                        '03001627': '1a6f615e8b1b5ae4dbbc9440457e303e'   # chair
                    }
                    
                    if category_id in fallback_models:
                        fallback_model_id = fallback_models[category_id]
                        path_ref = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{category_id}/{fallback_model_id}/easy/00.png"
                        print(f"警告：参考图像不存在，使用备用参考图像 {path_ref}")
                        # 同时更新model_id用于后续草图路径构建
                        model_id = fallback_model_id
                    else:
                        print(f"错误：未知类别 {category_id}，使用默认参考图像")
                        path_ref = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
            elif self.opt.dataset_mode == 'deepfashion':
                path_ref = os.path.join(self.opt.dataroot, path_ref)
            else:
                path_ref = os.path.dirname(image_path).replace(self.train_test_folder[1], self.train_test_folder[0]) + '/' + path_ref
            
            #参考图像
            image_ref = Image.open(path_ref).convert('RGB')
            #print(f"输入的参考图像路径是: {path_ref}")
            
            # 为参考图片构建对应的草图路径
            if "/ShapeNet/image/" in image_path:
            #     # 对于ShapeNet数据集，使用固定的草图路径
            #     # 根据类别ID选择对应的固定参考草图
            #     pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/00\.png"
            #     match = re.search(pattern, image_path)
            #     if match:
            #         category_id = match.group(1)
            #
            #         # 根据类别ID选择对应的固定参考草图
            #         if category_id == "02691156":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02691156.png"
            #         elif category_id == "02828884":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02828884.png"
            #         elif category_id == "02933112":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02933112.png"
            #         elif category_id == "02958343":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02958343.png"
            #         elif category_id == "03001627":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/03001627.png"
            #         elif category_id == "03211117":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/03211117.png"
            #         elif category_id == "03636649":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/03636649.png"
            #         elif category_id == "03691459":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/03691459.png"
            #         elif category_id == "04090263":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/04090263.png"
            #         elif category_id == "04256520":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/04256520.png"
            #         elif category_id == "04379243":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/04379243.png"
            #         elif category_id == "04401088":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/04401088.png"
            #         elif category_id == "04530566":
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/04530566.png"
            #         else:
            #             # 如果是其他类别，使用默认参考草图
            #             path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/1.png"
            #     else:
            #         # 如果无法提取类别ID，使用默认参考草图
            #         path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/1.png"
                
                # 根据参考图像路径构建对应草图路径
                if len(ref_parts) == 2:  # 确保格式为"类别ID/模型ID"
                    # 构建草图路径（使用可能已更新的model_id）
                    path_ref_label = f"/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic/{category_id}/{model_id}/sketches/render_0.png"
                    
                    # 检查草图是否存在，如果不存在则使用默认草图
                    if not os.path.exists(path_ref_label):
                        print(f"警告：参考草图不存在 {path_ref_label}，使用默认草图")
                        if category_id == "02691156":
                            path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02691156.png"
                        elif category_id == "02958343":
                            path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/02958343.png"
                        elif category_id == "03001627":
                            path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/03001627.png"
                        else:
                            path_ref_label = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/sketches/1.png"
            elif self.opt.dataset_mode != 'deepfashion':
                path_ref_label = path_ref.replace('.jpg', '.png')
                #参考图像所对应的草图路径
                path_ref_label = self.imgpath_to_labelpath(path_ref_label)
            else: 
                path_ref_label = self.imgpath_to_labelpath(path_ref)
            #参考图像所对应的草图
            label_ref_tensor, params = self.get_label_tensor(path_ref_label)
            transform_image = get_transform(self.opt, params)
            #参考图像经过转换之后的张量
            ref_tensor = transform_image(image_ref)
            #print(f"输入的ref_tensor是: {ref_tensor}")
            #ref_tensor = self.reference_transform(image_ref)
            #与参考图像同形状的标记
            self_ref_flag = torch.zeros_like(ref_tensor)
        else:  # False
            pair = False
            if self.opt.dataset_mode == 'deepfashion' and self.opt.video_like:
                # if self.opt.hdfs:
                #     key = image_path.split('DeepFashion.zip@/')[-1]
                # else:
                #     key = image_path.split('DeepFashion/')[-1]
                key = image_path.replace('\\', '/').split('DeepFashion/')[-1]
                val = self.ref_dict[key]
                ref_name = val[0]
                key_name = key
                if os.path.dirname(ref_name) == os.path.dirname(key_name) and os.path.basename(ref_name).split('_')[0] == os.path.basename(key_name).split('_')[0]:
                    path_ref = os.path.join(self.opt.dataroot, ref_name)
                    image_ref = Image.open(path_ref).convert('RGB')
                    label_ref_path = self.imgpath_to_labelpath(path_ref)
                    label_ref_tensor, params = self.get_label_tensor(label_ref_path)
                    transform_image = get_transform(self.opt, params)
                    ref_tensor = transform_image(image_ref) 
                    pair = True
            if not pair:
                label_ref_tensor, params = self.get_label_tensor(label_path)
                transform_image = get_transform(self.opt, params)
                ref_tensor = transform_image(image)
            #ref_tensor = self.reference_transform(image)
            self_ref_flag = torch.ones_like(ref_tensor)

        input_dict = {'label': label_tensor,#草图
                      'image': image_tensor,#草图所对应的彩图
                      'path': image_path,#草图所对应的彩图的路径
                      'label_path': label_path,#草图路径（新增）
                      'self_ref': self_ref_flag,#与参考图像同形状的标记
                      'ref': ref_tensor,#参考图像经过转换之后的张量
                      'label_ref': label_ref_tensor #参考图像所对应的草图
                      }

        # Give subclasses a chance to modify the final output
        self.postprocess(input_dict)

        return input_dict

    def postprocess(self, input_dict):
        return input_dict

    def __len__(self):
        return self.dataset_size

    def get_ref(self, opt):
        pass

    def imgpath_to_labelpath(self, path):
        return path