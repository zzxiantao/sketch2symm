# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import cv2
import torch
import numpy as np
from PIL import Image
from skimage import feature
from data.pix2pix_dataset import Pix2pixDataset
from data.base_dataset import get_params, get_transform
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import re

class CelebAHQEdgeDataset(Pix2pixDataset):
    
    debug_counter = 0
    
    def initialize(self, opt):
        # 初始化原始路径映射字典
        self.original_path_map = {}
        super().initialize(opt)
    
    #hair, skin, l_brow, r_blow, l_eye, r_eye, l_ear, r_ear, nose, u_lip, mouth, l_lip, neck, 
    @staticmethod
    def modify_commandline_options(parser, is_train):
        parser = Pix2pixDataset.modify_commandline_options(parser, is_train)
        parser.set_defaults(preprocess_mode='resize_and_crop')
        parser.set_defaults(no_pairing_check=True)
        if is_train:
            parser.set_defaults(load_size=224)
        else:
            parser.set_defaults(load_size=224)
        parser.set_defaults(crop_size=224)
        parser.set_defaults(display_winsize=224)
        #parser.set_defaults(label_nc=15)
        parser.set_defaults(label_nc=1)
        parser.set_defaults(contain_dontcare_label=False)
        parser.set_defaults(cache_filelist_read=False)
        parser.set_defaults(cache_filelist_write=False)
        return parser

    def get_paths(self, opt):
        if opt.phase == 'train':
            fd = open(os.path.join(opt.dataroot, 'train.txt'))
            lines = fd.readlines()
            fd.close()
        elif opt.phase == 'test':
            # 检查是否使用imgtest数据集
            # if 'imgtest' in opt.dataroot:
            #     # 使用imgtest数据集
            #     #fd = open(os.path.join(opt.dataroot, 'val.txt'))
            #     fd = open('/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/split/shapenet_test_s2m.txt')
            #     lines = fd.readlines()
            #     fd.close()
                
            #     image_paths = []
            #     label_paths = []
            #     for i in range(len(lines)):
            #         img_id = lines[i].strip()
            #         # imgtest数据集路径
            #         image_paths.append(os.path.join(opt.dataroot, 'images', img_id + '.png'))
            #         label_paths.append(os.path.join(opt.dataroot, 'sketches', img_id + '.png'))
                
            #     return label_paths, image_paths
            # else:
                # 使用ShapeNet数据集
                fd = open('')
                # fd = open('/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/split/shapenet_test_thirteen.txt')
                lines = fd.readlines()
                fd.close()

                fallback_models = {
                    '02691156': ['1a04e3eab45ca15dd86060f189eb133', '10155655850468db78d106ce0a280f87', '1021a0914a7207aff927ed529ad90a11'],  # airplane
                    '02958343': ['48723bca810f80cf7c84d27684eb86f7', '1a0bc9ab92c915167ae33d942430658c', '4f31142fb24b4814ff1370452e3a0154'],  # car
                    '03001627': ['1a6f615e8b1b5ae4dbbc9440457e303e', '1006be65e7bc937e9141f9b58470d646', '1007e20d5e811b308351982a6e40cf41']   # chair
                }
                default_image_path = ""
                # default_image_path = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
                
                image_paths = []
                label_paths = []
                skipped_count = 0
                used_fallback_count = 0
                
                for i in range(len(lines)):
                    line = lines[i].strip()
                    if not line:
                        continue

                    sketch_path_key = line

                    path_parts = sketch_path_key.split('/')
                    if len(path_parts) != 2:
                        print(f"警告：路径格式错误 {line}")
                        continue
                    
                    category_id = path_parts[0]
                    model_id = path_parts[1]
                    
                    # 构建草图路径
                    label_path = os.path.join('',category_id, model_id, 'sketches', 'render_0.png')
                    # label_path = os.path.join('/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic',
                    #                         category_id, model_id, 'sketches', 'render_0.png')

                    image_path = None
                    
                    # 第1层：尝试原始彩图
                    original_image_path = os.path.join('', category_id, model_id, 'easy', '00.png')
                    # original_image_path = os.path.join('/home/lab322/ourdata/LMJ/dataset/ShapeNet/image',
                    #                                  category_id, model_id, 'easy', '00.png')
                    
                    # 记录原始路径信息（用于参考键生成）
                    original_ref_key = f"{category_id}/{model_id}"
                    
                    if os.path.exists(original_image_path):
                        # 原始彩图存在，使用原始路径
                        image_path = original_image_path
                        # 记录映射关系
                        self.original_path_map[image_path] = original_ref_key
                    else:
                        # 第2层：尝试多个备用模型ID
                        if category_id in fallback_models:
                            for fallback_id in fallback_models[category_id]:
                                fallback_image_path = os.path.join('', category_id, fallback_id, 'easy', '00.png')
                                # fallback_image_path = os.path.join('/home/lab322/ourdata/LMJ/dataset/ShapeNet/image',
                                #                                  category_id, fallback_id, 'easy', '00.png')
                                if os.path.exists(fallback_image_path):
                                    image_path = fallback_image_path
                                    # 记录映射关系：备用路径 -> 原始键
                                    self.original_path_map[image_path] = original_ref_key
                                    used_fallback_count += 1
                                    if used_fallback_count <= 10:
                                        print(f"警告：使用备用彩图 {category_id}/{fallback_id} 替代 {original_image_path}")
                                    break
                        
                        # 第3层：如果所有备用都失败，使用默认图像
                        if image_path is None:
                            if os.path.exists(default_image_path):
                                image_path = default_image_path
                                # 记录映射关系：默认路径 -> 原始键
                                self.original_path_map[image_path] = original_ref_key
                                if used_fallback_count <= 10:
                                    print(f"警告：所有备用彩图都不存在，使用默认图像：{category_id}/{model_id}")
                            else:
                                print(f"严重错误：连默认图像都不存在，跳过 {category_id}/{model_id}")
                                skipped_count += 1
                                continue

                    if not os.path.exists(label_path):
                        # 尝试同类别的其他草图作为备用
                        if category_id in fallback_models:
                            fallback_label_found = False
                            for fallback_id in fallback_models[category_id]:
                                fallback_label_path = os.path.join('',category_id, fallback_id, 'sketches', 'render_0.png')
                                # fallback_label_path = os.path.join('/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic',
                                #                                  category_id, fallback_id, 'sketches', 'render_0.png')
                                if os.path.exists(fallback_label_path):
                                    label_path = fallback_label_path
                                    fallback_label_found = True
                                    if used_fallback_count <= 10:
                                        print(f"警告：使用备用草图 {category_id}/{fallback_id} 替代原始草图")
                                    break
                            
                            if not fallback_label_found:
                                print(f"警告：草图和所有备用草图都不存在，跳过 {category_id}/{model_id}")
                                skipped_count += 1
                                continue
                        else:
                            print(f"警告：草图不存在且无备用策略，跳过 {category_id}/{model_id}")
                            skipped_count += 1
                            continue
                    
                    # 添加到最终列表
                    label_paths.append(label_path)
                    image_paths.append(image_path)
                
                print(f"数据集加载完成：成功 {len(label_paths)} 个，跳过 {skipped_count} 个，使用备用策略 {used_fallback_count} 个")
                return label_paths, image_paths
        else:
            fd = open(os.path.join(opt.dataroot, 'train.txt'))
            lines = fd.readlines()
            fd.close()
        
        image_paths = []
        label_paths = []
        for i in range(len(lines)):
            # 构建彩图路径
            image_paths.append(os.path.join('', lines[i].strip() + '/easy/00.png'))
            # image_paths.append(os.path.join('/home/lab322/ourdata/LMJ/dataset/ShapeNet/image', lines[i].strip() + '/easy/00.png'))

            parts = lines[i].strip().split('/')
            if len(parts) == 2:
                category_id = parts[0]
                model_id = parts[1]
                label_paths.append(
                    os.path.join('',
                                 category_id, model_id, 'sketches', 'render_0.png'))
                # label_paths.append(os.path.join('/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic',
                #                               category_id, model_id, 'sketches', 'render_0.png'))
            else:
                print(f"警告：路径格式不正确: {lines[i].strip()}")
        
        return label_paths, image_paths

    def get_ref(self, opt):
        extra = ''
        if opt.phase == 'test':
            extra = '_test'
        with open('./data/sketch2img_ref{}.txt'.format(extra)) as fd:
            lines = fd.readlines()
        ref_dict = {}
        for i in range(len(lines)):
            items = lines[i].strip().split(',')
            key = items[0]
            if opt.phase == 'test':
                val = items[1:]
            else:
                val = [items[1], items[-1]]
            ref_dict[key] = val
        train_test_folder = ('', '')
        return ref_dict, train_test_folder
    
    def get_ref_key(self, image_path):
        # 检查是否是imgtest数据集
        if "/imgtest/" in image_path:
            filename = os.path.basename(image_path)
            return filename
        
        # 对于ShapeNet数据集，我们需要基于草图路径生成键
        # 从图像路径获取对应的草图路径
        sketch_path = self.imgpath_to_labelpath(image_path)
        
        if "/sketch/Sketch2Model/shapenet-synthetic/" in sketch_path:
            # 从草图路径提取类别ID和模型ID
            pattern = ""
            # pattern = r"/sketch/Sketch2Model/shapenet-synthetic/([^/]+)/([^/]+)/sketches/render_0\.png"
            match = re.search(pattern, sketch_path)
            if match:
                category_id = match.group(1)
                model_id = match.group(2)
                key = f"{category_id}/{model_id}"
                return key

        if "/ShapeNet/image/" in image_path:
            pattern = r"/ShapeNet/image/([^/]+/[^/]+)/easy/00\.png"
            match = re.search(pattern, image_path)
            if match:
                key = match.group(1)
                return key
        
        # 如果都失败，使用默认方式
        return os.path.basename(image_path)

    def get_edges(self, edge, t):
        edge[:,1:] = edge[:,1:] | (t[:,1:] != t[:,:-1])
        edge[:,:-1] = edge[:,:-1] | (t[:,1:] != t[:,:-1])
        edge[1:,:] = edge[1:,:] | (t[1:,:] != t[:-1,:])
        edge[:-1,:] = edge[:-1,:] | (t[1:,:] != t[:-1,:])
        return edge


    def get_label_tensor(self, path):
        # 新草图路径构造
        sketch_path = path.replace("CelebAMask-HQ-mask-anno", "sketches").split('{')[0] 
        #print(sketch_path)
        img_path = self.labelpath_to_imgpath(path)
        
        # 强化的备用策略：检查彩色图像是否存在，如果不存在则使用多层备用机制
        if not os.path.exists(img_path):
            # 从草图路径中提取类别ID，使用对应的备用图像
            if "/sketch/Sketch2Model/shapenet-synthetic/" in path:
                pattern = ""
                # pattern = r"/sketch/Sketch2Model/shapenet-synthetic/([^/]+)/([^/]+)/sketches/render_0\.png"
                match = re.search(pattern, path)
                if match:
                    category_id = match.group(1)
                    
                    # 定义每个类别的多层备用模型ID
                    fallback_models = {
                        '02691156': ['1a04e3eab45ca15dd86060f189eb133', '10155655850468db78d106ce0a280f87', '1021a0914a7207aff927ed529ad90a11'],  # airplane
                        '02958343': ['48723bca810f80cf7c84d27684eb86f7', '1a0bc9ab92c915167ae33d942430658c', '4f31142fb24b4814ff1370452e3a0154'],  # car  
                        '03001627': ['1a6f615e8b1b5ae4dbbc9440457e303e', '1006be65e7bc937e9141f9b58470d646', '1007e20d5e811b308351982a6e40cf41']   # chair
                    }
                    
                    # 默认备用彩图
                    default_image_path = ""
                    # default_image_path = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
                    
                    # 多层备用策略
                    backup_img_path = None
                    
                    if category_id in fallback_models:
                        # 尝试多个备用模型ID
                        for fallback_id in fallback_models[category_id]:
                            fallback_img_path = ""
                            # fallback_img_path = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{category_id}/{fallback_id}/easy/00.png"
                            if os.path.exists(fallback_img_path):
                                backup_img_path = fallback_img_path
                                print(f"警告：在get_label_tensor中使用备用彩图 {backup_img_path}")
                                break
                    
                    # 如果所有备用都失败，使用默认图像
                    if backup_img_path is None:
                        if os.path.exists(default_image_path):
                            backup_img_path = default_image_path
                            print(f"警告：在get_label_tensor中使用默认图像 {default_image_path}")
                        else:
                            print(f"严重错误：在get_label_tensor中连默认图像都不存在，类别 {category_id}")
                            backup_img_path = None
                    
                    if backup_img_path:
                        img_path = backup_img_path
                else:
                    print(f"警告：无法解析草图路径 {path}")
                    # 使用默认图像
                    efault_image_path = ""
                    # default_image_path = "/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/imgs/imgtest/images/1.png"
                    if os.path.exists(default_image_path):
                        img_path = default_image_path
        
        # 最终保底：如果img_path仍然不存在
        if not os.path.exists(img_path):
            print(f"严重错误：所有备用策略都失败，为 {path} 创建图像")
            # 创建一个纯白图像
            temp_img = Image.new('RGB', (self.opt.load_size, self.opt.load_size), color='white')
            img = temp_img
        else:
            img = Image.open(img_path).resize((self.opt.load_size, self.opt.load_size), resample=Image.BILINEAR)
        
        params = get_params(self.opt, img.size)
        transform_label = get_transform(self.opt, params, method=Image.NEAREST, normalize=False)
        transform_img = get_transform(self.opt, params, method=Image.BILINEAR, normalize=False)
        sketch = Image.open(sketch_path).convert('L').resize((self.opt.load_size, self.opt.load_size), resample=Image.BILINEAR)
        
        # 明确只使用草图的第一个通道
        sketch_array = np.array(sketch)
        if len(sketch_array.shape) > 2:
            # 如果是多通道图像，只取第一个通道
            canny_edges = sketch_array[:, :, 0]
        else:
            # 如果已经是单通道，直接使用
            canny_edges = sketch_array
        
        canny_edges = canny_edges.astype(np.uint8)
       
        label_tensor = transform_label(Image.fromarray(canny_edges))
        
        return label_tensor, params
    
    def imgpath_to_labelpath(self, path):
        # 统一路径分隔符为Linux格式
        path = path.replace("\\", "/")
        
        # 优先检查是否有原始路径映射（用于备用策略）
        if path in self.original_path_map:
            original_key = self.original_path_map[path]
            # 从原始键构建草图路径
            parts = original_key.split('/')
            if len(parts) == 2:
                category_id, model_id = parts
                # sketch_path = f"/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic/{category_id}/{model_id}/sketches/render_0.png"
                sketch_path = ""
                return sketch_path
        
        # 处理ShapeNet特定的路径格式
        if "/ShapeNet/image/" in path:
            # 提取类别ID和模型ID
            pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/\d+\.png"
            match = re.search(pattern, path)
            if match:
                category_id = match.group(1)
                model_id = match.group(2)
                # 构建草图路径
                sketch_path = ""
                # sketch_path = f"/home/lab322/ourdata/LMJ/dataset/sketch/Sketch2Model/shapenet-synthetic/{category_id}/{model_id}/sketches/render_0.png"
                return sketch_path
        
        # 原有逻辑，处理其他类型的图像路径
        if "images/" in path:
            root, name = path.split("images/", 1)
        else:
            root, name = os.path.split(path)
        # 提取无扩展名的主文件名（兼容多层扩展名）
        base_name = os.path.splitext(name)[0]
        # 正确拼接标签路径
        label_path = os.path.join(root, "sketches", f"{base_name}.png")
        return label_path

    def labelpath_to_imgpath(self, path):
        # 统一路径分隔符
        path = path.replace("\\", "/")
        
        # 处理ShapeNet特定的草图路径格式
        if "/sketch/Sketch2Model/shapenet-synthetic/" in path:
            # 提取类别ID和模型ID
            pattern = r"/sketch/Sketch2Model/shapenet-synthetic/([^/]+)/([^/]+)/sketches/render_0\.png"
            match = re.search(pattern, path)
            if match:
                category_id = match.group(1)
                model_id = match.group(2)
                # 构建彩图路径
                img_path = ""
                # img_path = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{category_id}/{model_id}/easy/00.png"
                return img_path
        
        # 原有逻辑，处理其他类型的路径
        # 分割根目录（兼容带/不带末尾斜杠的情况）
        root = path.split("sketches/", 1)[0].rstrip("/")
        # 提取文件名并去除所有扩展名
        filename = os.path.splitext(os.path.basename(path))[0]
        # 正确拼接图像路径（使用.png扩展名）
        img_path = os.path.join(root, "images", f"{filename}.png")
        return img_path
