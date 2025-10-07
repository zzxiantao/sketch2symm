# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
from collections import OrderedDict
import torch
import open3d as o3d
import torchvision.utils as vutils
import torch.nn.functional as F
import data
import numpy as np
from util.util import masktorgb
from options.test_options import TestOptions
from models.pix2pix_model import Pix2PixModel
from models.model import PointCloudNet
from util.utils import predict
import re
import cv2
from PIL import Image
import time
from datetime import datetime

def render_point_cloud(ply_path, image_save_path):
    try:
        # 读取点云
        print(f"尝试读取点云文件: {ply_path}")
        pcd = o3d.io.read_point_cloud(ply_path)
        
        # 检查点云是否为空
        if len(np.asarray(pcd.points)) == 0:
            print("警告: 点云为空")
            # 尝试直接从文件读取
            try:
                data = np.loadtxt(ply_path, skiprows=7)  # 跳过PLY头部
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data)
                print(f"从文件直接读取了 {len(data)} 个点")
            except Exception as e:
                print(f"直接读取失败: {e}")
                return
        else:
            print(f"成功读取了 {len(np.asarray(pcd.points))} 个点")
        
        # 设置可视化窗口
        vis = o3d.visualization.Visualizer()
        vis.create_window(width=800, height=600, visible=True)  # 设置为可见
        vis.add_geometry(pcd)
        
        # 设置渲染选项
        opt = vis.get_render_option()
        opt.background_color = np.asarray([1, 1, 1])  # 白色背景
        opt.point_size = 2  # 点的大小
        
        # 设置视角
        ctr = vis.get_view_control()
        ctr.set_zoom(0.8)
        ctr.set_front([0, 0, -1])
        ctr.set_lookat([0, 0, 0])
        ctr.set_up([0, -1, 0])
        
        # 等待用户关闭窗口
        vis.run()
        
        # 保存当前视图
        vis.capture_screen_image(image_save_path)
        vis.destroy_window()
        print(f"点云渲染图片已保存到: {image_save_path}")
    except Exception as e:
        print(f"渲染点云时出错: {e}")


opt = TestOptions().parse()

# 打印当前使用的模型轮次
model_path = os.path.join(opt.checkpoints_dir, opt.name, f'{opt.which_epoch}_net_G.pth')
print(f"\n=== 加载的模型路径: {model_path} ===\n")

torch.manual_seed(0)
dataloader = data.create_dataloader(opt)
dataloader.dataset[0]

model = Pix2PixModel(opt)
model.eval()

# 计算模型参数量
def count_parameters(model):
    """计算模型参数量"""
    total_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        param_count = param.numel()
        total_params += param_count
        if param.requires_grad:
            trainable_params += param_count
    
    return total_params, trainable_params

# 打印草图生成彩图模型的参数量信息
sketch2img_total_params, sketch2img_trainable_params = count_parameters(model)
print(f"\n=== 草图生成彩图模型参数量统计 ===")
print(f"总参数量: {sketch2img_total_params:,} ({sketch2img_total_params/1e6:.2f}M)")
print(f"可训练参数量: {sketch2img_trainable_params:,} ({sketch2img_trainable_params/1e6:.2f}M)")
print(f"不可训练参数量: {sketch2img_total_params - sketch2img_trainable_params:,} ({(sketch2img_total_params - sketch2img_trainable_params)/1e6:.2f}M)")
print("=" * 50)

# "./output"
save_root = os.path.join(os.path.dirname(opt.checkpoints_dir), 'output')

# test

sample_counter = 0  # 全局样本计数器，确保文件名唯一

# 时间统计变量 - 草图生成彩图阶段
sketch2img_total_time = 0.0
sketch2img_samples = 0
sketch2img_start_time = time.time()
sketch2img_batch_times = []

print(f"\n=== 开始草图生成彩图阶段 ===")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"预计处理样本数: {opt.how_many}")
print("=" * 50)

for batch_idx, data_i in enumerate(dataloader):
    print('{} / {}'.format(batch_idx, len(dataloader)))
    if batch_idx * opt.batchSize >= opt.how_many:
        break
    imgs_num = data_i['label'].shape[0]
    #data_i['stage1'] = torch.ones_like(data_i['stage1'])
    
    # 记录草图生成彩图推理开始时间
    batch_start_time = time.time()
    
    out = model(data_i, mode='inference')
    
    # 记录草图生成彩图推理结束时间
    batch_end_time = time.time()
    batch_inference_time = batch_end_time - batch_start_time
    sketch2img_batch_times.append(batch_inference_time)
    sketch2img_total_time += batch_inference_time
    sketch2img_samples += imgs_num
    
    # 打印当前batch的时间统计
    avg_time_per_sample = batch_inference_time / imgs_num
    print(f"Batch {batch_idx}: {imgs_num} 样本, 推理时间: {batch_inference_time:.4f}s, 单样本平均: {avg_time_per_sample:.4f}s")
    if opt.save_per_img:  # True
        root = save_root + '/testscft'  # "./output/test_per_img/"
        if not os.path.exists(root + opt.name):
            os.makedirs(root + opt.name)
            
        imgs = out['fake_image'].data.cpu()
        #print(out['fake_image'].data.cpu())
        try:
            imgs = (imgs + 1) / 2
            for sample_idx in range(imgs.shape[0]):
                # 优先使用原始模型ID信息确保文件名唯一性
                if 'original_model_id' in data_i and sample_idx < len(data_i['original_model_id']):
                    # 使用原始模型ID信息生成唯一文件名
                    original_model_id = data_i['original_model_id'][sample_idx]
                    category_id, model_id = original_model_id.split('/')
                    name = f"{category_id}_{model_id}_00.png"
                    print(f"使用原始模型ID: {original_model_id} -> {name}")
                else:
                    # 回退到原有逻辑
                    orig_path = data_i['path'][sample_idx]
                    
                    # 从路径中提取类别和ID，确保每个样本都有唯一的文件名
                    if "/ShapeNet/image/" in orig_path:
                        # 使用正则表达式提取类别和ID
                        pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/00\.png"
                        match = re.search(pattern, orig_path)
                        if match:
                            category_id = match.group(1)
                            model_id = match.group(2)
                            # 构建新的文件名格式：类别_ID_00.png
                            name = f"{category_id}_{model_id}_00.png"
                        else:
                            # 如果无法匹配，使用全局计数器确保唯一性
                            print(f"警告：无法解析路径格式 {orig_path}，使用备用命名")
                            name = f"unknown_sample{sample_counter:04d}_00.png"
                    elif "/imgtest/" in orig_path:
                        # 对于imgtest数据集，直接使用原始文件名
                        name = os.path.basename(orig_path)
                    else:
                        # 对于其他路径，使用全局计数器确保唯一性
                        print(f"警告：未识别的路径格式 {orig_path}，使用备用命名")
                        name = f"other_sample{sample_counter:04d}_00.png"
                
                # 增加全局计数器
                sample_counter += 1
                
                # 直接保存原始生成的图像
                final_save_path = root + opt.name + '/' + name
                vutils.save_image(imgs[sample_idx:sample_idx+1], final_save_path,  
                        nrow=1, padding=0, normalize=False)
                
        except OSError as err:
            print(err)
    else:  # False
        if not os.path.exists(save_root + '/test_three/' + opt.name):
            os.makedirs(save_root + '/test_three/' + opt.name)
        
        label = data_i['label'].expand(-1, 3, -1, -1).float()
        imgs = torch.cat((label.cpu(), data_i['ref'].cpu(), out['fake_image'].data.cpu()), 0)
        try:
            imgs = (imgs + 1) / 2
            
            # 从路径中提取类别和ID来构建文件名
            # 优先使用原始模型ID信息
            if 'original_model_id' in data_i and len(data_i['original_model_id']) > 0:
                # 使用第一个样本的原始模型ID信息
                original_model_id = data_i['original_model_id'][0]
                category_id, model_id = original_model_id.split('/')
                filename = f"{category_id}_{model_id}.png"
                print(f"test_three使用原始模型ID: {original_model_id} -> {filename}")
            else:
                # 回退到原有逻辑
                orig_path = data_i['path'][0]  # 取第一个路径作为代表
                if "/ShapeNet/image/" in orig_path:
                    # 使用正则表达式提取类别和ID
                    pattern = r"/ShapeNet/image/([^/]+)/([^/]+)/easy/00\.png"
                    match = re.search(pattern, orig_path)
                    if match:
                        category_id = match.group(1)
                        model_id = match.group(2)
                        # 构建新的文件名格式：类别_ID.png
                        filename = f"{category_id}_{model_id}.png"
                    else:
                        # 如果无法匹配，使用batch索引确保唯一性
                        filename = f"unknown_batch{batch_idx:04d}.png"
                elif "/imgtest/" in orig_path:
                    # 对于imgtest数据集，使用原始文件名（去掉扩展名后加.png）
                    base_name = os.path.splitext(os.path.basename(orig_path))[0]
                    filename = f"{base_name}.png"
                else:
                    # 对于其他路径，使用batch索引确保唯一性
                    filename = f"other_batch{batch_idx:04d}.png"
            
            vutils.save_image(imgs, save_root + '/test_three/' + opt.name + '/' + filename,
                    nrow=imgs_num, padding=0, normalize=False)
        except OSError as err:
            print(err)

# 计算草图生成彩图阶段的时间统计
sketch2img_end_time = time.time()
sketch2img_total_elapsed = sketch2img_end_time - sketch2img_start_time

print("\n" + "=" * 60)
print("=== 草图生成彩图阶段统计报告 ===")
print("=" * 60)
print(f"结束时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"总处理时间: {sketch2img_total_elapsed:.4f}s ({sketch2img_total_elapsed/60:.2f}分钟)")
print(f"总推理时间: {sketch2img_total_time:.4f}s ({sketch2img_total_time/60:.2f}分钟)")
print(f"总处理样本数: {sketch2img_samples}")
print(f"总batch数: {len(sketch2img_batch_times)}")

if sketch2img_samples > 0:
    print(f"\n--- 单样本时间统计 ---")
    print(f"单样本平均推理时间: {sketch2img_total_time/sketch2img_samples:.4f}s")
    print(f"单样本平均总时间: {sketch2img_total_elapsed/sketch2img_samples:.4f}s")
    print(f"推理效率: {sketch2img_total_time/sketch2img_total_elapsed*100:.2f}%")
    
    # 计算吞吐量
    samples_per_second = sketch2img_samples / sketch2img_total_time
    print(f"推理吞吐量: {samples_per_second:.2f} 样本/秒")

print("=" * 60)
print("草图生成彩图阶段完成！")

#遍历全部test_per_img中的图片
#使用非对称模型
# model_save_name = "checkpoints/(three_2048)best_model_epoch.pth"
model_save_name = "checkpoints/(thirteen_2048)best_model_symmertry_epoch.pth"

print(f"\n=== 开始彩图生成点云阶段 ===")
print(f"开始时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 50)

# 加载点云生成模型
pointcloud_model = PointCloudNet(num_views=1, point_cloud_size=2048, num_heads=4, dim_feedforward=2048, use_symmetry=True)
pointcloud_model.load_state_dict(torch.load(model_save_name)["model"])
pointcloud_model = pointcloud_model.cuda()  # 将模型移动到GPU
pointcloud_model.eval()

# 打印彩图生成点云模型的参数量信息
pointcloud_total_params, pointcloud_trainable_params = count_parameters(pointcloud_model)
print(f"\n=== 彩图生成点云模型参数量统计 ===")
print(f"总参数量: {pointcloud_total_params:,} ({pointcloud_total_params/1e6:.2f}M)")
print(f"可训练参数量: {pointcloud_trainable_params:,} ({pointcloud_trainable_params/1e6:.2f}M)")
print(f"不可训练参数量: {pointcloud_total_params - pointcloud_trainable_params:,} ({(pointcloud_total_params - pointcloud_trainable_params)/1e6:.2f}M)")

# 计算两阶段总参数量
total_combined_params = sketch2img_total_params + pointcloud_total_params
total_combined_trainable = sketch2img_trainable_params + pointcloud_trainable_params
print(f"\n=== 两阶段模型总参数量统计 ===")
print(f"草图生成彩图模型: {sketch2img_total_params:,} ({sketch2img_total_params/1e6:.2f}M)")
print(f"彩图生成点云模型: {pointcloud_total_params:,} ({pointcloud_total_params/1e6:.2f}M)")
print(f"两阶段总参数量: {total_combined_params:,} ({total_combined_params/1e6:.2f}M)")
print(f"两阶段可训练参数: {total_combined_trainable:,} ({total_combined_trainable/1e6:.2f}M)")
print("=" * 50)  

# 使用生成的图像生成点云
test_img_dir = save_root + '/testscft' + opt.name
point_cloud_dir = save_root + '/testscft'

# 确保点云保存目录存在
os.makedirs(point_cloud_dir, exist_ok=True)

print(f"正在处理 {test_img_dir} 中的图像...")
print(f"将生成的点云保存到 {point_cloud_dir}")

# 获取目录中的所有图像文件
if os.path.exists(test_img_dir):
    image_files = [f for f in os.listdir(test_img_dir) 
                   if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    for img_file in image_files:
        # 构建完整图像路径
        image_path = os.path.join(test_img_dir, img_file)
        
        # 根据原始图像名称创建点云文件名
        file_name = os.path.splitext(img_file)[0]  # 去掉文件扩展名
        save_path = os.path.join(point_cloud_dir, f"{file_name}.ply")
        
        print(f"处理图像: {image_path}")
        print(f"生成点云: {save_path}")
        
        # 记录单张图像点云生成开始时间
        single_start_time = time.time()
        
        # 生成点云
        predict(pointcloud_model, image_path, save_path)
        
        # 记录单张图像点云生成结束时间
        single_end_time = time.time()
        single_time = single_end_time - single_start_time
        print(f"已完成 {img_file} 的处理，耗时: {single_time:.4f}s")
        
        # 渲染点云并保存图片
        # render_image_path = save_path.replace('.ply', '_render.png')
        # render_point_cloud(save_path, render_image_path)
        
    print(f"所有图像处理完成，共处理了 {len(image_files)} 个文件")
    
    # 打印两阶段总体统计
    print("\n" + "=" * 60)
    print("=== 两阶段总体统计报告 ===")
    print("=" * 60)
    print(f"草图生成彩图阶段:")
    print(f"  - 样本数: {sketch2img_samples}")
    print(f"  - 推理时间: {sketch2img_total_time:.4f}s")
    print(f"  - 单样本平均: {sketch2img_total_time/sketch2img_samples:.4f}s" if sketch2img_samples > 0 else "  - 单样本平均: N/A")
    
    print(f"彩图生成点云阶段:")
    print(f"  - 样本数: {len(image_files)}")
    print(f"  - 单样本平均: 见上方各样本耗时")
    
    print(f"\n--- 模型参数量总结 ---")
    print(f"草图生成彩图模型: {sketch2img_total_params:,} ({sketch2img_total_params/1e6:.2f}M)")
    print(f"彩图生成点云模型: {pointcloud_total_params:,} ({pointcloud_total_params/1e6:.2f}M)")
    print(f"两阶段总参数量: {total_combined_params:,} ({total_combined_params/1e6:.2f}M)")
    
    print(f"\n--- 设备信息 ---")
    print(f"设备: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"GPU内存使用: {torch.cuda.max_memory_allocated()/1024**3:.2f}GB")
        print(f"GPU内存缓存: {torch.cuda.max_memory_reserved()/1024**3:.2f}GB")
    
    print("=" * 60)
    print("两阶段处理完成！")
    
else:
    print(f"目录不存在: {test_img_dir}")
    print("无法找到图像目录进行点云生成")
