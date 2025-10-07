import os
import numpy as np
from glob import glob
from tqdm import tqdm
import shutil

def process_pointcloud(input_path, output_base_path):
    # 确保输出目录存在
    os.makedirs(output_base_path, exist_ok=True)
    
    # 获取所有.npz文件
    npz_files = glob(os.path.join(input_path, "*/pointcloud.npz"))
    
    for npz_file in tqdm(npz_files, desc="Processing point clouds"):
        # 获取模型ID（从目录名中提取）
        model_id = os.path.basename(os.path.dirname(npz_file))
        
        # 读取点云数据
        with np.load(npz_file) as data:
            pointcloud = data['points']  # 假设点云数据存储在'points'键下
        
        # 创建输出目录
        output_dir = os.path.join(output_base_path, model_id)
        os.makedirs(output_dir, exist_ok=True)
        
        # 处理1024个点的版本
        if len(pointcloud) > 1024:
            indices = np.random.choice(len(pointcloud), 1024, replace=False)
            sampled_pointcloud = pointcloud[indices]
        else:
            indices = np.random.choice(len(pointcloud), 1024, replace=True)
            sampled_pointcloud = pointcloud[indices]
        
        output_file = os.path.join(output_dir, "pointcloud_1024.npy")
        np.save(output_file, sampled_pointcloud)
        
        # 处理2048个点的版本
        if len(pointcloud) > 2048:
            indices = np.random.choice(len(pointcloud), 2048, replace=False)
            sampled_pointcloud = pointcloud[indices]
        else:
            indices = np.random.choice(len(pointcloud), 2048, replace=True)
            sampled_pointcloud = pointcloud[indices]
        
        output_file = os.path.join(output_dir, "pointcloud_2048.npy")
        np.save(output_file, sampled_pointcloud)
        
        # 删除旧的2024点文件（如果存在）
        old_file = os.path.join(output_dir, "pointcloud_2024.npy")
        if os.path.exists(old_file):
            os.remove(old_file)

def main():
    # 设置输入和输出路径
    input_path = "/home/lab322/ourdata/yydw/dataset/dataset_small_v1.1/ShapeNet/03001627"
    output_base_path = "/home/lab322/ourdata/LMJ/dataset/ShapeNet/ShapeNet_pointclouds/03001627"
    
    # 处理所有点云文件
    process_pointcloud(input_path, output_base_path)
    
    print("处理完成！")
    print(f"总共处理了 {len(glob(os.path.join(output_base_path, '*', 'pointcloud_1024.npy')))} 个模型")

if __name__ == "__main__":
    main() 