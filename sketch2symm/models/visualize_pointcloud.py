import os
import numpy as np
import open3d as o3d
from glob import glob
import random

def visualize_pointcloud(pointcloud_path):
    # 读取点云数据
    pointcloud = np.load(pointcloud_path)
    
    # 创建Open3D点云对象
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pointcloud)
    
    # 设置点云颜色（这里设置为蓝色）
    pcd.paint_uniform_color([0, 0, 1])
    
    # 创建可视化窗口
    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name=f"Point Cloud: {os.path.basename(os.path.dirname(pointcloud_path))}")
    
    # 添加点云到窗口
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
    
    # 运行可视化
    vis.run()
    vis.destroy_window()

def main():
    # 设置点云路径
    pointcloud_dir = "/home/lab322/ourdata/LMJ/dataset/ShapeNet/ShapeNet_pointclouds/03001627"
    
    # 获取所有点云文件
    pointcloud_files = glob(os.path.join(pointcloud_dir, "*", "pointcloud_2048.npy"))
    
    # 随机选择5个点云进行可视化
    selected_files = random.sample(pointcloud_files, min(5, len(pointcloud_files)))
    
    print("开始可视化点云...")
    print("使用鼠标可以：")
    print("1. 左键拖动：旋转视角")
    print("2. 右键拖动：平移视角")
    print("3. 滚轮：缩放")
    print("4. 按ESC或关闭窗口查看下一个点云")
    
    for file in selected_files:
        print(f"\n正在显示: {os.path.basename(os.path.dirname(file))}")
        visualize_pointcloud(file)

if __name__ == "__main__":
    main() 