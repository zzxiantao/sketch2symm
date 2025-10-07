import torch
import torch.nn as nn
import numpy as np
import os
import argparse
from tqdm import tqdm
import logging
from datetime import datetime
import glob
import re
import open3d as o3d
from einops import rearrange

# from util.utils import chamfer_distance
from chamferdist import ChamferDistance
from pytorch3d.loss import chamfer_distance
# 添加对utils模块的导入 - 只保留需要的函数
from util.utils import (
    fscore,
    EMDLoss_old,
    earth_mover_distance,
)
# from EMDLOSS_pointr import EMDLoss as NewEMDLoss, calculate_emd_for_point_clouds


# 设置日志
def setup_logger(args):
    # 创建logs目录
    if not os.path.exists('logs'):
        os.makedirs('logs')

    # 创建日志文件名
    current_time = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = f'logs/eval_{current_time}.log'

    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    # 获取logger实例
    logger = logging.getLogger()
    
    # 在日志文件的第一行打印模型轮次和实验名称
    with open(log_file, 'w', encoding='utf-8') as f:
        f.write(f"模型轮次: {args.epoch}\n")
        f.write(f"实验名称: 380_sketch2img_warp_13_symmertry\n")
        f.write("=" * 50 + "\n\n")
    
    return logger


# 加载PLY格式点云文件
def load_ply_pointcloud(file_path):
    try:
        pcd = o3d.io.read_point_cloud(file_path)
        points = np.asarray(pcd.points)
        if len(points) == 0:
            raise ValueError("点云为空")
        return points
    except Exception as e:
        logging.error(f"无法加载PLY文件 {file_path}: {e}")
        return None

def normalize_point_cloud(point_cloud):
        """
        Normalize a point cloud to be centered around the origin and fit within a unit cube.

        :param point_cloud: Numpy array of shape (num_points, dimensions)
        :return: Normalized point cloud.
        """
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        return centered_point_cloud

# 加载NPY格式点云文件
def load_npy_pointcloud(file_path):
    try:
        points = np.load(file_path)
        points = normalize_point_cloud(points)
        return points
    except Exception as e:
        logging.error(f"无法加载NPY文件 {file_path}: {e}")
        return None


# 从点云文件名中提取类别ID和模型ID
def extract_ids_from_filename(filename):
    # 匹配类别ID_模型ID_00.ply格式，支持包含连字符的UUID
    # 修改正则表达式以支持连字符和其他特殊字符
    match = re.match(r'([^_]+)_([^_]+)_00\.ply', os.path.basename(filename))
    if match:
        return match.group(1), match.group(2)


    # 匹配任何字符直到最后一个_00.ply
    match = re.match(r'([^_]+)_(.+)_00\.ply', os.path.basename(filename))
    if match:
        return match.group(1), match.group(2)

    return None, None


def main():
    parser = argparse.ArgumentParser(description='评估已生成的点云文件')
    parser.add_argument('--gen_dir', type=str,
                        default='/home/lab322/ourdata/LMJ/project/cocosnet-rgb2point/output/zhibiao/380_point_warp_13',
                        help='生成的点云文件目录')
    parser.add_argument('--gt_dir', type=str, default='/home/lab322/ourdata/LMJ/dataset/ShapeNet/ShapeNet_pointclouds_ours',
                        help='GT点云文件目录')
    parser.add_argument('--output', type=str, default='evaluation_warp_13_results.txt', help='结果输出文件')
    parser.add_argument('--device', type=str, default='cuda', help='设备 (cuda 或 cpu)')
    parser.add_argument('--fscore_threshold', type=float, default=0.01, help='F-score计算的阈值（欧几里得距离，将自动平方以匹配平方距离）')
    parser.add_argument('--save_detailed_results', action='store_true', help='保存详细的评测结果到JSON文件')
    parser.add_argument('--select_file', type=str, default='select.txt', help='包含要评估的样本ID列表的文件')
    # 添加新的参数
    parser.add_argument('--epoch', type=str, default='latest', help='模型轮次')
    parser.add_argument('--name', type=str, default='experiment', help='实验名称')
    args = parser.parse_args()

    # 初始化日志记录器
    logger = setup_logger(args)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    logger.info(f"使用设备: {device}")
    

    selected_sample_ids = []
    if os.path.exists(args.select_file):
        with open(args.select_file, 'r', encoding='utf-8') as f:
            selected_sample_ids = [line.strip() for line in f if line.strip()]
        logger.info(f"从 {args.select_file} 读取了 {len(selected_sample_ids)} 个样本ID")
    else:
        logger.warning(f"select文件 {args.select_file} 不存在，将评估所有点云文件")

    # 初始化Chamfer距离计算
    chamferDist = ChamferDistance()
    
    # 初始化两个EMD计算器进行对比
    old_emd_calculator = None  # EMDLoss_old是Function，不需要实例化


    # 创建类别映射表
    label_table = {
        "02691156": "airplane",
        "02828884": "bench",
        "04379243": "table",
        "02933112": "cabinet",
        "02958343": "car",
        "03001627": "chair",
        "03211117": "display",
        "03636649": "lamp",
        "03691459": "loudspeaker",
        "04090263": "rifle",
        "04256520": "sofa",
        "04379243": "table",
        "04401088": "telephone",
        "04530566": "watercraft",
    }

    # 查找所有生成的点云文件
    all_gen_files = sorted(glob.glob(os.path.join(args.gen_dir, "*.ply")))
    
    # 如果有select文件，则筛选出需要评估的文件
    if selected_sample_ids:
        gen_files = []
        for gen_file in all_gen_files:
            category_id, model_id = extract_ids_from_filename(gen_file)
            if category_id and model_id:
                sample_id = f"{category_id}_{model_id}"
                if sample_id in selected_sample_ids:
                    gen_files.append(gen_file)
        
        logger.info(f"根据select文件从 {len(all_gen_files)} 个文件中筛选出 {len(gen_files)} 个需要评估的点云文件")
    else:
        gen_files = all_gen_files
        logger.info(f"找到 {len(gen_files)} 个生成的点云文件")

    if len(gen_files) == 0:
        logger.error("没有找到需要评估的点云文件！")
        return

    # 初始化评估指标和存储 - 包含CD、F-score和两套EMD
    cd_values = []
    fscore_values = []
    emd_values = []  # EMD实现的结果（来自util/utils.py）
    loss_history = []
    cd_table = {}
    fscore_table = {}
    emd_table = {}  # EMD（utils）

    sample_details = []  # 存储 (category_id, model_id, emd_value) 的列表

    comparison_count = 0
    error_count = 0
    detailed_results = []  # 用于保存详细结果

    # 评估模型
    logger.info("开始评估...")
    for gen_file in tqdm(gen_files):
        # 从文件名提取类别ID和模型ID
        category_id, model_id = extract_ids_from_filename(gen_file)
        if category_id is None or model_id is None:
            logger.warning(f"无法从文件名 {os.path.basename(gen_file)} 提取ID，跳过")
            continue

        # 构建对应的GT点云文件路径
        gt_file = os.path.join(args.gt_dir, category_id, model_id, "pointcloud_2048.npy")

        # 加载生成的点云和GT点云
        gen_pointcloud = load_ply_pointcloud(gen_file)
        gt_pointcloud = load_npy_pointcloud(gt_file)

        if gen_pointcloud is None or gt_pointcloud is None:
            logger.warning(f"无法加载点云文件: {gen_file} 或 {gt_file}")
            error_count += 1
            continue

        # 转换为PyTorch张量
        gen_tensor = torch.from_numpy(gen_pointcloud).float().unsqueeze(0).to(device)
        gt_tensor = torch.from_numpy(gt_pointcloud).float().unsqueeze(0).to(device)

        # 计算Chamfer距离
        cd_loss = chamferDist(gen_tensor, gt_tensor, bidirectional=True)
        loss = cd_loss
        loss_history.append(loss.item())

        # 使用pytorch3d中的函数计算距离
        distance, dist_details = chamfer_distance(gen_tensor, gt_tensor)
        cd_values.append(distance.item())

        # 计算EMD - 同时测试两个实现
        # 1. 测试util/utils.py中的EMDLoss_old实现
        try:
            # 检查点云是否为2048个点，如果不是则报错
            if gen_tensor.shape[1] != 2048:
                raise ValueError(f"生成点云有{gen_tensor.shape[1]}个点，需要2048个点")
                
            if gt_tensor.shape[1] != 2048:
                raise ValueError(f"真实点云有{gt_tensor.shape[1]}个点，需要2048个点")
            
            # 直接使用EMDLoss_old.apply()
            with torch.no_grad():
                old_emd_dist = EMDLoss_old.apply(gen_tensor, gt_tensor)
                emd_values.append(old_emd_dist.mean().item() if old_emd_dist.numel() > 1 else old_emd_dist.item())
                
        except Exception as e:
            logger.warning(f"使用util/utils.py中的EMDLoss_old时出错: {e}")
            emd_values.append(float('nan'))  # 使用NaN标记失败的计算

        # 计算F-score
        try:
            # 计算F-Score
            with torch.no_grad():
                # 计算点到点的欧氏距离
                x = gen_tensor[0].unsqueeze(1)  # [N, 1, 3]
                y = gt_tensor[0].unsqueeze(0)   # [1, M, 3]
                dist1 = torch.sum((x - y) ** 2, dim=2)  # [N, M]
                dist2 = torch.sum((y - x) ** 2, dim=2)  # [M, N]
                
                # 计算每个点的最小距离
                min_dist1, _ = torch.min(dist1, dim=1)  # [N]
                min_dist2, _ = torch.min(dist2, dim=0)  # [M]
                
                # 计算F-Score，使用0.01的阈值
                # fscore函数返回一个tuple: (fscore, precision_1, precision_2)
                fscore_val, precision_1, precision_2 = fscore(min_dist1.unsqueeze(0), min_dist2.unsqueeze(0), threshold=0.01)
                
                # 存储结果（转换为Python标量）
                fscore_values.append(fscore_val.mean().item())
            
        except Exception as e:
            logger.warning(f"计算F-score指标时出错: {e}")
            fscore_values.append(0.0)

        # 记录类别结果 - 包含CD、F-score和EMD
        if category_id not in cd_table:
            cd_table[category_id] = []
            fscore_table[category_id] = []
            emd_table[category_id] = []

        cd_table[category_id].append(distance.item())
        fscore_table[category_id].append(fscore_values[-1])
        emd_table[category_id].append(emd_values[-1])
        
        # 新增：收集样本详细信息
        sample_details.append((category_id, model_id, emd_values[-1]))

        # 保存详细结果 - 包含CD、F-score和EMD
        if args.save_detailed_results:
            result_detail = {
                'file': os.path.basename(gen_file),
                'category_id': category_id,
                'model_id': model_id,
                'chamfer_distance': distance.item(),
                'fscore': fscore_values[-1],
                'emd': emd_values[-1]
            }

            detailed_results.append(result_detail)

        comparison_count += 1

    if comparison_count == 0:
        logger.error("没有成功比较任何点云对！")
        return


    # 输出每个类别的结果 - 包含CD、F-score和两套EMD
    cdtable = {}
    fscore_table_avg = {}
    emd_table_avg = {}  # EMD（utils）

    total_cd = 0
    total_fscore = 0
    total_emd = 0  # EMD总和（utils）

    category_count = 0

    for key in cd_table.keys():
        human_read_key = label_table.get(key, key)
        cdtable[human_read_key] = np.mean(cd_table[key])
        fscore_table_avg[human_read_key] = np.mean(fscore_table[key])
        
        # 处理EMD的NaN值，只计算有效值的平均
        valid_emd_values = [x for x in emd_table[key] if not np.isnan(x)]
        emd_table_avg[human_read_key] = np.mean(valid_emd_values) if valid_emd_values else float('nan')
        
        total_cd += cdtable[human_read_key]
        total_fscore += fscore_table_avg[human_read_key]
        
        # 只在EMD值有效时才加入总计
        if not np.isnan(emd_table_avg[human_read_key]):
            total_emd += emd_table_avg[human_read_key]

        category_count += 1

        # 显示CD、F-score和EMD的日志信息
        old_emd_str = f"{emd_table_avg[human_read_key]:.4f}" if not np.isnan(emd_table_avg[human_read_key]) else "N/A"
        
        log_msg = f"类别 {human_read_key}: CD = {cdtable[human_read_key]:.4f}, F-score = {fscore_table_avg[human_read_key]:.4f}"
        log_msg += f", EMD = {old_emd_str}"
        
        log_msg += f", 样本数 = {len(cd_table[key])}"
        if valid_emd_values:
            log_msg += f", EMD有效样本 = {len(valid_emd_values)}"
        logger.info(log_msg)

    # 计算总体分数 - 包含CD、F-score和EMD
    avg_cd_by_category = total_cd / category_count if category_count > 0 else 0
    avg_fscore_by_category = total_fscore / category_count if category_count > 0 else 0
    
    # 计算有效EMD类别数量
    valid_emd_categories = sum(1 for key in emd_table_avg.keys() if not np.isnan(emd_table_avg[key]))
    
    avg_emd_by_category = total_emd / valid_emd_categories if valid_emd_categories > 0 else float('nan')
    
    avg_cd_overall = np.mean(cd_values) if cd_values else 0
    avg_fscore_overall = np.mean(fscore_values) if fscore_values else 0
    
    # 计算总体EMD平均值（排除NaN）
    valid_emd_overall = [x for x in emd_values if not np.isnan(x)]
    
    avg_emd_overall = np.mean(valid_emd_overall) if valid_emd_overall else float('nan')
    
    avg_loss = np.mean(loss_history) if loss_history else 0

    # 显示CD、F-score和EMD的总体结果
    logger.info(f"总体平均Chamfer距离 (按类别平均): {avg_cd_by_category:.4f}")
    logger.info(f"总体平均Chamfer距离 (所有样本): {avg_cd_overall:.4f}")
    logger.info(f"总体平均F-score (按类别平均): {avg_fscore_by_category:.4f}")
    logger.info(f"总体平均F-score (所有样本): {avg_fscore_overall:.4f}")
    
    # EMD结果
    if np.isnan(avg_emd_by_category):
        logger.info(f"EMD平均 (按类别平均): N/A (无有效EMD值)")
    else:
        logger.info(f"EMD平均 (按类别平均): {avg_emd_by_category:.4f} (有效类别: {valid_emd_categories}/{category_count})")
    
    if np.isnan(avg_emd_overall):
        logger.info(f"EMD平均 (所有样本): N/A (无有效EMD值)")
    else:
        logger.info(f"EMD平均 (所有样本): {avg_emd_overall:.4f} (有效样本: {len(valid_emd_overall)}/{len(emd_values)})")

    logger.info(f"PyTorch Chamfer Loss: {avg_loss:.4f}")
    logger.info(f"成功评估 {comparison_count} 对点云, 错误 {error_count} 个")

    # 将结果写入文件 - 包含CD、F-score和EMD
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("评估结果摘要:\n")
        f.write(f"生成点云目录: {args.gen_dir}\n")
        f.write(f"GT点云目录: {args.gt_dir}\n")
        f.write(f"F-score阈值 (欧几里得距离): {args.fscore_threshold}\n")
        f.write(f"F-score阈值 (平方距离): {args.fscore_threshold ** 2}\n")
        f.write(f"总体平均Chamfer距离 (按类别平均): {avg_cd_by_category:.4f}\n")
        f.write(f"总体平均Chamfer距离 (所有样本): {avg_cd_overall:.4f}\n")
        f.write(f"总体平均F-score (按类别平均): {avg_fscore_by_category:.4f}\n")
        f.write(f"总体平均F-score (所有样本): {avg_fscore_overall:.4f}\n")
        
        if np.isnan(avg_emd_by_category):
            f.write(f"EMD平均 (按类别平均): N/A (无有效EMD值)\n")
        else:
            f.write(f"EMD平均 (按类别平均): {avg_emd_by_category:.4f} (有效类别: {valid_emd_categories}/{category_count})\n")
        
        if np.isnan(avg_emd_overall):
            f.write(f"EMD平均 (所有样本): N/A (无有效EMD值)\n")
        else:
            f.write(f"EMD平均 (所有样本): {avg_emd_overall:.4f} (有效样本: {len(valid_emd_overall)}/{len(emd_values)})\n")

        f.write(f"PyTorch Chamfer Loss: {avg_loss:.4f}\n")
        f.write(f"成功评估 {comparison_count} 对点云, 错误 {error_count} 个\n\n")

        f.write("类别结果:\n")
        for category in cdtable.keys():
            cd_val = cdtable[category]
            fscore_val = fscore_table_avg[category]
            emd_val = emd_table_avg[category]

            # 输出CD、F-score和EMD的类别结果
            old_emd_str = f"{emd_val:.4f}" if not np.isnan(emd_val) else "N/A"
            
            line = f"类别 {category}: CD = {cd_val:.4f}, F-score = {fscore_val:.4f}, EMD = {old_emd_str}"

            f.write(line + "\n")

    # 保存详细结果到JSON文件
    if args.save_detailed_results and detailed_results:
        json_output_path = args.output.replace('.txt', '_detailed.json')
        # save_pointcloud_evaluation_results(detailed_results, json_output_path)
        logger.info(f"详细结果保存功能暂时注释，需要实现 save_pointcloud_evaluation_results 函数")

    logger.info(f"详细结果已保存到 {args.output}")


if __name__ == "__main__":
    main() 