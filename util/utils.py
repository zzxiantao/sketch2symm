from torch.utils.data import Dataset
from glob import glob
import os
import torch.cuda as cuda
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch.nn as nn
from PIL import Image
from geomloss import SamplesLoss
import open3d as o3d
import torch
import emd_cuda

class PCDataset(Dataset):
    def __init__(self, stage, transform=None):
        self.transform = transform
        self.stage = stage

        if stage == "train":
            image_paths = f"split/shapenet_train.txt"
        elif stage == "test":
            image_paths = f"split/shapenet_test.txt"

        with open(image_paths) as caption_file:
            self.filenames = caption_file.readlines()

        self.numbers_list = [f"{i:02}" for i in range(24)]

        labels = []
        category = set()
        for f in self.filenames:
            attr = f.split("/")
            labels.append(attr[1].strip())
            category.add(attr[0])

        category = list(category)
        self.labels = []
        self.data = []

        # for c in ["02958343", "02691156", "03001627"]:
        for c in ["03001627"]:
            for label in labels:
                # volume_path = f"C:\\Users\\appro\\Documents\\ShapeNet\\ShapeNet_pointclouds\\{c}\\{label}\\pointcloud_1024.npy"
                volume_path = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/ShapeNet_pointclouds/{c}/{label}/pointcloud_2048.npy"
                # volume_path = f"/home/lab322/ourdata/yydw/dataset/dataset_small_v1.1/ShapeNet/{c}/{label}/pointcloud.npz"
                files = glob(
                    # f"C:\\Users\\appro\\Documents\\ShapeNet\\ShapeNetRendering\\{c}\\{label}\\rendering\\*.png"
                    f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{c}/{label}/easy/*.png"
                )
                for file in files:
                    if self.stage == "train":
                        if os.path.exists(volume_path):
                            self.data.append([c, label, file])

                if self.stage == "test":
                    if os.path.exists(volume_path) and len(files) > 1:
                        # test_image_path = f"C:\\Users\\appro\\Documents\\ShapeNet\\ShapeNetRendering\\{c}\\{label}\\rendering\\00.png"
                        test_image_path = f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/image/{c}/{label}/easy/00.png"
                        self.data.append([c, label, test_image_path])

    def __len__(self):
        return len(self.data)

    def normalize_point_cloud(self, point_cloud):
        """
        Normalize a point cloud to be centered around the origin and fit within a unit cube.

        :param point_cloud: Numpy array of shape (num_points, dimensions)
        :return: Normalized point cloud.
        """
        centroid = np.mean(point_cloud, axis=0)
        centered_point_cloud = point_cloud - centroid
        if self.stage == "train":
            np.random.shuffle(centered_point_cloud)
        return centered_point_cloud

    def __getitem__(self, idx):
        data = self.data[idx]
        category = data[0]
        label = data[1]
        image = data[2]

        image_files = [image]
        pc = np.load(
            # f"C:\\Users\\appro\\Documents\\ShapeNet\\ShapeNet_pointclouds\\{category}\\{label}\\pointcloud_1024.npy"
            f"/home/lab322/ourdata/LMJ/dataset/ShapeNet/ShapeNet_pointclouds/{category}/{label}/pointcloud_2048.npy"
            # f"/home/lab322/ourdata/yydw/dataset/dataset_small_v1.1/ShapeNet/{category}/{label}/pointcloud.npz"
        )
        pc = self.normalize_point_cloud(pc)

        images = []
        for filename in image_files:
            image = Image.open(filename).convert("RGB")
            if self.transform:
                image = self.transform(image)
            images.append(image)

        name = f"{category}_{label}"
        images_tensor = torch.stack(images, dim=0)

        return images_tensor, torch.as_tensor(pc, dtype=torch.float32), name


def chamfer_distance(x, y, metric="l2", direction="bi"):
    """Chamfer distance between two point clouds

    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default 'l2'
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """

    if direction == "y_to_x":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
    elif direction == "x_to_y":
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
    elif direction == "bi":
        x_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(
            n_neighbors=1, leaf_size=1, algorithm="kd_tree", metric=metric
        ).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: 'y_x', 'x_y', 'bi'")

    return chamfer_dist


def fscore(dist1, dist2, threshold=0.01):
    """
    Calculates the F-score between two point clouds with the corresponding threshold value.
    :param dist1: Batch, N-Points
    :param dist2: Batch, N-Points
    :param th: float
    :return: fscore, precision, recall
    """
    # NB : In this depo, dist1 and dist2 are squared pointcloud euclidean distances, so you should adapt the threshold accordingly.
    precision_1 = torch.mean((dist1 < threshold).float(), dim=1)
    precision_2 = torch.mean((dist2 < threshold).float(), dim=1)
    fscore = 2 * precision_1 * precision_2 / (precision_1 + precision_2)
    fscore[torch.isnan(fscore)] = 0
    return fscore, precision_1, precision_2

#1-nn emd 3dqd
class EMDLoss_old(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        xyz1 = xyz1.contiguous()
        xyz2 = xyz2.contiguous()
        assert xyz1.is_cuda and xyz2.is_cuda, "Only support cuda currently."
        match = emd_cuda.approxmatch_forward(xyz1, xyz2)
        cost = emd_cuda.matchcost_forward(xyz1, xyz2, match)
        ctx.save_for_backward(xyz1, xyz2, match)
        return cost

    @staticmethod
    def backward(ctx, grad_cost):
        xyz1, xyz2, match = ctx.saved_tensors
        grad_cost = grad_cost.contiguous()
        grad_xyz1, grad_xyz2 = emd_cuda.matchcost_backward(grad_cost, xyz1, xyz2, match)
        print('------2------')
        return grad_xyz1, grad_xyz2
       


def earth_mover_distance(xyz1, xyz2, transpose=True):
    """Earth Mover Distance (Approx)

    Args:
        xyz1 (torch.Tensor): (b, 3, n1)
        xyz2 (torch.Tensor): (b, 3, n1)
        transpose (bool): whether to transpose inputs as it might be BCN format.
            Extensions only support BNC format.

    Returns:
        cost (torch.Tensor): (b)

    """
    if xyz1.dim() == 2:
        xyz1 = xyz1.unsqueeze(0)
    if xyz2.dim() == 2:
        xyz2 = xyz2.unsqueeze(0)
    if transpose:
        xyz1 = xyz1.transpose(1, 2)
        xyz2 = xyz2.transpose(1, 2)
    cost = EMDLoss_old.apply(xyz1, xyz2)
    return cost


#rgb emd
# class EMDLoss(nn.Module):
#     def __init__(self):
#         super(EMDLoss, self).__init__()

#     def forward(self, pred, target):
#         # pred and target are expected to have shape (batch_size, 2048, 3)
#         assert pred.shape == target.shape
#         assert pred.shape[1] == 2048 and pred.shape[2] == 3

#         batch_size = pred.shape[0]
#         num_points = pred.shape[1]

#         # Compute pairwise distances between all points
#         diff = pred.unsqueeze(2) - target.unsqueeze(1)
#         dist = torch.sum(diff**2, dim=-1)

#         # Solve the assignment problem using Hungarian algorithm
#         # Note: This is a simplified version and may not be the most efficient for large point clouds
#         assignment = torch.zeros_like(dist)
#         for b in range(batch_size):
#             _, indices = torch.topk(dist[b], k=num_points, largest=False, dim=1)
#             assignment[b] = torch.scatter(assignment[b], 1, indices, 1)

#         # Compute the EMD
#         emd = torch.sum(dist * assignment, dim=[1, 2]) / num_points

#         return emd.mean()


def iou(pred_points, gt_points, voxel_size=0.1, grid_size=64):
    """
    计算两个点云之间的IoU（Intersection over Union）
    通过将点云转换为体素网格来计算

    Args:
        pred_points: 预测点云 (batch_size, num_points, 3)
        gt_points: 真实点云 (batch_size, num_points, 3)
        voxel_size: 体素大小
        grid_size: 网格大小 (grid_size x grid_size x grid_size)

    Returns:
        IoU值 (batch_size,) 或 标量
    """
    device = pred_points.device
    batch_size = pred_points.shape[0]

    # 修复：将两个点云合并后一起计算全局边界框进行归一化
    def normalize_points_together(pred_pts, gt_pts):
        # 合并两个点云来计算全局边界框
        all_points = torch.cat([pred_pts, gt_pts], dim=1)  # (batch_size, num_points*2, 3)

        # 找到全局边界框
        min_coords = torch.min(all_points, dim=1, keepdim=True)[0]  # (batch_size, 1, 3)
        max_coords = torch.max(all_points, dim=1, keepdim=True)[0]  # (batch_size, 1, 3)

        # 计算范围，添加小的padding避免边界问题
        range_coords = max_coords - min_coords
        range_coords = torch.clamp(range_coords, min=1e-6)

        # 添加5%的padding
        padding = range_coords * 0.05
        min_coords = min_coords - padding
        range_coords = range_coords + 2 * padding

        # 基于全局边界框归一化两个点云
        pred_normalized = (pred_pts - min_coords) / range_coords
        gt_normalized = (gt_pts - min_coords) / range_coords

        # 缩放到[0, grid_size-1]
        pred_scaled = pred_normalized * (grid_size - 1)
        gt_scaled = gt_normalized * (grid_size - 1)

        # 确保坐标在有效范围内
        pred_scaled = torch.clamp(pred_scaled, 0, grid_size - 1).long()
        gt_scaled = torch.clamp(gt_scaled, 0, grid_size - 1).long()

        return pred_scaled, gt_scaled

    pred_voxel_coords, gt_voxel_coords = normalize_points_together(pred_points, gt_points)

    iou_values = []

    for b in range(batch_size):
        # 创建体素网格
        pred_grid = torch.zeros(grid_size, grid_size, grid_size, device=device, dtype=torch.bool)
        gt_grid = torch.zeros(grid_size, grid_size, grid_size, device=device, dtype=torch.bool)

        # 标记预测点云占用的体素
        pred_coords = pred_voxel_coords[b]
        # 移除重复的体素坐标
        pred_coords_unique = torch.unique(pred_coords, dim=0)
        if len(pred_coords_unique) > 0:
            pred_grid[pred_coords_unique[:, 0], pred_coords_unique[:, 1], pred_coords_unique[:, 2]] = True

        # 标记真实点云占用的体素
        gt_coords = gt_voxel_coords[b]
        # 移除重复的体素坐标
        gt_coords_unique = torch.unique(gt_coords, dim=0)
        if len(gt_coords_unique) > 0:
            gt_grid[gt_coords_unique[:, 0], gt_coords_unique[:, 1], gt_coords_unique[:, 2]] = True

        # 计算交集和并集
        intersection = torch.logical_and(pred_grid, gt_grid)
        union = torch.logical_or(pred_grid, gt_grid)

        # 计算IoU
        intersection_count = intersection.sum().float()
        union_count = union.sum().float()

        if union_count > 0:
            iou_val = intersection_count / union_count
        else:
            iou_val = torch.tensor(0.0, device=device)  # 修改：如果都为空，IoU应该是0而不是1

        iou_values.append(iou_val)

    # 返回批次的IoU值
    if batch_size == 1:
        return iou_values[0]
    else:
        return torch.stack(iou_values)


def export_to_ply(point_cloud, filename):
    """
    Export a point cloud to a PLY file using numpy.
    :param point_cloud: Numpy array or PyTorch tensor of shape (num_points, 3) representing the point cloud.
    :param filename: String, the name of the file to save the point cloud to.
    """
    # 如果是PyTorch张量，先转换为numpy数组
    if isinstance(point_cloud, torch.Tensor):
        point_cloud = point_cloud.detach().cpu().numpy()

    # 打印调试信息
    print(f"Point cloud shape: {point_cloud.shape}")
    print(f"Point cloud dtype: {point_cloud.dtype}")

    # 确保点云是正确的形状
    if len(point_cloud.shape) != 2 or point_cloud.shape[1] != 3:
        print(f"Warning: Expected point cloud shape (N, 3), got {point_cloud.shape}")
        if len(point_cloud.shape) == 3 and point_cloud.shape[0] == 1:
            point_cloud = point_cloud[0]  # 取第一个批次
            print(f"Reshaped to {point_cloud.shape}")

    # 使用numpy直接写PLY文件
    try:
        with open(filename, 'w') as f:
            # 写PLY头部
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {point_cloud.shape[0]}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("end_header\n")

            # 写点云数据
            for point in point_cloud:
                f.write(f"{point[0]} {point[1]} {point[2]}\n")

        print(f"Successfully saved point cloud to {filename}")
    except Exception as e:
        print(f"Error saving point cloud: {e}")
        # 尝试保存为其他格式
        try:
            np.savetxt(filename.replace('.ply', '.xyz'), point_cloud)
            print(f"Saved as XYZ file instead: {filename.replace('.ply', '.xyz')}")
        except Exception as e2:
            print(f"Failed to save as XYZ file: {e2}")


from torchvision import transforms
from PIL import Image


def predict(model, image_path, save_path):
    # Define the transformations
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )
    # Load the image
    image = Image.open(image_path).convert("RGB")

    # Apply the transformations
    input_tensor = transform(image)
    input_tensor = input_tensor.reshape(1, 1, 3, 224, 224)
    input_tensor = input_tensor.cuda()  # 将输入数据移动到GPU

    # Invoke the model
    with torch.no_grad():  # Disable gradient computation for inference
        output = model(input_tensor)

    # 打印模型输出信息，安全地处理None值
    print(f"Model output type: {type(output)}")
    print(f"Model output length: {len(output)}")
    for i, out in enumerate(output):
        if out is not None:
            print(f"Output {i} shape: {out.shape}, type: {type(out)}")
        else:
            print(f"Output {i}: None")

    # 使用第一个输出（原始点云）
    export_to_ply(output[0], save_path)
    print(f"Image from {image_path} saved to {save_path}")

