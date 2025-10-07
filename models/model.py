import torch.nn as nn
from torch.nn import MultiheadAttention
import torch
import timm

def get_reflection_operator(n_pl):
    """ The reflection operator is parametrized by the normal vector
    of the plane of symmetry passing through the origin. """
    norm_npl = torch.norm(n_pl, 2)
    n_x = n_pl[0, 0] / norm_npl
    n_y = torch.tensor(0.0).cuda()
    # n_y = torch.tensor(0.0, device=n_pl.device)
    n_z = n_pl[0, 1] / norm_npl
    refl_mat = torch.stack(
        [
            1 - 2 * n_x * n_x,
            -2 * n_x * n_y,
            -2 * n_x * n_z,
            -2 * n_x * n_y,
            1 - 2 * n_y * n_y,
            -2 * n_y * n_z,
            -2 * n_x * n_z,
            -2 * n_y * n_z,
            1 - 2 * n_z * n_z,
        ],
        dim=0,
    ).reshape(1, 3, 3)
    return refl_mat

def symmetry_pc(x, refl_mat):
    """ Apply reflection transformation to point cloud """
    refl_batch = refl_mat.repeat(x.shape[0], 1, 1)  # [B, 3, 3]
    x = x.permute(0, 2, 1)  # [B, 3, N]
    symmetry_x = torch.matmul(refl_batch, x)  # [B, 3, N]
    symmetry_x = symmetry_x.permute(0, 2, 1)  # [B, N, 3]
    return symmetry_x

class PointCloudGeneratorWithAttention(nn.Module):
    def __init__(
        self, input_feature_dim, point_cloud_size, num_heads=16, dim_feedforward=2048
    ):
        super(PointCloudGeneratorWithAttention, self).__init__()
        print(f"input_feature_dim:{input_feature_dim}")
        print(f"dim_feedforward:{dim_feedforward}")
        print(f"point_cloud_size:{point_cloud_size*3}")
        self.self_attention = MultiheadAttention(
            embed_dim=input_feature_dim, num_heads=num_heads
        )
        self.linear_layers = nn.Sequential(
            nn.Linear(input_feature_dim, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(dim_feedforward, dim_feedforward),
            nn.LeakyReLU(0.2),
            nn.Linear(
                dim_feedforward, point_cloud_size * 3
            ),  # Output layer for point cloud
        )
        self.point_cloud_size = point_cloud_size

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_feature_dim]
        # Transpose for the attention layer
        x = x.transpose(0, 1)  # Shape: [seq_length, batch_size, input_feature_dim]

        # Self-attention
        attn_output, _ = self.self_attention(x, x, x)

        # Transpose back
        attn_output = attn_output.transpose(
            0, 1
        )  # Shape: [batch_size, seq_length, input_feature_dim]

        # Pass through the linear layers
        point_cloud = self.linear_layers(attn_output.flatten(start_dim=1))

        # Reshape to (batch_size, point_cloud_size, 3)
        point_cloud = point_cloud.view(-1, self.point_cloud_size, 3)
        return point_cloud

class SymmetryPredictor(nn.Module):
    """ Predict symmetry plane normal vector from image features """
    def __init__(self, input_dim):
        super(SymmetryPredictor, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 2),  # Output 2D normal vector (x, z components)
            nn.Tanh()  # Normalize to [-1, 1]
        )
        
    def forward(self, x):
        return self.mlp(x)

class PointCloudNet(nn.Module):
    def __init__(self, num_views, point_cloud_size, num_heads, dim_feedforward, use_symmetry=True):
        super(PointCloudNet, self).__init__()
        self.use_symmetry = use_symmetry
        
        # Load the pretrained Vision Transformer model from timm
        self.vit = timm.create_model(
            "vit_base_patch16_224", pretrained=True, num_classes=0
        )
        for param in self.vit.parameters():
            param.requires_grad = False
        # Define the number of features from the ViT model
        num_features = self.vit.num_features

        # Aggregate features from different views
        out_features = 1024 * 4
        self.aggregator = nn.Linear(num_features, out_features)
        
        # Point cloud generator with attention
        self.point_cloud_generator = PointCloudGeneratorWithAttention(
            input_feature_dim=out_features,
            point_cloud_size=point_cloud_size,
            num_heads=num_heads,
            dim_feedforward=dim_feedforward,
        )
        
        # 只有在使用对称性时才初始化对称性相关组件
        if self.use_symmetry:
            # Symmetry predictor
            self.symmetry_predictor = SymmetryPredictor(num_features)
            
            # Initialize a learnable bias for symmetry prediction
            self.symmetry_bias = nn.Parameter(torch.zeros(1, 2).cuda())

    def forward(self, x):
        batch_size, num_views, C, H, W = x.shape  # b, 1, 3, 224, 224

        # Process all views in the batch
        x = x.view(batch_size * num_views, C, H, W)

        # Extract features from the views using ViT
        with torch.no_grad():
            features = self.vit(x)

        # Reshape features back to separate views
        features = features.view(batch_size, num_views, -1)

        # Compute the mean of features from all views
        mean_features = torch.mean(features, dim=1)

        # Aggregate features
        aggregated_features = self.aggregator(mean_features)  # [b, 1024*4]
        aggregated_features = aggregated_features.unsqueeze(1)  # [b, 1, 1024*4]

        # Generate point cloud
        point_cloud = self.point_cloud_generator(aggregated_features)  # [b, 3072]
        point_cloud = point_cloud.view(batch_size, -1, 3)  # [b, 1024, 3]
        
        if self.use_symmetry:
            # Predict symmetry plane normal vector from image features
            n_pl = self.symmetry_predictor(mean_features) + self.symmetry_bias  # [b, 2]
        
            # Get reflection matrix and apply symmetry
            refl_mat = get_reflection_operator(n_pl)  # [b, 3, 3]
            symmetry_point_cloud = symmetry_pc(point_cloud, refl_mat)  # [b, 1024, 3]

            return point_cloud, symmetry_point_cloud, n_pl
        else:
            # 如果不使用对称性，返回相同的点云和None
            return point_cloud, point_cloud, None