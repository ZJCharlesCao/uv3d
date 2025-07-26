from model.coordconv import CoordConv1d, CoordConv2d, CoordConv3d
import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['MVCNN', 'mvcnncoord']

# 检查CUDA是否可用
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")


class MVCNN(nn.Module):

    def __init__(self, num_classes=2):
        super(MVCNN, self).__init__()

        # 更复杂的CoordConv预处理层 for 512x512
        self.coordconv_512 = nn.Sequential(
            CoordConv2d(14, 32, 3, padding=1, with_r=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 更复杂的CoordConv预处理层 for 256x256
        self.coordconv_256 = nn.Sequential(
            CoordConv2d(14, 32, 3, padding=1, with_r=True),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # 特征提取器 for 512x512输入 (3个视图) - 输出特征降至1/4
        self.features_512 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 128, kernel_size=3, padding=1),  # 降低通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),   # 进一步降低通道数 64*4*4=1024
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 特征提取器 for 256x256输入 (6个视图) - 输出特征降至1/16
        self.features_256 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(128, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),  # 降低通道数
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 32, kernel_size=3, padding=1),   # 降低通道数
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),    # 进一步降低通道数 16*2*2=64
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        # 可学习的权重参数，用于512x512的3个视图加权池化
        # 初始化为递减的权重：[0.5, 0.3, 0.2]
        self.view_weights_512 = nn.Parameter(torch.tensor([0.5, 0.3, 0.2], device=device))

        # 分类器 - 调整输入维度为 1024 + 64 = 1088
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(18000, 2048),
            nn.BatchNorm1d(2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes),
        )

    def weighted_pooling_512(self, view_features):
        """
        对512x512的3个视图进行加权池化
        Args:
            view_features: list of 3 tensors, each of shape [batch, feature_dim]
        Returns:
            weighted pooled features [batch, feature_dim]
        """
        # 对权重进行softmax归一化，确保权重和为1
        normalized_weights = F.softmax(self.view_weights_512, dim=0)

        # 加权求和
        weighted_sum = normalized_weights[0] * view_features[0]
        for i in range(1, len(view_features)):
            weighted_sum += normalized_weights[i] * view_features[i]

        return weighted_sum

    def forward(self, inputs_512, inputs_256):
        """
        Args:
            inputs_512: [batch, 177, 512, 512] - batch个177通道(3x59)512x512的张量
            inputs_256: [batch, 354, 256, 256] - batch个354通道(6x59)256x256的张量
        """
        batch_size = inputs_512.shape[0]

        # 处理512x512的输入 (3个59通道的视图)
        # 将[batch, 177, 512, 512]重塑为[batch, 3, 59, 512, 512]
        inputs_512_reshaped = inputs_512.view(batch_size, 3, 59, 512, 512)

        view_pool_512 = []
        for i in range(3):  # 遍历3个视图
            v = inputs_512_reshaped[:, i, :, :, :]  # [batch, 59, 512, 512]

            # 选取前6个通道和最后8个通道 (索引0-5和51-58)
            selected_channels = torch.cat([v[:, :6, :, :], v[:, 51:59, :, :]], dim=1)  # [batch, 14, 512, 512]

            selected_channels = self.coordconv_512(selected_channels)  # [batch, 64, 512, 512]
            selected_channels = self.features_512(selected_channels)  # [batch, 64, 4, 4]
            selected_channels = selected_channels.view(batch_size, -1)  # [batch, 1024]
            view_pool_512.append(selected_channels)

        # 对512x512的特征进行加权池化
        pooled_512 = self.weighted_pooling_512(view_pool_512)  # [batch, 1024]

        # 处理256x256的输入 (6个59通道的视图)
        print(inputs_256.shape)  # [batch, 354, 256, 256]
        # 将[batch, 354, 256, 256]重塑为[batch, 6, 59, 256, 256]
        inputs_256_reshaped = inputs_256.view(batch_size, 6, 59, 256, 256)

        view_pool_256 = []
        for i in range(6):  # 遍历6个视图
            v = inputs_256_reshaped[:, i, :, :, :]  # [batch, 59, 256, 256]

            # 选取前6个通道和最后8个通道 (索引0-5和51-58)
            selected_channels = torch.cat([v[:, :6, :, :], v[:, 51:59, :, :]], dim=1)  # [batch, 14, 256, 256]

            selected_channels = self.coordconv_256(selected_channels)  # [batch, 64, 256, 256]
            selected_channels = self.features_256(selected_channels)  # [batch, 16, 2, 2]
            selected_channels = selected_channels.view(batch_size, -1)  # [batch, 64]
            view_pool_256.append(selected_channels)

        # 对256x256的特征进行最大池化
        pooled_256 = view_pool_256[0]  # [batch, 64]
        for i in range(1, len(view_pool_256)):
            pooled_256 = torch.max(pooled_256, view_pool_256[i])

        # 拼接两个池化结果
        concatenated_features = torch.cat([pooled_512, pooled_256], dim=1)  # [batch, 1088]

        # 通过分类器
        output = self.classifier(concatenated_features)
        return output

    def get_view_weights(self):
        """返回当前的视图权重（归一化后的）"""
        return F.softmax(self.view_weights_512, dim=0)


def mvcnncoord(num_classes=2, **kwargs):
    r"""MVCNN model architecture adapted for multi-resolution multi-view inputs.
    Args:
        num_classes (int): Number of output classes
    """
    model = MVCNN(num_classes=num_classes, **kwargs)
    return model


if __name__ == "__main__":

    # 创建模型并移到CUDA
    model = mvcnncoord(num_classes=2).to(device)
    model.eval()

    # 准备输入数据并移到CUDA
    inputs_512 = torch.randn(3, 14, 512, 512, device=device)
    inputs_256 = torch.randn(6, 14, 256, 256, device=device)

    # 前向传播
    with torch.cuda.amp.autocast():  # 使用混合精度加速训练
        output = model(inputs_512, inputs_256)

    # # 显示模型参数量
    # total_params = sum(p.numel() for p in model.parameters())
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"总参数量: {total_params:,}")
    # print(f"可训练参数量: {trainable_params:,}")