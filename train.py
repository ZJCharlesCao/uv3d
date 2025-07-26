import argparse
import os
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import open3d as o3d
from sklearn.metrics import accuracy_score, classification_report
import glob
from tqdm import tqdm
from pathlib import Path
from gaussian_model import GaussianModel
from multilayer import multilayer_encode
import cv2
import re
from model import understanding


def read_and_process_layer_images(folder_path):
    """
    读取layer_M_N格式图片并拼接成两个大张量

    Args:
        folder_path: 图片文件夹路径

    Returns:
        tuple: (group1_tensor, group2_tensor)
        - group1_tensor: 合并的张量，形状为 (H, W, 3*59) (M=1-3)
        - group2_tensor: 合并的张量，形状为 (H, W, 6*59) (M=4-9)
    """

    # 支持的图片格式
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}

    # 存储文件路径，按M和N分组
    files_dict = {}

    folder = Path(folder_path)
    for file_path in folder.iterdir():
        if file_path.is_file() and file_path.suffix.lower() in image_extensions:
            match = re.match(r'layer_(\d+)_(\d+)', file_path.stem)
            if match:
                M, N = int(match.group(1)), int(match.group(2))
                if 0 <= N <= 19:  # N范围0-19
                    if M not in files_dict:
                        files_dict[M] = {}
                    files_dict[M][N] = file_path

    def process_group(M_range):
        """处理指定M范围的图片组并合并为一个大张量"""
        all_group_tensors = []

        for M in M_range:
            if M not in files_dict:
                continue

            # 收集该M值的所有N图片
            images = []
            for N in range(20):  # N=0-19
                if N in files_dict[M]:
                    img = cv2.imread(str(files_dict[M][N]))
                    if img is not None:
                        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        images.append(img_rgb)

            if len(images) == 20:  # 确保有20张图片
                # 拼接成60通道 (H, W, 60)
                stacked = np.stack(images, axis=-1)  # (H, W, 3, 20)
                reshaped = stacked.reshape(stacked.shape[0], stacked.shape[1], -1)  # (H, W, 60)

                # 只取前59个通道
                tensor_59 = reshaped[:, :, :59].astype(np.float32)
                all_group_tensors.append(tensor_59)
                print(f"M={M}: 形状 {tensor_59.shape}")

        # 将所有张量按通道维度合并
        if all_group_tensors:
            merged_tensor = np.concatenate(all_group_tensors, axis=2)  # 按通道维度合并
            return merged_tensor
        else:
            return None

    # 处理两个组
    group1_tensor = process_group(range(1, 4))  # M=1-3，期望形状 (H, W, 3*59=177)
    group2_tensor = process_group(range(4, 10))  # M=4-9，期望形状 (H, W, 6*59=354)

    if group1_tensor is not None:
        print(f"组1 (M=1-3): 合并张量形状 {group1_tensor.shape}")
    else:
        print("组1 (M=1-3): 没有找到有效数据")

    if group2_tensor is not None:
        print(f"组2 (M=4-9): 合并张量形状 {group2_tensor.shape}")
    else:
        print("组2 (M=4-9): 没有找到有效数据")

    return group1_tensor, group2_tensor


def check_existing_images(folder_path, min_images_required=180):
    """
    检查文件夹中是否有足够的图像文件

    Args:
        folder_path: 文件夹路径
        min_images_required: 最少需要的图像数量

    Returns:
        bool: 是否有足够的图像
    """
    if not os.path.exists(folder_path):
        return False

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    image_files = []

    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)

    return len(image_files) >= min_images_required


class GaussianDataset(Dataset):
    """高斯数据集类"""

    def __init__(self, dataset_path, uv_path, sh_degree=3, max_samples_per_class=None):
        self.dataset_path = dataset_path
        self.uv_path = uv_path
        self.sh_degree = sh_degree
        self.data_files = []
        self.labels = []

        # 获取两个子目录
        class_dirs = [d for d in os.listdir(dataset_path)
                      if os.path.isdir(os.path.join(dataset_path, d))]

        if len(class_dirs) != 2:
            raise ValueError(f"数据集目录应包含恰好2个子目录，但找到了 {len(class_dirs)} 个")

        class_dirs.sort()  # 确保顺序一致
        print(f"找到类别: {class_dirs}")

        # 遍历每个类别目录
        for class_idx, class_dir in enumerate(class_dirs):
            class_path = os.path.join(dataset_path, class_dir)
            # 查找所有 .ply 文件
            ply_files = glob.glob(os.path.join(class_path, "*.ply"))

            if max_samples_per_class:
                ply_files = ply_files[:max_samples_per_class]

            print(f"类别 {class_idx} ({class_dir}): 找到 {len(ply_files)} 个样本")

            for ply_file in ply_files:
                self.data_files.append(ply_file)
                self.labels.append(class_idx)

        print(f"总计加载 {len(self.data_files)} 个样本")

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        """加载并处理单个高斯样本"""
        file_path = self.data_files[idx]
        label = self.labels[idx]


        # 加载高斯模型
        gaussian = GaussianModel(self.sh_degree)
        gaussian.load_ply(file_path)
        names = os.path.basename(file_path)
        name = os.path.splitext(names)[0]

        # 检查是否已有足够的图像，如果有则跳过处理
        name_dir = os.path.join(self.uv_path, name)
        if check_existing_images(name_dir):
            print(f"检测到 {name} 目录已有足够图像，跳过处理")
            # 直接读取已有的图像并返回
            group1_tensor, group2_tensor = read_and_process_layer_images(name_dir)
            # 将两个张量转换为torch张量并返回
            return torch.FloatTensor(group1_tensor), torch.FloatTensor(group2_tensor), torch.LongTensor([label])

        # 如果没有足够图像，创建目录并进行处理
        os.makedirs(name_dir, exist_ok=True)

        # 提取属性
        xyz = gaussian._xyz.detach().cpu().numpy()
        features = torch.cat((gaussian._features_dc, gaussian._features_rest), dim=1).detach().cpu().numpy()
        features = features.reshape(features.shape[0], -1)
        rotation = gaussian._rotation.detach().cpu().numpy()
        scale = gaussian._scaling.detach().cpu().numpy()
        opacity = gaussian._opacity.detach().cpu().numpy()

        # 合并所有属性
        attributes = np.concatenate((xyz, features, rotation, scale, opacity), axis=1)
        print(f"处理文件: {file_path}, 类别: {label}, 属性形状: {attributes.shape}")

        # 进行编解码处理
        processed_attributes = self.process(name, xyz, attributes)
        group1_tensor, group2_tensor = read_and_process_layer_images(name_dir)

        return torch.FloatTensor(group1_tensor),torch.FloatTensor(group2_tensor), torch.LongTensor([label])


    def process(self, dir_name, xyz, attributes):
        """处理单个样本的编解码"""
        # 转换为 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        # 进行编码处理
        encoded_data = multilayer_encode(pcd, attributes, os.path.join(self.uv_path, dir_name), 0.005)

        # 进行解码处理
        # decoded_attributes, _ = multilayer_attribute_decode(os.path.join(self.uv_path, dir_name + ".png"), 9)

        return encoded_data  # 修正：返回编码后的数据


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(dataloader, desc="训练中")

    for batch_idx, (data1,data2, targets) in enumerate(progress_bar):
        data1,data2, targets = data1.to(device),data2.to(device), targets.squeeze().to(device)

        # 前向传播
        optimizer.zero_grad()
        outputs = model(data1,data2)
        loss = criterion(outputs, targets)

        # 反向传播
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # 更新进度条
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })

    return total_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    """验证模型"""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():
        for data1, data2, targets in tqdm(dataloader, desc="验证中"):
            data1, data2, targets = data1.to(device), data2.to(device), targets.squeeze().to(device)
            outputs = model(data1, data2)
            loss = criterion(outputs, targets)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100. * correct / total
    return total_loss / len(dataloader), accuracy, all_predictions, all_targets


def main():
    parser = argparse.ArgumentParser(description="高斯点云分类训练工具")
    parser.add_argument("--dataset", "-d", required=True, help="数据集根目录路径")
    parser.add_argument("--uv", "-u", required=True, help="数据集存放目录路径")
    parser.add_argument("--output", "-o", default="classification_output", help="输出目录")
    parser.add_argument("--batch-size", "-b", type=int, default=2, help="批处理大小")
    parser.add_argument("--epochs", "-e", type=int, default=50, help="训练轮数")
    parser.add_argument("--learning-rate", "-lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--sh", "-s", type=int, default=3, help="球谐函数度数")
    parser.add_argument("--train-split", "-ts", type=float, default=0.8, help="训练集比例")
    parser.add_argument("--device", default="cuda", help="计算设备 (cuda/cpu)")

    args = parser.parse_args()

    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)

    # 设置设备
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 加载数据集
    print("加载数据集...")
    dataset = GaussianDataset(args.dataset, args.uv, args.sh)

    # 划分训练集和验证集
    train_size = int(args.train_split * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False
    )

    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 创建模型
    model = understanding.mvcnncoord(num_classes=2).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 修改学习率调度器：使用余弦退火调度器，更平滑的学习率下降
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # 训练循环
    best_val_acc = 0.0
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    print("\n开始训练...")
    start_time = time.time()

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print("-" * 50)

        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证
        val_loss, val_acc, val_predictions, val_targets = validate(model, val_loader, criterion, device)

        # 更新学习率
        scheduler.step()

        # 记录历史
        train_losses.append(train_loss)
        train_accuracies.append(train_acc)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        # 打印结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        print(f"学习率: {scheduler.get_last_lr()[0]:.6f}")

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),  # 保存调度器状态
                'best_val_acc': best_val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss
            }, os.path.join(args.output, 'best_model.pth'))
            print(f"保存最佳模型 (验证准确率: {best_val_acc:.2f}%)")

    training_time = time.time() - start_time

    # 最终评估
    print("\n" + "=" * 60)
    print("训练完成 - 最终评估")
    print("=" * 60)

    # 加载最佳模型进行最终评估
    checkpoint = torch.load(os.path.join(args.output, 'best_model.pth'))
    model.load_state_dict(checkpoint['model_state_dict'])

    _, final_acc, final_predictions, final_targets = validate(model, val_loader, criterion, device)

    # 分类报告
    class_report = classification_report(final_targets, final_predictions,
                                         target_names=['Class 0', 'Class 1'])
    print("分类报告:")
    print(class_report)

    # 保存训练统计
    stats = {
        'training_time': training_time,
        'best_val_accuracy': best_val_acc,
        'final_val_accuracy': final_acc,
        'total_epochs': args.epochs,
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'val_losses': val_losses,
        'val_accuracies': val_accuracies
    }

    stats_path = os.path.join(args.output, "training_stats.txt")
    with open(stats_path, 'w', encoding='utf-8') as f:
        f.write(f"训练时间: {training_time:.2f} 秒\n")
        f.write(f"最佳验证准确率: {best_val_acc:.2f}%\n")
        f.write(f"最终验证准确率: {final_acc:.2f}%\n")
        f.write(f"总训练轮数: {args.epochs}\n")
        f.write(f"训练集大小: {len(train_dataset)}\n")
        f.write(f"验证集大小: {len(val_dataset)}\n")
        f.write("\n分类报告:\n")
        f.write(class_report)

    # 保存模型参数
    torch.save(stats, os.path.join(args.output, 'training_history.pth'))

    print(f"\n训练完成！")
    print(f"总训练时间: {training_time:.2f} 秒")
    print(f"最佳验证准确率: {best_val_acc:.2f}%")
    print(f"结果已保存到: {args.output}")


if __name__ == "__main__":
    main()