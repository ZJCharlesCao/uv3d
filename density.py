import numpy as np
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from typing import Tuple, List
import random
import open3d as o3d


class PointCloudDensityController:
    """
    点云密度自适应控制器
    使用空间扫描方法自适应地控制点云密度
    """

    def __init__(self, grid_size: float = 1.0, k_neighbors: int = 20,
                 density_threshold: float = 1.5, preserve_ratio: float = 0.3):
        """
        初始化密度控制器

        参数:
        - grid_size: 扫描网格的大小
        - k_neighbors: 用于密度计算的邻近点数量
        - density_threshold: 密度阈值倍数（相对于平均密度）
        - preserve_ratio: 高密度区域的保留比例
        """
        self.grid_size = grid_size
        self.k_neighbors = k_neighbors
        self.density_threshold = density_threshold
        self.preserve_ratio = preserve_ratio

    def calculate_local_density(self, points: np.ndarray) -> np.ndarray:
        """
        计算每个点的局部密度
        使用k近邻距离的倒数作为密度度量
        """
        # 使用KNN计算每个点到第k个邻居的距离
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors + 1, algorithm='auto')
        nbrs.fit(points)
        distances, indices = nbrs.kneighbors(points)

        # 排除自身，取第k个邻居的距离
        k_distances = distances[:, -1]

        # 密度 = 1 / (k近邻距离 + 小常数避免除零)
        densities = 1.0 / (k_distances + 1e-8)

        return densities

    def create_spatial_grid(self, points: np.ndarray) -> Tuple[dict, np.ndarray]:
        """
        创建空间网格并将点分配到网格中
        """
        # 计算边界
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)

        # 计算网格索引
        grid_indices = np.floor((points - min_coords) / self.grid_size).astype(int)

        # 将点按网格分组
        grid_dict = {}
        for i, grid_idx in enumerate(grid_indices):
            grid_key = tuple(grid_idx)
            if grid_key not in grid_dict:
                grid_dict[grid_key] = []
            grid_dict[grid_key].append(i)

        return grid_dict, grid_indices

    def adaptive_density_control(self, points) -> Tuple[np.ndarray, np.ndarray]:
        """
        主要的密度自适应控制函数

        参数:
        - points: numpy数组或Open3D Vector3dVector

        返回:
        - 筛选后的点云 (numpy数组)
        - 保留点的索引
        """
        # 检查输入类型并转换为numpy数组
        if hasattr(points, 'cpu'):  # Open3D Vector3dVector
            points_np = np.asarray(points.cpu().numpy() if hasattr(points, 'cpu') else points)
        elif hasattr(points, '__array__') or isinstance(points, np.ndarray):
            points_np = np.asarray(points)
        else:
            # 尝试转换为numpy数组
            try:
                points_np = np.asarray(points)
            except:
                raise TypeError(f"不支持的点云数据类型: {type(points)}")

        print(f"原始点云数量: {len(points_np)}")

        # 1. 计算局部密度
        print("正在计算局部密度...")
        densities = self.calculate_local_density(points_np)
        mean_density = np.mean(densities)
        density_std = np.std(densities)

        print(f"平均密度: {mean_density:.4f}")
        print(f"密度标准差: {density_std:.4f}")

        # 2. 创建空间网格
        print("创建空间扫描网格...")
        grid_dict, grid_indices = self.create_spatial_grid(points_np)
        print(f"网格数量: {len(grid_dict)}")

        # 3. 自适应筛选
        preserved_indices = []

        for grid_key, point_indices in grid_dict.items():
            if len(point_indices) <= 1:
                # 如果网格中只有一个或没有点，直接保留
                preserved_indices.extend(point_indices)
                continue

            # 计算该网格的平均密度
            grid_densities = densities[point_indices]
            grid_mean_density = np.mean(grid_densities)

            # 判断是否为高密度区域
            if grid_mean_density > mean_density * self.density_threshold:
                # 高密度区域：按密度排序，保留密度较低的点和一些关键点
                sorted_indices = sorted(point_indices, key=lambda x: densities[x])

                # 计算保留数量
                preserve_count = max(1, int(len(point_indices) * self.preserve_ratio))

                # 保留密度最低的点 + 随机保留一些点以保持空间分布
                low_density_count = int(preserve_count * 0.7)
                random_count = preserve_count - low_density_count

                selected_indices = sorted_indices[:low_density_count]

                if random_count > 0:
                    remaining_indices = sorted_indices[low_density_count:]
                    if len(remaining_indices) > 0:
                        random_selected = random.sample(remaining_indices,
                                                        min(random_count, len(remaining_indices)))
                        selected_indices.extend(random_selected)

                preserved_indices.extend(selected_indices)
            else:
                # 正常密度区域：保留所有点
                preserved_indices.extend(point_indices)
        preserved_indices = sorted(preserved_indices)
        filtered_points = points_np[preserved_indices]
        print(f"筛选后点云数量: {len(filtered_points)}")
        print(f"保留比例: {len(filtered_points) / len(points_np) * 100:.2f}%")

        # Convert filtered points to Open3D PointCloud format
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        return filtered_pcd, np.array(preserved_indices)

    def process_open3d_pointcloud(self, pcd):
        """
        专门处理Open3D点云对象的方法

        参数:
        - pcd: Open3D PointCloud对象

        返回:
        - 筛选后的Open3D PointCloud对象
        - 保留点的索引
        """
        import open3d as o3d

        # 获取点云数据
        points_np = np.asarray(pcd.points)

        # 执行密度控制
        filtered_points, preserved_indices = self.adaptive_density_control(points_np)

        # 创建新的Open3D点云对象
        filtered_pcd = o3d.geometry.PointCloud()
        filtered_pcd.points = o3d.utility.Vector3dVector(filtered_points)

        # 如果原点云有颜色信息，也要保留
        if pcd.has_colors():
            colors_np = np.asarray(pcd.colors)
            filtered_pcd.colors = o3d.utility.Vector3dVector(colors_np[preserved_indices])

        # 如果原点云有法向量信息，也要保留
        if pcd.has_normals():
            normals_np = np.asarray(pcd.normals)
            filtered_pcd.normals = o3d.utility.Vector3dVector(normals_np[preserved_indices])

        return filtered_pcd, preserved_indices
