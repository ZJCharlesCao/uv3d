import numpy as np
import open3d as o3d
import os
import trimesh
import igl
import time
import argparse
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.sparse import csr_matrix
from collections import defaultdict
from PIL import Image, ImageDraw
from scipy.sparse import csr_matrix, diags
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
import warnings
from scipy.interpolate import griddata
from scipy.spatial import Delaunay
from collections import Counter
from sklearn.decomposition import PCA
from typing import List, Tuple, Optional, Set
from sklearn.neighbors import NearestNeighbors
import cv2


def compute_circumsphere_radius(tetra_points):
    """计算四面体的外接球半径"""
    try:
        p1, p2, p3, p4 = tetra_points
        
        # 构建矩阵求解外接球
        A = np.array([
            [p1[0], p1[1], p1[2], 1],
            [p2[0], p2[1], p2[2], 1],
            [p3[0], p3[1], p3[2], 1],
            [p4[0], p4[1], p4[2], 1]
        ])
        
        # 如果四点共面，返回大半径
        if abs(np.linalg.det(A)) < 1e-10:
            return float('inf')
        
        b = np.array([
            [np.sum(p1**2)],
            [np.sum(p2**2)],
            [np.sum(p3**2)],
            [np.sum(p4**2)]
        ])
        
        x = np.linalg.solve(A, b).flatten()
        center = x[:3] / 2
        radius = np.sqrt(np.sum(center**2) + x[3]/2)
        
        return radius
    except:
        return float('inf')

def alpha_shape_3d(pcd, alpha):
    """
    使用Alpha Shape算法从点云生成mesh
    
    Args:
        pcd: Open3D PointCloud对象
        alpha: Alpha参数值
        
    Returns:
        mesh: Open3D TriangleMesh对象
    """
    # 获取点云坐标
    points = np.asarray(pcd.points)
    
    # 执行Delaunay三角剖分
    tri = Delaunay(points)
    tetrahedra = tri.simplices
    
    # 根据alpha值过滤四面体
    valid_tetrahedra = []
    for tetra in tetrahedra:
        tetra_points = points[tetra]
        radius = compute_circumsphere_radius(tetra_points)
        
        if radius <= alpha:
            valid_tetrahedra.append(tetra)
    
    # 提取边界三角形
    face_count = Counter()
    
    # 统计每个面的出现次数
    for tetra in valid_tetrahedra:
        faces = [
            tuple(sorted([tetra[0], tetra[1], tetra[2]])),
            tuple(sorted([tetra[0], tetra[1], tetra[3]])),
            tuple(sorted([tetra[0], tetra[2], tetra[3]])),
            tuple(sorted([tetra[1], tetra[2], tetra[3]]))
        ]
        
        for face in faces:
            face_count[face] += 1
    
    # 边界面（只属于一个四面体的面）
    boundary_triangles = [face for face, count in face_count.items() if count == 1]
    
    # 创建mesh
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(points)
    mesh.triangles = o3d.utility.Vector3iVector(boundary_triangles)
    
    # 清理mesh
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    
    return mesh

def _initialize_conformal(vertices, faces):
    
    n_vertices = len(vertices)
    
    # 使用更稳健的球面参数化初始化
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    
    # 将顶点投影到单位球面
    norms = np.linalg.norm(centered_vertices, axis=1)
    norms = np.where(norms < 1e-10, 1, norms)  # 避免除零
    
    normalized_vertices = centered_vertices / norms[:, np.newaxis]
    
    # 使用球面到平面的立体投影
    # 选择北极点 (0, 0, 1) 作为投影中心
    z_coords = normalized_vertices[:, 2]
    
    # 避免投影点在北极点
    z_coords = np.clip(z_coords, -0.999, 0.999)
    
    # 立体投影公式
    uv_coords = np.zeros((n_vertices, 2))
    uv_coords[:, 0] = normalized_vertices[:, 0] / (1 - z_coords)
    uv_coords[:, 1] = normalized_vertices[:, 1] / (1 - z_coords)
    
    # 处理可能的异常值
    uv_coords = np.clip(uv_coords, -10, 10)
    
    # 归一化到合理范围
    uv_min = np.min(uv_coords, axis=0)
    uv_max = np.max(uv_coords, axis=0)
    uv_range = uv_max - uv_min
    uv_range = np.where(uv_range < 1e-6, 1, uv_range)
    
    uv_coords = (uv_coords - uv_min) / uv_range
    
    # 添加小的随机扰动避免退化
    noise = np.random.normal(0, 0.01, uv_coords.shape)
    uv_coords += noise
    
    print(f"立体投影初始化完成，UV坐标范围: [{np.min(uv_coords):.3f}, {np.max(uv_coords):.3f}]")
    
    return uv_coords

def _initialize_conformal(vertices,faces):
    """更稳健的立体投影"""
    centroid = np.mean(vertices, axis=0)
    centered_vertices = vertices - centroid
    
    # 使用PCA找到主方向
    pca = PCA(n_components=3)
    pca.fit(centered_vertices)
    
    # 将数据投影到PCA空间
    transformed = pca.transform(centered_vertices)
    
    # 投影到单位球面
    norms = np.linalg.norm(transformed, axis=1)
    norms = np.maximum(norms, 1e-10)
    normalized_vertices = transformed / norms[:, np.newaxis]
    
    # 改进的立体投影 - 使用适应性投影点
    z_coords = normalized_vertices[:, 2]
    
    # 动态选择投影点以减少聚集
    z_median = np.median(z_coords)
    projection_point = -np.sign(z_median) * 0.9  # 适应性投影点
    
    denominator = 1 - projection_point * z_coords
    denominator = np.maximum(np.abs(denominator), 0.01)
    
    uv_coords = np.zeros((len(vertices), 2))
    uv_coords[:, 0] = normalized_vertices[:, 0] / denominator
    uv_coords[:, 1] = normalized_vertices[:, 1] / denominator
    
    # 使用更宽松的限制
    scale_factor = np.percentile(np.abs(uv_coords), 95)
    if scale_factor > 0:
        uv_coords = uv_coords / scale_factor * 2
    
    return uv_coords

def _build_laplacian_matrix(vertices, faces):
    """构建拉普拉斯矩阵 - 快速版本"""
    n_vertices = len(vertices)
    
    # 使用混合权重：余切权重 + 均匀权重
    edges = {}
    face_areas = []
    
    for face in faces:
        v0, v1, v2 = vertices[face]
        
        # 计算面积
        cross = np.cross(v1 - v0, v2 - v0)
        area = 0.5 * np.linalg.norm(cross)
        area = max(area, 1e-12)  # 防止零面积
        face_areas.append(area)
        
        # 计算边长
        edges_in_face = [
            (face[0], face[1], np.linalg.norm(v1 - v0)),
            (face[1], face[2], np.linalg.norm(v2 - v1)),
            (face[2], face[0], np.linalg.norm(v0 - v2))
        ]
        
        # 计算余切权重但增加数值保护
        for i, (vi, vj, edge_len) in enumerate(edges_in_face):
            if edge_len < 1e-12:
                continue
                
            # 对角顶点
            vk = face[(i+2) % 3]
            opposite_vertex = vertices[vk]
            
            # 计算角度
            vec1 = vertices[vi] - opposite_vertex
            vec2 = vertices[vj] - opposite_vertex
            
            len1 = np.linalg.norm(vec1)
            len2 = np.linalg.norm(vec2)
            
            if len1 < 1e-12 or len2 < 1e-12:
                weight = 1.0 / edge_len  # 退化到均匀权重
            else:
                cos_angle = np.dot(vec1, vec2) / (len1 * len2)
                cos_angle = np.clip(cos_angle, -0.999, 0.999)
                
                sin_angle = np.sqrt(1 - cos_angle**2)
                if sin_angle < 1e-6:
                    weight = 1.0 / edge_len
                else:
                    cot_weight = cos_angle / sin_angle
                    # 限制余切权重的范围
                    cot_weight = np.clip(cot_weight, -10, 10)
                    
                    
                    uniform_weight = 1.0 / edge_len
                    beta = 1e-10
                    weight = (1-beta) * abs(cot_weight) + beta * uniform_weight

            edge_key = tuple(sorted([vi, vj]))
            edges[edge_key] = edges.get(edge_key, 0) + weight
    
    # 构建矩阵
    row_indices = []
    col_indices = []
    data = []
    
    for (vi, vj), weight in edges.items():
        # 确保权重为正且有限
        weight = max(abs(weight), 1e-8)
        if not np.isfinite(weight):
            weight = 1e-8
        
        row_indices.extend([vi, vj])
        col_indices.extend([vj, vi])
        data.extend([weight, weight])
    
    L = csr_matrix((data, (row_indices, col_indices)), shape=(n_vertices, n_vertices))
    
    # 设置对角线并添加正则化
    diagonal = np.array(L.sum(axis=1)).flatten()
    regularization = 1e-8
    L.setdiag(-diagonal - regularization)
    
    return L, np.array(face_areas)

def _compute_optimal_rotations(vertices, faces, uv_coords, face_areas):
    """计算最优旋转 - 快速版本"""
    rotations = []
    
    for face in faces:
        v0, v1, v2 = face
        
        # 3D局部坐标
        P = np.array([vertices[v1] - vertices[v0], vertices[v2] - vertices[v0]]).T
        # 2D局部坐标
        Q = np.array([uv_coords[v1] - uv_coords[v0], uv_coords[v2] - uv_coords[v0]]).T
        
        try:
            # 简化的旋转计算
            S = Q @ np.linalg.pinv(P)
            U, _, Vt = np.linalg.svd(S[:, :2])
            R = U @ Vt
            
            # 确保det(R) > 0
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
            
            rotations.append(R)
        except:
            rotations.append(np.eye(2))
    
    return rotations

def _solve_global_step(L, faces, rotations, face_areas, vertices, uv_coords):
    """快速全局步骤求解"""
    n_vertices = L.shape[0]
    b = np.zeros((n_vertices, 2))
    
    # 构建右端项
    for f_idx, face in enumerate(faces):
        R = rotations[f_idx]
        area = face_areas[f_idx]
        
        p0, p1, p2 = vertices[face]
        local_3d = np.array([p1 - p0, p2 - p0])
        rotated_2d = R @ local_3d[:, :2].T
        
        weight = area * 0.5
        b[face[0]] += weight * (-rotated_2d[0, :] - rotated_2d[1, :])
        b[face[1]] += weight * rotated_2d[0, :]
        b[face[2]] += weight * rotated_2d[1, :]
    
    # 简单固定第一个顶点求解
    new_uv_coords = uv_coords.copy()
    
    for dim in range(2):
        try:
            free_vertices = list(range(1, n_vertices))
            L_free = L[free_vertices][:, free_vertices]
            b_free = b[free_vertices, dim]
            
            # 减去固定顶点贡献
            L_fixed = L[free_vertices, 0].toarray().flatten()
            b_free -= L_fixed * uv_coords[0, dim]
            
            # 直接求解
            uv_free = spsolve(L_free, b_free)
            new_uv_coords[free_vertices, dim] = uv_free
        except:
            continue
    
    return new_uv_coords

def _normalize_uv_coordinates(uv_coords):
    """更稳健的UV坐标归一化"""
    uv_coords = np.clip(uv_coords, -50, 50)
    
    # 计算更稳健的范围
    center = np.median(uv_coords, axis=0)  # 使用中位数而非均值
    centered_coords = uv_coords - center
    
    # 使用MAD (Median Absolute Deviation) 而不是标准差
    mad = np.median(np.abs(centered_coords), axis=0)
    mad = np.where(mad < 1e-6, 1, mad)
    
    # 更温和的缩放
    scale_factor = np.maximum(mad * 3, np.max(np.abs(centered_coords), axis=0))
    normalized_uv = centered_coords / scale_factor
    
    # 移到[0,1]范围但保持更多的分散性
    normalized_uv = (normalized_uv + 1) * 0.4 + 0.1  # 映射到[0.1, 0.9]
    
    return np.clip(normalized_uv, 0, 1)


import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve


class RobustSLIMUV:
    def __init__(self, vertices, faces):
        """
        简化的SLIM UV参数化器

        Args:
            vertices: numpy array (N, 3) - 3D顶点坐标
            faces: numpy array (M, 3) - 面片顶点索引
        """
        self.V = np.array(vertices, dtype=np.float64)
        self.F = np.array(faces, dtype=np.int32)
        self.n_vertices = len(self.V)
        self.n_faces = len(self.F)

        # 初始化UV坐标
        self.UV = np.zeros((self.n_vertices, 2))

        # SLIM参数
        self.max_iterations = 10
        self.convergence_threshold = 1e-6
        self.regularization = 1e-8  # 正则化参数，防止数值不稳定

        # 预计算每个面片的局部坐标系和面积
        self._precompute_face_data()

    def _precompute_face_data(self):
        """预计算每个三角面片的局部2D坐标系和面积"""
        self.face_areas_3d = np.zeros(self.n_faces)
        self.face_local_coords = np.zeros((self.n_faces, 3, 2))

        for i, face in enumerate(self.F):
            v0, v1, v2 = self.V[face[0]], self.V[face[1]], self.V[face[2]]

            e1 = v1 - v0
            e2 = v2 - v0

            # 计算面积
            area_vec = np.cross(e1, e2)
            area = 0.5 * np.linalg.norm(area_vec)
            self.face_areas_3d[i] = area

            # 跳过退化三角形
            if area < 1e-12:
                continue

            # 构建局部坐标系，保持几何距离和角度
            e1_len = np.linalg.norm(e1)
            if e1_len < 1e-12:
                continue

            u_axis = e1 / e1_len
            normal = area_vec / (2.0 * area)
            v_axis = np.cross(normal, u_axis)

            # 计算局部2D坐标，保持原始距离
            self.face_local_coords[i, 0] = [0, 0]
            self.face_local_coords[i, 1] = [e1_len, 0]
            self.face_local_coords[i, 2] = [np.dot(e2, u_axis), np.dot(e2, v_axis)]

    def initialize_uv_pca(self):
        """使用PCA进行UV初始化，保持几何结构"""
        # 中心化顶点
        centered = self.V - np.mean(self.V, axis=0)

        # 计算协方差矩阵
        cov_matrix = centered.T @ centered / (self.n_vertices - 1)

        # 特征值分解，选择最大的两个主成分
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

        # 按特征值排序（降序）
        sorted_indices = np.argsort(eigenvalues)[::-1]
        principal_axes = eigenvectors[:, sorted_indices[:2]]

        # 投影到主平面
        self.UV = centered @ principal_axes

        # 归一化到[0,1]范围，但保持长宽比
        uv_min = np.min(self.UV, axis=0)
        uv_max = np.max(self.UV, axis=0)
        uv_range = uv_max - uv_min

        # 防止除零
        uv_range = np.maximum(uv_range, 1e-8)

        # 标准化到[0,1]
        self.UV = (self.UV - uv_min) / uv_range

        # 将UV坐标移动到[0.1, 0.9]范围内，避免边界问题
        self.UV = 0.05 + 0.9 * self.UV

    def compute_slim_energy(self):
        """计算SLIM能量，增加数值稳定性检查"""
        total_energy = 0.0
        valid_faces = 0

        for i, face in enumerate(self.F):
            if self.face_areas_3d[i] < 1e-12:
                continue

            p0, p1, p2 = self.face_local_coords[i]
            u0, u1, u2 = self.UV[face]

            # 检查UV坐标是否有效
            if np.any(np.isnan(self.UV[face])) or np.any(np.isinf(self.UV[face])):
                continue

            Ds = np.column_stack([p1 - p0, p2 - p0])
            Dm = np.column_stack([u1 - u0, u2 - u0])

            try:
                # 添加正则化以避免奇异矩阵
                Ds_reg = Ds + self.regularization * np.eye(2)
                Ds_inv = np.linalg.inv(Ds_reg)
                J = Dm @ Ds_inv

                # 计算奇异值
                s = np.linalg.svd(J, compute_uv=False)
                s = np.maximum(s, 1e-8)  # 防止奇异值过小
                s = np.minimum(s, 1e8)  # 防止奇异值过大

                # SLIM能量：对称狄利克雷能量
                energy = np.sum(s ** 2 + 1.0 / (s ** 2))

                if np.isfinite(energy):
                    total_energy += energy * self.face_areas_3d[i]
                    valid_faces += 1

            except (np.linalg.LinAlgError, ZeroDivisionError):
                # 对问题面片施加惩罚
                total_energy += 1e6

        return total_energy / max(valid_faces, 1)

    def optimize_slim(self):
        """SLIM优化主循环，增强数值稳定性"""
        self.initialize_uv_pca()

        prev_energy = self.compute_slim_energy()
        print(f"初始能量 = {prev_energy:.6f}")

        for iteration in range(self.max_iterations):
            # 备份当前UV坐标
            uv_backup = self.UV.copy()

            try:
                self._solve_slim_step()

                # 检查是否产生了NaN或无穷大
                if np.any(np.isnan(self.UV)) or np.any(np.isinf(self.UV)):
                    print(f"迭代 {iteration + 1}: 检测到数值不稳定，恢复到上一步")
                    self.UV = uv_backup
                    self.regularization *= 10  # 增加正则化
                    continue

                current_energy = self.compute_slim_energy()

                # # 如果能量增加太多，回退
                # if current_energy > prev_energy * 20:
                #     print(f"迭代 {iteration + 1}: 能量增加过大，恢复到上一步")
                #     self.UV = uv_backup
                #     self.regularization *= 2
                #     continue
                #
                print(f"迭代 {iteration + 1}: 能量 = {current_energy:.6f}")
                #
                # 收敛检查
                if abs(prev_energy - current_energy) / (prev_energy + 1e-8) < self.convergence_threshold:
                    print(f"在 {iteration + 1} 次迭代后收敛")
                    break

                prev_energy = current_energy

            except Exception as e:
                print(f"迭代 {iteration + 1} 出错: {e}，恢复到上一步")
                self.UV = uv_backup
                self.regularization *= 10

        print("SLIM 优化完成")

    def _solve_slim_step(self):
        """执行一步SLIM优化，增强数值稳定性"""
        n = self.n_vertices
        A = sp.lil_matrix((2 * n, 2 * n))
        b = np.zeros(2 * n)

        # 添加小的正则化到对角线
        for i in range(2 * n):
            A[i, i] += self.regularization

        # 对每个面片，计算其对全局系统的贡献
        for i, face in enumerate(self.F):
            if self.face_areas_3d[i] < 1e-12:
                continue
            try:
                self._add_slim_terms(i, face, A, b)
            except Exception as e:
                print(f"面片 {i} 处理失败: {e}")
                continue

        # 求解线性系统
        A = A.tocsr()
        try:
            # 使用更稳定的求解器
            solution = spsolve(A, b)

            # 检查解的有效性
            if np.any(np.isnan(solution)) or np.any(np.isinf(solution)):
                raise ValueError("求解得到无效结果")

            self.UV[:, 0] = solution[:n]
            self.UV[:, 1] = solution[n:]

        except Exception as e:
            print(f"线性系统求解失败: {e}")
            raise

    def _add_slim_terms(self, face_idx, face, A, b):
        """为单个面片添加SLIM能量项到线性系统"""
        n = self.n_vertices
        v_indices = face

        # 检查面片有效性
        if self.face_areas_3d[face_idx] < 1e-12:
            return

        # 局部步骤: 计算理想旋转
        p0, p1, p2 = self.face_local_coords[face_idx]
        u0, u1, u2 = self.UV[v_indices]

        Ds = np.column_stack([p1 - p0, p2 - p0])
        Dm = np.column_stack([u1 - u0, u2 - u0])

        try:
            # 添加正则化
            Ds_reg = Ds + self.regularization * np.eye(2)
            Ds_inv = np.linalg.inv(Ds_reg)
            J = Dm @ Ds_inv

            # SVD分解求理想旋转
            U, s, Vt = np.linalg.svd(J)

            # 限制奇异值范围，避免极端变形
            s = np.clip(s, 0.1, 10.0)

            R = U @ Vt  # 理想旋转矩阵

            # 确保是纯旋转 (det > 0)
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt

        except np.linalg.LinAlgError:
            R = np.eye(2)  # 使用单位矩阵作为fallback

        # 全局步骤: 构建线性系统贡献
        area_weight = self.face_areas_3d[face_idx]

        # 简化的余切权重计算
        for i in range(3):
            for j in range(i + 1, 3):
                vi, vj = v_indices[i], v_indices[j]

                # 基于面积的权重
                w = area_weight

                # 拉普拉斯项
                A[vi, vi] += w
                A[vj, vj] += w
                A[vi, vj] -= w
                A[vj, vi] -= w

                A[n + vi, n + vi] += w
                A[n + vj, n + vj] += w
                A[n + vi, n + vj] -= w
                A[n + vj, n + vi] -= w

                # 右手边项：目标变形
                p_diff = self.face_local_coords[face_idx, i] - self.face_local_coords[face_idx, j]
                target_uv_diff = R @ p_diff

                b_contrib_u = w * target_uv_diff[0]
                b_contrib_v = w * target_uv_diff[1]

                b[vi] += b_contrib_u
                b[vj] -= b_contrib_u

                b[n + vi] += b_contrib_v
                b[n + vj] -= b_contrib_v

    def get_uv_coordinates(self):
        """返回最终的UV坐标"""
        return self.UV.copy()
    
# def uv_parameterization(vertices, faces, max_iterations=100, tolerance=1e-6):
#     """
#     改进的闭合mesh ARAP参数化
#     """
#     n_vertices = len(vertices)
#     n_faces = len(faces)
    
#     print(f"开始闭合mesh ARAP参数化，顶点数: {n_vertices}, 面数: {n_faces}")
    
#     # 1. 改进的初始化
#     print("步骤1: 立体投影初始化...")
#     uv_coords = _initialize_conformal(vertices, faces)
    
#     # 2. 构建拉普拉斯矩阵
#     print("步骤2: 构建拉普拉斯矩阵...")
#     L, face_areas = _build_laplacian_matrix(vertices, faces)
    
#     # 3. ARAP迭代优化
#     print("步骤3: ARAP迭代优化...")
#     for iteration in range(max_iterations):
#         uv_coords_old = uv_coords.copy()
        
#         # 3a. 计算最优旋转
#         rotations = _compute_optimal_rotations(vertices, faces, uv_coords, face_areas)
        
#         # 3b. 全局步骤
#         uv_coords = _solve_global_step(L, faces, rotations, face_areas, vertices, uv_coords)
        
#         # 检查收敛
#         diff = np.linalg.norm(uv_coords - uv_coords_old)
#         if iteration % 10 == 0:
#             print(f"迭代 {iteration+1}: 误差 = {diff:.2e}")
            
#         if diff < tolerance:
#             print(f"ARAP收敛于第{iteration+1}次迭代")
#             break
    
#     # 4. 最终归一化
#     print("步骤4: 最终归一化...")
#     uv_coords = _normalize_uv_coordinates(uv_coords)
    
#     return uv_coords



def process_point_cloud(pcd, output_dir, image_size = None, ball_radius=0.03, visualization=False):
    """
    读取点云，进行Ball Pivoting重建，然后用UV参数化进行展开
    
    参数:
    - pcd: 输入的点云
    - output_dir: 输出结果的目录
    - ball_radius: Ball Pivoting算法的球半径
    - visualization: 是否可视化中间结果
    """
    
    if pcd.is_empty():
        raise ValueError("无法读取点云文件或文件为空")
    
    # 保存原始点云坐标
    original_points = np.asarray(pcd.points)
    print(f"原始点云有 {len(original_points)} 个点")
    
    # 计算法向量（如果没有）
    if not pcd.has_normals():
        print("计算点云法向量...")
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        pcd.orient_normals_consistent_tangent_plane(k=20)
    
    # 使用Ball Pivoting算法进行表面重建，直到网格封闭
    is_watertight = False
    count = 0

    if visualization:
        print("可视化点云")
        o3d.visualization.draw_geometries([pcd])
    if image_size < 200:
        # 使用Open3D的Alpha Shape进行三角化重建
        print("使用Alpha Shape进行三角化重建...")
        mesh = alpha_shape_3d(pcd, alpha=5)
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.fill_holes()

        
    else:
        print(f"使用Ball Pivoting进行表面重建, 球半径 = {ball_radius}...")
        radii = [ball_radius, ball_radius * 2, ball_radius * 4]  # 多尺度重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.remove_degenerate_faces()
        tri_mesh.remove_duplicate_faces()
        tri_mesh.fill_holes()
        


    # 可视化重建结果
    if visualization:
        print("可视化重建的表面")
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    
    # 保存重建的mesh
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "reconstructed_mesh.obj"), mesh)
    print(f"重建的网格有 {len(mesh.triangles)} 个面")
    
    
    # UV展开 - 使用ARAP方法
    print("开始ARAP UV参数化...")
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    slim = RobustSLIMUV(vertices, faces)
    slim.optimize_slim()
    # uv_coords = uv_parameterization(vertices, faces)
    uv_coords = slim.get_uv_coordinates()
    
    print(f"UV参数化完成，生成 {len(uv_coords)} 个UV坐标")
    return {
        "original_points": original_points,
        "uv_coords": uv_coords,
        "faces": faces,
        "vertices": vertices
    }


# def create_uv_rgb_grid(vertices, uv_coords, image_size=512, output_path="uv_rgb_texture.png", output_metadata=None):
#     """
#     直接生成UV RGB图像，将3D顶点的XYZ坐标映射到UV空间的RGB通道
#     图像尺寸由UV坐标分布自动决定，确保所有像素都能铺开
    
#     参数:
#     - vertices: 3D顶点坐标 (N, 3)
#     - uv_coords: UV坐标 (N, 2)
#     - min_image_size: 最小图像尺寸
#     - output_path: 保存路径
#     """
    
#     # 如果uv_coords是列表，取第一个元素
#     if isinstance(uv_coords, list):
#         uv_coords = uv_coords[0]
    
#     # 归一化XYZ坐标到[0,1]范围
#     x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
#     y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
#     z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
#     r = (vertices[:, 0] - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(vertices[:, 0])
#     g = (vertices[:, 1] - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(vertices[:, 1])
#     b = (vertices[:, 2] - z_min) / (z_max - z_min) if z_max != z_min else np.zeros_like(vertices[:, 2])
    
#     # 将RGB值转换到[0,255]范围
#     rgb_colors = np.stack([r, g, b], axis=1) * 255
#     rgb_colors = rgb_colors.astype(np.uint8)
    
#     # 归一化UV坐标到[0,1]范围
#     u_min, u_max = uv_coords[:, 0].min(), uv_coords[:, 0].max()
#     v_min, v_max = uv_coords[:, 1].min(), uv_coords[:, 1].max()
    
#     u_normalized = (uv_coords[:, 0] - u_min) / (u_max - u_min) if u_max != u_min else np.zeros_like(uv_coords[:, 0])
#     v_normalized = (uv_coords[:, 1] - v_min) / (v_max - v_min) if v_max != v_min else np.zeros_like(uv_coords[:, 1])
    
    
#     # 使用最佳尺寸转换像素坐标
#     pixel_x = (u_normalized * (image_size - 1)).astype(np.int32)
#     pixel_y = ((1 - v_normalized) * (image_size - 1)).astype(np.int32)
    
#     # 验证重叠情况
#     pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
#     unique_coords, counts = np.unique(pixel_coords, axis=0, return_counts=True)
#     num_repeated = np.sum(counts > 1)
#     print(f"有 {num_repeated} 个像素坐标有重复（被多个顶点映射）")
    
#     # 创建空白图像
#     rgb_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

#     # 直接填充顶点像素（避免重复填充，记录已填充位置）
#     grid_map = np.zeros((image_size, image_size), dtype=np.int32)
#     processed_indices = []
#     grid_vertices = []
#     overflow_vertices = []

#     for i, (x, y) in enumerate(zip(pixel_x, pixel_y)):
#         if 0 <= x < image_size and 0 <= y < image_size:
#             if grid_map[x, y] == 0:  # 该像素未被占用
#                 grid_map[x, y] = i + 1  # 存储顶点索引+1（避免0）
#                 rgb_image[y, x] = rgb_colors[i]
#                 grid_vertices.append(vertices[i])
#             else:
#                 # 该像素已被占用，产生重复
#                 overflow_vertices.append(vertices[i])
#         else:
#             # 坐标超出网格范围
#             overflow_vertices.append(vertices[i])
    
#     # 统计非空白像素数量
#     non_white_pixels = np.sum(np.any(rgb_image != 255, axis=2))

#     print(f"非空白像素数量: {non_white_pixels}")
#     print(f"总顶点数量: {len(vertices)}")
#     print(f"像素利用率: {non_white_pixels/len(vertices)*100:.2f}%")
#     metadata = {
#         'x_min': float(vertices[:, 0].min()),
#         'x_max': float(vertices[:, 0].max()),
#         'y_min': float(vertices[:, 1].min()),
#         'y_max': float(vertices[:, 1].max()),
#         'z_min': float(vertices[:, 2].min()),
#         'z_max': float(vertices[:, 2].max()),
#         'u_min': float(uv_coords[:, 0].min()),
#         'u_max': float(uv_coords[:, 0].max()),
#         'v_min': float(uv_coords[:, 1].min()),
#         'v_max': float(uv_coords[:, 1].max()),
#         'original_vertex_count': len(vertices),
#         'size': image_size  # 假设使用默认尺寸
#     }

#     import json

#     with open(output_metadata, 'w') as f:
#         json.dump(metadata, f, indent=2)

#     # 确保图像格式正确
#     if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
#         # 保存图像到本地
#         pil_image = Image.fromarray(rgb_image, mode='RGB')
#         pil_image.save(output_path)
#         print(f"UV RGB纹理图像已保存到: {output_path}")
#     else:
#         print(f"错误：图像数据格式不正确，形状为 {rgb_image.shape}")

#     return grid_vertices,  overflow_vertices


import json
import os
import numpy as np
import cv2


def create_attribute_rgb_layers(vertices, uv_coords, attributes, image_size, output_dir, layer_count):
    """
    将多维属性映射到RGB图像，使用分组归一化策略
    """

    # 转换到像素坐标
    pixel_x = (uv_coords[:, 0] * (image_size - 1)).astype(np.int32)
    pixel_y = ((1 - uv_coords[:, 1]) * (image_size - 1)).astype(np.int32)

    # 检查边界
    valid_mask = (pixel_x >= 0) & (pixel_x < image_size) & (pixel_y >= 0) & (pixel_y < image_size)
    pixel_x = pixel_x[valid_mask]
    pixel_y = pixel_y[valid_mask]
    valid_attributes = attributes[valid_mask]

    print(f"有效像素点数量: {len(pixel_x)}")

    num_attr = valid_attributes.shape[1]

    # 定义分组
    group1_end = min(3, num_attr)
    group2_end = min(51, num_attr)

    # 计算各组归一化参数
    norm_params = {"layer": layer_count, "groups": {}}

    if group1_end > 0:
        g1_data = valid_attributes[:, 0:group1_end]
        norm_params["groups"]["group1"] = {"min": float(g1_data.min()), "max": float(g1_data.max())}

    if group2_end > group1_end:
        g2_data = valid_attributes[:, group1_end:group2_end]
        norm_params["groups"]["group2"] = {"min": float(g2_data.min()), "max": float(g2_data.max())}

    if num_attr > group2_end:
        g3_data = valid_attributes[:, group2_end:]
        norm_params["groups"]["group3"] = {"min": float(g3_data.min()), "max": float(g3_data.max())}

    def normalize_data(data, min_val, max_val):
        if max_val != min_val:
            return ((data - min_val) / (max_val - min_val) * 255).astype(np.uint8)
        return np.zeros_like(data, dtype=np.uint8)

    # 生成RGB图像
    for i in range(0, num_attr, 3):
        img = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255
        grid_map = np.zeros((image_size, image_size), dtype=bool)

        # 处理RGB三个通道
        current_channels = []
        for c in range(3):
            if i + c < num_attr:
                attr_idx = i + c
                channel_data = valid_attributes[:, attr_idx]

                # 选择归一化参数
                if attr_idx < group1_end:
                    min_val, max_val = norm_params["groups"]["group1"]["min"], norm_params["groups"]["group1"]["max"]
                elif attr_idx < group2_end:
                    min_val, max_val = norm_params["groups"]["group2"]["min"], norm_params["groups"]["group2"]["max"]
                else:
                    min_val, max_val = norm_params["groups"]["group3"]["min"], norm_params["groups"]["group3"]["max"]

                current_channels.append(normalize_data(channel_data, min_val, max_val))
            else:
                current_channels.append(np.full(len(pixel_x), 255, dtype=np.uint8))

        # 填充像素
        filled_count = 0
        for j, (x, y) in enumerate(zip(pixel_x, pixel_y)):
            if not grid_map[y, x]:
                grid_map[y, x] = True
                img[y, x] = [current_channels[0][j], current_channels[1][j], current_channels[2][j]]
                filled_count += 1

        # 保存图片
        output_path = os.path.join(output_dir, f"layer_{layer_count}_{i // 3}.png")
        cv2.imwrite(output_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_PNG_COMPRESSION, 6])

        # print(f"图像 {i // 3}: 填充像素数 {filled_count}")

    # 保存归一化参数
    json_path = os.path.join(output_dir, f"layer_{layer_count}_norm_params.json")
    with open(json_path, 'w') as f:
        json.dump(norm_params, f, indent=2)

