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
from scipy.sparse.csgraph import shortest_path
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


def _build_laplacian_matrix(vertices, faces):
    """构建拉普拉斯矩阵和面积权重"""
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    # 计算每个面的面积
    face_areas = []
    for face in faces:
        v0, v1, v2 = vertices[face]
        edge1 = v1 - v0
        edge2 = v2 - v0
        area = 0.5 * np.linalg.norm(np.cross(edge1, edge2))
        face_areas.append(area)
    
    face_areas = np.array(face_areas)
    
    # 构建拉普拉斯矩阵 (cotangent weights)
    row_indices = []
    col_indices = []
    data = []
    
    for f_idx, face in enumerate(faces):
        v0, v1, v2 = face
        
        # 计算cotangent权重
        p0, p1, p2 = vertices[face]
        
        # 边向量
        e0 = p1 - p2  # 对面顶点v0的边
        e1 = p2 - p0  # 对面顶点v1的边  
        e2 = p0 - p1  # 对面顶点v2的边
        
        # 计算cotangent值
        def safe_cotangent(a, b):
            cos_angle = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            sin_angle = np.sqrt(1 - cos_angle**2 + 1e-10)
            return cos_angle / sin_angle
        
        cot0 = safe_cotangent(-e1, -e2)  # 顶点v0处的cotangent
        cot1 = safe_cotangent(-e2, -e0)  # 顶点v1处的cotangent
        cot2 = safe_cotangent(-e0, -e1)  # 顶点v2处的cotangent
        
        # 添加边权重
        # 边 v1-v2
        row_indices.extend([v1, v2])
        col_indices.extend([v2, v1])
        data.extend([cot0, cot0])
        
        # 边 v2-v0
        row_indices.extend([v2, v0])
        col_indices.extend([v0, v2])
        data.extend([cot1, cot1])
        
        # 边 v0-v1
        row_indices.extend([v0, v1])
        col_indices.extend([v1, v0])
        data.extend([cot2, cot2])
    
    # 构建稀疏矩阵
    L = csr_matrix((data, (row_indices, col_indices)), shape=(n_vertices, n_vertices))
    
    # 计算对角线元素（每行和为0）
    diagonal = np.array(L.sum(axis=1)).flatten()
    L.setdiag(-diagonal)
    
    return L, face_areas

def _compute_optimal_rotations(vertices, faces, uv_coords, face_areas):
    """计算每个面的最优旋转矩阵"""
    n_faces = len(faces)
    rotations = []
    
    for f_idx, face in enumerate(faces):
        v0, v1, v2 = face
        
        # 3D坐标
        p0, p1, p2 = vertices[face]
        P = np.array([p1 - p0, p2 - p0]).T  # 2x3
        
        # 2D UV坐标
        uv0, uv1, uv2 = uv_coords[face]
        Q = np.array([uv1 - uv0, uv2 - uv0]).T  # 2x2
        
        # 计算协方差矩阵
        try:
            # 使用Moore-Penrose伪逆
            P_pinv = np.linalg.pinv(P)
            S = Q @ P_pinv  # 2x3
            
            # 提取2x2子矩阵进行SVD
            S_2x2 = S[:, :2]
            
            # SVD分解获得最优旋转
            U, _, Vt = np.linalg.svd(S_2x2)
            R = U @ Vt
            
            # 确保是旋转矩阵（行列式为1）
            if np.linalg.det(R) < 0:
                U[:, -1] *= -1
                R = U @ Vt
                
        except np.linalg.LinAlgError:
            # 如果SVD失败，使用单位矩阵
            R = np.eye(2)
        
        rotations.append(R)
    
    return rotations

def _initialize_closed_mesh_conformal(vertices, faces):
    """改进的闭合mesh初始化"""
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

def _solve_global_step_closed_mesh(L, faces, rotations, face_areas, vertices, uv_coords):
    """求解闭合mesh的全局步骤线性系统"""
    n_vertices = L.shape[0]
    
    # 构建右端项
    b = np.zeros((n_vertices, 2))
    
    for f_idx, face in enumerate(faces):
        R = rotations[f_idx]
        area = face_areas[f_idx]
        
        # 获取3D坐标
        p0, p1, p2 = vertices[face]
        local_3d = np.array([p1 - p0, p2 - p0])
        
        # 应用旋转到3D坐标的前两个分量
        local_3d_2d = local_3d[:, :2].T  # 2x2
        rotated_2d = R @ local_3d_2d  # 2x2
        
        # 为每个顶点添加贡献
        weight = area * 0.5  # 面积权重
        
        b[face[0]] += weight * (-rotated_2d[0, :] - rotated_2d[1, :])
        b[face[1]] += weight * rotated_2d[0, :]
        b[face[2]] += weight * rotated_2d[1, :]
    
    # 固定质心以避免平移自由度
    centroid = np.mean(uv_coords, axis=0)
    
    # 构建约束：所有顶点的质心保持不变
    constraint_matrix = np.ones((1, n_vertices)) / n_vertices
    
    # 使用拉格朗日乘数法求解约束优化问题
    # 构建增广系统
    # [L   A^T] [x]   [b]
    # [A    0 ] [λ] = [c]
    
    A = csr_matrix(constraint_matrix)
    zero_block = csr_matrix((1, 1))
    
    # 构建块矩阵
    top_block = sp.hstack([L, A.T])
    bottom_block = sp.hstack([A, zero_block])
    augmented_matrix = sp.vstack([top_block, bottom_block])
    
    new_uv_coords = uv_coords.copy()
    
    for dim in range(2):
        # 构建右端向量
        augmented_b = np.zeros(n_vertices + 1)
        augmented_b[:n_vertices] = b[:, dim]
        augmented_b[n_vertices] = centroid[dim]
        
        try:
            solution = spsolve(augmented_matrix, augmented_b)
            new_uv_coords[:, dim] = solution[:n_vertices]
        except Exception as e:
            print(f"约束系统求解失败 (dim={dim}): {e}")
            # 如果求解失败，使用简化的方法
            try:
                # 固定第一个顶点
                fixed_vertex = 0
                free_vertices = list(range(1, n_vertices))
                
                L_free = L[free_vertices][:, free_vertices]
                b_free = b[free_vertices, dim]
                
                # 减去固定顶点的贡献
                L_fixed_contrib = L[free_vertices, fixed_vertex].toarray().flatten()
                b_free -= L_fixed_contrib * uv_coords[fixed_vertex, dim]
                
                uv_free = spsolve(L_free, b_free)
                new_uv_coords[free_vertices, dim] = uv_free
                
            except Exception as e2:
                print(f"简化求解也失败: {e2}")
                continue
    
    return new_uv_coords

def _normalize_uv_coordinates(uv_coords):
    """更稳健的UV坐标归一化"""
    # 移除异常值
    uv_coords = np.clip(uv_coords, -100, 100)
    
    # 计算有效范围（排除异常值）
    percentile_low = np.percentile(uv_coords, 5, axis=0)
    percentile_high = np.percentile(uv_coords, 95, axis=0)
    
    # 使用百分位数进行归一化
    uv_range = percentile_high - percentile_low
    uv_range = np.where(uv_range < 1e-6, 1, uv_range)
    
    normalized_uv = (uv_coords - percentile_low) / uv_range
    
    # 确保在[0,1]范围内
    normalized_uv = np.clip(normalized_uv, 0, 1)
    
    return normalized_uv

def uv_parameterization(vertices, faces, max_iterations=100, tolerance=1e-6):
    """
    改进的闭合mesh ARAP参数化
    """
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    print(f"开始闭合mesh ARAP参数化，顶点数: {n_vertices}, 面数: {n_faces}")
    
    # 1. 改进的初始化
    print("步骤1: 立体投影初始化...")
    uv_coords = _initialize_closed_mesh_conformal(vertices, faces)
    
    # 2. 构建拉普拉斯矩阵
    print("步骤2: 构建拉普拉斯矩阵...")
    L, face_areas = _build_laplacian_matrix(vertices, faces)
    
    # 3. ARAP迭代优化
    print("步骤3: ARAP迭代优化...")
    for iteration in range(max_iterations):
        uv_coords_old = uv_coords.copy()
        
        # 3a. 计算最优旋转
        rotations = _compute_optimal_rotations(vertices, faces, uv_coords, face_areas)
        
        # 3b. 全局步骤
        uv_coords = _solve_global_step_closed_mesh(L, faces, rotations, face_areas, vertices, uv_coords)
        
        # 检查收敛
        diff = np.linalg.norm(uv_coords - uv_coords_old)
        if iteration % 10 == 0:
            print(f"迭代 {iteration+1}: 误差 = {diff:.2e}")
            
        if diff < tolerance:
            print(f"ARAP收敛于第{iteration+1}次迭代")
            break
    
    # 4. 最终归一化
    print("步骤4: 最终归一化...")
    uv_coords = _normalize_uv_coordinates(uv_coords)
    
    return uv_coords

def process_point_cloud(pcd, output_dir, ball_radius=0.03, visualization=False):
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
    while not is_watertight:
        count += 1
        print(f"使用Ball Pivoting进行表面重建, 球半径 = {ball_radius}...")
        radii = [ball_radius, ball_radius * 2, ball_radius * 4]  # 多尺度重建
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))

        # 转换为trimesh格式，方便进行UV展开
        vertices = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        tri_mesh.remove_degenerate_faces()
        tri_mesh.remove_duplicate_faces()
        tri_mesh.fill_holes()
    
        is_watertight = tri_mesh.is_watertight

        print(f"网格是否封闭: {is_watertight}")
        if count > 1:
            # 使用Open3D的Alpha Shape进行三角化重建
            print("使用Alpha Shape进行三角化重建...")
            mesh = alpha_shape_3d(pcd, alpha = 0.18)
            # tetra_mesh, pt_map = o3d.geometry.TetraMesh.create_from_point_cloud(pcd)
            # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(
            #     pcd, ball_radius, tetra_mesh, pt_map
            # )
            print(np.asarray(pcd.points), np.asarray(mesh.vertices))
            # mesh.remove_duplicated_triangles()
            # mesh.remove_degenerate_triangles()
            # mesh.remove_duplicated_vertices()
            # mesh.remove_non_manifold_edges()

            # 转换为trimesh格式，方便进行UV展开
            vertices = np.asarray(mesh.vertices)
            faces = np.asarray(mesh.triangles)
            tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
            tri_mesh.fill_holes()
            is_watertight = tri_mesh.is_watertight


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
    
    uv_coords = uv_parameterization(vertices, faces)
    
    print(f"UV参数化完成，生成 {len(uv_coords)} 个UV坐标")
    return {
        "original_points": original_points,
        "uv_coords": uv_coords,
        "faces": faces,
        "vertices": vertices
    }


def create_uv_rgb_grid(vertices, uv_coords, image_size=512, output_path="uv_rgb_texture.png", output_metadata=None):
    """
    直接生成UV RGB图像，将3D顶点的XYZ坐标映射到UV空间的RGB通道
    图像尺寸由UV坐标分布自动决定，确保所有像素都能铺开
    
    参数:
    - vertices: 3D顶点坐标 (N, 3)
    - uv_coords: UV坐标 (N, 2)
    - min_image_size: 最小图像尺寸
    - output_path: 保存路径
    """
    
    # 如果uv_coords是列表，取第一个元素
    if isinstance(uv_coords, list):
        uv_coords = uv_coords[0]
    
    # 归一化XYZ坐标到[0,1]范围
    x_min, x_max = vertices[:, 0].min(), vertices[:, 0].max()
    y_min, y_max = vertices[:, 1].min(), vertices[:, 1].max()
    z_min, z_max = vertices[:, 2].min(), vertices[:, 2].max()
    
    r = (vertices[:, 0] - x_min) / (x_max - x_min) if x_max != x_min else np.zeros_like(vertices[:, 0])
    g = (vertices[:, 1] - y_min) / (y_max - y_min) if y_max != y_min else np.zeros_like(vertices[:, 1])
    b = (vertices[:, 2] - z_min) / (z_max - z_min) if z_max != z_min else np.zeros_like(vertices[:, 2])
    
    # 将RGB值转换到[0,255]范围
    rgb_colors = np.stack([r, g, b], axis=1) * 255
    rgb_colors = rgb_colors.astype(np.uint8)
    
    # 归一化UV坐标到[0,1]范围
    u_min, u_max = uv_coords[:, 0].min(), uv_coords[:, 0].max()
    v_min, v_max = uv_coords[:, 1].min(), uv_coords[:, 1].max()
    
    u_normalized = (uv_coords[:, 0] - u_min) / (u_max - u_min) if u_max != u_min else np.zeros_like(uv_coords[:, 0])
    v_normalized = (uv_coords[:, 1] - v_min) / (v_max - v_min) if v_max != v_min else np.zeros_like(uv_coords[:, 1])
    
    
    # 使用最佳尺寸转换像素坐标
    pixel_x = (u_normalized * (image_size - 1)).astype(np.int32)
    pixel_y = ((1 - v_normalized) * (image_size - 1)).astype(np.int32)
    
    # 验证重叠情况
    pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
    unique_coords, counts = np.unique(pixel_coords, axis=0, return_counts=True)
    num_repeated = np.sum(counts > 1)
    print(f"有 {num_repeated} 个像素坐标有重复（被多个顶点映射）")
    
    # 创建空白图像
    rgb_image = np.ones((image_size, image_size, 3), dtype=np.uint8) * 255

    # 直接填充顶点像素（避免重复填充，记录已填充位置）
    grid_map = np.zeros((image_size, image_size), dtype=np.int32)
    processed_indices = []
    grid_vertices = []
    overflow_vertices = []

    for i, (x, y) in enumerate(zip(pixel_x, pixel_y)):
        if 0 <= x < image_size and 0 <= y < image_size:
            if grid_map[x, y] == 0:  # 该像素未被占用
                grid_map[x, y] = i + 1  # 存储顶点索引+1（避免0）
                rgb_image[y, x] = rgb_colors[i]
                grid_vertices.append(vertices[i])
            else:
                # 该像素已被占用，产生重复
                overflow_vertices.append(vertices[i])
        else:
            # 坐标超出网格范围
            overflow_vertices.append(vertices[i])
    
    # 统计非空白像素数量
    non_white_pixels = np.sum(np.any(rgb_image != 255, axis=2))

    print(f"非空白像素数量: {non_white_pixels}")
    print(f"总顶点数量: {len(vertices)}")
    print(f"像素利用率: {non_white_pixels/len(vertices)*100:.2f}%")
    metadata = {
        'x_min': float(vertices[:, 0].min()),
        'x_max': float(vertices[:, 0].max()),
        'y_min': float(vertices[:, 1].min()),
        'y_max': float(vertices[:, 1].max()),
        'z_min': float(vertices[:, 2].min()),
        'z_max': float(vertices[:, 2].max()),
        'u_min': float(uv_coords[:, 0].min()),
        'u_max': float(uv_coords[:, 0].max()),
        'v_min': float(uv_coords[:, 1].min()),
        'v_max': float(uv_coords[:, 1].max()),
        'original_vertex_count': len(vertices),
        'size': image_size  # 假设使用默认尺寸
    }

    import json

    with open(output_metadata, 'w') as f:
        json.dump(metadata, f, indent=2)

    # 确保图像格式正确
    if rgb_image.ndim == 3 and rgb_image.shape[2] == 3:
        # 保存图像到本地
        pil_image = Image.fromarray(rgb_image, mode='RGB')
        pil_image.save(output_path)
        print(f"UV RGB纹理图像已保存到: {output_path}")
    else:
        print(f"错误：图像数据格式不正确，形状为 {rgb_image.shape}")

    return grid_vertices,  overflow_vertices
    

    

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="点云重建和LSCM UV展开工具")
#     parser.add_argument("--output", "-o", default="output", help="输出目录")
#     parser.add_argument("--radius", "-r", type=float, default=0.005, help="Ball Pivoting球半径")
#     parser.add_argument("--visualize", "-v", action="store_true", help="是否显示3D可视化结果")
    
#     args = parser.parse_args()
    
#     # 创建输出目录
#     if not os.path.exists(args.output):
#         os.makedirs(args.output)
    
#     # 使用bunny数据集作为示例
#     bunny = o3d.data.BunnyMesh()
#     gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
#     gt_mesh.compute_vertex_normals()
#     pcd = gt_mesh.sample_points_poisson_disk(3000)
#     start_time = time.time()
#     results = process_point_cloud(pcd, args.output, args.radius, args.visualize)
#     point = results["vertices"]
#     uv_coords = results["uv_coords"]
#     faces = results["faces"]


#     create_uv_rgb_grid(point, uv_coords, output_path=os.path.join(args.output, "uv_rgb_texture.png"))

#     print(f"总处理时间: {time.time() - start_time:.2f} 秒")
