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
from scipy.sparse.linalg import spsolve
from scipy.spatial.distance import cdist
import warnings

def build_adjacency_matrix(faces, num_vertices):
    """构建邻接矩阵用于最短路径计算"""
    edges = defaultdict(list)
    
    # 从面构建边
    for face in faces:
        for i in range(3):
            v1, v2 = face[i], face[(i + 1) % 3]
            edges[v1].append(v2)
            edges[v2].append(v1)
    
    # 构建邻接矩阵
    row_indices = []
    col_indices = []
    data = []
    
    for v1 in edges:
        for v2 in edges[v1]:
            row_indices.append(v1)
            col_indices.append(v2)
            data.append(1.0)  # 可以用实际边长度替代
    
    return csr_matrix((data, (row_indices, col_indices)), shape=(num_vertices, num_vertices))

def find_shortest_path(vertices, faces, start_vertex, end_vertex):
    """使用Dijkstra算法找到最短路径"""
    num_vertices = len(vertices)
    adj_matrix = build_adjacency_matrix(faces, num_vertices)
    
    # 计算最短路径
    dist_matrix, predecessors = shortest_path(
        adj_matrix, directed=False, indices=start_vertex, 
        return_predecessors=True
    )
    
    # 重构路径
    path = []
    current = end_vertex
    while current != -9999 and current != start_vertex:
        path.append(current)
        current = predecessors[current]
    
    if current == start_vertex:
        path.append(start_vertex)
        path.reverse()
        return path
    else:
        return [start_vertex, end_vertex]  # 如果找不到路径，返回直接连接

def uv_parameterization(vertices, faces):
    """
    使用LSCM (Least Squares Conformal Mapping) 进行UV参数化
    """
    
    print("使用LSCM进行UV参数化...")
    
    # 检查是否有边界
    boundary_loop = igl.boundary_loop(faces)
    
    if len(boundary_loop) > 0:
        print(f"检测到边界，边界顶点数量: {len(boundary_loop)}")
        print(vertices.shape, faces.shape)
        return uv_parameterization_arap_with_boundary(vertices, faces, boundary_loop)


# def uv_parameterization_lscm_with_boundary(vertices, faces, boundary_loop):
#     """对有边界的网格使用LSCM参数化"""

#     # 选择边界上的两个固定点进行约束
#     # 通常选择边界上距离最远的两个点
#     boundary_vertices = vertices[boundary_loop]
    
#     # 计算边界顶点之间的距离
#     distances = np.linalg.norm(
#         boundary_vertices[:, None] - boundary_vertices[None, :], axis=2
#     )
#     max_dist_idx = np.unravel_index(np.argmax(distances), distances.shape)
    
#     # 选择边界上距离最远的两个点作为固定点
#     fixed_idx1 = boundary_loop[max_dist_idx[0]]
#     fixed_idx2 = boundary_loop[max_dist_idx[1]]
    
#     # 设置固定点的UV坐标
#     fixed_indices = np.array([fixed_idx1, fixed_idx2], dtype=np.int32)
#     fixed_uv = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
    
#     print(f"LSCM固定点: {fixed_idx1}, {fixed_idx2}")
    
#     # 使用LSCM进行参数化
#     uv_coords = igl.lscm(vertices, faces, fixed_indices, fixed_uv)
#     print(uv_coords,"over")
#     # print(f"LSCM参数化成功，生成 {len(uv_coords)} 个UV坐标")
#     return uv_coords

def uv_parameterization_arap_with_boundary(vertices, faces, boundary_loop, 
                                         boundary_shape='circle', 
                                         max_iterations=50, 
                                         tolerance=1e-6):
    """
    使用ARAP方法进行UV参数化
    
    Parameters:
    -----------
    vertices : np.ndarray, shape (n_vertices, 3)
        3D顶点坐标
    faces : np.ndarray, shape (n_faces, 3)  
        三角面片索引
    boundary_loop : list or np.ndarray
        边界环的顶点索引（按顺序）
    boundary_shape : str, default 'circle'
        边界映射形状 ('circle' 或 'square')
    max_iterations : int, default 50
        最大迭代次数
    tolerance : float, default 1e-6
        收敛阈值
        
    Returns:
    --------
    uv_coords : np.ndarray, shape (n_vertices, 2)
        UV坐标
    """
    
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    # 1. 初始化边界UV坐标
    uv_coords = np.zeros((n_vertices, 2))
    boundary_uvs = _map_boundary_to_shape(boundary_loop, boundary_shape)
    uv_coords[boundary_loop] = boundary_uvs
    
    # 2. 构建拉普拉斯矩阵和面积权重
    L, face_areas = _build_laplacian_matrix(vertices, faces)
    
    # 3. 初始化内部顶点UV坐标（使用调和映射）
    uv_coords = _initialize_interior_uvs(L, boundary_loop, boundary_uvs, n_vertices)
    
    # 4. ARAP迭代优化
    for iteration in range(max_iterations):
        uv_coords_old = uv_coords.copy()
        
        # 4a. 局部步骤：计算最优旋转矩阵
        rotations = _compute_optimal_rotations(vertices, faces, uv_coords, face_areas)
        
        # 4b. 全局步骤：求解线性系统
        uv_coords = _solve_global_step(L, faces, rotations, face_areas, 
                                     boundary_loop, boundary_uvs, vertices, uv_coords)
        
        # 检查收敛
        diff = np.linalg.norm(uv_coords - uv_coords_old)
        if diff < tolerance:
            print(f"ARAP收敛于第{iteration+1}次迭代，误差: {diff:.2e}")
            break
    else:
        warnings.warn(f"ARAP未在{max_iterations}次迭代内收敛")
    
    return uv_coords

def _map_boundary_to_shape(boundary_loop, shape='circle'):
    """将边界顶点映射到指定形状"""
    n_boundary = len(boundary_loop)
    
    if shape == 'circle':
        # 映射到单位圆
        angles = np.linspace(0, 2*np.pi, n_boundary, endpoint=False)
        boundary_uvs = 0.5 * np.column_stack([np.cos(angles), np.sin(angles)]) + 0.5
    elif shape == 'square':
        # 映射到单位正方形边界
        boundary_uvs = np.zeros((n_boundary, 2))
        
        for i in range(n_boundary):
            t = i / n_boundary
            if t < 0.25:  # 底边
                boundary_uvs[i] = [4*t, 0]
            elif t < 0.5:  # 右边
                boundary_uvs[i] = [1, 4*(t-0.25)]
            elif t < 0.75:  # 顶边
                boundary_uvs[i] = [1-4*(t-0.5), 1]
            else:  # 左边
                boundary_uvs[i] = [0, 1-4*(t-0.75)]
    else:
        raise ValueError("boundary_shape必须是'circle'或'square'")
    
    return boundary_uvs

def _build_laplacian_matrix(vertices, faces):
    """构建Cotangent拉普拉斯矩阵"""
    n_vertices = len(vertices)
    n_faces = len(faces)
    
    # 计算每个面的面积和cotangent权重
    face_areas = np.zeros(n_faces)
    I, J, V = [], [], []
    
    for f_idx, face in enumerate(faces):
        v0, v1, v2 = vertices[face]
        
        # 计算边向量
        e0 = v1 - v0  # v0 -> v1
        e1 = v2 - v1  # v1 -> v2  
        e2 = v0 - v2  # v2 -> v0
        
        # 面积
        face_areas[f_idx] = 0.5 * np.linalg.norm(np.cross(e0, -e2))
        
        # Cotangent权重
        def cotangent(v1, v2):
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
            cos_angle = np.clip(cos_angle, -1, 1)
            sin_angle = np.sqrt(1 - cos_angle**2 + 1e-10)
            return cos_angle / sin_angle
        
        # 每条边的cotangent权重
        cot0 = cotangent(-e2, e0)  # 角度在v0
        cot1 = cotangent(-e0, e1)  # 角度在v1
        cot2 = cotangent(-e1, e2)  # 角度在v2
        
        # 添加到稀疏矩阵
        edges = [(face[1], face[2], cot0), (face[2], face[0], cot1), (face[0], face[1], cot2)]
        
        for i, j, cot in edges:
            I.extend([i, j])
            J.extend([j, i])
            V.extend([cot, cot])
    
    # 构建拉普拉斯矩阵
    L = csr_matrix((V, (I, J)), shape=(n_vertices, n_vertices))
    
    # 对角线元素
    L_diag = -np.array(L.sum(axis=1)).flatten()
    L = L + diags(L_diag, shape=(n_vertices, n_vertices))
    
    return L, face_areas

def _initialize_interior_uvs(L, boundary_loop, boundary_uvs, n_vertices):
    """使用调和映射初始化内部顶点的UV坐标"""
    uv_coords = np.zeros((n_vertices, 2))
    uv_coords[boundary_loop] = boundary_uvs
    
    # 找到内部顶点
    interior_vertices = [i for i in range(n_vertices) if i not in boundary_loop]
    
    if len(interior_vertices) == 0:
        return uv_coords
    
    # 构建内部顶点的线性系统
    L_interior = L[interior_vertices][:, interior_vertices]
    
    # 求解每个维度
    for dim in range(2):
        b = np.zeros(len(interior_vertices))
        # 添加边界条件的贡献
        for i, boundary_idx in enumerate(boundary_loop):
            boundary_contrib = L[interior_vertices, boundary_idx].toarray().flatten()
            b -= boundary_contrib * boundary_uvs[i, dim]
        
        # 求解线性系统
        if L_interior.shape[0] > 0:
            uv_interior = spsolve(L_interior, b)
            uv_coords[interior_vertices, dim] = uv_interior
    
    return uv_coords

def _compute_optimal_rotations(vertices, faces, uv_coords, face_areas):
    """计算每个面的最优旋转矩阵"""
    n_faces = len(faces)
    rotations = np.zeros((n_faces, 2, 2))
    
    for f_idx, face in enumerate(faces):
        # 计算3D面的局部坐标系
        v0, v1, v2 = vertices[face]
        e1_3d = v1 - v0
        e2_3d = v2 - v0
        
        # 构建3D面的正交基
        u1 = e1_3d / (np.linalg.norm(e1_3d) + 1e-10)
        normal = np.cross(e1_3d, e2_3d)
        normal = normal / (np.linalg.norm(normal) + 1e-10)
        u2 = np.cross(normal, u1)
        
        # 将3D坐标投影到2D
        X = np.array([
            [np.dot(e1_3d, u1), np.dot(e1_3d, u2)],
            [np.dot(e2_3d, u1), np.dot(e2_3d, u2)]
        ])
        
        # UV坐标差值
        uv0, uv1, uv2 = uv_coords[face]
        U = np.array([
            uv1 - uv0,
            uv2 - uv0
        ])
        
        # 计算雅可比矩阵
        if np.abs(np.linalg.det(U)) > 1e-10:
            J = X @ np.linalg.inv(U)
            
            # SVD分解找最优旋转
            try:
                U_svd, S, Vt = np.linalg.svd(J)
                R = U_svd @ Vt
                
                # 确保是旋转矩阵（行列式为正）
                if np.linalg.det(R) < 0:
                    U_svd[:, -1] *= -1
                    R = U_svd @ Vt
                    
                rotations[f_idx] = R
            except:
                rotations[f_idx] = np.eye(2)
        else:
            rotations[f_idx] = np.eye(2)
    
    return rotations

def _solve_global_step(L, faces, rotations, face_areas, boundary_loop, boundary_uvs, vertices, uv_coords):
    """求解全局步骤的线性系统"""
    n_vertices = L.shape[0]
    
    # 构建右端项
    b = np.zeros((n_vertices, 2))
    
    for f_idx, face in enumerate(faces):
        R = rotations[f_idx]
        area = face_areas[f_idx]
        
        # 获取当前UV坐标
        uv0, uv1, uv2 = uv_coords[face]
        
        # 计算局部UV坐标
        local_uvs = np.array([uv1 - uv0, uv2 - uv0])
        
        # 应用旋转
        rotated_uvs = R @ local_uvs
        
        # 为每个顶点添加贡献
        b[face[0]] += area * (-rotated_uvs[0] - rotated_uvs[1])
        b[face[1]] += area * rotated_uvs[0]
        b[face[2]] += area * rotated_uvs[1]
    
    # 应用边界条件
    interior_vertices = [i for i in range(n_vertices) if i not in boundary_loop]
    
    if len(interior_vertices) == 0:
        return uv_coords
    
    # 修改系统矩阵和右端项
    L_interior = L[interior_vertices][:, interior_vertices]
    b_interior = b[interior_vertices]
    
    # 减去边界贡献
    for i, boundary_idx in enumerate(boundary_loop):
        boundary_contrib = L[interior_vertices, boundary_idx].toarray().flatten()
        b_interior[:, 0] -= boundary_contrib * boundary_uvs[i, 0]
        b_interior[:, 1] -= boundary_contrib * boundary_uvs[i, 1]
    
    # 求解线性系统
    uv_interior = np.zeros((len(interior_vertices), 2))
    for dim in range(2):
        if L_interior.shape[0] > 0:
            uv_interior[:, dim] = spsolve(L_interior, b_interior[:, dim])
    
    # 组装完整的UV坐标
    new_uv_coords = uv_coords.copy()
    new_uv_coords[interior_vertices] = uv_interior
    
    return new_uv_coords




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
    
    # 使用Ball Pivoting算法进行表面重建
    print(f"使用Ball Pivoting进行表面重建, 球半径 = {ball_radius}...")
    radii = [ball_radius, ball_radius * 2, ball_radius * 4]  # 多尺度重建
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(radii))
    
    # 可视化重建结果
    if visualization:
        print("可视化重建的表面")
        o3d.visualization.draw_geometries([mesh], mesh_show_wireframe=True)
    
    # 保存重建的mesh
    o3d.io.write_triangle_mesh(os.path.join(output_dir, "reconstructed_mesh.obj"), mesh)
    print(f"重建的网格有 {len(mesh.triangles)} 个面")
    
    # 转换为trimesh格式，方便进行UV展开
    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.triangles)
    
    # 使用trimesh检查网格并修复
    print("检查网格并修复...")
    tri_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
    # tri_mesh.remove_duplicate_faces()
    # tri_mesh.remove_unreferenced_vertices()
    # tri_mesh.fill_holes()
    
    # 确保网格是2-manifold的
    if not tri_mesh.is_watertight:
        print("警告: 网格不是完全封闭的，可能会影响UV展开效果")
    
    # UV展开 - 使用LSCM方法
    print("开始LSCM UV参数化...")
    vertices = tri_mesh.vertices
    faces = tri_mesh.faces
    
    uv_coords = uv_parameterization(vertices, faces)
    
    if uv_coords is None:
        raise ValueError("LSCM UV参数化失败")
    

    # # 创建包含UV的OBJ格式文件
    # print("创建带有UV的OBJ文件...")
    # with open(os.path.join(output_dir, "mesh_with_uv.obj"), 'w') as f:
    #     # 写入顶点
    #     for v in vertices:
    #         f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        
    #     # 写入UV坐标
    #     for uv in uv_coords:
    #         f.write(f"vt {uv[0]} {uv[1]}\n")
        
    #     # 写入面和对应的UV索引
    #     for i, face in enumerate(faces):
    #         f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")
    
    # print(f"处理完成！所有文件已保存到 {output_dir}")
    return {
        "original_points": original_points,
        "uv_coords": uv_coords,
        "faces": faces,
        "vertices": vertices
    }

def create_uv_rgb_image(vertices, uv_coords, min_image_size=1000, output_path="uv_rgb_texture.png"):
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
    
    # 计算需要的最小图像尺寸以避免重叠
    def calculate_required_size(u_norm, v_norm, min_size):
        """计算避免像素重叠所需的最小图像尺寸"""
        best_size = min_size
        
        for size in range(min_size, min_size * 10, 100):  # 最多尝试到10倍最小尺寸
            # 转换为像素坐标
            pixel_x = (u_norm * (size - 1)).astype(np.int32)
            pixel_y = ((1 - v_norm) * (size - 1)).astype(np.int32)
            
            # 检查重叠
            pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
            unique_coords = np.unique(pixel_coords, axis=0)
            print(len(unique_coords), len(pixel_coords), size)
            if len(unique_coords) == len(pixel_coords):
                # 没有重叠，找到合适的尺寸
                best_size = size
                break
        
        return best_size
    
    # 计算最佳图像尺寸
    optimal_size = calculate_required_size(u_normalized, v_normalized, min_image_size)
    print(f"计算得出的最佳图像尺寸: {optimal_size}x{optimal_size}")
    
    # 使用最佳尺寸转换像素坐标
    pixel_x = (u_normalized * (optimal_size - 1)).astype(np.int32)
    pixel_y = ((1 - v_normalized) * (optimal_size - 1)).astype(np.int32)
    
    # 验证重叠情况
    pixel_coords = np.stack([pixel_x, pixel_y], axis=1)
    unique_coords, counts = np.unique(pixel_coords, axis=0, return_counts=True)
    num_repeated = np.sum(counts > 1)
    print(f"有 {num_repeated} 个像素坐标有重复（被多个顶点映射）")
    
    # 创建空白图像
    rgb_image = np.ones((optimal_size, optimal_size, 3), dtype=np.uint8) * 255

    # 直接填充顶点像素
    for i in range(len(pixel_x)):
        x, y = pixel_x[i], pixel_y[i]
        if 0 <= x < optimal_size and 0 <= y < optimal_size:
            rgb_image[y, x] = rgb_colors[i]
    
    # 统计非空白像素数量
    non_white_pixels = np.sum(np.any(rgb_image != 255, axis=2))
    print(f"非空白像素数量: {non_white_pixels}")
    print(f"总顶点数量: {len(vertices)}")
    print(f"像素利用率: {non_white_pixels/len(vertices)*100:.2f}%")
    
    # 保存图像到本地
    Image.fromarray(rgb_image).save(output_path)
    print(f"UV RGB纹理图像已保存到: {output_path}")
    

    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="点云重建和LSCM UV展开工具")
    parser.add_argument("--output", "-o", default="output", help="输出目录")
    parser.add_argument("--radius", "-r", type=float, default=0.005, help="Ball Pivoting球半径")
    parser.add_argument("--visualize", "-v", action="store_true", help="是否显示3D可视化结果")
    
    args = parser.parse_args()
    
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 使用bunny数据集作为示例
    bunny = o3d.data.BunnyMesh()
    gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
    gt_mesh.compute_vertex_normals()
    pcd = gt_mesh.sample_points_poisson_disk(3000)
    start_time = time.time()
    results = process_point_cloud(pcd, args.output, args.radius, args.visualize)
    point = results["vertices"]
    uv_coords = results["uv_coords"]
    faces = results["faces"]

    create_uv_rgb_image(point, uv_coords, output_path=os.path.join(args.output, "uv_rgb_texture.png"))

    print(f"总处理时间: {time.time() - start_time:.2f} 秒")
