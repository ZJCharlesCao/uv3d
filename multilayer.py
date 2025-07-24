import argparse
import os
import time
import numpy as np
import open3d as o3d
from typing import Dict, List, Tuple, Optional
from geometry import process_point_cloud, create_uv_rgb_grid
from decode_ply import decode_with_metadata,evaluate_quality
import math
from compression import ImageCompressor
import shutil
from gaussian_model import GaussianModel
from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist
from density import PointCloudDensityController

def find_unused_points(original_pcd, mesh_vertices):
    """
    使用向量化操作的高效版本
    
    Parameters:
    -----------
    original_pcd : numpy.ndarray or open3d.geometry.PointCloud
        原始点云
    mesh_vertices : numpy.ndarray
        重建后的mesh顶点
        
    Returns:
    --------
    dict
        包含未参与重建的点云信息
    """
    
    # 处理输入数据
    if isinstance(original_pcd, o3d.geometry.PointCloud):
        original_points = np.asarray(original_pcd.points)
    else:
        original_points = original_pcd
    
    # 确保数据类型一致
    original_points = original_points.astype(np.float64)
    mesh_vertices = np.asarray(mesh_vertices)
    mesh_vertices = mesh_vertices.astype(np.float64)
    n_original = len(original_points)
    n_vertices = len(mesh_vertices)
    
    print(f"原始点云数量: {n_original}")
    print(f"重建顶点数量: {n_vertices}")
    
    # 使用broadcasting进行向量化比较
    # 形状: (n_vertices, n_original, 3)
    diff = mesh_vertices[:, np.newaxis, :] - original_points[np.newaxis, :, :]
    
    # 计算欧氏距离
    distances = np.linalg.norm(diff, axis=2)
    
    # 找到每个mesh顶点对应的最近原始点
    closest_original_indices = np.argmin(distances, axis=1)
    
    # 检查最近距离是否足够小（表示是同一个点）
    min_distances = np.min(distances, axis=1)
    valid_matches = min_distances < 1e-10
    
    # 获取有效匹配的原始点索引
    used_indices = np.unique(closest_original_indices[valid_matches])
    
    # 创建掩码
    used_mask = np.zeros(n_original, dtype=bool)
    used_mask[used_indices] = True
    
    # 未使用的点
    unused_indices = np.where(~used_mask)[0]
    unused_points = original_points[unused_indices]
    
    # 统计信息
    stats = {
        'total_original_points': n_original,
        'total_vertices': n_vertices,
        'used_points_count': len(used_indices),
        'unused_points_count': len(unused_indices),
        'usage_rate': len(used_indices) / n_original * 100,
        'min_match_distance': np.min(min_distances[valid_matches]) if np.any(valid_matches) else 0,
        'max_match_distance': np.max(min_distances[valid_matches]) if np.any(valid_matches) else 0
    }
    
    print(f"未参与重建的点: {stats['unused_points_count']}")

    
    return {
        'unused_points': unused_points,
        'stats': stats,
    }


def multilayer_encode(input_pcd: o3d.geometry.PointCloud, 
                        output_dir: str, 
                        radius: float = 0.01, 
                        grid_size: int = 512,
                        max_iterations: int = 10,
                        max_layers: int = 8,
                        visualize: bool = False) -> Dict:
    """
    迭代处理管道
    
    Args:
        input_pcd: 输入点云
        output_dir: 输出目录
        radius: BPA球半径
        grid_size: 网格大小
        max_iterations: 最大迭代次数
        max_layers: 最大层数限制
        visualize: 是否可视化
        
    Returns:
        处理结果字典
    """
    print("=" * 60)
    print("开始迭代处理管道")
    print("=" * 60)
    
    current_pcd = input_pcd
    iteration = 0
    layer_count = 0


    initial_point_count = len(input_pcd.points)
    
    # 停止条件参数
    min_remaining_points = max(10, int(initial_point_count * 0.001))  # 至少10个点或初始点数的0.1%
    
    while len(current_pcd.points) > 0 and iteration < max_iterations and layer_count < max_layers:

        iteration += 1
        layer_count += 1
        
        print(f"\n第 {iteration} 轮迭代 (Layer {layer_count})")
        filtered_pcd = current_pcd
        if layer_count < 4:
            pd_controller = PointCloudDensityController(grid_size=0.2,
                                                        k_neighbors=15,
                                                        density_threshold=1,
                                                        preserve_ratio=0.33 * layer_count)
            filtered_pcd,_ = pd_controller.adaptive_density_control(current_pcd.points)
        print(f"当前处理点数: {len(filtered_pcd.points)}")
        ratio = math.log10(3 * iteration) + 1
        print(f"当前size: {int(grid_size/int(ratio))}")
        # 第一步：点云处理（BPA重建）
        current_layer_init = process_point_cloud(filtered_pcd, output_dir,image_size=int(grid_size/int(ratio)), ball_radius=radius, visualization=visualize)

        if len(current_layer_init["vertices"]) == 0:
            print("没有顶点参与重建，结束迭代")
            break
        # 第二步：网格处理
        current_layer, unused = create_uv_rgb_grid(current_layer_init["vertices"], current_layer_init["uv_coords"],
                                                    image_size=int(grid_size/int(ratio)), output_path=os.path.join(output_dir, f"uv_rgb_texture_layer_{layer_count}.png"), output_metadata=os.path.join(output_dir, f"uv_rgb_texture_layer_{layer_count}_metadata.json"))
        layer_dict = find_unused_points(current_pcd, current_layer)
        
        
        # 收集未处理的点云用于下一轮迭代
        unprocessed_points = layer_dict['unused_points']

        if len(unprocessed_points) > 0:
            radius = radius + 0.003  # 每轮迭代增大半径
            current_pcd = o3d.geometry.PointCloud()
            current_pcd.points = o3d.utility.Vector3dVector(unprocessed_points)
            
            current_remaining = len(current_pcd.points)
            
            # 更新停止条件检查
            # 1. 剩余点数过少
            if current_remaining < min_remaining_points:
                print(f"剩余点数过少 ({current_remaining} < {min_remaining_points})，结束迭代")
                break
            
            # 2. 达到最大层数限制
            if layer_count >= max_layers:
                print(f"达到最大层数限制 ({max_layers})，结束迭代")
                break
                
        else:
            print("所有点都已处理完成")
            break
        
    # 计算全程参与重建的点集（input_pcd - current_pcd）
    input_points = np.asarray(input_pcd.points)
    if len(current_pcd.points) > 0:
        remaining_points = np.asarray(current_pcd.points)
        # 找到input_points中不在remaining_points中的点（全程参与重建的点）
        # 使用广播和np.isclose避免浮点误差
        mask = np.ones(len(input_points), dtype=bool)
        for pt in remaining_points:
            close = np.isclose(input_points, pt, atol=1e-10).all(axis=1)
            mask[close] = False
        fully_used_points = input_points[mask]
    else:
        fully_used_points = input_points

    print(f"全程参与重建的点数: {len(fully_used_points)}")

    # 可选：将全程参与重建的点集保存为点云文件
    used_pcd = o3d.geometry.PointCloud()
    used_pcd.points = o3d.utility.Vector3dVector(fully_used_points)

    print(f"\n迭代处理完成，总轮次: {iteration}, 总层数: {layer_count}")
    print(f"剩余未处理点数: {len(current_pcd.points) if len(current_pcd.points) > 0 else 0}")
    
    return {
        "total_layers": layer_count,
        "initial_point_count": initial_point_count,
        "remaining_points": len(current_pcd.points) if len(current_pcd.points) > 0 else 0,
        "final_unprocessed_pcd": current_pcd if len(current_pcd.points) > 0 else None,
        "points_in_layer": used_pcd.points if len(used_pcd.points) > 0 else None,
    }

def multilayer_deocde(output_dir, layer_count):
    """
    解码多层点云并合并
    """
    print(f"\n开始解码多层点云...")
    
    all_decoded_points = []
    layer_info = []
    
    for layer_idx in range(1, layer_count + 1):
        image_path = os.path.join(output_dir, f"uv_rgb_texture_layer_{layer_idx}.jpeg")
        metadata_path = os.path.join(output_dir, f"uv_rgb_texture_layer_{layer_idx}_metadata.json")
        
        if not os.path.exists(image_path):
            print(f"层 {layer_idx} 的纹理图像不存在: {image_path}")
            continue
            
        if not os.path.exists(metadata_path):
            print(f"层 {layer_idx} 的元数据文件不存在: {metadata_path}")
            continue
        
        # 解码当前层
        layer_points = decode_with_metadata(image_path, metadata_path)
        
        if len(layer_points) > 0:
            all_decoded_points.append(layer_points)
            layer_info.append({
                'layer': layer_idx,
                'points_count': len(layer_points),
                'image_path': image_path,
                'metadata_path': metadata_path
            })
            print(f"层 {layer_idx}: 解码出 {len(layer_points)} 个点")
        else:
            print(f"层 {layer_idx}: 解码失败或无有效点")
    
    # 合并所有层的点云
    if all_decoded_points:
        merged_points = np.vstack(all_decoded_points)
        print(f"总共解码出 {len(merged_points)} 个点")
    else:
        merged_points = np.array([]).reshape(0, 3)
        print("解码失败，没有有效点")
    
    return merged_points, layer_info

def calculate_nearest_neighbor_distances(points):
    """
    计算每个点到其最近邻点的距离，并返回最大和最小的最近邻距离
    """
    if len(points) < 2:
        return 0, 0
    
    # 使用KNN找到每个点的最近邻（k=2，因为第一个邻居是自己）
    knn = NearestNeighbors(n_neighbors=2)
    knn.fit(points)
    
    # 获取每个点到其最近邻的距离
    distances, indices = knn.kneighbors(points)
    
    # distances[:, 0] 是到自己的距离（为0）
    # distances[:, 1] 是到最近邻的距离
    nearest_distances = distances[:, 1]
    
    min_nearest_distance = np.min(nearest_distances)
    max_nearest_distance = np.max(nearest_distances)
    
    return min_nearest_distance, max_nearest_distance

def main():
    parser = argparse.ArgumentParser(description="点云迭代处理工具")
    parser.add_argument("--input", "-i", help="输入点云文件路径（可选，默认使用bunny数据集）")
    parser.add_argument("--output", "-o", default="output", help="输出目录")
    parser.add_argument("--radius", "-r", type=float, default=0.005, help="Ball Pivoting球半径")
    parser.add_argument("--grid-size", "-g", type=int, default=512, help="网格大小")
    parser.add_argument("--max-iterations", "-m", type=int, default=10, help="最大迭代次数")
    parser.add_argument("--max-layers", "-l", type=int, default=10, help="最大层数限制")
    parser.add_argument("--visualize", "-v", action="store_true", help="是否显示3D可视化结果")
    parser.add_argument("--sh", "-s", type=int, default=3, help="球谐函数")
    
    args = parser.parse_args()
    # 创建输出目录
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    
    # 加载高斯数据
    if args.input:
        print(f"加载点云文件: {args.input}")
        gaussian = GaussianModel(args.sh)
        gaussian.load_ply(args.input)
        xyz = gaussian._xyz
        if hasattr(xyz, "detach"):  # 判断是否为torch.Tensor
            xyz = xyz.detach().cpu().numpy()
        print(f"点云包含 {len(xyz)} 个点")
        # knn = NearestNeighbors(n_neighbors=12)
        # knn.fit(xyz)

        # # Find the distances and indices of the neighbors for each point
        # distances, indices = knn.kneighbors(xyz)

        # # Determine a threshold to identify pcd1 that are part of the body
        # # This threshold will depend on your specific data
        # distance_threshold = np.mean(distances) + 0.8 * np.std(distances)

        # # Filter points
        # index = np.mean(distances, axis=1) < distance_threshold
        # filtered_points = xyz[index]
        print(calculate_nearest_neighbor_distances(xyz))


        
        # new_attributes = []
        # for attr in attributes:
        #     new_attributes.append(attr[index])
        # 转换为 open3d 点云对象
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
    else:
        print("使用bunny数据集作为示例")
        bunny = o3d.data.BunnyMesh()
        gt_mesh = o3d.io.read_triangle_mesh(bunny.path)
        gt_mesh.compute_vertex_normals()
        pcd = gt_mesh.sample_points_poisson_disk(5000)
    
    print(f"输入点云包含 {len(pcd.points)} 个点")
    
    # 开始处理
    start_time = time.time()
    
    results = multilayer_encode(
        pcd, 
        args.output, 
        args.radius, 
        args.grid_size,
        args.max_iterations,
        args.max_layers,
        args.visualize
    )
    
    processing_time = time.time() - start_time
    
    # 输出最终统计信息
    print("\n" + "=" * 60)
    print("处理完成统计")
    print("=" * 60)
    print(f"总处理时间: {processing_time:.2f} 秒")
    print(f"总层数: {results['total_layers']}")
    print(f"初始点数: {results['initial_point_count']}")
    print(f"剩余未处理点数: {results['remaining_points']}")
    
    # 保存最终统计信息
    stats_path = os.path.join(args.output, "processing_stats.txt")
    with open(stats_path, 'w') as f:
        f.write(f"总处理时间: {processing_time:.2f} 秒\n")
        f.write(f"总层数: {results['total_layers']}\n")
        f.write(f"初始点数: {results['initial_point_count']}\n")
        f.write(f"剩余未处理点数: {results['remaining_points']}\n")
    
    print(f"结果已保存到: {args.output}")
    print("\n=== 开始压缩阶段 ===")
    # 压缩图像
    compressor = ImageCompressor()
    compressor.batch_compress(args.output, args.output)
    print("\n=== 开始解码阶段 ===")
    
    # 遍历output下的所有子文件夹，对每个子文件夹中的图像分别进行解码和评估
    for subdir in os.listdir(args.output):
        subdir_path = os.path.join(args.output, subdir)
        if os.path.isdir(subdir_path):
            print(f"\n处理子文件夹: {subdir_path}")
            # 假设每个子文件夹都包含多层数据，尝试找到最大层数
            max_layer = 0
            for fname in os.listdir(subdir_path):
                if fname.startswith("uv_rgb_texture_layer_") and fname.endswith(".jpeg"):
                    try:
                        layer_num = int(fname.split("_")[-1].split(".")[0])
                        # 检查是否存在对应的元数据文件，如果没有则尝试转移/复制
                        meta_candidate = fname.replace(".jpeg", "_metadata.json")
                        meta_path = os.path.join(subdir_path, meta_candidate)
                        if not os.path.exists(meta_path):
                            # 尝试从主output目录转移元数据
                            src_meta = os.path.join(args.output, meta_candidate)
                            if os.path.exists(src_meta):
                                shutil.copy(src_meta, meta_path)
                                print(f"已转移元数据: {src_meta} -> {meta_path}")
                        if layer_num > max_layer:
                            max_layer = layer_num
                    except Exception:
                        continue
            if max_layer == 0:
                print(f"{subdir_path} 未找到有效的层数据，跳过。")
                continue

            # 解码
            decoded_points, layer_info = multilayer_deocde(subdir_path, max_layer)

            # 评估
            print("\n=== 开始评估阶段 ===")
            # 这里假设原始点云为results["points_in_layer"]，可根据实际情况调整
            evaluation_results = evaluate_quality(results["points_in_layer"], decoded_points)
            print(f"评估结果: {evaluation_results}")




if __name__ == "__main__":
    main()