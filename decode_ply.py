import numpy as np
from PIL import Image
from metric import calculate_hausdorff_distance, calculate_rmse

def decode_uv_rgb_to_points(image_path, original_bounds=None, uv_bounds=None):
    """
    从UV RGB纹理图像中解码出原始3D点云坐标
    
    参数:
    - image_path: RGB纹理图像路径
    - original_bounds: 原始XYZ坐标的边界值 [(x_min, x_max), (y_min, y_max), (z_min, z_max)]
    - uv_bounds: UV坐标的边界值 [(u_min, u_max), (v_min, v_max)]
    - return_uv_coords: 是否返回UV坐标
    
    返回:
    - vertices: 解码后的3D顶点坐标 (N, 3)
    - uv_coords: UV坐标 (N, 2) [可选]
    """
    
    # 读取图像
    image = Image.open(image_path).convert('RGB')
    rgb_array = np.array(image)
    
    print(f"图像尺寸: {rgb_array.shape}")
    
    # 找到非白色像素（有效像素）
    white_mask = np.all(rgb_array == 255, axis=2)
    non_white_mask = ~white_mask
    
    # 获取有效像素的坐标和RGB值
    valid_pixels = np.where(non_white_mask)
    pixel_y, pixel_x = valid_pixels[0], valid_pixels[1]
    rgb_values = rgb_array[pixel_y, pixel_x]  # (N, 3)
    
    print(f"有效像素数量: {len(pixel_x)}")
    
    # 将RGB值转换回[0,1]范围
    normalized_rgb = rgb_values.astype(np.float32) / 255.0
    
    # 从像素坐标恢复UV坐标
    image_height, image_width = rgb_array.shape[:2]
    u_normalized = pixel_x / (image_width - 1)
    v_normalized = 1 - (pixel_y / (image_height - 1))  # 注意这里要反转回来
    
    # 如果提供了UV边界，恢复原始UV坐标
    if uv_bounds is not None:
        u_min, u_max = uv_bounds[0]
        v_min, v_max = uv_bounds[1]
        
        u_coords = u_normalized * (u_max - u_min) + u_min
        v_coords = v_normalized * (v_max - v_min) + v_min
        uv_coords = np.stack([u_coords, v_coords], axis=1)
    else:
        uv_coords = np.stack([u_normalized, v_normalized], axis=1)
    
    # 如果提供了原始边界，恢复原始XYZ坐标
    if original_bounds is not None:
        x_min, x_max = original_bounds[0]
        y_min, y_max = original_bounds[1]
        z_min, z_max = original_bounds[2]
        
        x_coords = normalized_rgb[:, 0] * (x_max - x_min) + x_min
        y_coords = normalized_rgb[:, 1] * (y_max - y_min) + y_min
        z_coords = normalized_rgb[:, 2] * (z_max - z_min) + z_min
        
        vertices = np.stack([x_coords, y_coords, z_coords], axis=1)
    else:
        # 如果没有边界信息，直接使用归一化的RGB值作为坐标
        vertices = normalized_rgb
    
    print(f"解码后顶点数量: {len(vertices)}")
    
    return vertices

def decode_with_metadata(image_path, metadata_path=None, metadata_dict=None):
    """
    使用元数据文件进行解码
    
    参数:
    - image_path: RGB纹理图像路径
    - metadata_path: 元数据文件路径（JSON格式）
    - metadata_dict: 直接传入的元数据字典
    
    返回:
    - vertices: 解码后的3D顶点坐标
    - uv_coords: UV坐标
    """
    
    if metadata_path is not None:
        import json
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    elif metadata_dict is not None:
        metadata = metadata_dict
    else:
        raise ValueError("需要提供元数据文件路径或元数据字典")
    
    # 从元数据中提取边界信息
    original_bounds = [
        (metadata['x_min'], metadata['x_max']),
        (metadata['y_min'], metadata['y_max']),
        (metadata['z_min'], metadata['z_max'])
    ]
    
    uv_bounds = [
        (metadata['u_min'], metadata['u_max']),
        (metadata['v_min'], metadata['v_max'])
    ]
    
    return decode_uv_rgb_to_points(image_path, original_bounds, uv_bounds)


def visualize_decoded_points(vertices, output_path="decoded_points.ply"):
    """
    将解码后的点云保存为PLY格式文件
    
    参数:
    - vertices: 解码后的3D顶点坐标
    - output_path: 输出PLY文件路径
    """
    
    # 创建PLY文件头
    header = f"""ply
format ascii 1.0
element vertex {len(vertices)}
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
"""
    
    # 为可视化生成颜色（基于坐标位置）
    colors = ((vertices - vertices.min(axis=0)) / (vertices.max(axis=0) - vertices.min(axis=0)) * 255).astype(np.uint8)
    
    # 写入PLY文件
    with open(output_path, 'w') as f:
        f.write(header)
        for i, (vertex, color) in enumerate(zip(vertices, colors)):
            f.write(f"{vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f} {color[0]} {color[1]} {color[2]}\n")
    
    print(f"解码后的点云已保存到: {output_path}")

def evaluate_quality(original_points, decoded_points):
    """
    评估重建质量
    """
    print(f"\n=== 重建质量评估 ===")
    print(f"原始点云点数: {len(original_points)}")
    print(f"解码点云点数: {len(decoded_points)}")
    
    if len(decoded_points) == 0:
        print("解码点云为空，无法计算评估指标")
        return None
    
    # 计算豪斯多夫距离
    hausdorff_dist = calculate_hausdorff_distance(original_points, decoded_points)
    print(f"豪斯多夫距离: {hausdorff_dist:.6f}")
    
    # 计算RMSE
    rmse = calculate_rmse(original_points, decoded_points)
    print(f"RMSE: {rmse:.6f}")
    
    # 计算点数恢复率
    recovery_rate = len(decoded_points) / len(original_points) * 100
    print(f"点数恢复率: {recovery_rate:.2f}%")
    
    evaluation_results = {
        'original_points_count': len(original_points),
        'decoded_points_count': len(decoded_points),
        'hausdorff_distance': hausdorff_dist,
        'rmse': rmse,
        'recovery_rate': recovery_rate
    }
    
    return evaluation_results


# 使用示例
if __name__ == "__main__":
    # 示例1: 基本解码（需要手动提供边界信息）
    """
    original_bounds = [(-1.0, 1.0), (-1.0, 1.0), (-1.0, 1.0)]  # 示例边界
    uv_bounds = [(0.0, 1.0), (0.0, 1.0)]  # 示例UV边界
    
    vertices = decode_uv_rgb_to_points("uv_rgb_texture.png", original_bounds, uv_bounds)
    print(f"解码后顶点形状: {vertices.shape}")
    """
    
    # 示例2: 使用元数据文件解码
    """
    vertices, uv_coords = decode_with_metadata("uv_rgb_texture.png", "metadata.json")
    print(f"解码后顶点形状: {vertices.shape}")
    print(f"UV坐标形状: {uv_coords.shape}")
    
    # 保存为PLY文件用于可视化
    visualize_decoded_points(vertices)
    """
    
    print("解码器已准备就绪！")
    print("使用方法:")
    print("1. decode_uv_rgb_to_points() - 基本解码")
    print("2. decode_with_metadata() - 使用元数据解码")
    print("3. create_metadata_from_encoding() - 创建元数据文件")
    print("4. visualize_decoded_points() - 保存为PLY文件")