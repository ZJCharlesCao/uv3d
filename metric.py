from scipy.spatial.distance import directed_hausdorff
import open3d as o3d
import numpy as np
def calculate_hausdorff_distance(points1, points2):
    """
    计算两个点云之间的豪斯多夫距离
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # 计算双向豪斯多夫距离
    dist_1_to_2 = directed_hausdorff(points1, points2)[0]
    dist_2_to_1 = directed_hausdorff(points2, points1)[0]
    
    # 豪斯多夫距离是两个方向的最大值
    hausdorff_dist = max(dist_1_to_2, dist_2_to_1)
    
    return hausdorff_dist

def calculate_rmse(points1, points2):
    """
    计算两个点云之间的RMSE（需要点数相同或使用最近邻匹配）
    """
    if len(points1) == 0 or len(points2) == 0:
        return float('inf')
    
    # 创建Open3D点云对象
    pcd1 = o3d.geometry.PointCloud()
    pcd1.points = o3d.utility.Vector3dVector(points1)
    
    pcd2 = o3d.geometry.PointCloud()
    pcd2.points = o3d.utility.Vector3dVector(points2)
    
    # 计算点到点的距离
    distances = pcd1.compute_point_cloud_distance(pcd2)
    distances = np.asarray(distances)
    
    # 计算RMSE
    rmse = np.sqrt(np.mean(distances ** 2))
    
    return rmse