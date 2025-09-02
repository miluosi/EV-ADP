"""
Data loader module for GNN clustering project.
Provides functions to load and prepare OD demand data for graph neural network training.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from pathlib import Path
import warnings
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional, Dict, List, Union

# 导入GNN模型中的充电桩特征生成函数
try:
    import sys
    sys.path.append('src')
    from gnn_models import generate_charging_station_features
except ImportError:
    def generate_charging_station_features(coordinates, radius=0.01, num_stations_range=(5, 50), seed=42):
        """备用的充电桩特征生成函数"""
        np.random.seed(seed)
        num_locations = len(coordinates) if hasattr(coordinates, '__len__') else coordinates.shape[0]
        return torch.randint(num_stations_range[0], num_stations_range[1] + 1, (num_locations,), dtype=torch.long)

warnings.filterwarnings('ignore')


def load_od_data(data_folder="data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    加载weekday和weekend OD数据
    
    Args:
        data_folder: 数据文件夹路径
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (weekday_od, weekend_od)
    """
    data_path = Path(data_folder)
    xlsx_static_path = data_path / "xlsx" / "static"
    
    # 查找OD文件 (xlsx格式，在static文件夹中)
    weekday_files = list(xlsx_static_path.glob("*weekday*.xlsx"))
    weekend_files = list(xlsx_static_path.glob("*weekend*.xlsx"))
    
    if not weekday_files:
        raise FileNotFoundError(f"未找到weekday xlsx文件在 {xlsx_static_path}")
    if not weekend_files:
        raise FileNotFoundError(f"未找到weekend xlsx文件在 {xlsx_static_path}")
        
    # 使用最新的文件
    weekday_file = max(weekday_files, key=lambda x: x.stat().st_mtime)
    weekend_file = max(weekend_files, key=lambda x: x.stat().st_mtime)
    
    print(f"加载工作日数据: {weekday_file.name}")
    print(f"加载周末数据: {weekend_file.name}")
    
    # 读取xlsx文件
    weekday_od = pd.read_excel(weekday_file)
    weekend_od = pd.read_excel(weekend_file)
    
    print(f"工作日OD对数量: {len(weekday_od)}")
    print(f"周末OD对数量: {len(weekend_od)}")
    print(f"工作日数据列: {list(weekday_od.columns)}")
    
    return weekday_od, weekend_od


def load_adjacency_matrix(data_type="weekday", data_folder="data") -> torch.Tensor:
    """
    加载预计算的邻接矩阵
    
    Args:
        data_type: 数据类型，"weekday" 或 "weekend"
        data_folder: 数据文件夹路径
    
    Returns:
        torch.Tensor: 邻接矩阵张量
    """
    data_path = Path(data_folder)
    npy_static_path = data_path / "npy" / "static"
    
    # 根据数据类型选择文件
    if data_type == "weekday" or data_type == "工作日":
        adj_file = npy_static_path / "yellow_sod_weekday_adjacency_2025-05.npy"
    else:  # weekend
        adj_file = npy_static_path / "yellow_sod_weekend_adjacency_2025-05.npy"
    
    if not adj_file.exists():
        raise FileNotFoundError(f"邻接矩阵文件未找到: {adj_file}")
    
    print(f"加载邻接矩阵: {adj_file.name}")
    adj_matrix = np.load(adj_file)
    
    # 转换为PyTorch张量
    adj_tensor = torch.from_numpy(adj_matrix).float()
    
    print(f"邻接矩阵形状: {adj_tensor.shape}")
    print(f"邻接矩阵密度: {torch.sum(adj_tensor > 0).item() / (adj_tensor.shape[0] * adj_tensor.shape[1]):.4f}")
    
    return adj_tensor


def prepare_baseline_features(od_data: pd.DataFrame, coords_file="data/nyc_taxi_coordinates.csv") -> Tuple[np.ndarray, List, Dict]:
    """
    准备baseline算法的特征：基于需求的聚类，包含charge_num特征
    
    Args:
        od_data: OD数据DataFrame
        coords_file: 坐标文件路径
    
    Returns:
        Tuple[np.ndarray, List, Dict]: (features, all_locations, location_to_idx)
    """
    demand_col = 'daily_demand' if 'daily_demand' in od_data.columns else 'total_trips'
    pickup_col = 'pickup_location_id' if 'pickup_location_id' in od_data.columns else 'PULocationID'
    dropoff_col = 'dropoff_location_id' if 'dropoff_location_id' in od_data.columns else 'DOLocationID'
    
    # 获取所有唯一的位置ID
    all_locations = sorted(list(set(od_data[pickup_col].unique()) | 
                              set(od_data[dropoff_col].unique())))
    location_to_idx = {loc: idx for idx, loc in enumerate(all_locations)}
    
    # 为每个位置创建特征：[总出行需求, 总到达需求, 出行次数, 到达次数, 平均出行距离, 平均出行费用, 充电桩数量]
    features = np.zeros((len(all_locations), 7))
    
    for _, row in od_data.iterrows():
        origin_idx = location_to_idx[row[pickup_col]]
        dest_idx = location_to_idx[row[dropoff_col]]
        demand = row[demand_col]
        
        # 基本统计
        features[origin_idx, 0] += demand  # 总出行需求
        features[dest_idx, 1] += demand    # 总到达需求
        features[origin_idx, 2] += 1       # 出行次数
        features[dest_idx, 3] += 1         # 到达次数
        
        # 距离和费用（如果有的话）
        if 'total_distance' in od_data.columns:
            features[origin_idx, 4] += row['total_distance']
        if 'total_fare' in od_data.columns:
            features[origin_idx, 5] += row['total_fare']
    
    # 计算平均值
    for i in range(len(all_locations)):
        if features[i, 2] > 0:  # 避免除零
            features[i, 4] /= features[i, 2]  # 平均距离
            features[i, 5] /= features[i, 2]  # 平均费用
    
    # 添加充电桩数量特征
    try:
        coords_df = pd.read_csv(coords_file)
        print(f"为baseline算法加载坐标数据: {coords_file}")
        
        # 为每个位置获取坐标
        coordinates_list = []
        for location_id in all_locations:
            coord_row = coords_df[coords_df['zone_id'] == location_id]
            if not coord_row.empty:
                lat = coord_row.iloc[0]['lat']
                lon = coord_row.iloc[0]['lon']
                coordinates_list.append([lat, lon])
            else:
                # 使用默认坐标
                coordinates_list.append([40.7589, -73.9851])
        
        coordinates = np.array(coordinates_list)
        
        # 生成充电桩数量特征
        charge_num = generate_charging_station_features(
            coordinates=coordinates,
            radius=0.01,
            num_stations_range=(5, 50),
            seed=42
        )
        
        # 添加到特征矩阵中
        features[:, 6] = charge_num.numpy()
        print(f"Baseline特征中添加了充电桩数量特征: 平均 {charge_num.float().mean():.1f} 个/区域")
        
    except Exception as e:
        print(f"警告: 无法为baseline算法生成充电桩特征 ({e})，使用随机值")
        # 使用随机充电桩数量
        np.random.seed(42)
        features[:, 6] = np.random.randint(5, 51, len(all_locations))
    
    return features, all_locations, location_to_idx


def prepare_graph_data(od_data: pd.DataFrame, data_type="weekday", coords_file="data/nyc_taxi_coordinates.csv") -> Tuple[Data, Dict, List, Optional[torch.Tensor]]:
    """
    准备图数据用于GNN训练，包含charge_num特征
    
    Args:
        od_data: OD数据DataFrame
        data_type: 数据类型，"weekday" 或 "weekend"
        coords_file: 坐标文件路径
    
    Returns:
        Tuple[Data, Dict, List, Optional[torch.Tensor]]: (graph_data, location_to_idx, all_locations, charge_num)
    """
    print(f"\n创建{data_type}图数据...")
    
    # 检查列名并调整
    demand_col = 'daily_demand' if 'daily_demand' in od_data.columns else 'total_trips'
    pickup_col = 'pickup_location_id' if 'pickup_location_id' in od_data.columns else 'PULocationID'
    dropoff_col = 'dropoff_location_id' if 'dropoff_location_id' in od_data.columns else 'DOLocationID'
    
    # 获取所有唯一的位置ID
    all_locations = sorted(list(set(od_data[pickup_col].unique()) | 
                              set(od_data[dropoff_col].unique())))
    location_to_idx = {loc: idx for idx, loc in enumerate(all_locations)}
    
    print(f"节点数量: {len(all_locations)}")
    
    # 加载坐标数据并生成charge_num特征
    coordinates = None
    charge_num = None
    try:
        coords_df = pd.read_csv(coords_file)
        print(f"成功加载坐标数据: {coords_file}")
        
        # 为每个位置获取坐标
        coordinates_list = []
        valid_locations = []
        
        for location_id in all_locations:
            coord_row = coords_df[coords_df['zone_id'] == location_id]
            if not coord_row.empty:
                lat = coord_row.iloc[0]['lat']
                lon = coord_row.iloc[0]['lon']
                coordinates_list.append([lat, lon])
                valid_locations.append(location_id)
            else:
                # 如果没有找到坐标，使用默认值或跳过
                print(f"警告: 未找到位置 {location_id} 的坐标，使用默认坐标")
                coordinates_list.append([40.7589, -73.9851])  # 纽约市中心默认坐标
                valid_locations.append(location_id)
        
        coordinates = np.array(coordinates_list)
        print(f"坐标覆盖率: {len(valid_locations)}/{len(all_locations)} ({len(valid_locations)/len(all_locations)*100:.1f}%)")
        
        # 生成充电桩数量特征
        charge_num = generate_charging_station_features(
            coordinates=coordinates,
            radius=0.01,  # 约1km半径
            num_stations_range=(5, 50),  # 每个区域5-50个充电桩
            seed=42  # 固定随机种子确保可重复性
        )
        print(f"生成充电桩特征: 平均 {charge_num.float().mean():.1f} 个/区域, 范围 [{charge_num.min()}-{charge_num.max()}]")
        
    except Exception as e:
        print(f"警告: 无法加载坐标数据 ({e})，将使用随机充电桩特征")
        # 如果无法加载坐标，生成随机的充电桩数量
        charge_num = torch.randint(5, 51, (len(all_locations),), dtype=torch.long)
        torch.manual_seed(42)  # 确保可重复性
    
    # 创建边列表和边权重
    edge_index = []
    edge_weights = []
    
    for _, row in od_data.iterrows():
        origin_idx = location_to_idx[row[pickup_col]]
        dest_idx = location_to_idx[row[dropoff_col]]
        demand = row[demand_col]
        
        edge_index.append([origin_idx, dest_idx])
        edge_weights.append(demand)
    
    # 转换为tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # 创建节点特征 (使用节点度、流量统计和充电桩数量)
    num_nodes = len(all_locations)
    node_features = torch.zeros(num_nodes, 5)  # [出度, 入度, 出流量, 入流量, 充电桩数量]
    
    for _, row in od_data.iterrows():
        origin_idx = location_to_idx[row[pickup_col]]
        dest_idx = location_to_idx[row[dropoff_col]]
        demand = row[demand_col]
        
        # 出度和出流量
        node_features[origin_idx, 0] += 1  # 出度
        node_features[origin_idx, 2] += demand  # 出流量
        
        # 入度和入流量
        node_features[dest_idx, 1] += 1  # 入度
        node_features[dest_idx, 3] += demand  # 入流量
    
    # 添加充电桩数量特征（标准化）
    if charge_num is not None:
        charge_num_normalized = (charge_num.float() - charge_num.float().mean()) / (charge_num.float().std() + 1e-8)
        node_features[:, 4] = charge_num_normalized
    
    # 标准化其他特征
    node_features[:, :4] = F.normalize(node_features[:, :4], p=2, dim=1)
    
    # 创建PyTorch Geometric数据对象
    graph_data = Data(
        x=node_features,
        edge_index=edge_index,
        edge_attr=edge_weights.unsqueeze(1),
        num_nodes=num_nodes
    )
    
    print(f"边数量: {edge_index.shape[1]}")
    print(f"节点特征维度: {node_features.shape}")
    print(f"包含充电桩特征: {'是' if charge_num is not None else '否'}")
    
    return graph_data, location_to_idx, all_locations, charge_num


def save_clustering_results_with_coords(cluster_assignments: Union[np.ndarray, torch.Tensor], 
                                      all_locations: List, 
                                      data_type: str, 
                                      model_name: str, 
                                      coords_file="data/nyc_taxi_coordinates.csv") -> Tuple[str, pd.DataFrame, pd.DataFrame]:
    """
    保存聚类结果到xlsx文件，包含经纬度信息
    
    Args:
        cluster_assignments: 聚类分配结果
        all_locations: 所有位置ID列表
        data_type: 数据类型
        model_name: 模型名称
        coords_file: 坐标文件路径
    
    Returns:
        Tuple[str, pd.DataFrame, pd.DataFrame]: (文件路径, 结果DataFrame, 聚类统计DataFrame)
    """
    print(f"\n保存{data_type}_{model_name}聚类结果...")
    
    # 确保cluster_assignments是numpy数组
    if torch.is_tensor(cluster_assignments):
        cluster_assignments = cluster_assignments.cpu().numpy()
    
    # 加载坐标数据
    try:
        coords_df = pd.read_csv(coords_file)
        print(f"加载坐标数据: {coords_file}")
        print(f"坐标数据形状: {coords_df.shape}")
        print(f"坐标数据列: {list(coords_df.columns)}")
    except Exception as e:
        print(f"警告: 无法加载坐标文件 {coords_file}: {e}")
        coords_df = None
    
    # 创建结果DataFrame
    results_data = []
    for i, location_id in enumerate(all_locations):
        result_row = {
            'location_id': location_id,
            'cluster': cluster_assignments[i],
            'latitude': None,
            'longitude': None
        }
        
        # 添加坐标信息（如果可用）
        if coords_df is not None:
            coord_row = coords_df[coords_df['zone_id'] == location_id]
            if not coord_row.empty:
                result_row['latitude'] = coord_row.iloc[0]['lat']
                result_row['longitude'] = coord_row.iloc[0]['lon']
        
        results_data.append(result_row)
    
    # 创建DataFrame
    results_df = pd.DataFrame(results_data)
    
    # 添加聚类统计信息
    cluster_stats = []
    for cluster_id in sorted(results_df['cluster'].unique()):
        cluster_data = results_df[results_df['cluster'] == cluster_id]
        cluster_stats.append({
            'cluster_id': cluster_id,
            'num_locations': len(cluster_data),
            'percentage': len(cluster_data) / len(results_df) * 100,
            'avg_latitude': cluster_data['latitude'].mean() if cluster_data['latitude'].notna().any() else None,
            'avg_longitude': cluster_data['longitude'].mean() if cluster_data['longitude'].notna().any() else None
        })
    
    cluster_stats_df = pd.DataFrame(cluster_stats)
    
    # 保存到Excel文件
    results_path = Path("results")
    results_path.mkdir(exist_ok=True)
    
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    excel_filepath = results_path / f"{data_type}_{model_name}_clustering_results_{timestamp}.xlsx"
    
    with pd.ExcelWriter(excel_filepath, engine='openpyxl') as writer:
        results_df.to_excel(writer, sheet_name='聚类结果', index=False)
        cluster_stats_df.to_excel(writer, sheet_name='聚类统计', index=False)
    
    print(f"聚类结果已保存到: {excel_filepath}")
    print(f"总共{len(all_locations)}个位置, {len(cluster_stats_df)}个聚类")
    
    return str(excel_filepath), results_df, cluster_stats_df


def load_coordinates_data(coords_file="data/nyc_taxi_coordinates.csv") -> pd.DataFrame:
    """
    加载坐标数据
    
    Args:
        coords_file: 坐标文件路径
    
    Returns:
        pd.DataFrame: 坐标数据
    """
    try:
        coords_df = pd.read_csv(coords_file)
        print(f"成功加载坐标数据: {coords_file}")
        print(f"坐标数据形状: {coords_df.shape}")
        print(f"坐标数据列: {list(coords_df.columns)}")
        return coords_df
    except Exception as e:
        print(f"错误: 无法加载坐标文件 {coords_file}: {e}")
        raise


def validate_od_data(od_data: pd.DataFrame) -> Dict[str, any]:
    """
    验证OD数据的完整性和质量
    
    Args:
        od_data: OD数据DataFrame
    
    Returns:
        Dict[str, any]: 验证结果统计
    """
    stats = {}
    
    # 基本信息
    stats['total_records'] = len(od_data)
    stats['columns'] = list(od_data.columns)
    
    # 检查必要列
    demand_col = 'daily_demand' if 'daily_demand' in od_data.columns else 'total_trips'
    pickup_col = 'pickup_location_id' if 'pickup_location_id' in od_data.columns else 'PULocationID'
    dropoff_col = 'dropoff_location_id' if 'dropoff_location_id' in od_data.columns else 'DOLocationID'
    
    stats['demand_column'] = demand_col
    stats['pickup_column'] = pickup_col
    stats['dropoff_column'] = dropoff_col
    
    # 检查数据质量
    stats['missing_demand'] = od_data[demand_col].isna().sum()
    stats['missing_pickup'] = od_data[pickup_col].isna().sum()
    stats['missing_dropoff'] = od_data[dropoff_col].isna().sum()
    stats['zero_demand'] = (od_data[demand_col] == 0).sum()
    stats['negative_demand'] = (od_data[demand_col] < 0).sum()
    
    # 统计信息
    stats['unique_origins'] = od_data[pickup_col].nunique()
    stats['unique_destinations'] = od_data[dropoff_col].nunique()
    stats['unique_locations'] = len(set(od_data[pickup_col].unique()) | set(od_data[dropoff_col].unique()))
    
    stats['demand_stats'] = {
        'mean': od_data[demand_col].mean(),
        'median': od_data[demand_col].median(),
        'std': od_data[demand_col].std(),
        'min': od_data[demand_col].min(),
        'max': od_data[demand_col].max()
    }
    
    return stats


def standardize_features(features: np.ndarray, method='standard') -> Tuple[np.ndarray, StandardScaler]:
    """
    标准化特征
    
    Args:
        features: 特征矩阵
        method: 标准化方法，'standard' 或 'minmax'
    
    Returns:
        Tuple[np.ndarray, StandardScaler]: (标准化后的特征, 标准化器)
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
    else:
        raise ValueError("method 必须是 'standard' 或 'minmax'")
    
    features_scaled = scaler.fit_transform(features)
    return features_scaled, scaler


# 向后兼容性：为了保持与原代码的兼容性，提供一些别名和包装函数
def create_graph_data(*args, **kwargs):
    """向后兼容的函数名"""
    return prepare_graph_data(*args, **kwargs)


def load_and_prepare_data(data_folder="data", data_type="weekday", coords_file="data/nyc_taxi_coordinates.csv"):
    """
    一站式数据加载和准备函数
    
    Args:
        data_folder: 数据文件夹路径
        data_type: 数据类型，"weekday" 或 "weekend"
        coords_file: 坐标文件路径
    
    Returns:
        Dict: 包含所有加载的数据
    """
    print(f"开始加载和准备{data_type}数据...")
    
    # 加载OD数据
    weekday_od, weekend_od = load_od_data(data_folder)
    od_data = weekday_od if data_type == "weekday" else weekend_od
    
    # 验证数据
    validation_stats = validate_od_data(od_data)
    print(f"数据验证统计: {validation_stats['total_records']} 条记录, {validation_stats['unique_locations']} 个位置")
    
    # 准备图数据
    graph_data, location_to_idx, all_locations, charge_num = prepare_graph_data(od_data, data_type, coords_file)
    
    # 准备baseline特征
    baseline_features, _, _ = prepare_baseline_features(od_data, coords_file)
    
    return {
        'od_data': od_data,
        'graph_data': graph_data,
        'location_to_idx': location_to_idx,
        'all_locations': all_locations,
        'charge_num': charge_num,
        'baseline_features': baseline_features,
        'validation_stats': validation_stats,
        'weekday_od': weekday_od,
        'weekend_od': weekend_od
    }
