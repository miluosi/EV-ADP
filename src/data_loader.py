"""
Data Loader Module for NYC Taxi and Chicago Divvy Data

This module provides functions to load and process city Origin-Destination (OD) data
for GNN clustering and zone-based ADP algorithms.
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import logging
from typing import Tuple, Dict, List, Union, Optional

# 设置logger
logger = logging.getLogger(__name__)

def load_od_data(data_folder="data", data_type="nyc_taxi"):
    """
    加载OD数据（从CSV或XLSX文件）
    
    Args:
        data_folder: 数据文件夹路径
        data_type: 数据类型, "nyc_taxi" 或 "chicago_divvy"
        
    Returns:
        od_data: OD数据DataFrame
        coords_df: 坐标数据DataFrame
    """
    data_path = Path(data_folder)
    
    # 根据数据类型选择不同的文件路径
    if data_type == "nyc_taxi":
        od_file = data_path / "nyc_taxi_od_data.csv"
        coords_file = data_path / "nyc_taxi_coordinates.csv"
        
        # 如果没有找到OD文件，尝试使用样本文件
        if not od_file.exists():
            od_file = data_path / "nyc_taxi_sample.csv"
            logger.warning(f"主OD文件未找到，使用样本文件: {od_file}")
    
    elif data_type == "chicago_divvy":
        od_file = data_path / "chicago_divvy_od_data.csv"
        coords_file = data_path / "chicago_divvy_coordinates.csv"
        
        # 如果没有找到OD文件，尝试使用样本文件
        if not od_file.exists():
            od_file = data_path / "chicago_divvy_sample.csv"
            logger.warning(f"主OD文件未找到，使用样本文件: {od_file}")
            
    else:
        # 尝试加载xlsx文件（工作日/周末数据）
        xlsx_static_path = data_path / "xlsx" / "static"
        
        if not xlsx_static_path.exists():
            raise FileNotFoundError(f"数据目录不存在: {xlsx_static_path}")
        
        weekday_files = list(xlsx_static_path.glob("*weekday*.xlsx"))
        weekend_files = list(xlsx_static_path.glob("*weekend*.xlsx"))
        
        if not weekday_files:
            raise FileNotFoundError(f"未找到工作日xlsx文件在 {xlsx_static_path}")
        
        # 使用最新的工作日文件
        od_file = max(weekday_files, key=lambda x: x.stat().st_mtime)
        coords_file = data_path / "nyc_taxi_coordinates.csv"
        
        logger.info(f"使用工作日数据: {od_file.name}")
    
    # 确保文件存在
    if not od_file.exists():
        raise FileNotFoundError(f"OD数据文件未找到: {od_file}")
    
    # 根据文件类型读取数据
    if od_file.suffix == ".csv":
        logger.info(f"读取CSV格式OD数据: {od_file}")
        od_data = pd.read_csv(od_file)
    elif od_file.suffix == ".xlsx" or od_file.suffix == ".xls":
        logger.info(f"读取Excel格式OD数据: {od_file}")
        od_data = pd.read_excel(od_file)
    else:
        raise ValueError(f"不支持的文件格式: {od_file.suffix}")
    
    # 读取坐标文件（如果存在）
    coords_df = None
    if coords_file.exists():
        logger.info(f"读取坐标数据: {coords_file}")
        coords_df = pd.read_csv(coords_file)
    else:
        logger.warning(f"坐标文件未找到: {coords_file}")
    
    logger.info(f"OD数据加载完成: {len(od_data)}条记录")
    logger.info(f"OD数据列: {list(od_data.columns)}")
    
    return od_data, coords_df

def prepare_baseline_features(od_data, coords_df=None):
    """
    准备基于需求的特征，用于baseline聚类和GNN输入
    
    Args:
        od_data: OD数据DataFrame
        coords_df: 坐标数据DataFrame (可选)
        
    Returns:
        features: 节点特征数组
        all_locations: 所有位置ID列表
        location_to_idx: 位置ID到索引的映射
    """
    # 检测列名
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
        demand = row[demand_col] if demand_col in row else 1
        
        # 基本统计
        features[origin_idx, 0] += demand  # 总出行需求
        features[dest_idx, 1] += demand    # 总到达需求
        features[origin_idx, 2] += 1       # 出行次数
        features[dest_idx, 3] += 1         # 到达次数
        
        # 距离和费用（如果有的话）
        if 'total_distance' in od_data.columns and pd.notna(row['total_distance']):
            features[origin_idx, 4] += row['total_distance']
        if 'total_fare' in od_data.columns and pd.notna(row['total_fare']):
            features[origin_idx, 5] += row['total_fare']
    
    # 计算平均值
    for i in range(len(all_locations)):
        if features[i, 2] > 0:  # 避免除零
            features[i, 4] /= features[i, 2]  # 平均距离
            features[i, 5] /= features[i, 2]  # 平均费用
    
    # 添加充电桩数量特征
    if coords_df is not None:
        try:
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
            
            # 导入生成充电桩特征的函数
            try:
                from src.gnn_models import generate_charging_station_features
                
                # 生成充电桩数量特征
                charge_num = generate_charging_station_features(
                    coordinates=coordinates,
                    radius=0.01,
                    num_stations_range=(5, 50),
                    seed=42
                )
                
                # 添加到特征矩阵中
                features[:, 6] = charge_num.numpy()
                logger.info(f"添加了充电桩数量特征: 平均 {charge_num.float().mean():.1f} 个/区域")
                
            except ImportError:
                # 如果无法导入模块，使用随机值
                logger.warning("无法导入generate_charging_station_features函数，使用随机充电桩数量")
                np.random.seed(42)
                features[:, 6] = np.random.randint(5, 51, len(all_locations))
            
        except Exception as e:
            logger.warning(f"无法基于坐标生成充电桩特征 ({e})，使用随机值")
            # 使用随机充电桩数量
            np.random.seed(42)
            features[:, 6] = np.random.randint(5, 51, len(all_locations))
    else:
        # 如果没有坐标数据，使用随机充电桩数量
        np.random.seed(42)
        features[:, 6] = np.random.randint(5, 51, len(all_locations))
    
    return features, all_locations, location_to_idx

def prepare_graph_data(od_data, coords_df=None):
    """
    准备图数据用于GNN训练
    
    Args:
        od_data: OD数据DataFrame
        coords_df: 坐标数据DataFrame (可选)
        
    Returns:
        graph_data: PyTorch Geometric图数据对象
        location_to_idx: 位置ID到索引的映射
        all_locations: 所有位置ID列表
        charge_num: 充电桩数量特征
    """
    logger.info("创建图数据...")
    
    # 检查列名并调整
    demand_col = 'daily_demand' if 'daily_demand' in od_data.columns else 'total_trips'
    pickup_col = 'pickup_location_id' if 'pickup_location_id' in od_data.columns else 'PULocationID'
    dropoff_col = 'dropoff_location_id' if 'dropoff_location_id' in od_data.columns else 'DOLocationID'
    
    # 获取所有唯一的位置ID
    all_locations = sorted(list(set(od_data[pickup_col].unique()) | 
                          set(od_data[dropoff_col].unique())))
    location_to_idx = {loc: idx for idx, loc in enumerate(all_locations)}
    
    logger.info(f"节点数量: {len(all_locations)}")
    
    # 加载坐标数据并生成charge_num特征
    coordinates = None
    charge_num = None
    
    if coords_df is not None:
        try:
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
                    logger.warning(f"未找到位置 {location_id} 的坐标，使用默认坐标")
                    coordinates_list.append([40.7589, -73.9851])  # 纽约市中心默认坐标
                    valid_locations.append(location_id)
            
            coordinates = np.array(coordinates_list)
            logger.info(f"坐标覆盖率: {len(valid_locations)}/{len(all_locations)} ({len(valid_locations)/len(all_locations)*100:.1f}%)")
            
            # 生成充电桩数量特征
            try:
                from src.gnn_models import generate_charging_station_features
                
                charge_num = generate_charging_station_features(
                    coordinates=coordinates,
                    radius=0.01,  # 约1km半径
                    num_stations_range=(5, 50),  # 每个区域5-50个充电桩
                    seed=42  # 固定随机种子确保可重复性
                )
                logger.info(f"生成充电桩特征: 平均 {charge_num.float().mean():.1f} 个/区域, 范围 [{charge_num.min()}-{charge_num.max()}]")
            
            except ImportError:
                logger.warning("无法导入generate_charging_station_features函数，使用随机充电桩数量")
                charge_num = torch.randint(5, 51, (len(all_locations),), dtype=torch.long)
                torch.manual_seed(42)  # 确保可重复性
                
        except Exception as e:
            logger.warning(f"无法基于坐标生成充电桩特征 ({e})，使用随机值")
            # 使用随机充电桩数量
            charge_num = torch.randint(5, 51, (len(all_locations),), dtype=torch.long)
            torch.manual_seed(42)  # 确保可重复性
    else:
        # 如果无法加载坐标，生成随机的充电桩数量
        charge_num = torch.randint(5, 51, (len(all_locations),), dtype=torch.long)
        torch.manual_seed(42)  # 确保可重复性
    
    # 创建边列表和边权重
    edge_index = []
    edge_weights = []
    
    for _, row in od_data.iterrows():
        origin_idx = location_to_idx[row[pickup_col]]
        dest_idx = location_to_idx[row[dropoff_col]]
        demand = row[demand_col] if demand_col in row and pd.notna(row[demand_col]) else 1
        
        edge_index.append([origin_idx, dest_idx])
        edge_weights.append(demand)
      # 转换为tensor
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weights = torch.tensor(edge_weights, dtype=torch.float)
    
    # 验证边索引的有效性
    num_nodes = len(all_locations)
    if edge_index.numel() > 0:
        max_index = edge_index.max().item()
        min_index = edge_index.min().item()
        
        if max_index >= num_nodes or min_index < 0:
            logger.warning(f"边索引超出范围: [{min_index}, {max_index}], 节点数: {num_nodes}")
            
            # 过滤有效的边
            valid_mask = (edge_index[0] < num_nodes) & (edge_index[1] < num_nodes) & \
                        (edge_index[0] >= 0) & (edge_index[1] >= 0)
            
            if valid_mask.sum() > 0:
                edge_index = edge_index[:, valid_mask]
                edge_weights = edge_weights[valid_mask]
                logger.info(f"过滤后有效边数: {edge_index.size(1)}")
            else:
                # 创建最小连通图
                logger.warning("没有有效边，创建环形连通图...")
                if num_nodes > 1:
                    edge_list = [[i, (i+1) % num_nodes] for i in range(num_nodes)]
                    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
                    edge_weights = torch.ones(num_nodes)
                else:
                    # 单节点自环
                    edge_index = torch.tensor([[0], [0]], dtype=torch.long)
                    edge_weights = torch.ones(1)
    
    # 创建节点特征 (使用节点度、流量统计和充电桩数量)
    num_nodes = len(all_locations)
    node_features = torch.zeros(num_nodes, 5)  # [出度, 入度, 出流量, 入流量, 充电桩数量]
    
    for _, row in od_data.iterrows():
        origin_idx = location_to_idx[row[pickup_col]]
        dest_idx = location_to_idx[row[dropoff_col]]
        demand = row[demand_col] if demand_col in row and pd.notna(row[demand_col]) else 1
        
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
    node_features[:, :4] = F.normalize(node_features[:, :4], p=2, dim=0)
    
    # 创建PyTorch Geometric数据对象
    try:
        from torch_geometric.data import Data
        
        graph_data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_weights.unsqueeze(1),
            num_nodes=num_nodes
        )
        
        logger.info(f"边数量: {edge_index.shape[1]}")
        logger.info(f"节点特征维度: {node_features.shape}")
        logger.info(f"包含充电桩特征: {'是' if charge_num is not None else '否'}")
        
        return graph_data, location_to_idx, all_locations, charge_num
    
    except ImportError:
        logger.error("无法导入torch_geometric.data.Data，返回原始特征")
        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_weights': edge_weights,
            'num_nodes': num_nodes
        }, location_to_idx, all_locations, charge_num


def load_adjacency_matrix(data_type="nyc_taxi", data_folder="data"):
    """
    加载预计算的邻接矩阵或从OD数据创建
    
    Args:
        data_type: 数据类型, "nyc_taxi" 或 "chicago_divvy" 或 "weekday"
        data_folder: 数据文件夹路径
        
    Returns:
        adj_tensor: 邻接矩阵的PyTorch张量
    """
    data_path = Path(data_folder)
    
    # 如果有预计算的邻接矩阵文件
    if data_type in ["weekday", "weekend", "工作日", "周末"]:
        npy_static_path = data_path / "npy" / "static"
        
        if npy_static_path.exists():
            # 根据数据类型选择文件
            if data_type in ["weekday", "工作日"]:
                adj_files = list(npy_static_path.glob("*weekday*adjacency*.npy"))
            else:  # weekend
                adj_files = list(npy_static_path.glob("*weekend*adjacency*.npy"))
            
            if adj_files:
                adj_file = adj_files[0]  # 使用找到的第一个文件
                logger.info(f"加载预计算的邻接矩阵: {adj_file.name}")
                
                adj_matrix = np.load(adj_file)
                
                # 转换为PyTorch张量
                adj_tensor = torch.from_numpy(adj_matrix).float()
                
                logger.info(f"邻接矩阵形状: {adj_tensor.shape}")
                logger.info(f"邻接矩阵密度: {torch.sum(adj_tensor > 0).item() / (adj_tensor.shape[0] * adj_tensor.shape[1]):.4f}")
                
                return adj_tensor
    
    # 如果没有预计算文件或文件类型不匹配，提示需要从OD数据创建
    logger.warning(f"未找到预计算的邻接矩阵文件，需要从OD数据创建")
    return None


# 辅助函数
def load_nyc_taxi_data(use_sample=False):
    """
    专门加载NYC出租车数据的快捷方法
    
    Args:
        use_sample: 是否使用样本数据
        
    Returns:
        od_data: OD数据
        coords_df: 坐标数据
        graph_data: 图数据
        all_locations: 所有位置ID
        location_to_idx: 位置ID到索引的映射
    """
    data_folder = "data"
    
    # 确定使用的文件名
    od_filename = "nyc_taxi_sample.csv" if use_sample else "nyc_taxi_od_data.csv"
    coords_filename = "nyc_taxi_coordinates.csv"
    
    # 构建完整路径
    data_path = Path(data_folder)
    od_file = data_path / od_filename
    coords_file = data_path / coords_filename
    
    # 检查文件是否存在
    if not od_file.exists():
        # 如果主文件不存在且没有指定使用样本，则尝试使用样本文件
        if not use_sample:
            logger.warning(f"主OD文件{od_filename}未找到，尝试使用样本文件")
            od_file = data_path / "nyc_taxi_sample.csv"
            if not od_file.exists():
                raise FileNotFoundError(f"无法找到任何NYC出租车数据文件")
    
    # 读取数据
    logger.info(f"读取NYC出租车数据: {od_file}")
    od_data = pd.read_csv(od_file)
    
    # 读取坐标
    coords_df = None
    if coords_file.exists():
        logger.info(f"读取NYC坐标数据: {coords_file}")
        coords_df = pd.read_csv(coords_file)
    else:
        logger.warning(f"NYC坐标文件未找到: {coords_file}")
    
    # 准备图数据
    graph_data, location_to_idx, all_locations, _ = prepare_graph_data(od_data, coords_df)
    
    return od_data, coords_df, graph_data, all_locations, location_to_idx


def create_synthetic_od_data(num_locations=100, num_trips=1000, seed=42):
    """
    创建合成的OD数据用于测试
    
    Args:
        num_locations: 位置数量
        num_trips: 出行数量
        seed: 随机种子
        
    Returns:
        od_data: 合成的OD数据DataFrame
    """
    np.random.seed(seed)
    
    # 创建位置ID (从1开始)
    location_ids = np.arange(1, num_locations + 1)
    
    # 生成随机的OD对
    pickup_ids = np.random.choice(location_ids, size=num_trips)
    dropoff_ids = np.random.choice(location_ids, size=num_trips)
    
    # 确保没有自环 (pickup != dropoff)
    for i in range(num_trips):
        while dropoff_ids[i] == pickup_ids[i]:
            dropoff_ids[i] = np.random.choice(location_ids)
    
    # 生成随机需求
    demand = np.random.lognormal(mean=1.0, sigma=0.8, size=num_trips)
    demand = np.round(demand).astype(int) + 1  # 确保至少为1
    
    # 生成距离和费用
    distance = np.random.uniform(low=0.5, high=10.0, size=num_trips)
    fare = distance * np.random.uniform(low=2.0, high=4.0, size=num_trips)
    
    # 创建DataFrame
    od_data = pd.DataFrame({
        'PULocationID': pickup_ids,
        'DOLocationID': dropoff_ids,
        'total_trips': demand,
        'total_distance': distance,
        'total_fare': fare
    })
    
    # 创建合成坐标
    lat_base = 40.7
    lon_base = -74.0
    
    coords_data = {
        'zone_id': location_ids,
        'lat': lat_base + np.random.uniform(-0.1, 0.1, size=num_locations),
        'lon': lon_base + np.random.uniform(-0.1, 0.1, size=num_locations),
    }
    
    coords_df = pd.DataFrame(coords_data)
    
    return od_data, coords_df
