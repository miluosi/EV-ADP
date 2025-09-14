#!/usr/bin/env python3
"""
改进神经网络的Action区分度
增加显式的Action Type Embedding和更丰富的特征表示
"""

import torch
import torch.nn as nn
import torch.optim as optim
import sys
sys.path.append('src')

class ImprovedPathBasedNetwork(nn.Module):
    """
    改进的路径基础神经网络，增强Action类型区分度
    
    主要改进：
    1. 显式的Action Type Embedding
    2. 增强的特征表示
    3. Multi-Head Attention机制
    4. Action-specific的处理路径
    """
    
    def __init__(self, 
                 num_locations: int,
                 max_capacity: int,
                 embedding_dim: int = 128,
                 lstm_hidden: int = 256,
                 dense_hidden: int = 512,
                 pretrained_embeddings = None):
        super(ImprovedPathBasedNetwork, self).__init__()
        
        self.num_locations = num_locations
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        
        # 1. 显式的Action Type Embedding
        self.action_types = ['assign', 'charge', 'idle']
        self.action_type_embedding = nn.Embedding(
            num_embeddings=len(self.action_types) + 1,  # +1 for unknown
            embedding_dim=embedding_dim // 4,  # 较小的维度
            padding_idx=0
        )
        
        # 2. 位置嵌入（原有）
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        if pretrained_embeddings is not None:
            self.location_embedding.weight.data.copy_(pretrained_embeddings)
        
        # 3. 路径LSTM（原有）
        self.path_lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # location + delay
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # 4. 时间嵌入（原有）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ELU()
        )
        
        # 5. 距离嵌入（新增）
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 4),
            nn.ELU()
        )
        
        # 6. Multi-Head Attention for feature fusion
        feature_dim = lstm_hidden + embedding_dim // 2 + embedding_dim // 4 + embedding_dim // 4 + 4
        # 确保feature_dim能被num_heads整除
        num_heads = 8
        if feature_dim % num_heads != 0:
            # 调整feature_dim到最接近的能被num_heads整除的数
            feature_dim = ((feature_dim + num_heads - 1) // num_heads) * num_heads
        
        self.feature_dim = feature_dim
        self.num_heads = num_heads
        
        # 添加一个线性层来调整特征维度
        raw_feature_dim = lstm_hidden + embedding_dim // 2 + embedding_dim // 4 + embedding_dim // 4 + 4
        self.feature_projection = nn.Linear(raw_feature_dim, feature_dim)
        
        self.attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 7. Action-specific处理分支
        self.assign_branch = nn.Sequential(
            nn.Linear(feature_dim, dense_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden // 2, dense_hidden // 4)
        )
        
        self.charge_branch = nn.Sequential(
            nn.Linear(feature_dim, dense_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden // 2, dense_hidden // 4)
        )
        
        self.idle_branch = nn.Sequential(
            nn.Linear(feature_dim, dense_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden // 2, dense_hidden // 4)
        )
        
        # 8. 最终融合层
        final_input_dim = feature_dim + dense_hidden // 4  # main features + action branch
        self.final_layers = nn.Sequential(
            nn.Linear(final_input_dim, dense_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden, dense_hidden // 2),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """初始化网络权重"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight_ih' in name:
                    torch.nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name:
                    torch.nn.init.orthogonal_(param.data)
                elif 'bias' in name:
                    param.data.fill_(0)
    
    def _get_action_type_id(self, action_type: str) -> int:
        """将action_type字符串转换为ID"""
        if action_type.startswith('assign'):
            return 1
        elif action_type.startswith('charge'):
            return 2
        elif action_type == 'idle':
            return 3
        else:
            return 0  # unknown
    
    def _calculate_manhattan_distance(self, loc1: int, loc2: int, grid_size: int = 10) -> float:
        """计算曼哈顿距离"""
        x1, y1 = loc1 % grid_size, loc1 // grid_size
        x2, y2 = loc2 % grid_size, loc2 // grid_size
        return abs(x1 - x2) + abs(y1 - y2)
    
    def forward(self, 
                path_locations: torch.Tensor,
                path_delays: torch.Tensor,
                current_time: torch.Tensor,
                other_agents: torch.Tensor,
                num_requests: torch.Tensor,
                battery_level: torch.Tensor = None,
                request_value: torch.Tensor = None,
                action_type_str: str = None,
                vehicle_location: int = None,
                target_location: int = None) -> torch.Tensor:
        """
        改进的前向传播
        
        新增参数：
            action_type_str: Action类型字符串 ('assign', 'charge', 'idle')
            vehicle_location: 车辆当前位置（用于计算距离）
            target_location: 目标位置（用于计算距离）
        """
        batch_size = path_locations.size(0)
        
        # 1. 处理位置嵌入和路径LSTM（原有逻辑）
        location_embeds = self.location_embedding(path_locations)
        mask = (path_locations != 0).float().unsqueeze(-1)
        masked_delays = path_delays * mask
        path_input = torch.cat([location_embeds, masked_delays], dim=-1)
        
        lstm_out, (hidden, _) = self.path_lstm(path_input)
        path_representation = hidden[-1]  # [batch_size, lstm_hidden]
        
        # 2. 处理时间嵌入（原有逻辑）
        time_embed = self.time_embedding(current_time)  # [batch_size, embedding_dim//2]
        
        # 3. 处理Action Type嵌入（新增）
        if action_type_str is not None:
            action_type_id = self._get_action_type_id(action_type_str)
            action_type_ids = torch.full((batch_size,), action_type_id, dtype=torch.long).to(path_locations.device)
        else:
            # 如果没有提供action_type，尝试从路径推断
            action_type_ids = torch.zeros(batch_size, dtype=torch.long).to(path_locations.device)
        
        action_embed = self.action_type_embedding(action_type_ids)  # [batch_size, embedding_dim//4]
        
        # 4. 处理距离嵌入（新增）
        if vehicle_location is not None and target_location is not None:
            distance = self._calculate_manhattan_distance(vehicle_location, target_location)
            distance_tensor = torch.full((batch_size, 1), distance / 20.0).to(path_locations.device)  # 归一化
        else:
            distance_tensor = torch.zeros(batch_size, 1).to(path_locations.device)
        
        distance_embed = self.distance_embedding(distance_tensor)  # [batch_size, embedding_dim//4]
        
        # 5. 处理battery和request_value（原有逻辑）
        if battery_level is None:
            battery_level = torch.ones(current_time.size()).to(current_time.device)
        if request_value is None:
            request_value = torch.zeros(current_time.size()).to(current_time.device)
        
        # 6. 特征融合
        combined_features = torch.cat([
            path_representation,    # [lstm_hidden]
            time_embed,            # [embedding_dim//2]
            action_embed,          # [embedding_dim//4] - 新增
            distance_embed,        # [embedding_dim//4] - 新增
            other_agents,          # [1]
            num_requests,          # [1]
            battery_level,         # [1]
            request_value          # [1]
        ], dim=1)  # [batch_size, raw_feature_dim]
        
        # 投影到正确的特征维度
        projected_features = self.feature_projection(combined_features)  # [batch_size, feature_dim]
        
        # 7. Multi-Head Attention处理（新增）
        # 将特征扩展为序列形式以适应attention
        features_seq = projected_features.unsqueeze(1)  # [batch_size, 1, feature_dim]
        attended_features, _ = self.attention(features_seq, features_seq, features_seq)
        attended_features = attended_features.squeeze(1)  # [batch_size, feature_dim]
        
        # 8. Action-specific分支处理（新增）
        if action_type_str is not None:
            if action_type_str.startswith('assign'):
                action_branch_output = self.assign_branch(attended_features)
            elif action_type_str.startswith('charge'):
                action_branch_output = self.charge_branch(attended_features)
            elif action_type_str == 'idle':
                action_branch_output = self.idle_branch(attended_features)
            else:
                # 对于未知action类型，使用assign分支作为默认
                action_branch_output = self.assign_branch(attended_features)
        else:
            # 如果没有action_type信息，使用assign分支
            action_branch_output = self.assign_branch(attended_features)
        
        # 9. 最终特征融合
        final_features = torch.cat([attended_features, action_branch_output], dim=1)
        
        # 10. 最终输出
        value = self.final_layers(final_features)
        
        return value


def create_improved_value_function():
    """创建改进的值函数实例"""
    
    class ImprovedChargingValueFunction:
        """改进的充电环境值函数，支持增强的Action区分度"""
        
        def __init__(self, grid_size: int = 10, num_vehicles: int = 8, 
                     log_dir: str = "logs/improved_charging_nn", device: str = 'cpu',
                     episode_length: int = 300, max_requests: int = 1000):
            
            self.grid_size = grid_size
            self.num_vehicles = num_vehicles
            self.episode_length = episode_length
            self.max_requests = max_requests
            self.num_locations = grid_size * grid_size
            self.device = torch.device(device)
            
            # 使用改进的网络
            self.network = ImprovedPathBasedNetwork(
                num_locations=self.num_locations,
                max_capacity=6,
                embedding_dim=128,
                lstm_hidden=256,
                dense_hidden=512
            ).to(self.device)
            
            # 目标网络
            self.target_network = ImprovedPathBasedNetwork(
                num_locations=self.num_locations,
                max_capacity=6,
                embedding_dim=128,
                lstm_hidden=256,
                dense_hidden=512
            ).to(self.device)
            
            self.target_network.load_state_dict(self.network.state_dict())
            
            # 优化器
            self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
            self.loss_fn = nn.MSELoss()
            
            # 经验缓冲区
            from collections import deque
            self.experience_buffer = deque(maxlen=20000)
            
            print(f"✓ 改进的PyTorchChargingValueFunction初始化完成")
            print(f"   - 网络参数: {sum(p.numel() for p in self.network.parameters())}")
            print(f"   - 增强的Action区分度: Action Type Embedding + Multi-Head Attention")
        
        def get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                       target_location: int, current_time: float = 0.0, 
                       other_vehicles: int = 0, num_requests: int = 0, 
                       battery_level: float = 1.0, request_value: float = 0.0) -> float:
            """
            使用改进网络计算Q值，现在包含action_type和距离信息
            """
            # 准备输入
            inputs = self._prepare_enhanced_input(
                vehicle_location, target_location, current_time, 
                other_vehicles, num_requests, action_type, 
                battery_level, request_value
            )
            
            # 前向传播
            self.network.eval()
            with torch.no_grad():
                q_value = self.network(
                    path_locations=inputs['path_locations'],
                    path_delays=inputs['path_delays'],
                    current_time=inputs['current_time'],
                    other_agents=inputs['other_agents'],
                    num_requests=inputs['num_requests'],
                    battery_level=inputs['battery_level'],
                    request_value=inputs['request_value'],
                    action_type_str=action_type,  # 新增：直接传递action类型
                    vehicle_location=vehicle_location,  # 新增：车辆位置
                    target_location=target_location     # 新增：目标位置
                )
                
                return float(q_value.item())
        
        def _prepare_enhanced_input(self, vehicle_location: int, target_location: int, 
                                   current_time: float, other_vehicles: int, 
                                   num_requests: int, action_type: str, 
                                   battery_level: float = 1.0, request_value: float = 0.0):
            """准备增强的网络输入"""
            
            # 路径序列（简化版本）
            path_locations = torch.zeros(1, 3, dtype=torch.long)
            path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
            
            # 确保位置索引在有效范围内
            safe_vehicle_location = max(0, min(vehicle_location, self.num_locations - 1))
            safe_target_location = max(0, min(target_location, self.num_locations - 1))
            
            path_locations[0, 0] = safe_vehicle_location + 1
            path_locations[0, 1] = safe_target_location + 1
            path_locations[0, 2] = 0  # End token
            
            # 根据action类型设置不同的延迟模式（更显著的区分）
            if action_type.startswith('assign'):
                path_delays[0, 0, 0] = 0.0
                path_delays[0, 1, 0] = 0.2 + max(0.0, (self.episode_length - current_time) / self.episode_length) * 0.3
            elif action_type.startswith('charge'):
                path_delays[0, 0, 0] = 0.0
                path_delays[0, 1, 0] = 0.8  # 显著的充电延迟
            elif action_type == 'idle':
                path_delays[0, 0, 0] = 0.0
                path_delays[0, 1, 0] = 0.01  # 很小的idle延迟
            else:
                path_delays[0, 0, 0] = 0.0
                path_delays[0, 1, 0] = 0.1
            
            # 归一化其他特征
            time_tensor = torch.tensor([[current_time / self.episode_length]], dtype=torch.float32)
            others_tensor = torch.tensor([[min(other_vehicles, self.num_vehicles) / self.num_vehicles]], dtype=torch.float32)
            requests_tensor = torch.tensor([[min(num_requests, self.max_requests) / self.max_requests]], dtype=torch.float32)
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32)
            value_tensor = torch.tensor([[request_value / 100.0]], dtype=torch.float32)
            
            return {
                'path_locations': path_locations.to(self.device),
                'path_delays': path_delays.to(self.device),
                'current_time': time_tensor.to(self.device),
                'other_agents': others_tensor.to(self.device),
                'num_requests': requests_tensor.to(self.device),
                'battery_level': battery_tensor.to(self.device),
                'request_value': value_tensor.to(self.device)
            }
        
        def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                                  vehicle_location: int, target_location: int, 
                                  current_time: float = 0.0, other_vehicles: int = 0, 
                                  num_requests: int = 0, battery_level: float = 1.0,
                                  request_value: float = 0.0) -> float:
            """计算assignment Q值"""
            return self.get_q_value(vehicle_id, f"assign_{target_id}", 
                                   vehicle_location, target_location, current_time, 
                                   other_vehicles, num_requests, battery_level, request_value)
        
        def get_charging_q_value(self, vehicle_id: int, station_id: int,
                               vehicle_location: int, station_location: int,
                               current_time: float = 0.0, other_vehicles: int = 0,
                               num_requests: int = 0, battery_level: float = 1.0) -> float:
            """计算charging Q值"""
            return self.get_q_value(vehicle_id, f"charge_{station_id}",
                                   vehicle_location, station_location, current_time,
                                   other_vehicles, num_requests, battery_level)
        
        def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, 
                            battery_level: float, current_time: float = 0.0, 
                            other_vehicles: int = 0, num_requests: int = 0) -> float:
            """计算idle Q值"""
            return self.get_q_value(vehicle_id, "idle", vehicle_location, vehicle_location, 
                                   current_time, other_vehicles, num_requests, battery_level)
    
    return ImprovedChargingValueFunction


if __name__ == "__main__":
    print("测试改进的神经网络Action区分度...")
    
    # 创建改进的值函数
    improved_vf = create_improved_value_function()()
    
    # 测试参数
    vehicle_location = 45
    target_location_assign = 67
    target_location_charge = 23
    current_time = 150.0
    other_vehicles = 2
    num_requests = 8
    battery_level = 0.6
    request_value = 25.0
    
    print(f"\n测试不同Action类型的Q值计算:")
    print(f"车辆位置: {vehicle_location}, 请求位置: {target_location_assign}, 充电站位置: {target_location_charge}")
    print(f"电池电量: {battery_level}, 请求价值: {request_value}")
    
    # 计算Q值
    q_assign = improved_vf.get_assignment_q_value(
        0, 1, vehicle_location, target_location_assign, 
        current_time, other_vehicles, num_requests, battery_level, request_value
    )
    
    q_charge = improved_vf.get_charging_q_value(
        0, 0, vehicle_location, target_location_charge,
        current_time, other_vehicles, num_requests, battery_level
    )
    
    q_idle = improved_vf.get_idle_q_value(
        0, vehicle_location, battery_level, 
        current_time, other_vehicles, num_requests
    )
    
    print(f"\n改进后的Q值:")
    print(f"  Assign: {q_assign:.6f}")
    print(f"  Charge: {q_charge:.6f}")
    print(f"  Idle:   {q_idle:.6f}")
    
    # 显示排序
    q_values = [("Assign", q_assign), ("Charge", q_charge), ("Idle", q_idle)]
    q_values.sort(key=lambda x: x[1], reverse=True)
    print(f"\nQ值排序:")
    for i, (action, q_val) in enumerate(q_values):
        print(f"  {i+1}. {action}: {q_val:.6f}")
    
    print(f"\n✓ 改进的神经网络包含以下增强特征:")
    print(f"  1. 显式的Action Type Embedding")
    print(f"  2. 距离信息嵌入")
    print(f"  3. Multi-Head Attention机制")
    print(f"  4. Action-specific处理分支")
    print(f"  5. 更显著的延迟模式区分")
