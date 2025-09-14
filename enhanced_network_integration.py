#!/usr/bin/env python3
"""
替换现有神经网络结构以增强Action区分度的方案
这个文件包含可以直接集成到现有ValueFunction_pytorch.py中的改进
"""

import torch
import torch.nn as nn

class EnhancedPathBasedNetwork(nn.Module):
    """
    增强版路径基础神经网络 - 可直接替换现有的PyTorchPathBasedNetwork
    
    主要改进：
    1. 显式的Action Type Embedding
    2. 增强的延迟模式区分
    3. Action-specific特征处理
    4. 更好的特征融合
    """
    
    def __init__(self, 
                 num_locations: int,
                 max_capacity: int,
                 embedding_dim: int = 128,
                 lstm_hidden: int = 256,
                 dense_hidden: int = 512,
                 pretrained_embeddings = None):
        super(EnhancedPathBasedNetwork, self).__init__()
        
        self.num_locations = num_locations
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        
        # 1. Action Type Embedding - 显式区分不同动作类型
        self.action_type_to_id = {
            'assign': 1,
            'charge': 2, 
            'idle': 3,
            'unknown': 0
        }
        self.action_type_embedding = nn.Embedding(
            num_embeddings=4,  # assign, charge, idle, unknown
            embedding_dim=embedding_dim // 4,
            padding_idx=0
        )
        
        # 2. 位置嵌入（保持原有）
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        if pretrained_embeddings is not None:
            self.location_embedding.weight.data.copy_(pretrained_embeddings)
        
        # 3. 路径LSTM（保持原有）
        self.path_lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # location embedding + delay
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # 4. 时间嵌入（保持原有）
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim // 2),
            nn.ELU()
        )
        
        # 5. 上下文嵌入 - 处理环境状态
        self.context_embedding = nn.Sequential(
            nn.Linear(4, embedding_dim // 4),  # other_agents + num_requests + battery + request_value
            nn.ELU()
        )
        
        # 6. Action-specific特征处理
        action_feature_dim = embedding_dim // 4
        self.assign_features = nn.Sequential(
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.ELU(),
            nn.Linear(action_feature_dim, action_feature_dim)
        )
        
        self.charge_features = nn.Sequential(
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.ELU(),
            nn.Linear(action_feature_dim, action_feature_dim)
        )
        
        self.idle_features = nn.Sequential(
            nn.Linear(action_feature_dim, action_feature_dim),
            nn.ELU(),
            nn.Linear(action_feature_dim, action_feature_dim)
        )
        
        # 7. 最终状态嵌入
        # 计算总特征维度: path + time + context + action_specific
        total_feature_dim = lstm_hidden + embedding_dim // 2 + embedding_dim // 4 + action_feature_dim
        
        self.state_embedding = nn.Sequential(
            nn.Linear(total_feature_dim, dense_hidden),
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
    
    def _get_action_type_id(self, action_type_str: str) -> int:
        """从action_type字符串获取ID"""
        if action_type_str.startswith('assign'):
            return self.action_type_to_id['assign']
        elif action_type_str.startswith('charge'):
            return self.action_type_to_id['charge']
        elif action_type_str == 'idle':
            return self.action_type_to_id['idle']
        else:
            return self.action_type_to_id['unknown']
    
    def forward(self, 
                path_locations: torch.Tensor,
                path_delays: torch.Tensor,
                current_time: torch.Tensor,
                other_agents: torch.Tensor,
                num_requests: torch.Tensor,
                battery_level: torch.Tensor = None,
                request_value: torch.Tensor = None,
                action_type_str: str = None) -> torch.Tensor:
        """
        增强的前向传播，支持action_type信息
        
        新增参数：
            action_type_str: Action类型字符串，用于显式区分动作类型
        """
        batch_size = path_locations.size(0)
        
        # 1. 路径处理（保持原有逻辑）
        location_embeds = self.location_embedding(path_locations)
        mask = (path_locations != 0).float().unsqueeze(-1)
        masked_delays = path_delays * mask
        path_input = torch.cat([location_embeds, masked_delays], dim=-1)
        
        lstm_out, (hidden, _) = self.path_lstm(path_input)
        path_representation = hidden[-1]  # [batch_size, lstm_hidden]
        
        # 2. 时间处理（保持原有逻辑）
        time_embed = self.time_embedding(current_time)
        
        # 3. 处理缺失值
        if battery_level is None:
            battery_level = torch.ones(current_time.size()).to(current_time.device)
        if request_value is None:
            request_value = torch.zeros(current_time.size()).to(current_time.device)
        
        # 4. 上下文特征嵌入
        context_features = torch.cat([
            other_agents, num_requests, battery_level, request_value
        ], dim=1)
        context_embed = self.context_embedding(context_features)
        
        # 5. Action Type嵌入和特定处理
        if action_type_str is not None:
            action_type_id = self._get_action_type_id(action_type_str)
            action_type_ids = torch.full((batch_size,), action_type_id, dtype=torch.long).to(path_locations.device)
            action_type_embed = self.action_type_embedding(action_type_ids)
            
            # Action-specific特征处理
            if action_type_str.startswith('assign'):
                action_specific_features = self.assign_features(action_type_embed)
            elif action_type_str.startswith('charge'):
                action_specific_features = self.charge_features(action_type_embed)
            elif action_type_str == 'idle':
                action_specific_features = self.idle_features(action_type_embed)
            else:
                action_specific_features = action_type_embed  # 直接使用嵌入
        else:
            # 如果没有action_type信息，使用零向量
            action_specific_features = torch.zeros(batch_size, self.embedding_dim // 4).to(path_locations.device)
        
        # 6. 特征融合
        combined_features = torch.cat([
            path_representation,      # 路径表示
            time_embed,              # 时间嵌入
            context_embed,           # 上下文嵌入
            action_specific_features # Action特定特征
        ], dim=1)
        
        # 7. 最终值预测
        value = self.state_embedding(combined_features)
        
        return value


def create_enhanced_input_preparation_methods():
    """
    创建增强的输入准备方法，可以集成到现有的PyTorchChargingValueFunction中
    """
    
    def _prepare_enhanced_network_input_with_action_type(self, vehicle_location: int, target_location: int, 
                                                        current_time: float, other_vehicles: int, 
                                                        num_requests: int, action_type: str, 
                                                        battery_level: float = 1.0, request_value: float = 0.0):
        """
        增强的网络输入准备方法，显著提高不同action类型的区分度
        """
        path_locations = torch.zeros(1, 3, dtype=torch.long)
        path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
        
        # 确保位置索引有效
        safe_vehicle_location = max(0, min(vehicle_location, self.num_locations - 1))
        safe_target_location = max(0, min(target_location, self.num_locations - 1))
        
        path_locations[0, 0] = safe_vehicle_location + 1
        path_locations[0, 1] = safe_target_location + 1
        path_locations[0, 2] = 0  # End token
        
        # 显著增强不同action类型的延迟模式区分
        if action_type.startswith('assign'):
            # ASSIGN: 基于请求价值和紧急度的延迟
            urgency = max(0.0, (self.episode_length - current_time) / self.episode_length)
            value_factor = min(request_value / 50.0, 1.0)  # 归一化请求价值
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.1 + urgency * 0.3 + value_factor * 0.2  # 0.1-0.6 范围
            
        elif action_type.startswith('charge'):
            # CHARGE: 基于电池电量的延迟
            battery_urgency = max(0.0, (0.3 - battery_level) / 0.3)  # 电量越低越紧急
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.7 + battery_urgency * 0.2  # 0.7-0.9 范围（高延迟）
            
        elif action_type == 'idle':
            # IDLE: 很小的固定延迟
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.01  # 很小的固定值
            
        else:
            # 其他情况: 中等延迟
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.15
        
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
            'request_value': value_tensor.to(self.device),
            'action_type_str': action_type  # 传递action类型信息
        }
    
    def enhanced_get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                           target_location: int, current_time: float = 0.0, 
                           other_vehicles: int = 0, num_requests: int = 0, 
                           battery_level: float = 1.0, request_value: float = 0.0) -> float:
        """
        增强的Q值计算方法，支持action_type信息传递
        """
        # 使用增强的输入准备方法
        inputs = self._prepare_enhanced_network_input_with_action_type(
            vehicle_location, target_location, current_time, 
            other_vehicles, num_requests, action_type, battery_level, request_value
        )
        
        # 前向传播时传递action_type信息
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
                action_type_str=action_type  # 新增：传递action类型
            )
            
            return float(q_value.item())
    
    return _prepare_enhanced_network_input_with_action_type, enhanced_get_q_value


def integration_instructions():
    """
    集成指导：如何将增强功能集成到现有系统中
    """
    instructions = """
    📋 集成增强Action区分度的步骤：
    
    1️⃣ 替换神经网络类：
       将现有的 PyTorchPathBasedNetwork 替换为 EnhancedPathBasedNetwork
    
    2️⃣ 更新网络实例化：
       在 PyTorchChargingValueFunction.__init__ 中：
       self.network = EnhancedPathBasedNetwork(...)
       self.target_network = EnhancedPathBasedNetwork(...)
    
    3️⃣ 更新输入准备方法：
       添加 _prepare_enhanced_network_input_with_action_type 方法
    
    4️⃣ 更新Q值计算方法：
       修改 get_q_value 方法以传递 action_type_str 参数
    
    5️⃣ 更新训练方法：
       在 train_step 中确保传递 action_type_str 信息
    
    🔧 主要改进特性：
    ✅ 显式的Action Type Embedding
    ✅ 增强的延迟模式区分 (assign: 0.1-0.6, charge: 0.7-0.9, idle: 0.01)
    ✅ Action-specific特征处理分支
    ✅ 更好的上下文信息融合
    ✅ 保持与现有接口的兼容性
    
    📊 预期效果：
    - 不同action类型的Q值将有更明显的区分
    - 神经网络能够学习到action-specific的模式
    - 减少assign Q值总是最低的问题
    """
    return instructions


if __name__ == "__main__":
    print("🚀 增强Action区分度的神经网络改进方案")
    print("=" * 60)
    
    # 显示集成指导
    print(integration_instructions())
    
    # 测试网络结构
    print("\n🧪 测试增强网络结构...")
    
    enhanced_network = EnhancedPathBasedNetwork(
        num_locations=100,
        max_capacity=6,
        embedding_dim=128,
        lstm_hidden=256,
        dense_hidden=512
    )
    
    print(f"✅ 网络参数总数: {sum(p.numel() for p in enhanced_network.parameters())}")
    
    # 模拟输入测试
    batch_size = 1
    path_locations = torch.randint(1, 101, (batch_size, 3))
    path_delays = torch.rand(batch_size, 3, 1)
    current_time = torch.rand(batch_size, 1)
    other_agents = torch.rand(batch_size, 1)
    num_requests = torch.rand(batch_size, 1)
    battery_level = torch.rand(batch_size, 1)
    request_value = torch.rand(batch_size, 1)
    
    # 测试不同action类型
    action_types = ['assign_1', 'charge_0', 'idle']
    
    print(f"\n📊 测试不同Action类型的输出:")
    for action_type in action_types:
        output = enhanced_network(
            path_locations, path_delays, current_time,
            other_agents, num_requests, battery_level, request_value,
            action_type_str=action_type
        )
        print(f"  {action_type:10s}: {output.item():.6f}")
    
    print(f"\n✅ 增强网络测试完成！")
    print(f"💡 下一步：将这些改进集成到现有的ValueFunction_pytorch.py中")
