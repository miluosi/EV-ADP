"""
PyTorch-based Value Function for ADP with Gym Integration

This module replaces the original Keras/TensorFlow implementation with PyTorch,
while maintaining the core ADP algorithm concepts.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
import gym
from gym import spaces
from typing import List, Tuple, Dict, Any, Optional
from abc import ABC, abstractmethod
from collections import deque
import random
import pickle
from pathlib import Path as PathlibPath
import logging

# Use fallback logger instead of tensorboard to avoid protobuf issues
TENSORBOARD_AVAILABLE = False
print("Using fallback logging instead of TensorBoard to avoid protobuf compatibility issues")

class SummaryWriter:
    """Fallback logger when TensorBoard is not available"""
    def __init__(self, *args, **kwargs):
        self.log_dir = args[0] if args else "logs"
        print(f"Fallback logger initialized for {self.log_dir}")
    
    def add_scalar(self, tag, value, step):
        print(f"[{step}] {tag}: {value:.4f}")
    
    def flush(self):
        pass
    
    def close(self):
        pass

# Import existing modules (modified for compatibility)
try:
    from LearningAgent import LearningAgent
    from Action import Action  
    from Environment import Environment
    from Experience import Experience
    from CentralAgent import CentralAgent
    from Request import Request
except ImportError:
    # Define placeholder classes if imports fail
    class LearningAgent: pass
    class Action: pass
    class Environment: pass
    class Experience: pass
    class CentralAgent: pass
    class Request: pass
    class Path: pass
    class Experience: pass
    class CentralAgent: pass
    class Request: pass


class PyTorchReplayBuffer:
    """PyTorch-based replay buffer for experience storage"""
    
    def __init__(self, capacity: int, device: str = 'cpu'):
        self.capacity = capacity
        self.device = torch.device(device)
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def add(self, experience: Experience, priority: float = 1.0):
        """Add experience to buffer"""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, beta: float = 0.4):
        """Sample batch with prioritized sampling"""
        if len(self.buffer) < batch_size:
            return [], [], []
            
        # Convert priorities to probabilities
        priorities = np.array(self.priorities)
        probs = priorities ** 0.6  # Alpha parameter
        probs = probs / probs.sum()
        
        # Sample indices
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices]
        
        # Importance sampling weights
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights = weights / weights.max()
        
        return experiences, weights, indices
    
    def update_priorities(self, indices: List[int], priorities: List[float]):
        """Update priorities for experiences"""
        for idx, priority in zip(indices, priorities):
            if 0 <= idx < len(self.priorities):
                self.priorities[idx] = priority + 1e-6
    
    def __len__(self):
        return len(self.buffer)


class PyTorchValueFunction(ABC, nn.Module):
    """Abstract base class for PyTorch-based value functions"""
    
    def __init__(self, log_dir: str, device: str = 'cpu'):
        super(PyTorchValueFunction, self).__init__()
        self.device = torch.device(device)
        
        # Setup logging
        log_path = PathlibPath(log_dir) / type(self).__name__
        log_path.mkdir(parents=True, exist_ok=True)
        self.writer = SummaryWriter(str(log_path))
        
        # Training statistics
        self.training_step = 0
        
    def add_to_logs(self, tag: str, value: float, step: int):
        """Add scalar to tensorboard logs"""
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()
    
    @abstractmethod
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Get value estimates for experiences"""
        raise NotImplementedError
    
    @abstractmethod
    def update(self, central_agent: CentralAgent):
        """Update value function parameters"""
        raise NotImplementedError
    
    @abstractmethod
    def remember(self, experience: Experience):
        """Store experience for learning"""
        raise NotImplementedError


class PyTorchRewardPlusDelay(PyTorchValueFunction):
    """Simple reward + delay value function (no learning required)"""
    
    def __init__(self, delay_coefficient: float, log_dir: str = "logs/reward_plus_delay", device: str = 'cpu'):
        super().__init__(log_dir=log_dir, device=device)
        self.delay_coefficient = delay_coefficient
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initialized PyTorchRewardPlusDelay with delay_coefficient={delay_coefficient}")
    
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Compute value as immediate reward plus delay bonus"""
        scored_actions_all_agents = []
        
        for experience in experiences:
            for feasible_actions in experience.feasible_actions_all_agents:
                scored_actions = []
                for action in feasible_actions:
                    if hasattr(action, 'new_path') and action.new_path:
                        immediate_reward = sum([getattr(request, 'value', 0) 
                                              for request in getattr(action, 'requests', [])])
                        delay_bonus = self.delay_coefficient * getattr(action.new_path, 'total_delay', 0)
                        score = immediate_reward + delay_bonus
                    else:
                        score = 0.0
                    
                    scored_actions.append((action, score))
                scored_actions_all_agents.append(scored_actions)
        
        return scored_actions_all_agents
    
    def get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                   target_location: int, current_time: float = 0.0) -> float:
        """
        Simple Q-value calculation for ChargingIntegratedEnvironment
        
        Args:
            vehicle_id: ID of the vehicle
            action_type: Type of action ('assign', 'charge', 'move', 'idle')
            vehicle_location: Current location of vehicle
            target_location: Target location (request pickup or charging station)
            current_time: Current simulation time
            
        Returns:
            Q-value for the state-action pair
        """
        # Base reward calculation
        base_reward = 0.0
        
        # Distance penalty
        if hasattr(self, '_calculate_distance'):
            distance = self._calculate_distance(vehicle_location, target_location)
        else:
            # Simple Manhattan distance approximation
            distance = abs(vehicle_location - target_location)
        
        distance_penalty = distance * 0.1
        
        # Action-specific rewards
        if action_type.startswith('assign'):
            # Passenger service reward
            base_reward = 10.0  # Base reward for serving passenger
            # Add urgency bonus based on delay
            urgency_bonus = self.delay_coefficient * max(0, 300 - current_time)  # 300 is max delay
            base_reward += urgency_bonus
            
        elif action_type.startswith('charge'):
            # Charging reward - depends on battery level (if available)
            base_reward = 5.0  # Base charging benefit
            # Negative delay penalty for charging (opportunity cost)
            charging_delay_penalty = self.delay_coefficient * 30  # Assume 30 time units charging
            base_reward -= charging_delay_penalty
            
        elif action_type == 'move':
            # Movement has small cost
            base_reward = -1.0
            
        elif action_type == 'idle':
            # Idle has minimal cost
            base_reward = -0.5
        
        # Final Q-value calculation
        q_value = base_reward - distance_penalty
        
        return float(q_value)
    
    def _calculate_distance(self, loc1: int, loc2: int, grid_size: int = 10) -> float:
        """Calculate Manhattan distance between two locations"""
        x1, y1 = loc1 // grid_size, loc1 % grid_size
        x2, y2 = loc2 // grid_size, loc2 % grid_size
        return abs(x1 - x2) + abs(y1 - y2)
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int, 
                              current_time: float = 0.0) -> float:
        """Get Q-value for vehicle assignment to request"""
        return self.get_q_value(vehicle_id, f"assign_{target_id}", 
                               vehicle_location, target_location, current_time)
    
    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int,
                           current_time: float = 0.0) -> float:
        """Get Q-value for vehicle charging decision"""
        return self.get_q_value(vehicle_id, f"charge_{station_id}",
                               vehicle_location, station_location, current_time)

    def update(self, *args, **kwargs):
        """No learning required for this value function"""
        pass
    
    def remember(self, *args, **kwargs):
        """No experience storage required"""
        pass


class PyTorchChargingValueFunction(PyTorchValueFunction):
    """Neural network-based value function for ChargingIntegratedEnvironment using PyTorchPathBasedNetwork"""
    
    def __init__(self, grid_size: int = 10, num_vehicles: int = 8, 
                 log_dir: str = "logs/charging_nn", device: str = 'cpu',
                 episode_length: int = 300, max_requests: int = 1000):
        super().__init__(log_dir=log_dir, device=device)
        
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.episode_length = episode_length  # 实际episode长度
        self.max_requests = max_requests      # 最大预期请求数
        self.num_locations = grid_size * grid_size
        
        # Initialize the neural network with increased capacity for complex environment
        self.network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            num_vehicles=num_vehicles,  # 添加车辆数量参数
            max_capacity=6,  # Increased capacity for longer paths
            embedding_dim=128,  # Larger embedding for complex environment
            lstm_hidden=256,   # Larger LSTM for complex patterns
            dense_hidden=512,   # Larger dense layer
            pretrained_embeddings=None  # Explicitly set to None to ensure gradients
        ).to(self.device)
        
        # Target network for stable DQN training
        self.target_network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            num_vehicles=num_vehicles,  # 添加车辆数量参数
            max_capacity=6,
            embedding_dim=128,
            lstm_hidden=256,
            dense_hidden=512,
            pretrained_embeddings=None
        ).to(self.device)
        
        # Copy weights from main network to target network
        self.target_network.load_state_dict(self.network.state_dict())
        self.target_update_frequency = 100  # Update target network every 100 steps
        
        # Ensure all parameters require gradients
        total_params = 0
        grad_params = 0
        for name, param in self.network.named_parameters():
            param.requires_grad = True
            total_params += 1
            if param.requires_grad:
                grad_params += 1
        
        print(f"   Parameters requiring gradients: {grad_params}/{total_params}")
        
        # Optimizer for training - reduced learning rate for stable learning
        self.optimizer = optim.Adam(self.network.parameters(), lr=2e-3, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # 修复学习率调度器：更保守的设置，避免学习率过快下降
        # 原设置：factor=0.7, patience=50, min_lr=1e-4 太激进
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.9, patience=200, 
            min_lr=1e-3, verbose=True  # 保持最小学习率为1e-3，避免过度降低
        )
        
        # Training data buffer - increased size for more diverse experiences
        self.experience_buffer = deque(maxlen=20000)  # Doubled buffer size
        
        # Training metrics tracking
        self.training_losses = []
        self.q_values_history = []
        self.training_step = 0
        
        print(f"✓ PyTorchChargingValueFunction initialized with neural network")
        print(f"   - Grid size: {grid_size}x{grid_size}")
        print(f"   - Network parameters: {sum(p.numel() for p in self.network.parameters())}")
    
    def get_q_value(self, vehicle_id: int, action_type: str, vehicle_location: int, 
                   target_location: int, current_time: float = 0.0, 
                   other_vehicles: int = 0, num_requests: int = 0, 
                   battery_level: float = 1.0, request_value: float = 0.0) -> float:
        """
        Neural network-based Q-value calculation using PyTorchPathBasedNetwork
        现在支持vehicle_id、battery_level、request_value和action_type参数
        """
        # 将action_type字符串转换为数值编码
        if action_type == 'idle':
            action_type_id = 1
        elif action_type.startswith('assign'):
            action_type_id = 2
        elif action_type.startswith('charge'):
            action_type_id = 3
        else:
            action_type_id = 2  # 默认为assign
        
        # 从Environment中获取车辆类型（需要从外部传入或者推断）
        # 假设vehicle_id为偶数是EV，奇数是AEV（简化处理）
        # 实际应用中应该从环境或配置中获取
        vehicle_type_id = 1 if vehicle_id % 2 == 0 else 2  # 1=EV, 2=AEV
        
        # 使用支持battery和request_value的输入准备方法
        inputs = self._prepare_network_input_with_battery(
            vehicle_location, target_location, current_time, 
            other_vehicles, num_requests, action_type, battery_level, request_value
        )
        
        # 处理返回的输入（可能包含或不包含battery和request_value）
        if len(inputs) == 7:  # 包含battery和request_value
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor, value_tensor = inputs
        elif len(inputs) == 6:  # 只包含battery
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor = inputs
            value_tensor = torch.tensor([[request_value]], dtype=torch.float32).to(self.device)
        else:  # 不包含battery（向后兼容）
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = inputs
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
            value_tensor = torch.tensor([[request_value]], dtype=torch.float32).to(self.device)
        
        # 创建vehicle和action相关的tensors
        action_type_tensor = torch.tensor([[action_type_id]], dtype=torch.long).to(self.device)
        vehicle_id_tensor = torch.tensor([[vehicle_id + 1]], dtype=torch.long).to(self.device)  # +1因为0是padding
        vehicle_type_tensor = torch.tensor([[vehicle_type_id]], dtype=torch.long).to(self.device)
        
        # Forward pass through network
        self.network.eval()
        with torch.no_grad():
            q_value = self.network(
                path_locations=path_locations,
                path_delays=path_delays,
                current_time=time_tensor,
                other_agents=others_tensor,
                num_requests=requests_tensor,
                battery_level=battery_tensor,
                request_value=value_tensor,
                action_type=action_type_tensor,
                vehicle_id=vehicle_id_tensor,
                vehicle_type=vehicle_type_tensor
            )
            
            # Apply clipping to prevent extreme Q-values that can dominate the objective
            raw_q_value = float(q_value.item())
            
            # Clip Q-values to reasonable range to prevent optimization instability
            # This ensures Q-values don't overwhelm request values in the objective function
            clipped_q_value = max(-50.0, min(50.0, raw_q_value))
            
            #if abs(raw_q_value - clipped_q_value) > 1e-6:  # Only log if actual clipping occurred
                #print(f"Q-value clipped: {raw_q_value:.3f} -> {clipped_q_value:.3f} for action {action_type}")
            
            return clipped_q_value
    
    def _prepare_network_input(self, vehicle_location: int, target_location: int, 
                              current_time: float, other_vehicles: int, num_requests: int,
                              action_type: str):
        """Prepare input tensors for the neural network"""
        # Create path sequence: current location -> target location
        path_locations = torch.zeros(1, 3, dtype=torch.long)  # batch_size=1, seq_len=3
        path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
        
        # Set path: current -> target -> end (with boundary checking)
        # Ensure indices are within valid range [0, num_locations-1]
        safe_vehicle_location = max(0, min(vehicle_location, self.num_locations - 1))
        safe_target_location = max(0, min(target_location, self.num_locations - 1))
        
        # Debug: Log if we had to clamp any values
        # if vehicle_location != safe_vehicle_location or target_location != safe_target_location:
        #     print(f"WARNING: Clamped location indices - vehicle: {vehicle_location}->{safe_vehicle_location}, target: {target_location}->{safe_target_location}, max_allowed: {self.num_locations-1}")
        
        path_locations[0, 0] = safe_vehicle_location + 1  # +1 because 0 is padding
        path_locations[0, 1] = safe_target_location + 1
        path_locations[0, 2] = 0  # End token
        
        # Set delays based on action type
        if action_type.startswith('assign'):
            # Passenger service - delays based on urgency
            path_delays[0, 0, 0] = 0.0  # No delay at current location
            path_delays[0, 1, 0] = max(0.0, (self.episode_length - current_time) / self.episode_length)  # Normalized urgency
        elif action_type.startswith('charge'):
            # Charging action - charging time penalty
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.8  # High delay for charging (opportunity cost)
        else:
            # Movement or idle
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.1  # Small delay for movement
        
        # Normalize time (0-1 range)
        time_tensor = torch.tensor([[current_time / self.episode_length]], dtype=torch.float32)
        
        # Normalize other metrics
        others_tensor = torch.tensor([[min(other_vehicles, self.num_vehicles) / self.num_vehicles]], dtype=torch.float32)
        requests_tensor = torch.tensor([[min(num_requests, self.max_requests) / self.max_requests]], dtype=torch.float32)
        
        # Debug: Log extreme values for monitoring
        if other_vehicles > self.num_vehicles:
            print(f"WARNING: other_vehicles ({other_vehicles}) > num_vehicles ({self.num_vehicles})")
        if num_requests > self.max_requests:
            print(f"WARNING: num_requests ({num_requests}) > max_requests ({self.max_requests}), clamping to {self.max_requests}")
        
        # Move to device
        return (path_locations.to(self.device), 
                path_delays.to(self.device),
                time_tensor.to(self.device),
                others_tensor.to(self.device),
                requests_tensor.to(self.device))
    
    def validate_normalization_params(self):
        """验证归一化参数的合理性"""
        print("=== Normalization Parameters Validation ===")
        print(f"Grid size: {self.grid_size}")
        print(f"Number of vehicles: {self.num_vehicles}")
        print(f"Episode length: {self.episode_length}")
        print(f"Max requests: {self.max_requests}")
        print(f"Number of locations: {self.num_locations}")
        
        # 检查参数合理性
        issues = []
        if self.episode_length <= 0:
            issues.append("Episode length must be positive")
        if self.num_vehicles <= 0:
            issues.append("Number of vehicles must be positive")
        if self.max_requests <= 0:
            issues.append("Max requests must be positive")
            
        if issues:
            print("⚠️ Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("✓ All normalization parameters are valid")
        print("=" * 45)
    
    def _prepare_network_input_with_battery(self, vehicle_location: int, target_location: int, 
                                           current_time: float, other_vehicles: int, 
                                           num_requests: int, action_type: str, 
                                           battery_level: float = 1.0, request_value: float = 0.0):
        """
        Prepare input tensors for the neural network including battery and request value information
        
        Args:
            vehicle_location: 车辆当前位置
            target_location: 目标位置
            current_time: 当前时间
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            action_type: 动作类型
            battery_level: 电池电量 (0-1)
            request_value: 请求价值 (只对assign动作有效)
        """
        # 根据动作类型选择合适的输入准备方法
        if action_type == 'idle':
            # 对于idle状态，处理目标位置为当前位置
            path_locations = torch.zeros(1, 3, dtype=torch.long)  # batch_size=1, seq_len=3
            path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
            
            # 设置路径：当前位置 -> 当前位置（表示停留）-> 结束 (with boundary checking)
            # Ensure indices are within valid range [0, num_locations-1]
            safe_vehicle_location = max(0, min(vehicle_location, self.num_locations - 1))
            
            path_locations[0, 0] = safe_vehicle_location + 1  # +1 because 0 is padding
            path_locations[0, 1] = safe_vehicle_location + 1  # 同样的位置表示idle
            path_locations[0, 2] = 0  # End token
            
            # 设置延迟 - idle状态的延迟模式
            path_delays[0, 0, 0] = 0.0  # 当前位置无延迟
            path_delays[0, 1, 0] = 0.05  # idle的小延迟（等待成本）
            path_delays[0, 2, 0] = 0.0  # 结束位置无延迟
            
            # 归一化时间 (0-1 range)
            time_tensor = torch.tensor([[current_time / self.episode_length]], dtype=torch.float32)
            
            # 归一化其他指标
            others_tensor = torch.tensor([[min(other_vehicles, self.num_vehicles) / self.num_vehicles]], dtype=torch.float32)
            requests_tensor = torch.tensor([[min(num_requests, self.max_requests) / self.max_requests]], dtype=torch.float32)
            
            # Debug: Log extreme values for monitoring
            if other_vehicles > self.num_vehicles:
                print(f"WARNING: other_vehicles ({other_vehicles}) > num_vehicles ({self.num_vehicles})")
            if num_requests > self.max_requests:
                print(f"WARNING: num_requests ({num_requests}) > max_requests ({self.max_requests}), clamping to {self.max_requests}")
            
            # 归一化电池电量
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32)
            
            # 归一化请求价值 (对idle动作，request_value应该为0)
            value_tensor = torch.tensor([[request_value / 100.0]], dtype=torch.float32)  # 假设最大价值100
            
            # Move to device
            return (path_locations.to(self.device), 
                    path_delays.to(self.device),
                    time_tensor.to(self.device),
                    others_tensor.to(self.device),
                    requests_tensor.to(self.device),
                    battery_tensor.to(self.device),
                    value_tensor.to(self.device))
        else:
            # 对于非idle动作，使用标准方法并添加battery和request_value信息
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = self._prepare_network_input(
                vehicle_location, target_location, current_time, 
                other_vehicles, num_requests, action_type
            )
            
            # 添加battery信息
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
            
            # 添加request_value信息 (归一化)
            normalized_value = request_value / 100.0 if action_type.startswith('assign') else 0.0
            value_tensor = torch.tensor([[normalized_value]], dtype=torch.float32).to(self.device)
            
            return (path_locations, path_delays, time_tensor, 
                   others_tensor, requests_tensor, battery_tensor, value_tensor)
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int, 
                              current_time: float = 0.0, other_vehicles: int = 0, 
                              num_requests: int = 0, battery_level: float = 1.0,
                              request_value: float = 0.0) -> float:
        """
        Enhanced Q-value for vehicle assignment to request using neural network
        现在包含更丰富的上下文信息和优化的计算逻辑
        """
        # 基础Q值计算
        base_q_value = self.get_q_value(vehicle_id, f"assign_{target_id}", 
                                       vehicle_location, target_location, current_time, 
                                       other_vehicles, num_requests, battery_level, request_value)
        
        # # 增强的上下文调整因子
        # context_adjustment = self._calculate_context_adjustment(
        #     vehicle_id, vehicle_location, target_location, battery_level, 
        #     request_value, other_vehicles, num_requests, current_time
        # )
        
        # 返回调整后的Q值
        return base_q_value 
        
    def _calculate_context_adjustment(self, vehicle_id: int, vehicle_location: int, 
                                    target_location: int, battery_level: float,
                                    request_value: float, other_vehicles: int, 
                                    num_requests: int, current_time: float) -> float:
        """
        计算基于上下文的Q值调整因子
        考虑车辆类型、电池状态、竞争环境、请求价值等因素
        """
        adjustment = 0.0
        
        # 1. 电池电量对分配的影响
        if battery_level < 0.3:  # 低电量时
            # 计算到充电站的距离影响
            grid_size = int(math.sqrt(max(vehicle_location, target_location)) + 1)
            distance_to_target = self._calculate_manhattan_distance(vehicle_location, target_location, grid_size)
            # 距离越远，电量越低，Q值调整越负
            battery_penalty = -0.2 * (0.3 - battery_level) * (distance_to_target / 10.0)
            adjustment += battery_penalty
            
        # 2. 请求价值对分配的影响
        if request_value > 0:
            # 高价值请求获得奖励
            value_bonus = min(0.1 * (request_value / 50.0), 0.5)  # 最大奖励0.5
            adjustment += value_bonus
            
        # 3. 竞争环境的影响
        if other_vehicles > 0 and num_requests > 0:
            competition_ratio = other_vehicles / max(num_requests, 1)
            if competition_ratio > 1.0:  # 车辆多于请求
                # 竞争激烈时，距离近的分配获得更多奖励
                grid_size = int(math.sqrt(max(vehicle_location, target_location)) + 1)
                distance = self._calculate_manhattan_distance(vehicle_location, target_location, grid_size)
                distance_bonus = max(0, 0.2 - 0.02 * distance)  # 距离越近奖励越高
                adjustment += distance_bonus
                
        # 4. 时间因素的影响（紧急请求）
        # 假设current_time可以反映请求的紧急程度
        if current_time > 0:
            time_factor = min(current_time / 100.0, 1.0)  # 时间标准化
            urgency_bonus = 0.1 * time_factor  # 时间越长越紧急
            adjustment += urgency_bonus
            
        # 5. 车辆类型的影响
        vehicle_type_id = 1 if vehicle_id % 2 == 0 else 2  # 简化的车辆类型判断
        if vehicle_type_id == 2:  # AEV类型车辆
            # AEV在某些情况下可能有优势
            aev_bonus = 0.05 if battery_level > 0.7 else -0.05
            adjustment += aev_bonus
            
        return adjustment
    def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = 0.0, 
                        other_vehicles: int = 0, num_requests: int = 0) -> float:
        """
        Get Q-value for idle action with random movement
        Idle action involves moving to a random nearby location, not staying in place
        """
        import random
        
        # Convert location index to coordinates for random target generation
        current_x = vehicle_location % self.grid_size
        current_y = vehicle_location // self.grid_size
        
        # Generate random target coordinates within 2 steps (matching Environment logic)
        target_x = max(0, min(self.grid_size-1, 
                                    current_x + random.randint(-1, 1)))
        target_y = max(0, min(self.grid_size-1, 
                                    current_y + random.randint(-1, 1)))
        target_location = target_y * self.grid_size + target_x
        
        # Use the generated random target location for Q-value calculation
        return self.get_q_value(vehicle_id, "idle", vehicle_location, target_location, 
                               current_time, other_vehicles, num_requests, battery_level)

    def get_waiting_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = 0.0, 
                        other_vehicles: int = 0, num_requests: int = 0) -> float:
        """
        Get Q-value for waiting action (staying in place)
        Unlike idle action, waiting means the vehicle stays at the current location
        """
        # For waiting action, target location equals current location (no movement)
        return self.get_q_value(vehicle_id, "idle", vehicle_location, vehicle_location, 
                               current_time, other_vehicles, num_requests, battery_level)



    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int,
                           current_time: float = 0.0, other_vehicles: int = 0,
                           num_requests: int = 0, battery_level: float = 1.0) -> float:
        """
        Get Q-value for vehicle charging decision using neural network
        现在支持battery_level参数
        """
        return self.get_q_value(vehicle_id, f"charge_{station_id}",
                               vehicle_location, station_location, current_time,
                               other_vehicles, num_requests, battery_level)
    
    def store_experience(self, vehicle_id: int, action_type: str, vehicle_location: int,
                        target_location: int, current_time: float, reward: float,
                        next_vehicle_location: int, battery_level: float = 1.0, 
                        next_battery_level: float = 1.0, other_vehicles: int = 0, 
                        num_requests: int = 0, request_value: float = 0.0,
                        next_action_type: str = None, next_request_value: float = 0.0):
        """
        Store experience for training - 现在支持vehicle_id、battery和request_value信息
        
        Args:
            vehicle_id: 车辆ID
            action_type: 动作类型
            vehicle_location: 车辆当前位置
            target_location: 目标位置
            current_time: 当前时间
            reward: 获得的奖励
            next_vehicle_location: 下一状态的车辆位置
            battery_level: 当前电池电量 (默认1.0为向后兼容)
            next_battery_level: 下一状态的电池电量 (默认1.0为向后兼容)
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            request_value: 请求价值 (只对assign动作有效，默认0.0)
            next_action_type: 下一个动作类型 (车辆完成当前动作后根据ILP分配的动作标签)
        """
        # 从vehicle_id推断车辆类型（简化处理）
        vehicle_type = 1 if vehicle_id % 2 == 0 else 2  # 1=EV, 2=AEV
        
        experience = {
            'vehicle_id': vehicle_id,
            'vehicle_type': vehicle_type,  # 添加车辆类型
            'action_type': action_type,
            'vehicle_location': vehicle_location,
            'target_location': target_location,
            'battery_level': battery_level,  # 添加当前电池电量
            'current_time': current_time,
            'reward': reward,
            'next_vehicle_location': next_vehicle_location,
            'next_battery_level': next_battery_level,  # 添加下一状态电池电量
            'next_action_type': next_action_type if next_action_type is not None else action_type,  # 添加下一动作类型，默认为当前动作类型
            'other_vehicles': other_vehicles,
            'num_requests': num_requests,
            'request_value': request_value,  # 添加请求价值信息
            'next_request_value': next_request_value,  # 下一状态请求价值
            'is_idle': action_type == 'idle'  # 自动标记idle状态
        }
        self.experience_buffer.append(experience)
    
    def store_idle_experience(self, vehicle_id: int, vehicle_location: int, 
                            battery_level: float, current_time: float, reward: float,
                            next_vehicle_location: int, next_battery_level: float,
                            other_vehicles: int = 0, num_requests: int = 0, request_value: float = 0.0):
        """
        Store idle experience for training - 专门为idle动作存储经验
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 当前电池电量
            current_time: 当前时间
            reward: 获得的奖励
            next_vehicle_location: 下一状态的车辆位置
            next_battery_level: 下一状态的电池电量
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            request_value: 请求价值 (idle时为0.0)
        """
        experience = {
            'vehicle_id': vehicle_id,
            'action_type': 'idle',
            'vehicle_location': vehicle_location,
            'target_location': vehicle_location,  # idle时目标位置就是当前位置
            'battery_level': battery_level,
            'current_time': current_time,
            'reward': reward,
            'next_vehicle_location': next_vehicle_location,
            'next_battery_level': next_battery_level,
            'other_vehicles': other_vehicles,
            'num_requests': num_requests,
            'request_value': request_value,  # 添加请求价值信息（idle时为0）
            'is_idle': True  # 标记这是一个idle经验
        }
        self.experience_buffer.append(experience)
    
    def _advanced_sample(self, batch_size: int, method: str = "balanced"):
        """
        简化的采样策略：只保留balanced和importance采样
        """
        experiences = list(self.experience_buffer)
        
        if method == "importance":
            return self._importance_sampling(experiences, batch_size)
        else:
            return self._balanced_sample(batch_size)
    
    def _importance_sampling(self, experiences, batch_size: int):
        """
        重要性采样：根据经验的重要性权重进行采样
        重要性基于：TD误差、奖励稀有性、动作类型稀有性
        """
        if len(experiences) == 0:
            return []
        
        # 计算每个经验的重要性权重
        weights = []
        action_counts = {'idle': 0, 'assign': 0, 'charge': 0}
        reward_values = [exp['reward'] for exp in experiences]
        
        # 统计动作类型分布
        for exp in experiences:
            action_type = exp['action_type']
            if action_type == 'idle':
                action_counts['idle'] += 1
            elif action_type.startswith('assign'):
                action_counts['assign'] += 1
            elif action_type.startswith('charge'):
                action_counts['charge'] += 1
        
        total_experiences = len(experiences)
        reward_std = np.std(reward_values) if len(reward_values) > 1 else 1.0
        
        for i, exp in enumerate(experiences):
            # 1. 动作稀有性权重
            action_type = exp['action_type']
            if action_type == 'idle':
                action_rarity = total_experiences / max(1, action_counts['idle'])
            elif action_type.startswith('assign'):
                action_rarity = total_experiences / max(1, action_counts['assign'])
            elif action_type.startswith('charge'):
                action_rarity = total_experiences / max(1, action_counts['charge'])
            else:
                action_rarity = 1.0
            
            # 2. 奖励稀有性权重
            reward = exp['reward']
            reward_rarity = abs(reward) / (reward_std + 1e-8)
            
            # 3. 时间权重（最近的经验更重要）
            time_weight = 0.5 + 0.5 * (i / max(1, len(experiences) - 1))
            
            # 4. 如果是高价值assign动作，给予额外权重
            if action_type.startswith('assign') and reward > 10:
                assign_bonus = 2.0
            else:
                assign_bonus = 1.0
            
            # 组合权重
            total_weight = action_rarity * reward_rarity * time_weight * assign_bonus
            weights.append(max(0.1, total_weight))  # 最小权重防止0权重
        
        # 归一化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 根据权重采样
        indices = np.random.choice(len(experiences), size=min(batch_size, len(experiences)), 
                                 replace=False, p=weights)
        
        sampled_experiences = [experiences[i] for i in indices]
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            print(f"📊 Importance sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}")
        
        return sampled_experiences
    
    def _thompson_sampling(self, experiences, batch_size: int):
        """
        Thompson采样：基于贝叶斯优化的探索-利用平衡
        为每种动作类型维护一个Beta分布
        """
        if len(experiences) == 0:
            return []
        
        # 为每种动作类型维护成功/失败计数
        action_stats = {
            'idle': {'success': 1, 'failure': 1},      # 先验参数
            'assign': {'success': 1, 'failure': 1},
            'charge': {'success': 1, 'failure': 1}
        }
        
        # 更新统计数据
        for exp in experiences:
            action_type = exp['action_type']
            reward = exp['reward']
            
            if action_type == 'idle':
                key = 'idle'
            elif action_type.startswith('assign'):
                key = 'assign'
            elif action_type.startswith('charge'):
                key = 'charge'
            else:
                continue
            
            # 定义成功的标准
            if reward > 0:
                action_stats[key]['success'] += 1
            else:
                action_stats[key]['failure'] += 1
        
        # 从Beta分布采样获得每种动作的期望回报
        action_expectations = {}
        for action_type, stats in action_stats.items():
            # Beta分布采样
            alpha = stats['success']
            beta = stats['failure']
            expectation = np.random.beta(alpha, beta)
            action_expectations[action_type] = expectation
        
        print(f"🎲 Thompson sampling expectations: {action_expectations}")
        
        # 基于期望回报分配采样概率
        total_expectation = sum(action_expectations.values())
        if total_expectation > 0:
            sampling_probs = {k: v/total_expectation for k, v in action_expectations.items()}
        else:
            sampling_probs = {k: 1.0/3 for k in action_expectations.keys()}
        
        # 分别从每种动作类型中采样
        sampled_experiences = []
        for action_type, prob in sampling_probs.items():
            target_count = int(batch_size * prob)
            
            # 找到该动作类型的所有经验
            if action_type == 'idle':
                type_experiences = [exp for exp in experiences if exp['action_type'] == 'idle']
            elif action_type == 'assign':
                type_experiences = [exp for exp in experiences if exp['action_type'].startswith('assign')]
            elif action_type == 'charge':
                type_experiences = [exp for exp in experiences if exp['action_type'].startswith('charge')]
            
            if type_experiences and target_count > 0:
                actual_count = min(target_count, len(type_experiences))
                sampled = random.sample(type_experiences, actual_count)
                sampled_experiences.extend(sampled)
        
        # 如果采样不足，随机补充
        remaining = batch_size - len(sampled_experiences)
        if remaining > 0:
            remaining_experiences = [exp for exp in experiences if exp not in sampled_experiences]
            if remaining_experiences:
                additional = random.sample(remaining_experiences, min(remaining, len(remaining_experiences)))
                sampled_experiences.extend(additional)
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 10000 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            print(f"📊 Thompson sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}")
        
        return sampled_experiences
    
    def _prioritized_sampling(self, experiences, batch_size: int):
        """
        优先经验回放：基于TD误差的优先级采样
        优先级 = |TD误差| + 动作价值 + 探索奖励
        """
        if len(experiences) == 0:
            return []
        
        priorities = []
        
        for exp in experiences:
            # 1. 基于奖励的基础优先级
            reward = exp['reward']
            base_priority = abs(reward) + 1e-6  # 避免0优先级
            
            # 2. 动作类型奖励
            action_type = exp['action_type']
            if action_type.startswith('assign'):
                action_bonus = 2.0  # assign动作更重要
            elif action_type.startswith('charge'):
                action_bonus = 1.5  # charge动作中等重要
            else:  # idle
                action_bonus = 1.0
            
            # 3. 稀有动作奖励
            rarity_bonus = 1.0
            if action_type.startswith('assign') and reward > 10:
                rarity_bonus = 3.0  # 高价值assign动作
            elif action_type.startswith('charge') and reward > 0:
                rarity_bonus = 2.0  # 有正回报的charge动作
            
            # 组合优先级
            priority = base_priority * action_bonus * rarity_bonus
            priorities.append(priority)
        
        # 转换为概率分布
        priorities = np.array(priorities)
        
        # 使用alpha参数控制优先级强度
        alpha = 0.6  # 0表示均匀采样，1表示纯优先级采样
        priorities = priorities ** alpha
        
        # 归一化
        probabilities = priorities / np.sum(priorities)
        
        # 采样
        indices = np.random.choice(len(experiences), size=min(batch_size, len(experiences)), 
                                 replace=False, p=probabilities)
        
        sampled_experiences = [experiences[i] for i in indices]
        
        # 调试信息 - 只在每100步输出一次
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            action_types = [exp['action_type'] for exp in sampled_experiences]
            assign_count = sum(1 for a in action_types if a.startswith('assign'))
            idle_count = sum(1 for a in action_types if a == 'idle')
            charge_count = sum(1 for a in action_types if a.startswith('charge'))
            
            avg_priority = np.mean([priorities[i] for i in indices])
            print(f"📊 Prioritized sampling: Assign={assign_count}, Idle={idle_count}, Charge={charge_count}, Avg Priority={avg_priority:.3f}")
        
        return sampled_experiences
        """
        平衡采样策略：确保正样本和负样本的比例均衡
        
        Args:
            batch_size: 批次大小
            
        Returns:
            均衡采样的经验列表
        """
    def _balanced_sample(self, batch_size: int):
        experiences = list(self.experience_buffer)
        
        # 根据奖励将经验分为正样本和负样本
        positive_samples = []  # 正奖励样本
        negative_samples = []  # 负奖励样本
        neutral_samples = []   # 接近零的奖励样本
        reward_threshold = 0
        reward_threshold_positive = 1.0   # 正样本阈值 - 只有明显的正奖励
        reward_threshold_negative = -0.1  # 负样本阈值 - 包含大部分负奖励
        
        for exp in experiences:
            reward = exp['reward']
            if reward > reward_threshold_positive:
                positive_samples.append(exp)
            elif reward < reward_threshold_negative:
                negative_samples.append(exp)
            else:
                neutral_samples.append(exp)
        
        # 计算采样比例
        total_positive = len(positive_samples)
        total_negative = len(negative_samples)
        total_neutral = len(neutral_samples)
        
        if total_positive == 0 and total_negative == 0:
            # 如果没有明确的正负样本，使用随机采样
            return random.sample(experiences, min(batch_size, len(experiences)))
        
        # 计算期望的采样数量 - 优先保证正负样本均衡
        if total_positive > 0 and total_negative > 0:
            # 有正负样本时，采用平衡策略
            positive_count = min(batch_size // 3, total_positive)  # 1/3 正样本
            negative_count = min(batch_size // 3, total_negative)  # 1/3 负样本
            neutral_count = min(batch_size - positive_count - negative_count, total_neutral)  # 剩余为中性样本
        elif total_positive > 0:
            # 只有正样本时
            positive_count = min(batch_size // 2, total_positive)
            negative_count = 0
            neutral_count = min(batch_size - positive_count, total_neutral)
        else:
            # 只有负样本时
            positive_count = 0
            negative_count = min(batch_size // 2, total_negative)
            neutral_count = min(batch_size - negative_count, total_neutral)
        
        # 执行采样
        sampled_batch = []
        
        if positive_count > 0:
            sampled_batch.extend(random.sample(positive_samples, positive_count))
        
        if negative_count > 0:
            sampled_batch.extend(random.sample(negative_samples, negative_count))
        
        if neutral_count > 0:
            sampled_batch.extend(random.sample(neutral_samples, neutral_count))
        
        # 如果采样数量不足，从所有样本中补充
        remaining_needed = batch_size - len(sampled_batch)
        if remaining_needed > 0:
            remaining_experiences = [exp for exp in experiences if exp not in sampled_batch]
            if remaining_experiences:
                additional_samples = random.sample(
                    remaining_experiences, 
                    min(remaining_needed, len(remaining_experiences))
                )
                sampled_batch.extend(additional_samples)
        
        # 打印采样统计信息（每100步打印一次）
        if hasattr(self, 'training_step') and self.training_step % 100 == 0:
            pos_in_batch = sum(1 for exp in sampled_batch if exp['reward'] > reward_threshold)
            neg_in_batch = sum(1 for exp in sampled_batch if exp['reward'] < reward_threshold)
            neu_in_batch = len(sampled_batch) - pos_in_batch - neg_in_batch
            
            print(f"  📊 Balanced sampling: Pos={pos_in_batch}, Neg={neg_in_batch}, Neutral={neu_in_batch}")
            print(f"     Buffer stats: Pos={total_positive}, Neg={total_negative}, Neutral={total_neutral}")
        
        return sampled_batch

    def train_step(self, batch_size: int = 64, tau: float = 0.01):  # 软更新系数，推荐0.001~0.01，可调
        """Perform one training step using stored experiences with proper DQN algorithm"""
        if len(self.experience_buffer) < batch_size * 2:  # Wait for more experiences
            return 0.0
        
        # 原始随机采样方法（已替换为高级采样）
        batch = random.sample(list(self.experience_buffer), batch_size)


        # if self.training_step < 500:
        #     # 初期使用平衡采样建立基础
        #     batch = self._advanced_sample(batch_size, method="balanced")
        # else:
        #     # 后期使用重要性采样
        #     batch = self._advanced_sample(batch_size, method="importance")

    
        # Separate current states and next states for batch processing
        current_states = []
        next_states = []
        rewards = []
        
        for exp in batch:
            # Current state - 使用支持battery和request_value的输入准备方法
            current_battery = exp.get('battery_level', 0.5)  # 向后兼容
            current_request_value = exp.get('request_value', 0.0)  # 提取请求价值
            current_inputs = self._prepare_network_input_with_battery(
                exp['vehicle_location'], exp['target_location'], exp['current_time'], 
                exp['other_vehicles'], exp['num_requests'], exp['action_type'], 
                current_battery, current_request_value
            )
            
            # 处理返回的输入（现在包含battery和request_value）
            if len(current_inputs) == 7:  # 包含battery和request_value
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor, current_value_tensor = current_inputs
            elif len(current_inputs) == 6:  # 包含battery但没有request_value
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor = current_inputs
                current_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
            else:  # 不包含battery和request_value（向后兼容）
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor = current_inputs
                current_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                current_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
            
            current_states.append({
                'path_locations': current_path_locations.squeeze(0),
                'path_delays': current_path_delays.squeeze(0),
                'current_time': current_time_tensor.squeeze(0),
                'other_agents': current_others_tensor.squeeze(0),
                'num_requests': current_requests_tensor.squeeze(0),
                'battery_level': current_battery_tensor.squeeze(0),  # 添加battery信息
                'request_value': current_value_tensor.squeeze(0),  # 添加request_value信息
                'action_type': exp['action_type'],  # 添加action_type信息
                'vehicle_id': exp['vehicle_id'],    # 添加vehicle_id信息
                'vehicle_type': exp.get('vehicle_type', 1)  # 添加vehicle_type信息（向后兼容）
            })
            
            # Next state (for target calculation) - 使用支持battery和request_value的输入准备方法
            next_battery = exp.get('next_battery_level', 1.0)  # 向后兼容
            next_request_value = exp.get('next_request_value', 0.0)  # 下一状态请求价值
            next_action_type = exp.get('next_action_type', exp['action_type'])  # 获取下一个动作类型，如果没有则使用当前动作类型作为备用
            next_inputs = self._prepare_network_input_with_battery(
                exp['next_vehicle_location'], exp['target_location'], 
                exp['current_time'] + 1, exp['other_vehicles'], exp['num_requests'], 
                next_action_type, next_battery, next_request_value
            )
            
            
            # 处理next state的返回值（现在包含battery和request_value）
            if len(next_inputs) == 7:  # 包含battery和request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor, next_value_tensor = next_inputs
            elif len(next_inputs) == 6:  # 包含battery但没有request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor = next_inputs
            else:  # 不包含battery和request_value（向后兼容）
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor = next_inputs
                next_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
            
            next_states.append({
                'path_locations': next_path_locations.squeeze(0),
                'path_delays': next_path_delays.squeeze(0),
                'current_time': next_time_tensor.squeeze(0),
                'other_agents': next_others_tensor.squeeze(0),
                'num_requests': next_requests_tensor.squeeze(0),
                'battery_level': next_battery_tensor.squeeze(0),  # 添加battery信息
                'request_value': next_value_tensor.squeeze(0),  # 添加request_value信息
                'action_type': next_action_type,  # 使用下一个动作类型而不是当前动作类型
                'vehicle_id': exp['vehicle_id'],    # 添加vehicle_id信息
                'vehicle_type': exp.get('vehicle_type', 1)  # 添加vehicle_type信息（向后兼容）
            })
            
            rewards.append(exp['reward'])
        
        # Stack batch inputs for current states
        current_batch_path_locations = torch.stack([state['path_locations'] for state in current_states])
        current_batch_path_delays = torch.stack([state['path_delays'] for state in current_states])
        current_batch_current_time = torch.stack([state['current_time'] for state in current_states])
        current_batch_other_agents = torch.stack([state['other_agents'] for state in current_states])
        current_batch_num_requests = torch.stack([state['num_requests'] for state in current_states])
        current_batch_battery_levels = torch.stack([state['battery_level'] for state in current_states])  # 添加battery批处理
        current_batch_request_values = torch.stack([state['request_value'] for state in current_states])  # 添加request_value批处理
        
        # Convert action_type strings to tensors for current states
        current_action_types = []
        current_vehicle_ids = []
        current_vehicle_types = []
        for state in current_states:
            action_type_str = state['action_type']
            if action_type_str == 'idle':
                action_type_id = 1
            elif action_type_str.startswith('assign'):
                action_type_id = 2
            elif action_type_str.startswith('charge'):
                action_type_id = 3
            else:
                action_type_id = 2  # 默认为assign
            current_action_types.append(action_type_id)
            current_vehicle_ids.append(state['vehicle_id'] + 1)  # +1因为0是padding
            current_vehicle_types.append(state['vehicle_type'])
        
        current_batch_action_types = torch.tensor(current_action_types, dtype=torch.long).to(self.device)
        current_batch_vehicle_ids = torch.tensor(current_vehicle_ids, dtype=torch.long).to(self.device)
        current_batch_vehicle_types = torch.tensor(current_vehicle_types, dtype=torch.long).to(self.device)
        
        # Stack batch inputs for next states
        next_batch_path_locations = torch.stack([state['path_locations'] for state in next_states])
        next_batch_path_delays = torch.stack([state['path_delays'] for state in next_states])
        next_batch_current_time = torch.stack([state['current_time'] for state in next_states])
        next_batch_other_agents = torch.stack([state['other_agents'] for state in next_states])
        next_batch_num_requests = torch.stack([state['num_requests'] for state in next_states])
        next_batch_battery_levels = torch.stack([state['battery_level'] for state in next_states])  # 添加next states的battery批处理
        next_batch_request_values = torch.stack([state['request_value'] for state in next_states])  # 添加next states的request_value批处理
        # Convert action_type strings to tensors for next states

        next_vehicle_ids = []
        next_vehicle_types = []
        next_action_types = []
        for state in next_states:
            action_type_str = state['action_type']
            next_vehicle_ids.append(state['vehicle_id'] )  # +1因为0是padding
            next_vehicle_types.append(state['vehicle_type'])
            if action_type_str == 'idle':
                action_type_id = 1
            elif action_type_str.startswith('assign'):
                action_type_id = 2
            elif action_type_str.startswith('charge'):
                action_type_id = 3
            else:
                action_type_id = 2  # 默认为assign
            next_action_types.append(action_type_id)


        next_batch_vehicle_ids = torch.tensor(next_vehicle_ids, dtype=torch.long).to(self.device)
        next_batch_vehicle_types = torch.tensor(next_vehicle_types, dtype=torch.long).to(self.device)
        next_batch_action_types = torch.tensor(next_action_types, dtype=torch.long).to(self.device)
        # Current Q-values (with gradients) - 现在包含所有特征信息
        self.network.train()
        current_q_values = self.network(
            path_locations=current_batch_path_locations,
            path_delays=current_batch_path_delays,
            current_time=current_batch_current_time,
            other_agents=current_batch_other_agents,
            num_requests=current_batch_num_requests,
            battery_level=current_batch_battery_levels,
            request_value=current_batch_request_values,
            action_type=current_batch_action_types.unsqueeze(1),
            vehicle_id=current_batch_vehicle_ids.unsqueeze(1),
            vehicle_type=current_batch_vehicle_types.unsqueeze(1)
        )
        
        # Next Q-values using target network (without gradients) - 现在包含所有特征信息
        with torch.no_grad():
            self.target_network.eval()
            next_q_values = self.target_network(
                path_locations=next_batch_path_locations,
                path_delays=next_batch_path_delays,
                current_time=next_batch_current_time,
                other_agents=next_batch_other_agents,
                num_requests=next_batch_num_requests,
                battery_level=next_batch_battery_levels,
                request_value=next_batch_request_values,
                action_type=next_batch_action_types.unsqueeze(1),
                vehicle_id=next_batch_vehicle_ids.unsqueeze(1),
                vehicle_type=next_batch_vehicle_types.unsqueeze(1)
            )
        
        # Calculate TD targets without normalization
        gamma = 0.95  # Slightly lower discount factor for stability
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Calculate target Q-values directly without normalization
        with torch.no_grad():
            target_q_values = rewards_tensor + gamma * next_q_values
            
            # 添加数值稳定性检查
            if torch.isnan(target_q_values).any() or torch.isinf(target_q_values).any():
                print(f"WARNING: Invalid target Q-values detected!")
                print(f"  Rewards range: [{rewards_tensor.min():.3f}, {rewards_tensor.max():.3f}]")
                print(f"  Next Q-values range: [{next_q_values.min():.3f}, {next_q_values.max():.3f}]")
                return 0.0
            
        # 添加数值稳定性检查
        if torch.isnan(current_q_values).any() or torch.isinf(current_q_values).any():
            print(f"WARNING: Invalid current Q-values detected!")
            return 0.0
            
        # Compute loss with raw values
        loss = self.loss_fn(current_q_values, target_q_values)
        loss_value = loss.item()  # Define loss_value immediately after loss computation
        
        # 检查损失是否异常
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: Invalid loss detected: {loss_value}")
            return 0.0
        
        # Additional safety check
        if not loss.requires_grad:
            print("WARNING: Loss does not require gradients!")
            return 0.0
        
        # Backpropagation with gradient monitoring
        self.optimizer.zero_grad()
        loss.backward()
        
        # Monitor gradients for debugging
        total_grad_norm = 0.0
        for name, param in self.network.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_grad_norm += param_norm.item() ** 2
        total_grad_norm = total_grad_norm ** (1. / 2)
        
        # 修复梯度裁剪：从0.5增加到10.0，避免过度裁剪
        # 原来的0.5太小，导致梯度被严重裁剪，学习能力受限
        # torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update learning rate scheduler based on loss
        self.scheduler.step(loss_value)
        
        # Update target network periodically (key DQN component)
        
        if self.training_step % self.target_update_frequency == 0:
            for target_param, param in zip(self.target_network.parameters(), self.network.parameters()):
                target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
            print(f"🔄 Target network soft-updated at step {self.training_step} with tau={tau}")
        
        # Record training metrics
        self.training_losses.append(loss_value)
        
        # Record Q-values statistics
        with torch.no_grad():
            q_mean = current_q_values.mean().item()
            q_std = current_q_values.std().item()
            q_max = current_q_values.max().item()
            q_min = current_q_values.min().item()
            self.q_values_history.append({
                'mean': q_mean, 'std': q_std, 'max': q_max, 'min': q_min
            })
        
        self.training_step += 1
        
        # Print training progress occasionally  
        if self.training_step % 100 == 0:
            current_lr = self.optimizer.param_groups[0]['lr']
            
            print(f"Training step {self.training_step}: Loss={loss_value:.4f}, Q_mean={q_mean:.4f}, Q_std={q_std:.4f}, Q_range=[{q_min:.4f}, {q_max:.4f}], LR={current_lr:.6f}")
            print(f"  Gradient norm: {total_grad_norm:.4f}, No normalization - using raw Q-values")
        
        return loss_value

    def train_step_supervised(self, env, num_samples: int = 64):
        """
        Supervised training: minimize MSE between optimizer-induced option values (labels)
        and network predictions for the same actions, aligning with src_2's "planner-in-the-loop" idea.

        This does NOT use optimizer to assign vehicles in the environment step; it only calls the
        optimizer here to get a one-shot assignment to define labels. If optimizer is unavailable,
        it falls back to a heuristic assignment.
        """
        import torch
        import random
        from collections import deque

        # Sanity: environment must provide vehicles, requests, and evaluators
        if env is None or not hasattr(env, 'vehicles'):
            return 0.0

        # 1) Build a pool of candidate idle vehicles similar to simulate_motion
        vehicles_to_rebalance = []
        for vehicle_id, vehicle in env.vehicles.items():
            if ((vehicle.get('assigned_request') is None and
                 vehicle.get('passenger_onboard') is None and
                 vehicle.get('charging_station') is None and
                 vehicle.get('idle_target') is None) or
                (vehicle.get('battery', 1.0) <= getattr(env, 'rebalance_battery_threshold', 0.5)) or
                vehicle.get('is_stationary', False)):
                vehicles_to_rebalance.append(vehicle_id)

        if not vehicles_to_rebalance:
            return 0.0

        # 2) Get an assignment mapping using optimizer (preferred) or heuristic fallback
        assignments = {}
        try:
            if not hasattr(env, 'gurobi_optimizer'):
                from src.GurobiOptimizer import GurobiOptimizer
                env.gurobi_optimizer = GurobiOptimizer(env)
            assignments = env.gurobi_optimizer.optimize_vehicle_rebalancing_reject(vehicles_to_rebalance)
        except Exception:
            # Heuristic fallback without modifying optimizer implementation
            try:
                available_requests = list(getattr(env, 'active_requests', {}).values())
                charging_stations = [st for st in getattr(env, 'charging_manager').stations.values() if st.available_slots > 0] if hasattr(env, 'charging_manager') else []
                assignments = env.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance, available_requests, charging_stations)  # type: ignore
            except Exception:
                return 0.0

        if not assignments:
            return 0.0

        # 3) Build supervised mini-batch from assignments (action -> label)
        #    Label = env.evaluate_service_option / evaluate_charging_option (option completion value)
        #    Match the network inputs for those actions
        inputs_list = []
        labels_list = []

        # Helper: pack a single sample
        def _append_sample(vehicle_id: int, action_type: str, veh_loc: int, tgt_loc: int,
                           current_time: float, battery: float, request_value: float, label: float):
            # Raw counts for normalization inside helper
            other_vehicles_raw = max(0, sum(1 for v in env.vehicles.values() if v.get('assigned_request') is None and v.get('passenger_onboard') is None and v.get('charging_station') is None) - 1)
            num_requests_raw = len(getattr(env, 'active_requests', {}))

            # Prepare full set of inputs with battery/request value; tensors are already normalized as in other code paths
            path_locations_b, path_delays_b, time_b, others_b, requests_b, battery_b, value_b = self._prepare_network_input_with_battery(
                veh_loc, tgt_loc, current_time, other_vehicles_raw, num_requests_raw, action_type, battery, request_value
            )
            # Package tensors in expected dict form
            sample = {
                'path_locations': path_locations_b,
                'path_delays': path_delays_b,
                'current_time': time_b,
                'other_agents': others_b,
                'num_requests': requests_b,
                'battery_level': battery_b,
                'request_value': value_b,
                'action_type': torch.tensor([[1 if action_type=='idle' else (2 if action_type.startswith('assign') else 3)]], dtype=torch.long, device=self.device),
                'vehicle_id': torch.tensor([[vehicle_id + 1]], dtype=torch.long, device=self.device),
                'vehicle_type': torch.tensor([[1 if vehicle_id % 2 == 0 else 2]], dtype=torch.long, device=self.device)
            }
            inputs_list.append(sample)
            labels_list.append(label)

        # Fill samples from assignments
        for vehicle_id, target in assignments.items():
            vehicle = env.vehicles.get(vehicle_id)
            if not vehicle:
                continue
            veh_loc = vehicle.get('location', 0)
            battery = vehicle.get('battery', 1.0)

            # Service assignment
            if target and hasattr(target, 'pickup') and hasattr(target, 'dropoff'):
                try:
                    label_val = env.evaluate_service_option(vehicle_id, target)
                except Exception:
                    label_val = 0.0
                # Use pickup as target_location for the assign action
                tgt_loc = getattr(target, 'pickup', veh_loc)
                req_val = getattr(target, 'final_value', getattr(target, 'value', 0.0))
                _append_sample(vehicle_id, f"assign_{getattr(target, 'request_id', 0)}", veh_loc, tgt_loc, env.current_time, battery, req_val, label_val)
            # Charging assignment
            elif target and hasattr(target, 'id') and hasattr(target, 'location'):
                try:
                    label_val = env.evaluate_charging_option(vehicle_id, target)
                except Exception:
                    label_val = 0.0
                tgt_loc = getattr(target, 'location', veh_loc)
                _append_sample(vehicle_id, f"charge_{getattr(target, 'id', 0)}", veh_loc, tgt_loc, env.current_time, battery, 0.0, label_val)
            else:
                # Idle/no-op: optionally skip, to focus on meaningful supervised labels
                continue

        if not inputs_list:
            return 0.0

        # Downsample to num_samples if necessary
        if len(inputs_list) > num_samples:
            idxs = random.sample(range(len(inputs_list)), num_samples)
            inputs_list = [inputs_list[i] for i in idxs]
            labels_list = [labels_list[i] for i in idxs]

        # 4) Batch the tensors and train with MSE
        def _stack(key):
            return torch.cat([s[key] for s in inputs_list], dim=0)

        self.network.train()
        preds = self.network(
            path_locations=_stack('path_locations'),
            path_delays=_stack('path_delays'),
            current_time=_stack('current_time'),
            other_agents=_stack('other_agents'),
            num_requests=_stack('num_requests'),
            battery_level=_stack('battery_level'),
            request_value=_stack('request_value'),
            action_type=_stack('action_type'),
            vehicle_id=_stack('vehicle_id'),
            vehicle_type=_stack('vehicle_type')
        )

        labels_tensor = torch.tensor(labels_list, dtype=torch.float32, device=self.device).unsqueeze(1)
        loss = self.loss_fn(preds, labels_tensor)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()

        loss_value = float(loss.item())
        self.training_losses.append(loss_value)
        self.training_step += 1

        # Track Q stats
        with torch.no_grad():
            self.q_values_history.append({
                'mean': preds.mean().item(),
                'std': preds.std().item(),
                'max': preds.max().item(),
                'min': preds.min().item()
            })

        return loss_value
    
    def get_value(self, experiences: List[Experience]) -> List[List[Tuple[Action, float]]]:
        """Compatibility method for Experience-based interface"""
        # This is a simplified implementation for compatibility
        return []
    
    def update(self, *args, **kwargs):
        """Update the neural network"""
        if len(self.experience_buffer) > 100:
            loss = self.train_step()
            self.add_to_logs('training_loss', loss, self.training_step)
            self.training_step += 1
    
    def remember(self, experience: Experience):
        """简化的经验存储，依赖Environment进行筛选"""
        try:
            # 从Experience中提取相关信息
            for agent_id, actions_info in experience.action_to_take_all_agents.items():
                if not actions_info:
                    continue
                
                action, reward = actions_info[0] if len(actions_info) > 0 else (None, 0.0)
                if action is None:
                    continue
                
                # 获取当前状态信息
                current_state = experience.current_states.get(agent_id)
                next_state = experience.next_states.get(agent_id) if hasattr(experience, 'next_states') else None
                
                if current_state is None:
                    continue
                
                # 创建简化的经验记录
                enhanced_experience = {
                    'vehicle_id': agent_id,
                    'vehicle_location': getattr(current_state, 'location', 0),
                    'target_location': 0,  # 将从action中提取
                    'current_time': experience.current_time,
                    'reward': reward,
                    'next_vehicle_location': getattr(next_state, 'location', 0) if next_state else 0,
                    'other_vehicles': len(experience.current_states) - 1,
                    'num_requests': len(getattr(experience, 'active_requests', [])),
                    'battery_level': getattr(current_state, 'battery', 1.0),
                    'next_battery_level': getattr(next_state, 'battery', 1.0) if next_state else 1.0,
                    'request_value': 0.0,
                    'action_type': 'idle',  # 默认值
                }
                
                # 根据action类型更新相关信息
                if hasattr(action, 'requests') and action.requests:
                    # Service action
                    request = list(action.requests)[0]
                    enhanced_experience['target_location'] = getattr(request, 'pickup', 0)
                    enhanced_experience['request_value'] = getattr(request, 'final_value', 0.0)
                    enhanced_experience['action_type'] = 'assign'
                elif hasattr(action, 'charging_station_id'):
                    # Charging action
                    enhanced_experience['action_type'] = 'charge'
                    enhanced_experience['target_location'] = getattr(action, 'charging_station_id', 0)
                else:
                    # Idle action
                    enhanced_experience['action_type'] = 'idle'
                    if hasattr(action, 'target_coords'):
                        target_x, target_y = action.target_coords
                        grid_size = int(math.sqrt(enhanced_experience['vehicle_location']) + 1)
                        enhanced_experience['target_location'] = target_y * grid_size + target_x
                
                # 直接存储，依赖Environment进行预筛选
                self.experience_buffer.append(enhanced_experience)
                    
                # 定期进行训练
                if len(self.experience_buffer) % 50 == 0:
                    self.train_step(batch_size=32)
                        
        except Exception as e:
            print(f"Warning: Error in simplified remember method: {e}")
            pass
    
    def _evaluate_assignment_quality(self, action, reward: float) -> float:
        """
        重新定义分配质量评估：专注于订单完成能力
        成功的assignment = 能够完成整个服务流程的分配
        """
        # 1. 最高优先级：实际完成了订单（获得了final_value奖励）
        if reward >= 15:  # 完成订单的典型奖励范围
            return 1.0  # 完美质量 - 这是我们最想学习的经验
        
        # 2. 高优先级：部分完成但有正向进展
        elif reward >= 5:  # 可能完成了pickup但还未dropoff
            return 0.8  # 高质量 - 展示了完成能力
        
        # 3. 中等优先级：成功分配但还在执行中
        elif reward > 0:  # 成功分配，正在执行
            return 0.6  # 中等质量 - 有潜力完成
        
        # 4. 低优先级：分配被拒绝或失败
        elif reward == 0:  # 分配失败或被拒绝
            return 0.2  # 低质量 - 可以学习为什么失败
        
        # 5. 负面案例：电池耗尽、无法完成等
        else:  # 负奖励 - 电池耗尽、乘客滞留等
            return 0.0  # 零质量 - 避免学习这类经验
    
    def _analyze_competitive_context(self, experience: Experience) -> float:
        """分析竞争环境上下文"""
        num_vehicles = len(experience.current_states) if hasattr(experience, 'current_states') else 1
        num_requests = len(getattr(experience, 'active_requests', []))
        
        if num_requests == 0:
            return 0.0  # 无请求环境
        
        competition_ratio = num_vehicles / num_requests
        if competition_ratio > 2.0:
            return 1.0  # 高竞争
        elif competition_ratio > 1.0:
            return 0.6  # 中等竞争
        else:
            return 0.2  # 低竞争

    def _assess_order_completion_potential(self, action, current_state, reward: float) -> float:
        """
        评估订单完成潜力：预测这个分配决策能否成功完成订单
        """
        # 基础完成潜力评估
        completion_potential = 0.0
        
        # 1. 电池充足度对完成潜力的影响
        battery_level = getattr(current_state, 'battery', 1.0)
        if battery_level > 0.5:
            completion_potential += 0.4  # 高电量 = 高完成潜力
        elif battery_level > 0.3:
            completion_potential += 0.2  # 中等电量 = 中等完成潜力
        else:
            completion_potential += 0.0  # 低电量 = 低完成潜力
        
        # 2. 如果是assignment action，考虑距离因素
        if hasattr(action, 'requests') and action.requests:
            request = list(action.requests)[0]
            pickup_location = getattr(request, 'pickup', 0)
            current_location = getattr(current_state, 'location', 0)
            
            # 简化的距离计算（假设grid_size=40）
            grid_size = 40
            pickup_x, pickup_y = pickup_location % grid_size, pickup_location // grid_size
            current_x, current_y = current_location % grid_size, current_location // grid_size
            distance = abs(pickup_x - current_x) + abs(pickup_y - current_y)
            
            # 距离越近，完成潜力越高
            if distance <= 3:
                completion_potential += 0.3  # 很近
            elif distance <= 6:
                completion_potential += 0.2  # 较近
            elif distance <= 10:
                completion_potential += 0.1  # 中等距离
            # 远距离不加分
        
        # 3. 实际奖励反馈的完成潜力
        if reward >= 15:  # 已完成订单
            completion_potential = 1.0  # 确定完成
        elif reward >= 5:  # 部分完成
            completion_potential = max(completion_potential, 0.8)
        elif reward > 0:  # 正在执行
            completion_potential = max(completion_potential, 0.6)
        
        return min(1.0, completion_potential)
    
    def _is_order_completion_valuable_experience(self, experience: dict) -> bool:
        """
        严格控制experience存储：只存储关键决策点
        - 完成订单的experience（最终收益）
        - 充电决策的experience  
        - idle移动决策的experience
        - 排除pickup/dropoff执行过程中的experience
        """
        reward = experience['reward']
        action_type = experience['action_type']
        assignment_quality = experience['assignment_quality']
        
        # 1. 【最高优先级】完成订单的experience - 这是最终的成功决策结果
        if reward >= 15 and assignment_quality >= 0.8:
            print(f"✓ Storing COMPLETED ORDER experience: reward={reward}, vehicle={experience['vehicle_id']}")
            return True
        
        # 2. 【充电决策】- 电池管理的关键决策点
        if action_type.startswith('charge'):
            battery_level = experience.get('battery_level', 1.0)
            # 只存储真正需要充电的决策（低电量）
            if battery_level < 0.5:
                print(f"✓ Storing CHARGING decision experience: battery={battery_level}, vehicle={experience['vehicle_id']}")
                return True
            return False
        
        # 3. 【Idle决策】- 空闲状态的移动决策
        if action_type == 'idle':
            # 存储所有idle决策，因为这些是重要的定位决策
            return True
        
        # 4. 【初始assignment决策】- 只存储刚开始分配的决策，不存储执行过程
        if action_type.startswith('assign'):
            # 只存储真正的分配决策时刻（高质量或负面教训）
            if assignment_quality >= 0.6:  # 成功的分配决策
                print(f"✓ Storing SUCCESSFUL assignment decision: quality={assignment_quality}, reward={reward}")
                return True
            elif assignment_quality == 0.0 and reward <= 0:  # 失败的分配决策（学习教训）
                print(f"✓ Storing FAILED assignment decision for learning: quality={assignment_quality}, reward={reward}")
                return True
            else:
                # 排除执行过程中的中间状态（pickup进行中、dropoff进行中等）
                return False
        
        # 5. 其他情况：不存储
        return False
    
    def plot_training_metrics(self, save_path: str = None):
        """Plot training losses and Q-values over time"""
        import matplotlib.pyplot as plt
        
        if not self.training_losses:
            print("No training data to plot")
            return
        
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot training loss
        ax1.plot(self.training_losses, label='Training Loss', color='red', alpha=0.7)
        ax1.set_title('Neural Network Training Loss')
        ax1.set_xlabel('Training Steps')
        ax1.set_ylabel('MSE Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot Q-values mean
        if self.q_values_history:
            q_means = [q['mean'] for q in self.q_values_history]
            q_stds = [q['std'] for q in self.q_values_history]
            
            ax2.plot(q_means, label='Q-value Mean', color='blue', alpha=0.7)
            ax2.fill_between(range(len(q_means)), 
                           [m - s for m, s in zip(q_means, q_stds)],
                           [m + s for m, s in zip(q_means, q_stds)],
                           alpha=0.2, color='blue', label='±1 Std')
            ax2.set_title('Q-Values Statistics')
            ax2.set_xlabel('Training Steps')
            ax2.set_ylabel('Q-Value')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            # Plot Q-values standard deviation
            ax3.plot(q_stds, label='Q-value Std Dev', color='green', alpha=0.7)
            ax3.set_title('Q-Values Standard Deviation')
            ax3.set_xlabel('Training Steps')
            ax3.set_ylabel('Standard Deviation')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Training metrics plot saved to: {save_path}")
        else:
            plt.savefig('results/training_metrics.png', dpi=300, bbox_inches='tight')
            print("Training metrics plot saved to: results/training_metrics.png")
        
        plt.show()
        return fig


class PyTorchPathBasedNetwork(nn.Module):
    """PyTorch implementation of path-based neural network with action type and vehicle embedding"""
    
    def __init__(self, 
                 num_locations: int,
                 num_vehicles: int,
                 max_capacity: int,
                 embedding_dim: int = 100,
                 lstm_hidden: int = 200,
                 dense_hidden: int = 300,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(PyTorchPathBasedNetwork, self).__init__()
        
        self.num_locations = num_locations
        self.num_vehicles = num_vehicles
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        
        # Location embedding layer
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim,
            padding_idx=0
        )
        
        # Vehicle ID embedding layer
        self.vehicle_embedding = nn.Embedding(
            num_embeddings=num_vehicles + 1,  # +1 for padding/unknown vehicles
            embedding_dim=embedding_dim // 4,  # 较小的维度，专注于车辆特征
            padding_idx=0
        )
        
        # Vehicle type embedding layer (EV vs AEV)
        # 0: unknown, 1: EV, 2: AEV
        self.vehicle_type_embedding = nn.Embedding(
            num_embeddings=3,
            embedding_dim=embedding_dim // 4,
            padding_idx=0
        )
        
        # Action type embedding layer
        # 0: padding, 1: idle, 2: assign, 3: charge
        self.action_type_embedding = nn.Embedding(
            num_embeddings=4,
            embedding_dim=embedding_dim // 2,
            padding_idx=0
        )
        
        # Initialize with pretrained embeddings if available
        if pretrained_embeddings is not None:
            self.location_embedding.weight.data.copy_(pretrained_embeddings)
            self.location_embedding.weight.requires_grad = False
        
        # LSTM for path processing
        self.path_lstm = nn.LSTM(
            input_size=embedding_dim + 1,  # embedding + delay
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(1, embedding_dim),
            nn.ELU()
        )
        
        # Context embedding for action-specific features
        self.context_embedding = nn.Sequential(
            nn.Linear(2, embedding_dim // 2),  # battery + request_value
            nn.ELU(),
            nn.Dropout(0.1)
        )
        
        # Vehicle-specific feature processing
        self.vehicle_feature_embedding = nn.Sequential(
            nn.Linear(embedding_dim // 4 + embedding_dim // 4, embedding_dim // 2),  # vehicle_id + vehicle_type
            nn.ELU(),
            nn.Dropout(0.1)
        )
        
        # State embedding layers - 包含所有特征
        state_input_dim = (lstm_hidden + embedding_dim + 2 +      # path + time + other_agents + num_requests
                          embedding_dim // 2 +                    # action_type_embedding
                          embedding_dim // 2 +                    # context_embedding (battery + request_value)
                          embedding_dim // 2)                     # vehicle_feature_embedding (vehicle_id + type)
        
        self.state_embedding = nn.Sequential(
            nn.Linear(state_input_dim, dense_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden, dense_hidden),
            nn.ELU(),
            nn.Dropout(0.1),
            nn.Linear(dense_hidden, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LSTM):
            for name, param in module.named_parameters():
                if 'weight' in name:
                    nn.init.xavier_uniform_(param)
                elif 'bias' in name:
                    nn.init.zeros_(param)
    
    def forward(self, 
                path_locations: torch.Tensor,
                path_delays: torch.Tensor,
                current_time: torch.Tensor,
                other_agents: torch.Tensor,
                num_requests: torch.Tensor,
                battery_level: torch.Tensor = None,
                request_value: torch.Tensor = None,
                action_type: torch.Tensor = None,
                vehicle_id: torch.Tensor = None,
                vehicle_type: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            path_locations: [batch_size, seq_len] - Location IDs in path
            path_delays: [batch_size, seq_len, 1] - Delay information
            current_time: [batch_size, 1] - Current time
            other_agents: [batch_size, 1] - Number of other agents nearby
            num_requests: [batch_size, 1] - Number of current requests
            battery_level: [batch_size, 1] - Battery level (0-1), optional
            request_value: [batch_size, 1] - Request value (0-1), optional
            action_type: [batch_size, 1] - Action type (1=idle, 2=assign, 3=charge), optional
            vehicle_id: [batch_size, 1] - Vehicle ID (1-num_vehicles), optional
            vehicle_type: [batch_size, 1] - Vehicle type (1=EV, 2=AEV), optional
        """
        batch_size = path_locations.size(0)
        
        # Get location embeddings
        location_embeds = self.location_embedding(path_locations)  # [batch_size, seq_len, embedding_dim]
        
        # Create mask for padding
        mask = (path_locations != 0).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
        
        # Apply mask to delays  
        masked_delays = path_delays * mask
        
        # Combine location embeddings with delays
        path_input = torch.cat([location_embeds, masked_delays], dim=-1)  # [batch_size, seq_len, embedding_dim + 1]
        
        # Process through LSTM
        lstm_out, (hidden, _) = self.path_lstm(path_input)  # [batch_size, seq_len, lstm_hidden]
        
        # Use final hidden state as path representation
        path_representation = hidden[-1]  # [batch_size, lstm_hidden]
        
        # Process time
        time_embed = self.time_embedding(current_time)  # [batch_size, embedding_dim]
        
        # Handle battery level - 如果没有提供battery_level，使用默认值1.0
        if battery_level is None:
            battery_level = torch.ones(current_time.size()).to(current_time.device)
        
        # Handle request value - 如果没有提供request_value，使用默认值0.0
        if request_value is None:
            request_value = torch.zeros(current_time.size()).to(current_time.device)
        
        # Handle vehicle_id - 如果没有提供，使用默认值1
        if vehicle_id is None:
            vehicle_id = torch.ones(current_time.size(), dtype=torch.long).to(current_time.device)
        
        # Handle vehicle_type - 如果没有提供，默认为EV (1)
        if vehicle_type is None:
            vehicle_type = torch.ones(current_time.size(), dtype=torch.long).to(current_time.device)
        
        # Handle action type - 如果没有提供action_type，尝试从路径推断
        if action_type is None:
            # 从路径模式推断action type
            # idle: 路径中第一个位置 == 第二个位置
            # assign/charge: 路径中第一个位置 != 第二个位置
            is_idle = (path_locations[:, 0] == path_locations[:, 1]).long()
            action_type = torch.where(is_idle, 
                                    torch.ones_like(is_idle),  # idle = 1
                                    torch.full_like(is_idle, 2))  # assign/charge = 2 (默认assign)
            action_type = action_type.unsqueeze(1)  # [batch_size, 1]
        
        # Get embeddings
        action_embed = self.action_type_embedding(action_type.squeeze(1))  # [batch_size, embedding_dim//2]
        vehicle_id_embed = self.vehicle_embedding(vehicle_id.squeeze(1))   # [batch_size, embedding_dim//4]
        vehicle_type_embed = self.vehicle_type_embedding(vehicle_type.squeeze(1))  # [batch_size, embedding_dim//4]
        
        # Process context features (battery + request_value)
        context_features = torch.cat([battery_level, request_value], dim=1)  # [batch_size, 2]
        context_embed = self.context_embedding(context_features)  # [batch_size, embedding_dim//2]
        
        # Process vehicle features (vehicle_id + vehicle_type)
        vehicle_features = torch.cat([vehicle_id_embed, vehicle_type_embed], dim=1)  # [batch_size, embedding_dim//2]
        vehicle_embed = self.vehicle_feature_embedding(vehicle_features)  # [batch_size, embedding_dim//2]
        
        # Combine all features
        combined_features = torch.cat([
            path_representation,     # [batch_size, lstm_hidden]
            time_embed,             # [batch_size, embedding_dim]
            other_agents,           # [batch_size, 1]
            num_requests,           # [batch_size, 1]
            action_embed,           # [batch_size, embedding_dim//2]
            context_embed,          # [batch_size, embedding_dim//2]
            vehicle_embed           # [batch_size, embedding_dim//2]
        ], dim=1)  # [batch_size, total_features]
        
        # Get final value prediction
        value = self.state_embedding(combined_features)  # [batch_size, 1]
        
        return value


class PyTorchNeuralNetworkBased(PyTorchValueFunction):
    """PyTorch implementation of neural network-based value function"""
    
    def __init__(self,
                 envt: Environment,
                 num_locations: int,
                 max_capacity: int,
                 load_model_path: str = '',
                 log_dir: str = 'logs/nn_based',
                 gamma: float = 0.99,
                 learning_rate: float = 1e-3,
                 batch_size_fit: int = 32,
                 batch_size_predict: int = 1024,
                 target_update_tau: float = 0.001,
                 device: str = 'cpu'):
        
        super().__init__(log_dir, device)
        
        # Environment and hyperparameters
        self.envt = envt
        self.num_locations = num_locations
        self.max_capacity = max_capacity
        self.gamma = gamma
        self.batch_size_fit = batch_size_fit
        self.batch_size_predict = batch_size_predict
        self.target_update_tau = target_update_tau
        
        # Initialize networks
        self.value_network = self._init_network().to(self.device)
        self.target_network = self._init_network().to(self.device)
        
        # Load pretrained model if specified
        if load_model_path and PathlibPath(load_model_path).exists():
            self.load_model(load_model_path)
        
        # Copy weights to target network
        self.target_network.load_state_dict(self.value_network.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.value_network.parameters(), lr=learning_rate)
        
        # Replay buffer
        buffer_size = max(int(1e6 / getattr(envt, 'NUM_AGENTS', 10)), 10000)
        self.replay_buffer = PyTorchReplayBuffer(buffer_size, str(self.device))
        
    def _init_network(self) -> PyTorchPathBasedNetwork:
        """Initialize the neural network"""
        # Try to load pretrained embeddings
        pretrained_embeddings = None
        if hasattr(self.envt, 'DATA_DIR'):
            embedding_path = PathlibPath(self.envt.DATA_DIR) / 'embedding_weights.pkl'
            if embedding_path.exists():
                try:
                    with open(embedding_path, 'rb') as f:
                        weights = pickle.load(f)
                    pretrained_embeddings = torch.FloatTensor(weights[0])
                except Exception as e:
                    logging.warning(f"Failed to load pretrained embeddings: {e}")
        
        return PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            max_capacity=self.max_capacity,
            pretrained_embeddings=pretrained_embeddings
        )
    
    def _format_input_batch(self, experiences: List[Experience]) -> Dict[str, torch.Tensor]:
        """Format experiences into network input tensors"""
        batch_size = len(experiences)
        max_seq_len = self.max_capacity * 2 + 1
        
        # Initialize tensors
        path_locations = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        path_delays = torch.zeros(batch_size, max_seq_len, 1, dtype=torch.float32)
        current_times = torch.zeros(batch_size, 1, dtype=torch.float32)
        other_agents = torch.zeros(batch_size, 1, dtype=torch.float32)
        num_requests = torch.zeros(batch_size, 1, dtype=torch.float32)
        
        for i, experience in enumerate(experiences):
            # Extract time information
            if hasattr(experience, 'time'):
                normalized_time = self._normalize_time(experience.time)
                current_times[i, 0] = normalized_time
            
            # Extract request count
            if hasattr(experience, 'num_requests'):
                normalized_requests = experience.num_requests / getattr(self.envt, 'NUM_AGENTS', 10)
                num_requests[i, 0] = normalized_requests
            
            # Process agents (simplified - take first agent for demo)
            if hasattr(experience, 'agents') and len(experience.agents) > 0:
                agent = experience.agents[0]
                
                # Extract path information
                if hasattr(agent, 'path') and hasattr(agent.path, 'request_order'):
                    self._extract_path_features(agent, path_locations[i], path_delays[i])
                
                # Count other agents (simplified)
                other_agents[i, 0] = len(experience.agents) / getattr(self.envt, 'NUM_AGENTS', 10)
        
        return {
            'path_locations': path_locations.to(self.device),
            'path_delays': path_delays.to(self.device),
            'current_time': current_times.to(self.device),
            'other_agents': other_agents.to(self.device),
            'num_requests': num_requests.to(self.device)
        }
    
    def _normalize_time(self, time: float) -> float:
        """Normalize time to [0, 1] range"""
        start_time = getattr(self.envt, 'START_EPOCH', 0)
        end_time = getattr(self.envt, 'STOP_EPOCH', 86400)
        return (time - start_time) / (end_time - start_time) if end_time > start_time else 0.0
    
    def _extract_path_features(self, agent: LearningAgent, 
                              path_locations: torch.Tensor, 
                              path_delays: torch.Tensor):
        """Extract path features from agent"""
        # Add current location
        if hasattr(agent, 'position') and hasattr(agent.position, 'next_location'):
            path_locations[0] = agent.position.next_location + 1
            path_delays[0, 0] = 1.0
        
        # Add path nodes (simplified extraction)
        if hasattr(agent, 'path') and hasattr(agent.path, 'request_order'):
            for idx, node in enumerate(agent.path.request_order):
                if idx >= len(path_locations) - 1:
                    break
                
                # Extract location and delay information
                try:
                    location, deadline = agent.path.get_info(node)
                    visit_time = getattr(node, 'expected_visit_time', 0)
                    
                    path_locations[idx + 1] = location + 1
                    # Normalize delay
                    max_delay = getattr(Request, 'MAX_DROPOFF_DELAY', 3600)
                    normalized_delay = (deadline - visit_time) / max_delay
                    path_delays[idx + 1, 0] = normalized_delay
                except Exception:
                    # Handle cases where path info extraction fails
                    break
    
    def get_value(self, experiences: List[Experience], 
                 use_target: bool = False) -> List[List[Tuple[Action, float]]]:
        """Get value estimates for experiences"""
        if not experiences:
            return []
        
        # Format input batch
        inputs = self._format_input_batch(experiences)
        
        # Get network predictions
        network = self.target_network if use_target else self.value_network
        network.eval()
        
        with torch.no_grad():
            values = network(**inputs)  # [batch_size, 1]
            values = values.cpu().numpy().flatten()
        
        # Format output to match expected interface
        scored_actions_all_agents = []
        value_idx = 0
        
        for experience in experiences:
            if hasattr(experience, 'feasible_actions_all_agents'):
                for feasible_actions in experience.feasible_actions_all_agents:
                    scored_actions = []
                    for action in feasible_actions:
                        # Get immediate reward
                        immediate_reward = self._get_immediate_reward(action)
                        
                        # Add discounted future value
                        future_value = values[value_idx] if value_idx < len(values) else 0.0
                        total_value = immediate_reward + self.gamma * future_value
                        
                        scored_actions.append((action, total_value))
                    
                    scored_actions_all_agents.append(scored_actions)
                    value_idx += 1
        
        return scored_actions_all_agents
    
    def _get_immediate_reward(self, action: Action) -> float:
        """Get immediate reward for an action"""
        if hasattr(self.envt, 'get_reward'):
            return self.envt.get_reward(action)
        elif hasattr(action, 'requests'):
            return sum([getattr(req, 'value', 0) for req in action.requests])
        else:
            return 0.0
    
    def remember(self, experience: Experience):
        """Store experience in replay buffer"""
        self.replay_buffer.add(experience)
    
    def update(self, central_agent: CentralAgent, num_samples: int = 3):
        """Update value function using sampled experiences"""
        # Check if enough experiences for training
        min_samples = max(self.batch_size_fit, 100)
        if len(self.replay_buffer) < min_samples:
            return
        
        # Sample experiences
        experiences, weights, indices = self.replay_buffer.sample(num_samples)
        if not experiences:
            return
        
        # Prepare training data
        self.value_network.train()
        
        # Get current value predictions
        inputs = self._format_input_batch(experiences)
        current_values = self.value_network(**inputs)
        
        # Get target values using target network
        target_values = self._compute_target_values(experiences, central_agent)
        
        # Compute loss with importance sampling
        weights_tensor = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        td_errors = F.mse_loss(current_values, target_values, reduction='none')
        weighted_loss = (td_errors * weights_tensor).mean()
        
        # Backward pass
        self.optimizer.zero_grad()
        weighted_loss.backward()
        # 修复梯度裁剪：从1.0增加到10.0，避免过度裁剪
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update target network
        self._soft_update_target()
        
        # Update replay buffer priorities
        priorities = td_errors.detach().cpu().numpy().flatten() + 1e-6
        self.replay_buffer.update_priorities(indices, priorities.tolist())
        
        # Log training statistics
        self.add_to_logs('loss', weighted_loss.item(), self.training_step)
        self.add_to_logs('mean_value', current_values.mean().item(), self.training_step)
        self.training_step += 1
    
    def _compute_target_values(self, experiences: List[Experience], 
                              central_agent: CentralAgent) -> torch.Tensor:
        """Compute target values for training"""
        target_values = []
        
        for experience in experiences:
            # Get next state values using target network
            next_state_values = self.get_value([experience], use_target=True)
            
            # Use central agent to get optimal actions and their values
            if next_state_values and hasattr(central_agent, 'choose_actions'):
                try:
                    optimal_actions = central_agent.choose_actions(
                        next_state_values, 
                        is_training=False
                    )
                    target_value = sum([score for _, score in optimal_actions]) / len(optimal_actions)
                except Exception:
                    target_value = 0.0
            else:
                target_value = 0.0
            
            target_values.append(target_value)
        
        return torch.FloatTensor(target_values).unsqueeze(1).to(self.device)
    
    def _soft_update_target(self):
        """Soft update of target network"""
        for target_param, param in zip(self.target_network.parameters(), 
                                     self.value_network.parameters()):
            target_param.data.copy_(
                self.target_update_tau * param.data + 
                (1 - self.target_update_tau) * target_param.data
            )
    
    def save_model(self, filepath: str):
        """Save model state"""
        torch.save({
            'value_network': self.value_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'training_step': self.training_step
        }, filepath)
    
    def load_model(self, filepath: str):
        """Load model state"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.value_network.load_state_dict(checkpoint['value_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.training_step = checkpoint.get('training_step', 0)


# Compatibility aliases for existing code
ValueFunction = PyTorchValueFunction
RewardPlusDelay = PyTorchRewardPlusDelay
ImmediateReward = lambda: PyTorchRewardPlusDelay(delay_coefficient=0)
NeuralNetworkBased = PyTorchNeuralNetworkBased
PathBasedNN = PyTorchNeuralNetworkBased


def main():
    """Test the PyTorch value function implementation"""
    logging.basicConfig(level=logging.INFO)
    
    # Create dummy environment for testing
    class DummyEnvironment:
        NUM_AGENTS = 5
        NUM_LOCATIONS = 100
        MAX_CAPACITY = 4
        START_EPOCH = 0
        STOP_EPOCH = 86400
        DATA_DIR = 'data/'
        
        def get_reward(self, action):
            return random.uniform(0, 10)
    
    env = DummyEnvironment()
    
    # Test PyTorch value function
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    value_function = PyTorchNeuralNetworkBased(
        envt=env,
        num_locations=env.NUM_LOCATIONS,
        max_capacity=env.MAX_CAPACITY,
        device=device
    )
    
    print("PyTorch Value Function initialized successfully!")
    print(f"Network parameters: {sum(p.numel() for p in value_function.value_network.parameters())}")
    print(f"Device: {device}")


if __name__ == "__main__":
    main()
