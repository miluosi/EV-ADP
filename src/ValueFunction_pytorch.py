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
            
            # Return raw Q-value without any normalization
            return float(q_value.item())
    
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
        Get Q-value for vehicle assignment to request using neural network
        现在支持battery_level和request_value参数
        """
        return self.get_q_value(vehicle_id, f"assign_{target_id}", 
                               vehicle_location, target_location, current_time, 
                               other_vehicles, num_requests, battery_level, request_value)
    def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = 0.0, 
                        other_vehicles: int = 0, num_requests: int = 0) -> float:
        """
        Get Q-value for vehicle idle action using neural network
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 电池电量 (0-1)
            current_time: 当前时间
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            
        Returns:
            float: idle动作的Q值
        """
        # 使用统一的get_q_value方法，保持与其他动作的一致性
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
                        num_requests: int = 0, request_value: float = 0.0):
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
            'other_vehicles': other_vehicles,
            'num_requests': num_requests,
            'request_value': request_value,  # 添加请求价值信息
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
    
    def _advanced_sample(self, batch_size: int, method: str = "importance"):
        """
        高级采样策略：重要性采样、Thompson采样、优先经验回放
        
        Args:
            batch_size: 批次大小
            method: 采样方法 ("importance", "thompson", "prioritized", "balanced")
            
        Returns:
            采样的经验列表
        """
        experiences = list(self.experience_buffer)
        
        if method == "importance":
            return self._importance_sampling(experiences, batch_size)
        elif method == "thompson":
            return self._thompson_sampling(experiences, batch_size)
        elif method == "prioritized":
            return self._prioritized_sampling(experiences, batch_size)
        else:
            return self._balanced_sample(batch_size)  # 回退到平衡采样
    
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
    
    def train_step(self, batch_size: int = 64):  # Increased batch size
        """Perform one training step using stored experiences with proper DQN algorithm"""
        if len(self.experience_buffer) < batch_size * 2:  # Wait for more experiences
            return 0.0
        
        # 原始随机采样方法（已替换为高级采样）
        # batch = random.sample(list(self.experience_buffer), batch_size)

        # 使用高级采样策略，根据训练步数选择不同的方法
        if self.training_step < 2000:
            # 初期使用平衡采样建立基础
            batch = self._advanced_sample(batch_size, method="importance")
        else:
            # 中期使用重要性采样
            batch = self._advanced_sample(batch_size, method="balanced")


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
            next_request_value = exp.get('request_value', 0.0)  # 下一状态也使用相同的请求价值
            next_inputs = self._prepare_network_input_with_battery(
                exp['next_vehicle_location'], exp['target_location'], 
                exp['current_time'] + 1, exp['other_vehicles'], exp['num_requests'], 
                exp['action_type'], next_battery, next_request_value
            )
            
            
            # 处理next state的返回值（现在包含battery和request_value）
            if len(next_inputs) == 7:  # 包含battery和request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor, next_value_tensor = next_inputs
            elif len(next_inputs) == 6:  # 包含battery但没有request_value
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor = next_inputs
                next_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
            else:  # 不包含battery和request_value（向后兼容）
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor = next_inputs
                next_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
                next_value_tensor = torch.tensor([[0.0]], dtype=torch.float32).to(self.device)
            
            next_states.append({
                'path_locations': next_path_locations.squeeze(0),
                'path_delays': next_path_delays.squeeze(0),
                'current_time': next_time_tensor.squeeze(0),
                'other_agents': next_others_tensor.squeeze(0),
                'num_requests': next_requests_tensor.squeeze(0),
                'battery_level': next_battery_tensor.squeeze(0),  # 添加battery信息
                'request_value': next_value_tensor.squeeze(0),  # 添加request_value信息
                'action_type': exp['action_type'],  # 添加action_type信息
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
        next_action_types = []
        next_vehicle_ids = []
        next_vehicle_types = []
        for state in next_states:
            action_type_str = state['action_type']
            if action_type_str == 'idle':
                action_type_id = 1
            elif action_type_str.startswith('assign'):
                action_type_id = 2
            elif action_type_str.startswith('charge'):
                action_type_id = 3
            else:
                action_type_id = 2  # 默认为assign
            next_action_types.append(action_type_id)
            next_vehicle_ids.append(state['vehicle_id'] + 1)  # +1因为0是padding
            next_vehicle_types.append(state['vehicle_type'])
        
        next_batch_action_types = torch.tensor(next_action_types, dtype=torch.long).to(self.device)
        next_batch_vehicle_ids = torch.tensor(next_vehicle_ids, dtype=torch.long).to(self.device)
        next_batch_vehicle_types = torch.tensor(next_vehicle_types, dtype=torch.long).to(self.device)
        
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
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=10.0)
        self.optimizer.step()
        
        # Update learning rate scheduler based on loss
        self.scheduler.step(loss_value)
        
        # Update target network periodically (key DQN component)
        if self.training_step % self.target_update_frequency == 0:
            self.target_network.load_state_dict(self.network.state_dict())
            print(f"🔄 Target network updated at step {self.training_step}")
        
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
        """Store experience (compatibility method)"""
        # Convert Experience to our format if needed
        pass
    
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
