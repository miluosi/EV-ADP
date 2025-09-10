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
                 log_dir: str = "logs/charging_nn", device: str = 'cpu'):
        super().__init__(log_dir=log_dir, device=device)
        
        self.grid_size = grid_size
        self.num_vehicles = num_vehicles
        self.num_locations = grid_size * grid_size
        
        # Initialize the neural network with increased capacity for complex environment
        self.network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
            max_capacity=6,  # Increased capacity for longer paths
            embedding_dim=128,  # Larger embedding for complex environment
            lstm_hidden=256,   # Larger LSTM for complex patterns
            dense_hidden=512,   # Larger dense layer
            pretrained_embeddings=None  # Explicitly set to None to ensure gradients
        ).to(self.device)
        
        # Target network for stable DQN training
        self.target_network = PyTorchPathBasedNetwork(
            num_locations=self.num_locations,
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
        self.optimizer = optim.Adam(self.network.parameters(), lr=1e-3, weight_decay=1e-5)
        self.loss_fn = nn.MSELoss()
        
        # Learning rate scheduler for adaptive learning - more conservative
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=50, 
            min_lr=1e-4, verbose=True
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
                   battery_level: float = 1.0) -> float:
        """
        Neural network-based Q-value calculation using PyTorchPathBasedNetwork
        现在支持battery_level参数
        """
        # 使用支持battery的输入准备方法
        inputs = self._prepare_network_input_with_battery(
            vehicle_location, target_location, current_time, 
            other_vehicles, num_requests, action_type, battery_level
        )
        
        # 处理返回的输入（可能包含或不包含battery）
        if len(inputs) == 6:  # 包含battery
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor, battery_tensor = inputs
        else:  # 不包含battery（向后兼容）
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = inputs
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
        
        # Forward pass through network
        self.network.eval()
        with torch.no_grad():
            q_value = self.network(
                path_locations=path_locations,
                path_delays=path_delays,
                current_time=time_tensor,
                other_agents=others_tensor,
                num_requests=requests_tensor,
                battery_level=battery_tensor
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
        
        # Set path: current -> target -> end
        path_locations[0, 0] = vehicle_location + 1  # +1 because 0 is padding
        path_locations[0, 1] = target_location + 1
        path_locations[0, 2] = 0  # End token
        
        # Set delays based on action type
        if action_type.startswith('assign'):
            # Passenger service - delays based on urgency
            path_delays[0, 0, 0] = 0.0  # No delay at current location
            path_delays[0, 1, 0] = max(0.0, (300 - current_time) / 300)  # Normalized urgency
        elif action_type.startswith('charge'):
            # Charging action - charging time penalty
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.8  # High delay for charging (opportunity cost)
        else:
            # Movement or idle
            path_delays[0, 0, 0] = 0.0
            path_delays[0, 1, 0] = 0.1  # Small delay for movement
        
        # Normalize time (0-1 range)
        time_tensor = torch.tensor([[current_time / 100.0]], dtype=torch.float32)  # Assume episode_length=100
        
        # Normalize other metrics
        others_tensor = torch.tensor([[other_vehicles / self.num_vehicles]], dtype=torch.float32)
        requests_tensor = torch.tensor([[num_requests / 10.0]], dtype=torch.float32)  # Assume max 10 requests
        
        # Move to device
        return (path_locations.to(self.device), 
                path_delays.to(self.device),
                time_tensor.to(self.device),
                others_tensor.to(self.device),
                requests_tensor.to(self.device))
    
    def _prepare_network_input_with_battery(self, vehicle_location: int, target_location: int, 
                                           current_time: float, other_vehicles: int, 
                                           num_requests: int, action_type: str, 
                                           battery_level: float = 1.0):
        """
        Prepare input tensors for the neural network including battery information
        
        Args:
            vehicle_location: 车辆当前位置
            target_location: 目标位置
            current_time: 当前时间
            other_vehicles: 附近其他车辆数量
            num_requests: 当前请求数量
            action_type: 动作类型
            battery_level: 电池电量 (0-1)
        """
        # 根据动作类型选择合适的输入准备方法
        if action_type == 'idle':
            # 对于idle状态，处理目标位置为当前位置
            path_locations = torch.zeros(1, 3, dtype=torch.long)  # batch_size=1, seq_len=3
            path_delays = torch.zeros(1, 3, 1, dtype=torch.float32)
            
            # 设置路径：当前位置 -> 当前位置（表示停留）-> 结束
            path_locations[0, 0] = vehicle_location + 1  # +1 because 0 is padding
            path_locations[0, 1] = vehicle_location + 1  # 同样的位置表示idle
            path_locations[0, 2] = 0  # End token
            
            # 设置延迟 - idle状态的延迟模式
            path_delays[0, 0, 0] = 0.0  # 当前位置无延迟
            path_delays[0, 1, 0] = 0.05  # idle的小延迟（等待成本）
            path_delays[0, 2, 0] = 0.0  # 结束位置无延迟
            
            # 归一化时间 (0-1 range)
            time_tensor = torch.tensor([[current_time / 100.0]], dtype=torch.float32)
            
            # 归一化其他指标
            others_tensor = torch.tensor([[other_vehicles / self.num_vehicles]], dtype=torch.float32)
            requests_tensor = torch.tensor([[num_requests / 10.0]], dtype=torch.float32)
            
            # 归一化电池电量
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32)
            
            # Move to device
            return (path_locations.to(self.device), 
                    path_delays.to(self.device),
                    time_tensor.to(self.device),
                    others_tensor.to(self.device),
                    requests_tensor.to(self.device),
                    battery_tensor.to(self.device))
        else:
            # 对于非idle动作，使用标准方法并添加battery信息
            path_locations, path_delays, time_tensor, others_tensor, requests_tensor = self._prepare_network_input(
                vehicle_location, target_location, current_time, 
                other_vehicles, num_requests, action_type
            )
            
            # 添加battery信息
            battery_tensor = torch.tensor([[battery_level]], dtype=torch.float32).to(self.device)
            
            return (path_locations, path_delays, time_tensor, 
                   others_tensor, requests_tensor, battery_tensor)
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int, 
                              current_time: float = 0.0, other_vehicles: int = 0, 
                              num_requests: int = 0, battery_level: float = 1.0) -> float:
        """
        Get Q-value for vehicle assignment to request using neural network
        现在支持battery_level参数
        """
        return self.get_q_value(vehicle_id, f"assign_{target_id}", 
                               vehicle_location, target_location, current_time, 
                               other_vehicles, num_requests, battery_level)
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
                        num_requests: int = 0):
        """
        Store experience for training - 现在支持battery信息
        
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
        """
        experience = {
            'vehicle_id': vehicle_id,
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
            'is_idle': action_type == 'idle'  # 自动标记idle状态
        }
        self.experience_buffer.append(experience)
    
    def store_idle_experience(self, vehicle_id: int, vehicle_location: int, 
                            battery_level: float, current_time: float, reward: float,
                            next_vehicle_location: int, next_battery_level: float,
                            other_vehicles: int = 0, num_requests: int = 0):
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
            'is_idle': True  # 标记这是一个idle经验
        }
        self.experience_buffer.append(experience)
    
    def train_step(self, batch_size: int = 64):  # Increased batch size
        """Perform one training step using stored experiences with proper DQN algorithm"""
        if len(self.experience_buffer) < batch_size * 2:  # Wait for more experiences
            return 0.0
        
        # Sample random batch (Experience Replay)
        batch = random.sample(list(self.experience_buffer), batch_size)
        
        # Separate current states and next states for batch processing
        current_states = []
        next_states = []
        rewards = []
        
        for exp in batch:
            # Current state - 使用支持battery的输入准备方法
            current_battery = exp.get('battery_level', 0.5)  # 向后兼容
            current_inputs = self._prepare_network_input_with_battery(
                exp['vehicle_location'], exp['target_location'], exp['current_time'], 
                exp['other_vehicles'], exp['num_requests'], exp['action_type'], current_battery
            )
            
            # 处理返回的输入（可能包含或不包含battery）
            if len(current_inputs) == 6:  # 包含battery
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor, current_battery_tensor = current_inputs
            else:  # 不包含battery（向后兼容）
                current_path_locations, current_path_delays, current_time_tensor, current_others_tensor, current_requests_tensor = current_inputs
                current_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
            
            current_states.append({
                'path_locations': current_path_locations.squeeze(0),
                'path_delays': current_path_delays.squeeze(0),
                'current_time': current_time_tensor.squeeze(0),
                'other_agents': current_others_tensor.squeeze(0),
                'num_requests': current_requests_tensor.squeeze(0),
                'battery_level': current_battery_tensor.squeeze(0)  # 添加battery信息
            })
            
            # Next state (for target calculation) - 使用支持battery的输入准备方法
            next_battery = exp.get('next_battery_level', 1.0)  # 向后兼容
            next_inputs = self._prepare_network_input_with_battery(
                exp['next_vehicle_location'], exp['target_location'], 
                exp['current_time'] + 1, exp['other_vehicles'], exp['num_requests'], 
                exp['action_type'], next_battery
            )
            
            
            # 处理next state的返回值
            if len(next_inputs) == 6:  # 包含battery
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor, next_battery_tensor = next_inputs
            else:  # 不包含battery（向后兼容）
                next_path_locations, next_path_delays, next_time_tensor, next_others_tensor, next_requests_tensor = next_inputs
                next_battery_tensor = torch.tensor([[1.0]], dtype=torch.float32).to(self.device)
            
            next_states.append({
                'path_locations': next_path_locations.squeeze(0),
                'path_delays': next_path_delays.squeeze(0),
                'current_time': next_time_tensor.squeeze(0),
                'other_agents': next_others_tensor.squeeze(0),
                'num_requests': next_requests_tensor.squeeze(0),
                'battery_level': next_battery_tensor.squeeze(0)  # 添加battery信息
            })
            
            rewards.append(exp['reward'])
        
        # Stack batch inputs for current states
        current_batch_path_locations = torch.stack([state['path_locations'] for state in current_states])
        current_batch_path_delays = torch.stack([state['path_delays'] for state in current_states])
        current_batch_current_time = torch.stack([state['current_time'] for state in current_states])
        current_batch_other_agents = torch.stack([state['other_agents'] for state in current_states])
        current_batch_num_requests = torch.stack([state['num_requests'] for state in current_states])
        current_batch_battery_levels = torch.stack([state['battery_level'] for state in current_states])  # 添加battery批处理
        
        # Stack batch inputs for next states
        next_batch_path_locations = torch.stack([state['path_locations'] for state in next_states])
        next_batch_path_delays = torch.stack([state['path_delays'] for state in next_states])
        next_batch_current_time = torch.stack([state['current_time'] for state in next_states])
        next_batch_other_agents = torch.stack([state['other_agents'] for state in next_states])
        next_batch_num_requests = torch.stack([state['num_requests'] for state in next_states])
        next_batch_battery_levels = torch.stack([state['battery_level'] for state in next_states])  # 添加next states的battery批处理
        
        # Current Q-values (with gradients) - 现在包含battery信息
        self.network.train()
        current_q_values = self.network(
            path_locations=current_batch_path_locations,
            path_delays=current_batch_path_delays,
            current_time=current_batch_current_time,
            other_agents=current_batch_other_agents,
            num_requests=current_batch_num_requests,
            battery_level=current_batch_battery_levels  # 添加battery参数
        )
        
        # Next Q-values using target network (without gradients) - 现在包含battery信息
        with torch.no_grad():
            self.target_network.eval()
            next_q_values = self.target_network(
                path_locations=next_batch_path_locations,
                path_delays=next_batch_path_delays,
                current_time=next_batch_current_time,
                other_agents=next_batch_other_agents,
                num_requests=next_batch_num_requests,
                battery_level=next_batch_battery_levels  # 添加battery参数
            )
        
        # Calculate TD targets without normalization
        gamma = 0.95  # Slightly lower discount factor for stability
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32).to(self.device).unsqueeze(1)
        
        # Calculate target Q-values directly without normalization
        with torch.no_grad():
            target_q_values = rewards_tensor + gamma * next_q_values
            
        # Compute loss with raw values
        loss = self.loss_fn(current_q_values, target_q_values)
        loss_value = loss.item()  # Define loss_value immediately after loss computation
        
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
        
        # Clip gradients more aggressively
        torch.nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=0.5)
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
    """PyTorch implementation of path-based neural network"""
    
    def __init__(self, 
                 num_locations: int,
                 max_capacity: int,
                 embedding_dim: int = 100,
                 lstm_hidden: int = 200,
                 dense_hidden: int = 300,
                 pretrained_embeddings: Optional[torch.Tensor] = None):
        super(PyTorchPathBasedNetwork, self).__init__()
        
        self.num_locations = num_locations
        self.max_capacity = max_capacity
        self.embedding_dim = embedding_dim
        
        # Location embedding layer
        self.location_embedding = nn.Embedding(
            num_embeddings=num_locations + 1,
            embedding_dim=embedding_dim,
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
        
        # State embedding layers - 增加battery维度
        state_input_dim = lstm_hidden + embedding_dim + 3  # path + time + other_agents + num_requests + battery
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
                battery_level: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            path_locations: [batch_size, seq_len] - Location IDs in path
            path_delays: [batch_size, seq_len, 1] - Delay information
            current_time: [batch_size, 1] - Current time
            other_agents: [batch_size, 1] - Number of other agents nearby
            num_requests: [batch_size, 1] - Number of current requests
            battery_level: [batch_size, 1] - Battery level (0-1), optional
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
        
        # Combine all features
        combined_features = torch.cat([
            path_representation,
            time_embed,
            other_agents,
            num_requests,
            battery_level  # 添加battery_level到特征组合中
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
        torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), max_norm=1.0)
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
