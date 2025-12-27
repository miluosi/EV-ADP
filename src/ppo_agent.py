"""
PPO Agent for Vehicle Dispatch Optimization
Proximal Policy Optimization implementation for multi-agent vehicle routing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
from typing import Dict, List, Tuple, Optional
import random


class ActorCriticNetwork(nn.Module):
    """
    Actor-Critic网络架构，用于PPO算法
    Actor输出动作概率分布，Critic输出状态价值
    """
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorCriticNetwork, self).__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Shared feature extractor
        self.shared_layers = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )
        
        # Actor head (policy network)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Critic head (value network)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Xavier初始化"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            state: 状态张量 [batch_size, state_dim]
        
        Returns:
            action_logits: 动作logits [batch_size, action_dim]
            value: 状态价值 [batch_size, 1]
        """
        features = self.shared_layers(state)
        action_logits = self.actor(features)
        value = self.critic(features)
        return action_logits, value
    
    def get_action_and_value(self, state: torch.Tensor, action: Optional[torch.Tensor] = None):
        """
        获取动作、动作log概率和状态价值
        
        Args:
            state: 状态张量
            action: 如果提供，计算该动作的log概率；否则采样新动作
        
        Returns:
            action: 采样的动作
            log_prob: 动作的log概率
            entropy: 策略熵
            value: 状态价值
        """
        action_logits, value = self.forward(state)
        probs = F.softmax(action_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, value


class PPOMemory:
    """
    PPO经验回放缓冲区
    存储轨迹数据用于批量更新
    """
    def __init__(self, batch_size: int = 256):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        self.batch_size = batch_size
    
    def store(self, state, action, log_prob, reward, value, done):
        """存储单步经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
    
    def clear(self):
        """清空缓冲区"""
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
    
    def get_batches(self):
        """
        将存储的经验转换为批次
        
        Returns:
            dict: 包含所有经验的字典
        """
        if len(self.states) == 0:
            return None
        
        return {
            'states': torch.FloatTensor(np.array(self.states)),
            'actions': torch.LongTensor(self.actions),
            'old_log_probs': torch.FloatTensor(self.log_probs),
            'rewards': torch.FloatTensor(self.rewards),
            'values': torch.FloatTensor(self.values),
            'dones': torch.FloatTensor(self.dones)
        }
    
    def __len__(self):
        return len(self.states)


class PPOAgent:
    """
    PPO Agent for vehicle dispatch optimization
    支持EV和AEV两种车辆类型的协同决策
    """
    def __init__(
        self,
        state_dim: int = 64,
        action_dim: int = 32,
        lr: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        update_epochs: int = 10,
        batch_size: int = 256,
        device: str = 'cuda'
    ):
        """
        初始化PPO Agent
        
        Args:
            state_dim: 状态维度
            action_dim: 动作维度
            lr: 学习率
            gamma: 折扣因子
            gae_lambda: GAE优势估计的lambda参数
            clip_epsilon: PPO裁剪参数
            value_coef: 价值损失系数
            entropy_coef: 熵正则化系数
            max_grad_norm: 梯度裁剪阈值
            update_epochs: 每次更新的训练轮数
            batch_size: 批次大小
            device: 计算设备
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        # 超参数
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.update_epochs = update_epochs
        self.batch_size = batch_size
        
        # 网络
        self.policy = ActorCriticNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        
        # 经验缓冲区
        self.memory = PPOMemory(batch_size)
        
        # 训练统计
        self.training_step = 0
        self.policy_losses = []
        self.value_losses = []
        self.entropy_losses = []
        self.clip_fractions = []
        
        print(f"✓ PPO Agent initialized on device: {self.device}")
        print(f"  - State dim: {state_dim}, Action dim: {action_dim}")
        print(f"  - Clip epsilon: {clip_epsilon}, GAE lambda: {gae_lambda}")
        print(f"  - Learning rate: {lr}, Batch size: {batch_size}")
    
    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[int, float, float]:
        """
        根据当前状态选择动作
        
        Args:
            state: 当前状态
            deterministic: 是否使用确定性策略
            mask: 动作掩码（无效动作设为0）
        
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_logits, value = self.policy(state_tensor)
            
            # 应用动作掩码
            if mask is not None:
                mask_tensor = torch.FloatTensor(mask).to(self.device)
                action_logits = action_logits.masked_fill(mask_tensor == 0, -1e9)
            
            probs = F.softmax(action_logits, dim=-1)
            
            if deterministic:
                action = torch.argmax(probs, dim=-1)
            else:
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            
            if not deterministic:
                return action.item(), log_prob.item(), value.item()
            else:
                return action.item(), 0.0, value.item()
    
    def store_transition(
        self,
        state: np.ndarray,
        action: int,
        log_prob: float,
        reward: float,
        value: float,
        done: bool
    ):
        """
        存储转换到经验缓冲区
        
        Args:
            state: 当前状态
            action: 执行的动作
            log_prob: 动作的对数概率
            reward: 获得的奖励
            value: 状态价值估计
            done: 是否结束
        """
        self.memory.store(state, action, log_prob, reward, value, done)
    
    def compute_gae(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor,
        next_value: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算Generalized Advantage Estimation (GAE)
        
        Args:
            rewards: 奖励序列
            values: 价值估计序列
            dones: 结束标志序列
            next_value: 下一个状态的价值估计
        
        Returns:
            advantages: 优势函数
            returns: 回报
        """
        advantages = torch.zeros_like(rewards).to(self.device)
        lastgaelam = 0
        
        values_list = values.tolist() + [next_value]
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_nonterminal = 1.0 - dones[t]
                nextvalues = next_value
            else:
                next_nonterminal = 1.0 - dones[t]
                nextvalues = values_list[t + 1]
            
            delta = rewards[t] + self.gamma * nextvalues * next_nonterminal - values_list[t]
            advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * next_nonterminal * lastgaelam
        
        returns = advantages + values
        return advantages, returns
    
    def update(self) -> Dict[str, float]:
        """
        使用收集的经验更新策略和价值网络
        
        Returns:
            stats: 训练统计信息
        """
        if len(self.memory) < self.batch_size:
            return {'status': 'insufficient_data', 'buffer_size': len(self.memory)}
        
        # 获取批次数据
        batch = self.memory.get_batches()
        if batch is None:
            return {'status': 'no_data'}
        
        states = batch['states'].to(self.device)
        actions = batch['actions'].to(self.device)
        old_log_probs = batch['old_log_probs'].to(self.device)
        rewards = batch['rewards'].to(self.device)
        values = batch['values'].to(self.device)
        dones = batch['dones'].to(self.device)
        
        # 计算GAE优势和回报
        with torch.no_grad():
            # 获取最后一个状态的价值估计
            if len(states) > 0:
                _, last_value = self.policy(states[-1:])
                last_value = last_value.item()
            else:
                last_value = 0.0
            
            advantages, returns = self.compute_gae(rewards, values, dones, last_value)
            # 标准化优势
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # 多轮更新
        policy_losses = []
        value_losses = []
        entropy_losses = []
        clip_fractions = []
        
        for epoch in range(self.update_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))
            
            # 分批训练
            for start_idx in range(0, len(states), self.batch_size):
                end_idx = min(start_idx + self.batch_size, len(states))
                batch_indices = indices[start_idx:end_idx]
                
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                
                # 前向传播
                action_logits, new_values = self.policy(batch_states)
                probs = F.softmax(action_logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # PPO损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                value_loss = F.mse_loss(new_values.squeeze(), batch_returns)
                
                # 总损失
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                # 统计
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy.item())
                
                # 计算裁剪比例
                with torch.no_grad():
                    clip_fraction = torch.mean((torch.abs(ratio - 1.0) > self.clip_epsilon).float()).item()
                    clip_fractions.append(clip_fraction)
        
        # 清空缓冲区
        self.memory.clear()
        
        # 更新统计
        self.training_step += 1
        avg_policy_loss = np.mean(policy_losses)
        avg_value_loss = np.mean(value_losses)
        avg_entropy = np.mean(entropy_losses)
        avg_clip_fraction = np.mean(clip_fractions)
        
        self.policy_losses.append(avg_policy_loss)
        self.value_losses.append(avg_value_loss)
        self.entropy_losses.append(avg_entropy)
        self.clip_fractions.append(avg_clip_fraction)
        
        return {
            'status': 'success',
            'training_step': self.training_step,
            'policy_loss': avg_policy_loss,
            'value_loss': avg_value_loss,
            'entropy': avg_entropy,
            'clip_fraction': avg_clip_fraction,
            'samples_trained': len(states)
        }
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_step': self.training_step,
            'policy_losses': self.policy_losses,
            'value_losses': self.value_losses,
        }, path)
        print(f"✓ PPO model saved to {path}")
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_step = checkpoint.get('training_step', 0)
        self.policy_losses = checkpoint.get('policy_losses', [])
        self.value_losses = checkpoint.get('value_losses', [])
        print(f"✓ PPO model loaded from {path}")
        print(f"  - Training step: {self.training_step}")


def create_ppo_state_features(env, vehicle_id: int, current_time: float) -> np.ndarray:
    """
    为PPO创建状态特征向量
    
    Args:
        env: 环境对象
        vehicle_id: 车辆ID
        current_time: 当前时间
    
    Returns:
        state: 状态特征向量
    """
    vehicle = env.vehicles.get(vehicle_id)
    if vehicle is None:
        return np.zeros(64)
    
    # 车辆特征
    vloc = vehicle.get('location', 0)
    vbat = vehicle.get('battery', vehicle.get('battery_level', 1.0))
    vtype = 1.0 if vehicle.get('type') == 1 else 0.0  # EV=1, AEV=0
    
    # 车辆坐标
    vx = vloc % env.grid_size
    vy = vloc // env.grid_size
    
    # 车辆状态标志
    has_request = 1.0 if vehicle.get('assigned_request') is not None else 0.0
    has_passenger = 1.0 if vehicle.get('passenger_onboard') is not None else 0.0
    is_charging = 1.0 if vehicle.get('charging_station') is not None else 0.0
    
    # 请求特征
    active_reqs = list(env.active_requests.values())
    num_requests = len(active_reqs)
    
    # 最近请求距离和价值
    if active_reqs and vloc is not None:
        distances = [env._manhattan_distance_loc(vloc, getattr(req, 'pickup', 0)) for req in active_reqs]
        values = [float(getattr(req, 'final_value', getattr(req, 'value', 0.0))) for req in active_reqs]
        min_dist = min(distances) if distances else 0.0
        max_value = max(values) if values else 0.0
        avg_value = np.mean(values) if values else 0.0
    else:
        min_dist = 0.0
        max_value = 0.0
        avg_value = 0.0
    
    # 充电站特征
    if hasattr(env, 'charging_manager') and env.charging_manager.stations:
        station_locs = [st.location for st in env.charging_manager.stations.values()]
        station_dists = [env._manhattan_distance_loc(vloc, sloc) for sloc in station_locs]
        nearest_station_dist = min(station_dists) if station_dists else 0.0
    else:
        nearest_station_dist = 0.0
    
    # 时间特征
    time_progress = current_time / env.episode_length if hasattr(env, 'episode_length') else 0.0
    
    # 其他车辆统计
    other_vehicles = sum(1 for v in env.vehicles.values() if v.get('assigned_request') is not None and v != vehicle)
    
    # 组合特征向量 (64维)
    state = np.array([
        # 车辆基本特征 (10)
        vx / env.grid_size, vy / env.grid_size,
        vbat, vtype,
        has_request, has_passenger, is_charging,
        time_progress,
        nearest_station_dist / env.grid_size,
        other_vehicles / len(env.vehicles),
        
        # 请求特征 (10)
        num_requests / 50.0,  # 归一化
        min_dist / env.grid_size,
        max_value / 1000.0,  # 假设最大价值1000
        avg_value / 1000.0,
        *([0.0] * 6),  # 预留
        
        # 扩展特征 (44)
        *([0.0] * 44)  # 预留给更复杂的特征
    ], dtype=np.float32)
    
    return state
