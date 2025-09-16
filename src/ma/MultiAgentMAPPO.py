"""
Multi-Agent Proximal Policy Optimization (MAPPO) implementation
for multi-vehicle charging coordination in EV-ADP system.

MAPPO uses a shared critic with individual actors and employs
PPO's clipped objective for stable multi-agent learning.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple, Any, Optional
from torch.distributions import Categorical


class MAPPOActor(nn.Module):
    """Actor network for MAPPO - outputs action probabilities"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(MAPPOActor, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=0.01)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, avail_actions: torch.Tensor = None) -> Categorical:
        """
        Forward pass returning action distribution
        Args:
            obs: Observations [batch_size, obs_dim]
            avail_actions: Available actions mask [batch_size, action_dim]
        Returns:
            Action distribution
        """
        logits = self.network(obs)
        
        # Apply action masking
        if avail_actions is not None:
            logits = logits - 1e8 * (1 - avail_actions)
        
        return Categorical(logits=logits)


class MAPPOCritic(nn.Module):
    """Shared critic network for MAPPO - evaluates global state"""

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super(MAPPOCritic, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight, gain=1.0)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Return value estimate for global state"""
        return self.network(state)


class MAPPOBuffer:
    """Rollout buffer for MAPPO"""

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int, action_dim: int, 
                 buffer_size: int = 2048):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.buffer_size = buffer_size
        self.ptr = 0
        self.size = 0
        
        # Storage
        self.observations = np.zeros((buffer_size, num_agents, obs_dim), dtype=np.float32)
        self.states = np.zeros((buffer_size, state_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, num_agents), dtype=np.int64)
        self.log_probs = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.rewards = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)
        self.avail_actions = np.zeros((buffer_size, num_agents, action_dim), dtype=np.float32)
        
        # For GAE computation
        self.advantages = np.zeros((buffer_size, num_agents), dtype=np.float32)
        self.returns = np.zeros((buffer_size, num_agents), dtype=np.float32)

    def store(self, obs: List[np.ndarray], state: np.ndarray, actions: List[int],
              log_probs: List[float], rewards: List[float], value: float, done: bool,
              avail_actions: List[List[int]] = None):
        """Store a transition"""
        idx = self.ptr
        
        # Store observations and state
        for i, ob in enumerate(obs):
            self.observations[idx, i] = ob
        self.states[idx] = state
        
        # Store actions and log probabilities
        for i, (action, log_prob) in enumerate(zip(actions, log_probs)):
            self.actions[idx, i] = action
            self.log_probs[idx, i] = log_prob
        
        # Store rewards and value
        for i, reward in enumerate(rewards):
            self.rewards[idx, i] = reward
        self.values[idx] = value
        self.dones[idx] = float(done)
        
        # Store available actions
        if avail_actions:
            for i, avail in enumerate(avail_actions):
                avail_mask = np.zeros(self.action_dim)
                avail_mask[avail] = 1.0
                self.avail_actions[idx, i] = avail_mask
        else:
            self.avail_actions[idx] = 1.0  # All actions available
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def compute_gae(self, next_value: float, gamma: float = 0.99, gae_lambda: float = 0.95):
        """Compute Generalized Advantage Estimation"""
        # Add next value for bootstrapping
        values = np.append(self.values[:self.size], next_value)
        
        # Compute advantages using GAE
        advantages = np.zeros_like(self.rewards[:self.size])
        last_gae_lambda = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = values[t + 1]
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = values[t + 1]
            
            # Compute TD error for each agent
            for i in range(self.num_agents):
                delta = self.rewards[t, i] + gamma * next_value * next_non_terminal - values[t]
                advantages[t, i] = last_gae_lambda = delta + gamma * gae_lambda * next_non_terminal * last_gae_lambda
        
        # Compute returns
        self.advantages[:self.size] = advantages
        self.returns[:self.size] = advantages + np.expand_dims(values[:self.size], axis=1)

    def get_batch(self, batch_size: int = None):
        """Get a batch of experiences"""
        if batch_size is None:
            batch_size = self.size
        
        indices = np.random.choice(self.size, batch_size, replace=False)
        
        return {
            'observations': torch.FloatTensor(self.observations[indices]),
            'states': torch.FloatTensor(self.states[indices]),
            'actions': torch.LongTensor(self.actions[indices]),
            'log_probs': torch.FloatTensor(self.log_probs[indices]),
            'rewards': torch.FloatTensor(self.rewards[indices]),
            'values': torch.FloatTensor(self.values[indices]),
            'dones': torch.FloatTensor(self.dones[indices]),
            'avail_actions': torch.FloatTensor(self.avail_actions[indices]),
            'advantages': torch.FloatTensor(self.advantages[indices]),
            'returns': torch.FloatTensor(self.returns[indices])
        }

    def clear(self):
        """Clear the buffer"""
        self.ptr = 0
        self.size = 0


class MAPPO:
    """
    Multi-Agent Proximal Policy Optimization implementation
    """

    def __init__(self, num_agents: int, obs_dim: int, state_dim: int, action_dim: int,
                 hidden_dim: int = 128, lr_actor: float = 3e-4, lr_critic: float = 3e-4,
                 gamma: float = 0.99, gae_lambda: float = 0.95, clip_param: float = 0.2,
                 value_loss_coef: float = 0.5, entropy_coef: float = 0.01, 
                 max_grad_norm: float = 0.5, ppo_epochs: int = 10, batch_size: int = 256,
                 buffer_size: int = 2048, device: str = 'cpu'):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        
        # Create actor networks (one for each agent)
        self.actors = nn.ModuleList([
            MAPPOActor(obs_dim, action_dim, hidden_dim).to(device)
            for _ in range(num_agents)
        ])
        
        # Create shared critic network
        self.critic = MAPPOCritic(state_dim, hidden_dim).to(device)
        
        # Optimizers
        self.actor_optimizer = optim.Adam([
            param for actor in self.actors for param in actor.parameters()
        ], lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        # Rollout buffer
        self.buffer = MAPPOBuffer(num_agents, obs_dim, state_dim, action_dim, buffer_size)
        
        # Training statistics
        self.training_step = 0

    def select_actions(self, obs: List[np.ndarray], state: np.ndarray,
                      avail_actions: List[List[int]] = None, deterministic: bool = False) -> Tuple[List[int], List[float], float]:
        """
        Select actions for all agents
        Returns:
            actions: List of selected actions
            log_probs: List of log probabilities
            value: State value estimate
        """
        actions = []
        log_probs = []
        
        # Get state value
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            value = self.critic(state_tensor).item()
        
        # Select action for each agent
        for i in range(self.num_agents):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            
            # Create available actions mask
            avail_mask = None
            if avail_actions and avail_actions[i]:
                avail_mask = torch.zeros(1, self.action_dim).to(self.device)
                avail_mask[0, avail_actions[i]] = 1.0
            
            with torch.no_grad():
                dist = self.actors[i](obs_tensor, avail_mask)
                
                if deterministic:
                    action = dist.probs.argmax(dim=-1).item()
                else:
                    action = dist.sample().item()
                
                log_prob = dist.log_prob(torch.tensor([action])).item()
            
            actions.append(action)
            log_probs.append(log_prob)
        
        return actions, log_probs, value

    def store_transition(self, obs: List[np.ndarray], state: np.ndarray, actions: List[int],
                        log_probs: List[float], rewards: List[float], value: float, done: bool,
                        avail_actions: List[List[int]] = None):
        """Store transition in buffer"""
        self.buffer.store(obs, state, actions, log_probs, rewards, value, done, avail_actions)

    def update(self, next_state: np.ndarray = None) -> Dict[str, float]:
        """Update networks using PPO"""
        # Get next value for GAE computation
        next_value = 0.0
        if next_state is not None:
            with torch.no_grad():
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                next_value = self.critic(next_state_tensor).item()
        
        # Compute advantages and returns
        self.buffer.compute_gae(next_value, self.gamma, self.gae_lambda)
        
        # Training statistics
        total_actor_loss = 0.0
        total_critic_loss = 0.0
        total_entropy_loss = 0.0
        
        # Multiple epochs of PPO updates
        for epoch in range(self.ppo_epochs):
            batch = self.buffer.get_batch(self.batch_size)
            
            # Move to device
            for key in batch:
                batch[key] = batch[key].to(self.device)
            
            # === Update Actors ===
            actor_losses = []
            entropy_losses = []
            
            for i in range(self.num_agents):
                # Get current policy distribution
                dist = self.actors[i](batch['observations'][:, i], batch['avail_actions'][:, i])
                
                # Compute probability ratio
                new_log_probs = dist.log_prob(batch['actions'][:, i])
                old_log_probs = batch['log_probs'][:, i]
                ratio = torch.exp(new_log_probs - old_log_probs)
                
                # Compute clipped surrogate loss
                advantages = batch['advantages'][:, i]
                # Normalize advantages per agent
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                surr1 = ratio * advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Entropy loss
                entropy_loss = -dist.entropy().mean()
                
                actor_losses.append(actor_loss)
                entropy_losses.append(entropy_loss)
            
            # Combine actor losses
            total_actor_loss_batch = sum(actor_losses) / self.num_agents
            total_entropy_loss_batch = sum(entropy_losses) / self.num_agents
            
            # === Update Critic ===
            current_values = self.critic(batch['states'])
            returns = batch['returns'].mean(dim=1)  # Average returns across agents
            
            # Value loss (clipped)
            value_pred_clipped = batch['values'] + (current_values.squeeze() - batch['values']).clamp(
                -self.clip_param, self.clip_param)
            value_losses = (current_values.squeeze() - returns).pow(2)
            value_losses_clipped = (value_pred_clipped - returns).pow(2)
            critic_loss = 0.5 * torch.max(value_losses, value_losses_clipped).mean()
            
            # Combined loss
            total_loss = (total_actor_loss_batch + 
                         self.value_loss_coef * critic_loss + 
                         self.entropy_coef * total_entropy_loss_batch)
            
            # Update networks
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            nn.utils.clip_grad_norm_([param for actor in self.actors for param in actor.parameters()], 
                                   self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            
            # Accumulate losses
            total_actor_loss += total_actor_loss_batch.item()
            total_critic_loss += critic_loss.item()
            total_entropy_loss += total_entropy_loss_batch.item()
        
        # Clear buffer
        self.buffer.clear()
        
        self.training_step += 1
        
        return {
            'actor_loss': total_actor_loss / self.ppo_epochs,
            'critic_loss': total_critic_loss / self.ppo_epochs,
            'entropy_loss': total_entropy_loss / self.ppo_epochs
        }

    def save_models(self, path: str):
        """Save all models"""
        torch.save({
            'actors': [actor.state_dict() for actor in self.actors],
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'training_step': self.training_step
        }, path)

    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path)
        
        for i, actor_state in enumerate(checkpoint['actors']):
            self.actors[i].load_state_dict(actor_state)
        
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])
        self.training_step = checkpoint['training_step']


class MAPPOEnvironmentWrapper:
    """Environment wrapper for MAPPO - same as QMIX wrapper"""

    def __init__(self, env, num_agents: int):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = 6  # Individual observation dimension
        self.action_dim = 3  # Discrete actions: 0=idle, 1=service, 2=charge  
        self.state_dim = self._get_state_dim()

    def _get_state_dim(self) -> int:
        """Get global state dimension"""
        return self.num_agents * 6 + 10  # Vehicle states + global info

    def reset(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Reset and return global state and individual observations"""
        env_state = self.env.reset()
        
        global_state = self._extract_global_state(env_state)
        obs = []
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(env_state, i)
            obs.append(agent_obs)
        
        return global_state, obs

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[float], bool, Dict]:
        """Execute actions and return results"""
        env_actions = self._convert_actions_to_env(actions)
        next_env_state, rewards, done, info = self.env.step(env_actions)
        
        next_global_state = self._extract_global_state(next_env_state)
        next_obs = []
        agent_rewards = []
        
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(next_env_state, i)
            agent_reward = self._extract_agent_reward(rewards, i)
            
            next_obs.append(agent_obs)
            agent_rewards.append(agent_reward)
        
        return next_global_state, next_obs, agent_rewards, done, info

    def _extract_global_state(self, env_state: Dict) -> np.ndarray:
        """Extract global state (same as QMIX wrapper)"""
        state_features = []
        
        if isinstance(env_state, dict) and 'vehicles' in env_state:
            for i in range(self.num_agents):
                if i < len(env_state['vehicles']):
                    vehicle = env_state['vehicles'][i]
                    state_features.extend([
                        float(vehicle.get('location', 0)) / 100.0,
                        float(vehicle.get('battery', 0.5)),
                        1.0 if vehicle.get('assigned_request') else 0.0,
                        1.0 if vehicle.get('charging_station') else 0.0,
                        float(vehicle.get('coordinates', [0, 0])[0]) / 12.0,
                        float(vehicle.get('coordinates', [0, 0])[1]) / 12.0,
                    ])
                else:
                    state_features.extend([0.0] * 6)
        else:
            state_features.extend([0.0] * (self.num_agents * 6))
        
        # Add global information
        state_features.extend([
            float(env_state.get('current_time', 0)) / 100.0,
            float(len(env_state.get('pending_requests', []))) / 10.0,
            float(env_state.get('total_reward', 0)) / 100.0,
        ])
        
        # Pad to fixed size
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        return np.array(state_features[:self.state_dim], dtype=np.float32)

    def _extract_agent_obs(self, env_state: Dict, agent_id: int) -> np.ndarray:
        """Extract individual agent observation"""
        if isinstance(env_state, dict) and 'vehicles' in env_state and agent_id < len(env_state['vehicles']):
            vehicle = env_state['vehicles'][agent_id]
            obs = np.array([
                float(vehicle.get('location', 0)) / 100.0,
                float(vehicle.get('battery', 0.5)),
                1.0 if vehicle.get('assigned_request') else 0.0,
                1.0 if vehicle.get('charging_station') else 0.0,
                float(vehicle.get('coordinates', [0, 0])[0]) / 12.0,
                float(vehicle.get('coordinates', [0, 0])[1]) / 12.0,
            ])
            return obs
        else:
            return np.zeros(6, dtype=np.float32)

    def _extract_agent_reward(self, rewards: Dict, agent_id: int) -> float:
        """Extract reward for specific agent"""
        return rewards.get(agent_id, 0.0)

    def _convert_actions_to_env(self, actions: List[int]) -> Dict:
        """Convert discrete actions to environment actions"""
        from Action import ServiceAction, ChargingAction, IdleAction
        import random
        
        env_actions = {}
        vehicles = getattr(self.env, 'vehicles', {})
        current_requests = getattr(self.env, 'requests', [])
        charging_stations = getattr(self.env, 'charging_stations', {})
        
        for i, action in enumerate(actions):
            current_pos = (0, 0)
            if i in vehicles:
                vehicle = vehicles[i]
                current_pos = vehicle.get('coordinates', (0, 0))
            
            if action == 0:  # Idle action
                # Move to a random nearby location
                target_x = min(11, max(0, current_pos[0] + random.randint(-2, 2)))
                target_y = min(11, max(0, current_pos[1] + random.randint(-2, 2)))
                target_pos = (target_x, target_y)
                env_actions[i] = IdleAction([], current_pos, target_pos)
                
            elif action == 1:  # Service action
                if current_requests:
                    # Accept a random available request
                    selected_request = random.choice(list(current_requests.values()))
                    env_actions[i] = ServiceAction([selected_request], selected_request.id)
                else:
                    # No requests available, default to idle
                    target_pos = (current_pos[0], current_pos[1])
                    env_actions[i] = IdleAction([], current_pos, target_pos)
                    
            elif action == 2:  # Charge action
                if charging_stations:
                    # Go to nearest or random charging station
                    station_id = random.choice(list(charging_stations.keys()))
                    env_actions[i] = ChargingAction([], station_id, 30.0)
                else:
                    # No charging stations available, default to idle
                    target_pos = (current_pos[0], current_pos[1])
                    env_actions[i] = IdleAction([], current_pos, target_pos)
            else:
                # Invalid action, default to idle
                target_pos = (current_pos[0], current_pos[1])
                env_actions[i] = IdleAction([], current_pos, target_pos)
        
        return env_actions

    def get_avail_actions(self, obs: List[np.ndarray]) -> List[List[int]]:
        """Get available actions for each agent"""
        avail_actions = []
        for i in range(self.num_agents):
            avail_actions.append([0, 1, 2])  # All actions available
        return avail_actions