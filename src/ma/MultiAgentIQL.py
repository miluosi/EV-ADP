"""
Independent Q-Learning (IQL) implementation
for multi-vehicle charging coordination in EV-ADP system.

IQL treats each agent independently with its own Q-network,
serving as a baseline for multi-agent coordination comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple, Any, Optional


class IQLAgent(nn.Module):
    """Independent Q-Learning Agent"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(IQLAgent, self).__init__()
        
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
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Return Q-values for all actions"""
        return self.network(obs)


class IQLReplayBuffer:
    """Independent replay buffer for each agent"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: np.ndarray, action: int, reward: float, 
             next_obs: np.ndarray, done: bool, avail_actions: List[int] = None):
        """Store a single agent transition"""
        transition = {
            'obs': obs.copy(),
            'action': action,
            'reward': reward,
            'next_obs': next_obs.copy(),
            'done': done,
            'avail_actions': avail_actions.copy() if avail_actions else None
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        
        return {
            'obs': np.array([t['obs'] for t in batch]),
            'actions': np.array([t['action'] for t in batch]),
            'rewards': np.array([t['reward'] for t in batch]),
            'next_obs': np.array([t['next_obs'] for t in batch]),
            'dones': np.array([t['done'] for t in batch]),
            'avail_actions': [t['avail_actions'] for t in batch]
        }

    def __len__(self):
        return len(self.buffer)


class IQL:
    """
    Independent Q-Learning for multi-agent systems
    Each agent learns independently without coordination
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 hidden_dim: int = 128, lr: float = 5e-4, gamma: float = 0.99, 
                 tau: float = 0.005, buffer_size: int = 100000, batch_size: int = 32,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, 
                 epsilon_decay: int = 50000, device: str = 'cpu'):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau
        
        # Epsilon-greedy exploration
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.epsilon = epsilon_start
        self.steps_done = 0
        
        # Create independent agents
        self.agents = nn.ModuleList([
            IQLAgent(obs_dim, action_dim, hidden_dim).to(device) 
            for _ in range(num_agents)
        ])
        
        # Create target networks
        self.target_agents = nn.ModuleList([
            IQLAgent(obs_dim, action_dim, hidden_dim).to(device) 
            for _ in range(num_agents)
        ])
        
        # Initialize target networks
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
            for param in target_agent.parameters():
                param.requires_grad = False
        
        # Independent optimizers for each agent
        self.optimizers = [
            optim.Adam(agent.parameters(), lr=lr) 
            for agent in self.agents
        ]
        
        # Independent replay buffers
        self.replay_buffers = [
            IQLReplayBuffer(buffer_size) 
            for _ in range(num_agents)
        ]
        
        # Training statistics
        self.training_step = 0

    def select_actions(self, obs: List[np.ndarray], avail_actions: List[List[int]] = None, 
                      test_mode: bool = False) -> List[int]:
        """Select actions for all agents independently"""
        actions = []
        
        for i in range(self.num_agents):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.agents[i](obs_tensor)
            
            # Apply action masking
            if avail_actions and avail_actions[i]:
                masked_q_values = q_values.clone()
                masked_q_values[0, [j for j in range(self.action_dim) if j not in avail_actions[i]]] = -float('inf')
                q_values = masked_q_values
            
            # Epsilon-greedy action selection
            if test_mode or np.random.random() > self.epsilon:
                action = q_values.argmax(dim=1).item()
            else:
                if avail_actions and avail_actions[i]:
                    action = np.random.choice(avail_actions[i])
                else:
                    action = np.random.randint(self.action_dim)
            
            actions.append(action)
        
        return actions

    def update_epsilon(self):
        """Update epsilon for exploration"""
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
                      np.exp(-1. * self.steps_done / self.epsilon_decay)

    def store_transition(self, obs: List[np.ndarray], actions: List[int],
                        rewards: List[float], next_obs: List[np.ndarray],
                        dones: List[bool], avail_actions: List[List[int]] = None):
        """Store transitions for all agents independently"""
        for i in range(self.num_agents):
            avail = avail_actions[i] if avail_actions else None
            self.replay_buffers[i].push(
                obs[i], actions[i], rewards[i], next_obs[i], dones[i], avail
            )

    def train_step(self) -> Dict[str, float]:
        """Train all agents independently"""
        if len(self.replay_buffers[0]) < self.batch_size:
            return {'loss': 0.0}
        
        self.training_step += 1
        
        total_loss = 0.0
        agents_trained = 0
        
        # Train each agent independently
        for i in range(self.num_agents):
            batch = self.replay_buffers[i].sample(self.batch_size)
            if batch is None:
                continue
            
            # Convert to tensors
            obs = torch.FloatTensor(batch['obs']).to(self.device)
            actions = torch.LongTensor(batch['actions']).to(self.device)
            rewards = torch.FloatTensor(batch['rewards']).to(self.device)
            next_obs = torch.FloatTensor(batch['next_obs']).to(self.device)
            dones = torch.FloatTensor(batch['dones']).to(self.device)
            
            # Current Q-values
            current_q_values = self.agents[i](obs)
            current_q = current_q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
            
            # Target Q-values
            with torch.no_grad():
                # Double DQN: use main network to select actions, target network to evaluate
                next_q_values = self.agents[i](next_obs)
                next_actions = next_q_values.argmax(dim=1)
                
                target_next_q_values = self.target_agents[i](next_obs)
                target_q = target_next_q_values.gather(1, next_actions.unsqueeze(1)).squeeze(1)
                
                # Compute target
                target = rewards + self.gamma * target_q * (1 - dones)
            
            # Compute loss
            loss = F.mse_loss(current_q, target)
            total_loss += loss.item()
            agents_trained += 1
            
            # Optimize agent
            self.optimizers[i].zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.agents[i].parameters(), 10.0)
            self.optimizers[i].step()
        
        # Update target networks
        self.update_target_networks()
        
        return {'loss': total_loss / max(agents_trained, 1)}

    def update_target_networks(self):
        """Soft update of target networks"""
        for agent, target_agent in zip(self.agents, self.target_agents):
            for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save_models(self, path: str):
        """Save all models"""
        torch.save({
            'agents': [agent.state_dict() for agent in self.agents],
            'target_agents': [agent.state_dict() for agent in self.target_agents],
            'optimizers': [optimizer.state_dict() for optimizer in self.optimizers],
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'training_step': self.training_step
        }, path)

    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path)
        
        for i, agent_state in enumerate(checkpoint['agents']):
            self.agents[i].load_state_dict(agent_state)
        
        for i, target_agent_state in enumerate(checkpoint['target_agents']):
            self.target_agents[i].load_state_dict(target_agent_state)
        
        for i, optimizer_state in enumerate(checkpoint['optimizers']):
            self.optimizers[i].load_state_dict(optimizer_state)
        
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.training_step = checkpoint['training_step']


class IQLEnvironmentWrapper:
    """Environment wrapper for IQL - same as other wrappers but simplified"""

    def __init__(self, env, num_agents: int):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = 6
        self.action_dim = 3  # 0=idle, 1=service, 2=charge

    def reset(self) -> List[np.ndarray]:
        """Reset and return individual observations"""
        env_state = self.env.reset()
        
        obs = []
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(env_state, i)
            obs.append(agent_obs)
        
        return obs

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute actions and return results"""
        env_actions = self._convert_actions_to_env(actions)
        next_env_state, rewards, done, info = self.env.step(env_actions)
        
        next_obs = []
        agent_rewards = []
        agent_dones = []
        
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(next_env_state, i)
            agent_reward = self._extract_agent_reward(rewards, i)
            agent_done = done  # Shared done signal
            
            next_obs.append(agent_obs)
            agent_rewards.append(agent_reward)
            agent_dones.append(agent_done)
        
        return next_obs, agent_rewards, agent_dones, info

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
            # Basic action filtering based on battery level
            battery_level = obs[i][1]  # Battery is second feature
            
            if battery_level < 0.2:
                # Low battery: prioritize charging
                avail_actions.append([0, 2])  # idle or charge
            elif battery_level > 0.8:
                # High battery: avoid charging
                avail_actions.append([0, 1])  # idle or service
            else:
                # Normal: all actions available
                avail_actions.append([0, 1, 2])
        
        return avail_actions