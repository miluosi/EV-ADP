"""
QMIX (Monotonic Value Function Factorisation) implementation
for multi-vehicle charging coordination in EV-ADP system.

QMIX learns decentralized policies with centralized training using
a monotonic mixing network to combine individual Q-values.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
from typing import List, Dict, Tuple, Any, Optional


class QMIXAgent(nn.Module):
    """Individual agent Q-network for QMIX"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 128):
        super(QMIXAgent, self).__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
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
        """Return Q-values for all actions given observation"""
        return self.network(obs)


class MixingNetwork(nn.Module):
    """Monotonic mixing network that combines individual Q-values"""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 32, hypernet_layers: int = 2, hypernet_embed: int = 64):
        super(MixingNetwork, self).__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        
        # Hypernetworks for generating mixing network weights
        self.hyper_w_1 = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, num_agents * embed_dim)
        )
        
        self.hyper_w_final = nn.Sequential(
            nn.Linear(state_dim, hypernet_embed),
            nn.ReLU(),
            nn.Linear(hypernet_embed, embed_dim)
        )
        
        # State-dependent bias
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        
        # V(s) - state value function
        self.V = nn.Sequential(
            nn.Linear(state_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """
        Mixing network forward pass
        Args:
            agent_qs: Individual Q-values [batch_size, num_agents]
            states: Global state [batch_size, state_dim]
        Returns:
            Mixed Q-value [batch_size, 1]
        """
        batch_size = agent_qs.size(0)
        
        # Generate weights for first layer (ensure monotonicity with abs)
        w1 = torch.abs(self.hyper_w_1(states))  # [batch_size, num_agents * embed_dim]
        w1 = w1.view(batch_size, self.num_agents, self.embed_dim)  # [batch_size, num_agents, embed_dim]
        
        # Generate bias for first layer
        b1 = self.hyper_b_1(states)  # [batch_size, embed_dim]
        b1 = b1.view(batch_size, 1, self.embed_dim)  # [batch_size, 1, embed_dim]
        
        # First layer computation
        agent_qs = agent_qs.view(batch_size, 1, self.num_agents)  # [batch_size, 1, num_agents]
        hidden = F.elu(torch.bmm(agent_qs, w1) + b1)  # [batch_size, 1, embed_dim]
        
        # Generate weights for final layer (ensure monotonicity with abs)
        w_final = torch.abs(self.hyper_w_final(states))  # [batch_size, embed_dim]
        w_final = w_final.view(batch_size, self.embed_dim, 1)  # [batch_size, embed_dim, 1]
        
        # Final layer computation
        y = torch.bmm(hidden, w_final)  # [batch_size, 1, 1]
        
        # Add state-dependent bias V(s)
        v = self.V(states).view(batch_size, 1, 1)  # [batch_size, 1, 1]
        
        return (y + v).view(batch_size, 1)  # [batch_size, 1]


class QMIXReplayBuffer:
    """Experience replay buffer for QMIX"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, obs: List[np.ndarray], actions: List[int],
             rewards: List[float], next_state: np.ndarray, next_obs: List[np.ndarray], 
             dones: List[bool], avail_actions: List[List[int]] = None):
        """Store a multi-agent transition"""
        transition = {
            'state': state.copy(),
            'obs': [o.copy() for o in obs],
            'actions': actions.copy(),
            'rewards': rewards.copy(),
            'next_state': next_state.copy(),
            'next_obs': [o.copy() for o in next_obs],
            'dones': dones.copy(),
            'avail_actions': avail_actions.copy() if avail_actions else None
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict:
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)
        
        return {
            'state': np.array([t['state'] for t in batch]),
            'obs': np.array([[t['obs'][i] for t in batch] for i in range(len(batch[0]['obs']))]),
            'actions': np.array([[t['actions'][i] for t in batch] for i in range(len(batch[0]['actions']))]),
            'rewards': np.array([[t['rewards'][i] for t in batch] for i in range(len(batch[0]['rewards']))]),
            'next_state': np.array([t['next_state'] for t in batch]),
            'next_obs': np.array([[t['next_obs'][i] for t in batch] for i in range(len(batch[0]['next_obs']))]),
            'dones': np.array([[t['dones'][i] for t in batch] for i in range(len(batch[0]['dones']))]),
            'avail_actions': np.array([[t['avail_actions'][i] for t in batch] for i in range(len(batch[0]['avail_actions']))]) if batch[0]['avail_actions'] else None
        }

    def __len__(self):
        return len(self.buffer)


class QMIX:
    """
    QMIX implementation for multi-agent coordination
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int, state_dim: int,
                 hidden_dim: int = 128, lr: float = 5e-4, gamma: float = 0.99, 
                 tau: float = 0.005, buffer_size: int = 100000, batch_size: int = 32,
                 epsilon_start: float = 1.0, epsilon_end: float = 0.05, epsilon_decay: int = 50000,
                 device: str = 'cpu'):
        
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
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
        
        # Create agent networks
        self.agents = nn.ModuleList([
            QMIXAgent(obs_dim, action_dim, hidden_dim).to(device) 
            for _ in range(num_agents)
        ])
        
        # Create target agent networks
        self.target_agents = nn.ModuleList([
            QMIXAgent(obs_dim, action_dim, hidden_dim).to(device) 
            for _ in range(num_agents)
        ])
        
        # Initialize target networks
        for agent, target_agent in zip(self.agents, self.target_agents):
            target_agent.load_state_dict(agent.state_dict())
            for param in target_agent.parameters():
                param.requires_grad = False
        
        # Create mixing networks
        self.mixer = MixingNetwork(num_agents, state_dim).to(device)
        self.target_mixer = MixingNetwork(num_agents, state_dim).to(device)
        
        # Initialize target mixer
        self.target_mixer.load_state_dict(self.mixer.state_dict())
        for param in self.target_mixer.parameters():
            param.requires_grad = False
        
        # Optimizers
        self.agent_params = list(self.agents.parameters())
        self.mixer_params = list(self.mixer.parameters())
        self.params = self.agent_params + self.mixer_params
        self.optimizer = optim.Adam(self.params, lr=lr)
        
        # Replay buffer
        self.replay_buffer = QMIXReplayBuffer(buffer_size)
        
        # Training statistics
        self.training_step = 0

    def select_actions(self, obs: List[np.ndarray], avail_actions: List[List[int]] = None, 
                      test_mode: bool = False) -> List[int]:
        """Select actions using epsilon-greedy policy"""
        actions = []
        
        for i in range(self.num_agents):
            obs_tensor = torch.FloatTensor(obs[i]).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                q_values = self.agents[i](obs_tensor)
            
            # Apply action masking if available
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

    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {'loss': 0.0}
        
        self.training_step += 1
        
        # Convert to tensors
        state = torch.FloatTensor(batch['state']).to(self.device)
        obs = [torch.FloatTensor(batch['obs'][i]).to(self.device) for i in range(self.num_agents)]
        actions = [torch.LongTensor(batch['actions'][i]).to(self.device) for i in range(self.num_agents)]
        rewards = torch.FloatTensor(batch['rewards']).sum(dim=0).to(self.device)  # Sum individual rewards
        next_state = torch.FloatTensor(batch['next_state']).to(self.device)
        next_obs = [torch.FloatTensor(batch['next_obs'][i]).to(self.device) for i in range(self.num_agents)]
        dones = torch.FloatTensor(batch['dones']).max(dim=0)[0].to(self.device)  # Episode done if any agent done
        
        # Current Q-values
        current_q_values = []
        for i in range(self.num_agents):
            q_vals = self.agents[i](obs[i])
            current_q = q_vals.gather(1, actions[i].unsqueeze(1)).squeeze(1)
            current_q_values.append(current_q)
        
        current_q_values = torch.stack(current_q_values, dim=1)  # [batch_size, num_agents]
        
        # Current mixed Q-value
        current_q_total = self.mixer(current_q_values, state)
        
        # Target Q-values
        with torch.no_grad():
            target_q_values = []
            for i in range(self.num_agents):
                target_q_vals = self.target_agents[i](next_obs[i])
                target_q = target_q_vals.max(dim=1)[0]
                target_q_values.append(target_q)
            
            target_q_values = torch.stack(target_q_values, dim=1)  # [batch_size, num_agents]
            
            # Target mixed Q-value
            target_q_total = self.target_mixer(target_q_values, next_state)
            
            # Compute target
            targets = rewards.unsqueeze(1) + self.gamma * target_q_total * (1 - dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_total, targets)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.params, 10.0)
        self.optimizer.step()
        
        # Update target networks
        self.update_target_networks()
        
        return {'loss': loss.item()}

    def update_target_networks(self):
        """Soft update of target networks"""
        for agent, target_agent in zip(self.agents, self.target_agents):
            for target_param, param in zip(target_agent.parameters(), agent.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.target_mixer.parameters(), self.mixer.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def store_transition(self, state: np.ndarray, obs: List[np.ndarray], actions: List[int],
                        rewards: List[float], next_state: np.ndarray, next_obs: List[np.ndarray],
                        dones: List[bool], avail_actions: List[List[int]] = None):
        """Store transition in replay buffer"""
        self.replay_buffer.push(state, obs, actions, rewards, next_state, next_obs, dones, avail_actions)

    def save_models(self, path: str):
        """Save all models"""
        torch.save({
            'agents': [agent.state_dict() for agent in self.agents],
            'target_agents': [agent.state_dict() for agent in self.target_agents],
            'mixer': self.mixer.state_dict(),
            'target_mixer': self.target_mixer.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done
        }, path)

    def load_models(self, path: str):
        """Load all models"""
        checkpoint = torch.load(path)
        
        for i, agent_state in enumerate(checkpoint['agents']):
            self.agents[i].load_state_dict(agent_state)
        
        for i, target_agent_state in enumerate(checkpoint['target_agents']):
            self.target_agents[i].load_state_dict(target_agent_state)
        
        self.mixer.load_state_dict(checkpoint['mixer'])
        self.target_mixer.load_state_dict(checkpoint['target_mixer'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']


class QMIXEnvironmentWrapper:
    """Environment wrapper for QMIX that provides global state"""

    def __init__(self, env, num_agents: int):
        self.env = env
        self.num_agents = num_agents
        self.obs_dim = 6  # Same as MADDPG wrapper
        self.action_dim = 3  # Discrete actions: 0=idle, 1=service, 2=charge
        self.state_dim = self._get_state_dim()

    def _get_state_dim(self) -> int:
        """Get global state dimension"""
        # Global state includes: all vehicle positions, batteries, charging station status
        return self.num_agents * 6 + 10  # Vehicle states + global info

    def reset(self) -> Tuple[np.ndarray, List[np.ndarray]]:
        """Reset and return global state and individual observations"""
        env_state = self.env.reset()
        
        # Extract global state
        global_state = self._extract_global_state(env_state)
        
        # Extract individual observations
        obs = []
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(env_state, i)
            obs.append(agent_obs)
        
        return global_state, obs

    def step(self, actions: List[int]) -> Tuple[np.ndarray, List[np.ndarray], List[float], List[bool], Dict]:
        """Execute actions and return results"""
        # Convert discrete actions to environment actions
        env_actions = self._convert_actions_to_env(actions)
        
        next_env_state, rewards, done, info = self.env.step(env_actions)
        
        # Extract next global state and observations
        next_global_state = self._extract_global_state(next_env_state)
        next_obs = []
        agent_rewards = []
        agent_dones = []
        
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(next_env_state, i)
            agent_reward = self._extract_agent_reward(rewards, i)
            agent_done = done  # Assuming shared done signal
            
            next_obs.append(agent_obs)
            agent_rewards.append(agent_reward)
            agent_dones.append(agent_done)
        
        return next_global_state, next_obs, agent_rewards, agent_dones, info

    def _extract_global_state(self, env_state: Dict) -> np.ndarray:
        """Extract global state from environment state"""
        state_features = []
        
        # Add all vehicle states
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
        
        # Add global information (charging stations, time, etc.)
        state_features.extend([
            float(env_state.get('current_time', 0)) / 100.0,
            float(len(env_state.get('pending_requests', []))) / 10.0,
            float(env_state.get('total_reward', 0)) / 100.0,
            # Add more global features as needed
        ])
        
        # Pad to fixed size
        while len(state_features) < self.state_dim:
            state_features.append(0.0)
        
        return np.array(state_features[:self.state_dim], dtype=np.float32)

    def _extract_agent_obs(self, env_state: Dict, agent_id: int) -> np.ndarray:
        """Extract individual agent observation (same as MADDPG wrapper)"""
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
        # For simplicity, assume all actions are always available
        # In practice, this could depend on battery level, location, etc.
        avail_actions = []
        for i in range(self.num_agents):
            # All agents can always idle, service, or charge
            avail_actions.append([0, 1, 2])
        
        return avail_actions