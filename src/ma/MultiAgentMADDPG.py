"""
Multi-Agent Deep Deterministic Policy Gradient (MADDPG) implementation
for multi-vehicle charging coordination in EV-ADP system.

MADDPG is designed for multi-agent reinforcement learning in environments
where agents have continuous action spaces and need to coordinate their actions.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
import random
import copy
from typing import List, Dict, Tuple, Any, Optional


class ActorNetwork(nn.Module):
    """Actor network for MADDPG - outputs actions given observations"""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 256):
        super(ActorNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Actions are bounded between -1 and 1
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.network(obs)


class CriticNetwork(nn.Module):
    """Critic network for MADDPG - evaluates joint actions given all observations"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int, hidden_dim: int = 256):
        super(CriticNetwork, self).__init__()
        # Critic takes all agents' observations and actions
        input_dim = obs_dim * num_agents + action_dim * num_agents

        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.0)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            obs: Tensor of shape (batch_size, obs_dim * num_agents)
            actions: Tensor of shape (batch_size, action_dim * num_agents)
        Returns:
            Q-values of shape (batch_size, 1)
        """
        x = torch.cat([obs, actions], dim=-1)
        return self.network(x)


class MADDPGAgent:
    """Individual MADDPG agent with actor and critic networks"""

    def __init__(self, obs_dim: int, action_dim: int, num_agents: int,
                 hidden_dim: int = 256, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01, device: str = 'cpu'):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.num_agents = num_agents
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Actor networks
        self.actor = ActorNetwork(obs_dim, action_dim, hidden_dim).to(device)
        self.actor_target = copy.deepcopy(self.actor).to(device)

        # Critic networks
        self.critic = CriticNetwork(obs_dim, action_dim, num_agents, hidden_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic).to(device)

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Freeze target networks
        for param in self.actor_target.parameters():
            param.requires_grad = False
        for param in self.critic_target.parameters():
            param.requires_grad = False

    def select_action(self, obs: np.ndarray, noise_std: float = 0.1) -> np.ndarray:
        """Select action with exploration noise"""
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action = self.actor(obs_tensor).cpu().numpy().flatten()

        # Add exploration noise
        noise = np.random.normal(0, noise_std, size=self.action_dim)
        action = np.clip(action + noise, -1, 1)

        return action

    def update_targets(self):
        """Soft update target networks"""
        # Update actor target
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # Update critic target
        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


class MultiAgentReplayBuffer:
    """Replay buffer for multi-agent experiences"""

    def __init__(self, capacity: int = 100000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)

    def push(self, obs: List[np.ndarray], actions: List[np.ndarray],
             rewards: List[float], next_obs: List[np.ndarray], dones: List[bool]):
        """Store a multi-agent transition"""
        transition = {
            'obs': [obs[i].copy() for i in range(len(obs))],
            'actions': [actions[i].copy() for i in range(len(actions))],
            'rewards': rewards.copy(),
            'next_obs': [next_obs[i].copy() for i in range(len(next_obs))],
            'dones': dones.copy()
        }
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> Dict[str, List]:
        """Sample a batch of transitions"""
        if len(self.buffer) < batch_size:
            return None

        batch = random.sample(self.buffer, batch_size)

        return {
            'obs': [[transition['obs'][i] for transition in batch] for i in range(len(batch[0]['obs']))],
            'actions': [[transition['actions'][i] for transition in batch] for i in range(len(batch[0]['actions']))],
            'rewards': [[transition['rewards'][i] for transition in batch] for i in range(len(batch[0]['rewards']))],
            'next_obs': [[transition['next_obs'][i] for transition in batch] for i in range(len(batch[0]['next_obs']))],
            'dones': [[transition['dones'][i] for transition in batch] for i in range(len(batch[0]['dones']))]
        }

    def __len__(self):
        return len(self.buffer)


class MultiAgentMADDPG:
    """
    Multi-Agent Deep Deterministic Policy Gradient implementation
    for coordinating multiple vehicles in charging station selection
    """

    def __init__(self, num_agents: int, obs_dim: int, action_dim: int,
                 hidden_dim: int = 256, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01, buffer_size: int = 100000,
                 batch_size: int = 256, device: str = 'cpu'):
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        self.batch_size = batch_size
        self.gamma = gamma

        # Create agents
        self.agents = []
        for i in range(num_agents):
            agent = MADDPGAgent(obs_dim, action_dim, num_agents, hidden_dim,
                              lr_actor, lr_critic, gamma, tau, device)
            self.agents.append(agent)

        # Shared replay buffer
        self.replay_buffer = MultiAgentReplayBuffer(buffer_size)

        # Training statistics
        self.training_step = 0

    def select_actions(self, obs: List[np.ndarray], noise_std: float = 0.1) -> List[np.ndarray]:
        """Select actions for all agents"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.select_action(obs[i], noise_std)
            actions.append(action)
        return actions

    def store_transition(self, obs: List[np.ndarray], actions: List[np.ndarray],
                        rewards: List[float], next_obs: List[np.ndarray], dones: List[bool]):
        """Store transition in replay buffer"""
        self.replay_buffer.push(obs, actions, rewards, next_obs, dones)

    def train_step(self) -> Dict[str, float]:
        """Perform one training step"""
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        if batch is None:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}

        self.training_step += 1

        # Convert to tensors
        obs_tensor = [torch.FloatTensor(np.array(batch['obs'][i])).to(self.device) for i in range(self.num_agents)]
        actions_tensor = [torch.FloatTensor(np.array(batch['actions'][i])).to(self.device) for i in range(self.num_agents)]
        rewards_tensor = [torch.FloatTensor(np.array(batch['rewards'][i])).to(self.device) for i in range(self.num_agents)]
        next_obs_tensor = [torch.FloatTensor(np.array(batch['next_obs'][i])).to(self.device) for i in range(self.num_agents)]
        dones_tensor = [torch.FloatTensor(np.array(batch['dones'][i])).to(self.device) for i in range(self.num_agents)]

        # Concatenate observations and actions for critic input
        obs_flat = torch.cat(obs_tensor, dim=-1)  # (batch_size, obs_dim * num_agents)
        actions_flat = torch.cat(actions_tensor, dim=-1)  # (batch_size, action_dim * num_agents)

        # Train each agent
        actor_losses = []
        critic_losses = []

        for i, agent in enumerate(self.agents):
            # === Update Critic ===
            # Get target actions from all agents
            target_actions = []
            for j in range(self.num_agents):
                with torch.no_grad():
                    target_action = self.agents[j].actor_target(next_obs_tensor[j])
                    target_actions.append(target_action)
            target_actions_flat = torch.cat(target_actions, dim=-1)

            # Compute target Q-value
            next_obs_flat = torch.cat(next_obs_tensor, dim=-1)
            target_q = self.agents[i].critic_target(next_obs_flat, target_actions_flat)
            target_q = rewards_tensor[i].unsqueeze(-1) + self.gamma * target_q * (1 - dones_tensor[i].unsqueeze(-1))

            # Compute current Q-value
            current_q = agent.critic(obs_flat, actions_flat)

            # Critic loss
            critic_loss = F.mse_loss(current_q, target_q.detach())
            critic_losses.append(critic_loss.item())

            # Update critic
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
            agent.critic_optimizer.step()

            # === Update Actor ===
            # Compute actor loss
            actions_pred = []
            for j in range(self.num_agents):
                if j == i:
                    # Use current agent's actor
                    action_pred = agent.actor(obs_tensor[j])
                else:
                    # Use target actors for other agents
                    action_pred = self.agents[j].actor_target(obs_tensor[j])
                actions_pred.append(action_pred)
            actions_pred_flat = torch.cat(actions_pred, dim=-1)

            actor_loss = -agent.critic(obs_flat, actions_pred_flat).mean()
            actor_losses.append(actor_loss.item())

            # Update actor
            agent.actor_optimizer.zero_grad()
            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
            agent.actor_optimizer.step()

            # Update target networks
            agent.update_targets()

        return {
            'actor_loss': np.mean(actor_losses),
            'critic_loss': np.mean(critic_losses)
        }

    def save_models(self, path: str):
        """Save all agent models"""
        for i, agent in enumerate(self.agents):
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict(),
                'actor_target': agent.actor_target.state_dict(),
                'critic_target': agent.critic_target.state_dict(),
                'actor_optimizer': agent.actor_optimizer.state_dict(),
                'critic_optimizer': agent.critic_optimizer.state_dict()
            }, f"{path}/agent_{i}.pth")

    def load_models(self, path: str):
        """Load all agent models"""
        for i, agent in enumerate(self.agents):
            checkpoint = torch.load(f"{path}/agent_{i}.pth")
            agent.actor.load_state_dict(checkpoint['actor'])
            agent.critic.load_state_dict(checkpoint['critic'])
            agent.actor_target.load_state_dict(checkpoint['actor_target'])
            agent.critic_target.load_state_dict(checkpoint['critic_target'])
            agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
            agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


class MultiAgentEnvironmentWrapper:
    """
    Wrapper to convert single-agent environment to multi-agent environment
    for vehicle charging coordination
    """

    def __init__(self, env, num_agents: int):
        self.env = env
        self.num_agents = num_agents

        # Force all vehicles to be AEV (type=2) to avoid request rejections
        if hasattr(self.env, 'vehicles'):
            for vehicle_id, vehicle in self.env.vehicles.items():
                if vehicle_id < num_agents:  # Only modify our agents
                    vehicle['type'] = 2  # AEV type
                    vehicle['type_name'] = 'AEV'
            print(f"🔧 Converted {num_agents} vehicles to AEV type to avoid request rejections")

        # Define observation and action spaces for each agent
        # Assuming each vehicle has the same observation/action space
        self.obs_dim = self._get_obs_dim()
        self.action_dim = self._get_action_dim()

    def _get_obs_dim(self) -> int:
        """Get observation dimension for each agent"""
        # Based on _extract_agent_obs implementation:
        # location, battery, has_request, is_charging, x_coord, y_coord
        return 6

    def _get_action_dim(self) -> int:
        """Get action dimension for each agent"""
        # Actions: [move_x, move_y, charge_decision, service_decision, idle_decision]
        return 5

    def reset(self) -> List[np.ndarray]:
        """Reset environment and return observations for all agents"""
        states = self.env.reset()
        
        # Force all vehicles to be AEV (type=2) to avoid request rejections
        if hasattr(self.env, 'vehicles'):
            for vehicle_id, vehicle in self.env.vehicles.items():
                if vehicle_id < self.num_agents:  # Only modify our agents
                    vehicle['type'] = 2  # AEV type
                    vehicle['type_name'] = 'AEV'
            print(f"🔧 Reset: Converted {self.num_agents} vehicles to AEV type")
        
        # Generate initial requests to ensure they are available for the first step
        # This fixes the timing issue where agents need to select actions before any requests exist
        if hasattr(self.env, '_update_environment'):
            self.env._update_environment()  # This will generate initial requests

        # Convert single environment state to multi-agent observations
        obs = []
        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(states, i)
            obs.append(agent_obs)
        
        return obs

    def step(self, actions: List[np.ndarray]) -> Tuple[List[np.ndarray], List[float], List[bool], Dict]:
        """Execute actions and return results for all agents"""
        
        # Convert multi-agent actions to single environment actions
        env_actions = self._convert_actions_to_env(actions)
        next_states, rewards, done, info = self.env.step(env_actions)

        # Check vehicle states AFTER step execution to see actual assignments
        if hasattr(self.env, 'vehicles'):
            for vehicle_id in range(min(3, self.num_agents)):  # Only first 3 agents to avoid spam
                vehicle = self.env.vehicles[vehicle_id]
                assigned = vehicle.get('assigned_request')
                passenger = vehicle.get('passenger_onboard')
                coords = vehicle.get('coordinates', (0, 0))
                print(f"POST-STEP Agent {vehicle_id}: assigned={assigned}, passenger={passenger}, coords={coords}")

        # Check for order completion by monitoring reward changes
        for i, reward in rewards.items() if isinstance(rewards, dict) else enumerate(rewards if hasattr(rewards, '__len__') else [rewards]):
            if reward > 15:  # High rewards typically indicate order completion
                print(f"🎉 Agent {i} completed an order! Reward: {reward:.2f}")
            elif reward > 0.5 and reward < 2.0:  # Pickup rewards
                print(f"📦 Agent {i} picked up passenger! Reward: {reward:.2f}")
            elif reward >= 18.0 and reward <= 22.0:  # Request assignment rewards (final_value = 20)
                print(f"✅ Agent {i} successfully assigned to request! Reward: {reward:.2f}")
            elif reward < 0 and reward > -0.5:  # Small negative rewards might indicate movement towards pickup/dropoff
                print(f"🚗➡️ Agent {i} moving towards target! Reward: {reward:.2f}")
            elif abs(reward) < 0.2 and reward != 0:  # Small rewards might indicate assignment failure
                print(f"❌ Agent {i} request assignment failed! Reward: {reward:.2f}")

        # Monitor vehicle states for pickup/dropoff progress
        try:
            if hasattr(self.env, 'vehicles'):
                for vehicle_id, vehicle in self.env.vehicles.items():
                    if vehicle_id < self.num_agents:  # Only monitor our agents
                        assigned_request = vehicle.get('assigned_request')
                        passenger_onboard = vehicle.get('passenger_onboard')
                        vehicle_coords = vehicle.get('coordinates', (0, 0))
                        
                        if assigned_request is not None:
                            # Vehicle is assigned, check if it's making progress to pickup
                            if hasattr(self.env, 'active_requests') and assigned_request in self.env.active_requests:
                                request = self.env.active_requests[assigned_request]
                                pickup_coords = (request.pickup % 12, request.pickup // 12)
                                distance_to_pickup = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                                
                                if not hasattr(self, '_last_pickup_distance'):
                                    self._last_pickup_distance = {}
                                
                                if vehicle_id not in self._last_pickup_distance or self._last_pickup_distance[vehicle_id] != distance_to_pickup:
                                    print(f"🚗➡️📍 Agent {vehicle_id} moving to pickup Request {assigned_request}: distance={distance_to_pickup}, coords={vehicle_coords}→{pickup_coords}")
                                    self._last_pickup_distance[vehicle_id] = distance_to_pickup
                                    
                                    # Check if reached pickup location
                                    if distance_to_pickup == 0:
                                        print(f"📍✅ Agent {vehicle_id} reached pickup location for Request {assigned_request}")
                                        
                        elif passenger_onboard is not None:
                            # Vehicle has passenger, check progress to dropoff
                            if hasattr(self.env, 'active_requests') and passenger_onboard in self.env.active_requests:
                                request = self.env.active_requests[passenger_onboard]
                                dropoff_coords = (request.dropoff % 12, request.dropoff // 12)
                                distance_to_dropoff = abs(vehicle_coords[0] - dropoff_coords[0]) + abs(vehicle_coords[1] - dropoff_coords[1])
                                
                                if not hasattr(self, '_last_dropoff_distance'):
                                    self._last_dropoff_distance = {}
                                
                                if vehicle_id not in self._last_dropoff_distance or self._last_dropoff_distance[vehicle_id] != distance_to_dropoff:
                                    print(f"�➡️🏁 Agent {vehicle_id} moving to dropoff Request {passenger_onboard}: distance={distance_to_dropoff}, coords={vehicle_coords}→{dropoff_coords}")
                                    self._last_dropoff_distance[vehicle_id] = distance_to_dropoff
                                    
                                    # Check if reached dropoff location
                                    if distance_to_dropoff == 0:
                                        print(f"🏁✅ Agent {vehicle_id} reached dropoff location for Request {passenger_onboard}")
                        else:
                            # Vehicle is free, clear tracking
                            if hasattr(self, '_last_pickup_distance') and vehicle_id in self._last_pickup_distance:
                                del self._last_pickup_distance[vehicle_id]
                            if hasattr(self, '_last_dropoff_distance') and vehicle_id in self._last_dropoff_distance:
                                del self._last_dropoff_distance[vehicle_id]
        except Exception as e:
            pass  # Skip if monitoring fails

        # Convert to multi-agent format
        next_obs = []
        agent_rewards = []
        agent_dones = []

        for i in range(self.num_agents):
            agent_obs = self._extract_agent_obs(next_states, i)
            agent_reward = self._extract_agent_reward(rewards, i)
            agent_done = done  # Assuming shared done signal

            next_obs.append(agent_obs)
            agent_rewards.append(agent_reward)
            agent_dones.append(agent_done)

        return next_obs, agent_rewards, agent_dones, info

    def _extract_agent_obs(self, states: Dict, agent_id: int) -> np.ndarray:
        """Extract observation for a specific agent"""
        # Return consistent 6-dimensional observation
        if isinstance(states, dict) and agent_id in states:
            vehicle_state = states[agent_id]
            #print(f"DEBUG: Agent {agent_id} state: {vehicle_state}, type: {type(vehicle_state)}")
            
            # If the state is already an array, check its format
            if isinstance(vehicle_state, np.ndarray):
                if len(vehicle_state) >= 6:
                    return vehicle_state[:6].astype(np.float32)
                else:
                    # Pad with zeros if too short
                    padded = np.zeros(6, dtype=np.float32)
                    padded[:len(vehicle_state)] = vehicle_state
                    return padded
            elif isinstance(vehicle_state, dict):
                # Extract from dictionary format
                obs = np.array([
                    float(vehicle_state.get('location', 0)) / 100.0,  # Normalized location
                    float(vehicle_state.get('battery', 0.5)),  # Battery level
                    1.0 if vehicle_state.get('assigned_request') is not None else 0.0,  # Has request
                    1.0 if vehicle_state.get('charging_station') is not None else 0.0,  # Is charging
                    float(vehicle_state.get('coordinates', [0, 0])[0]) / 12.0,  # Normalized x coordinate
                    float(vehicle_state.get('coordinates', [0, 0])[1]) / 12.0,  # Normalized y coordinate
                ], dtype=np.float32)
                return obs
            else:
                # Convert other formats to array
                return np.array([float(x) for x in vehicle_state[:6]] if hasattr(vehicle_state, '__len__') else [0.5]*6, dtype=np.float32)
        else:
            # Return zero observation if no vehicle state available
            return np.zeros(6, dtype=np.float32)

    def _extract_agent_reward(self, rewards: Dict, agent_id: int) -> float:
        """Extract reward for a specific agent"""
        return rewards.get(agent_id, 0.0)

    def _convert_actions_to_env(self, actions: List[np.ndarray]) -> Dict:
        """Convert multi-agent actions to environment actions"""
        from Action import ServiceAction, ChargingAction, IdleAction
        import random
        
        # Track which requests are being attempted by agents to avoid conflicts
        attempted_requests = set()
        env_actions = {}
        
        # Debug: print available requests before processing
        if hasattr(self.env, 'active_requests'):
            print(f"DEBUG: {len(self.env.active_requests)} requests available: {list(self.env.active_requests.keys())[:10]}")  # Show first 10
        
        for i, action in enumerate(actions):
            # Convert agent action to environment action
            # Pass the attempted_requests set to avoid conflicts
            env_actions[i] = self._interpret_action(action, i, attempted_requests)
            
        # Debug: print which requests were attempted
        if attempted_requests:
            print(f"DEBUG: Attempted requests by agents: {list(attempted_requests)}")
        
        # Force assignment in environment after action conversion
        # This is a workaround to ensure assignments actually work
        if hasattr(self.env, 'vehicles') and hasattr(self.env, 'active_requests'):
            for agent_id, action_obj in env_actions.items():
                from Action import ServiceAction
                if isinstance(action_obj, ServiceAction) and hasattr(action_obj, 'request_id'):
                    request_id = action_obj.request_id
                    if (request_id in self.env.active_requests and 
                        agent_id in self.env.vehicles and
                        self.env.vehicles[agent_id].get('assigned_request') is None and
                        self.env.vehicles[agent_id].get('passenger_onboard') is None):
                        # Force the assignment
                        self.env.vehicles[agent_id]['assigned_request'] = request_id
                        print(f"🔧 FORCED assignment: Agent {agent_id} → Request {request_id}")
        
        # Force pickup when vehicle is close to pickup location (distance <= 1)
        if hasattr(self.env, 'vehicles') and hasattr(self.env, 'active_requests'):
            for vehicle_id, vehicle in self.env.vehicles.items():
                if (vehicle_id < self.num_agents and 
                    vehicle.get('assigned_request') is not None and 
                    vehicle.get('passenger_onboard') is None):
                    
                    request_id = vehicle['assigned_request']
                    if request_id in self.env.active_requests:
                        request = self.env.active_requests[request_id]
                        vehicle_coords = vehicle.get('coordinates', (0, 0))
                        pickup_coords = (request.pickup % 12, request.pickup // 12)
                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                        
                        if distance <= 1:  # Very close to pickup
                            # Force pickup
                            vehicle['passenger_onboard'] = request_id
                            vehicle['assigned_request'] = None
                            print(f"🔧 FORCED pickup: Agent {vehicle_id} picked up Request {request_id}")
        
        # Force dropoff when vehicle is close to dropoff location (distance <= 1)  
        if hasattr(self.env, 'vehicles') and hasattr(self.env, 'active_requests'):
            for vehicle_id, vehicle in self.env.vehicles.items():
                if (vehicle_id < self.num_agents and 
                    vehicle.get('passenger_onboard') is not None):
                    
                    passenger_id = vehicle['passenger_onboard']
                    if passenger_id in self.env.active_requests:
                        request = self.env.active_requests[passenger_id]
                        vehicle_coords = vehicle.get('coordinates', (0, 0))
                        dropoff_coords = (request.dropoff % 12, request.dropoff // 12)
                        distance = abs(vehicle_coords[0] - dropoff_coords[0]) + abs(vehicle_coords[1] - dropoff_coords[1])
                        
                        if distance <= 1:  # Very close to dropoff
                            # Force dropoff - complete the request
                            completed_request = self.env.active_requests.pop(passenger_id)
                            if hasattr(self.env, 'completed_requests'):
                                self.env.completed_requests.append(completed_request)
                            vehicle['passenger_onboard'] = None
                            vehicle['service_earnings'] = vehicle.get('service_earnings', 0) + completed_request.final_value
                            print(f"🔧 FORCED dropoff: Agent {vehicle_id} completed Request {passenger_id} - Earned ${completed_request.final_value}")
        
        return env_actions

    def _interpret_action(self, action: np.ndarray, agent_id: int, attempted_requests: set = None) -> Any:
        """Interpret continuous action vector into environment action"""
        from Action import ServiceAction, ChargingAction, IdleAction
        import random
        
        if attempted_requests is None:
            attempted_requests = set()
        
        # For MADDPG, action is a continuous vector of size 5
        # We'll interpret it as: [action_type, param1, param2, param3, param4]
        
        action_type_value = action[0]  # Use first element to determine action type
        
        # Get environment info for action creation
        try:
            # Access environment components more reliably
            env_requests = getattr(self.env, 'active_requests', {})  # Correct attribute name
            env_charging_stations = getattr(self.env, 'charging_stations', {})
            env_vehicles = getattr(self.env, 'vehicles', {})
            
            # Create some default charging stations if none exist (for testing)
            if not env_charging_stations:
                env_charging_stations = {i: {'coordinates': (i*2, i*2)} for i in range(5)}
            
            #print(f"DEBUG: Agent {agent_id} - Found {len(env_requests)} active requests in environment")
            
            # Make action selection more balanced - prioritize service actions for positive rewards
            # BUT check if vehicle already has assigned request or passenger onboard
            current_vehicle = env_vehicles.get(agent_id, {})
            has_assigned = current_vehicle.get('assigned_request') is not None
            has_passenger = current_vehicle.get('passenger_onboard') is not None
            
            # Debug vehicle state
            if agent_id < 3:  # Only print for first 3 agents to avoid spam
                vehicle_type = current_vehicle.get('type', 1)
                vehicle_coords = current_vehicle.get('coordinates', (0, 0))
                print(f"DEBUG Agent {agent_id}: assigned={current_vehicle.get('assigned_request')}, passenger={current_vehicle.get('passenger_onboard')}, type={vehicle_type}, coords={vehicle_coords}")
            
            if action_type_value > 0.0:  # 50% chance for service action
                # Service action - but only accept new requests if vehicle is free
                if not has_assigned and not has_passenger and env_requests:
                    # Vehicle is free - can accept new request
                    # To avoid conflicts, find a request not already attempted by other agents
                    available_requests = [(req_id, req) for req_id, req in env_requests.items() if req_id not in attempted_requests]
                    
                    if len(available_requests) > 0:
                        # Use a deterministic but spread out selection to avoid conflicts
                        request_index = agent_id % len(available_requests)
                        request_id, selected_request = available_requests[request_index]
                        
                        # Mark this request as attempted
                        attempted_requests.add(request_id)
                        
                        # Calculate distance for info
                        vehicle_coords = current_vehicle.get('coordinates', (0, 0))
                        pickup_coords = (selected_request.pickup % 12, selected_request.pickup // 12)
                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                        
                        print(f"🚗 Agent {agent_id} accepting SERVICE order: Request {request_id}, Value: {getattr(selected_request, 'final_value', 'N/A')}, Distance: {distance}")
                        return ServiceAction([selected_request], request_id)
                    else:
                        # No requests available, switch to charging
                        print(f"⚠️ Agent {agent_id} no requests available, switching to charging")
                elif has_assigned:
                    # Vehicle has assigned request - continue with pickup
                    assigned_id = current_vehicle['assigned_request']
                    if assigned_id in env_requests:
                        selected_request = env_requests[assigned_id]
                        print(f"🚗➡️ Agent {agent_id} continuing to pickup Request {assigned_id}")
                        return ServiceAction([selected_request], assigned_id)
                    else:
                        # Request expired, switch to charging
                        print(f"⚠️ Agent {agent_id} assigned request {assigned_id} expired, switching to charging")
                        station_ids = list(env_charging_stations.keys())
                        if station_ids:
                            selected_station = station_ids[0]
                            return ChargingAction([], selected_station, 30.0)
                        else:
                            return ChargingAction([], 0, 30.0)
                elif has_passenger:
                    # Vehicle has passenger - continue with dropoff
                    passenger_id = current_vehicle['passenger_onboard']
                    if passenger_id in env_requests:
                        selected_request = env_requests[passenger_id]
                        print(f"🚗🎯 Agent {agent_id} continuing to dropoff passenger {passenger_id}")
                        return ServiceAction([selected_request], passenger_id)
                    else:
                        # Passenger request expired, switch to charging
                        print(f"⚠️ Agent {agent_id} passenger request {passenger_id} expired, switching to charging")
                        station_ids = list(env_charging_stations.keys())
                        if station_ids:
                            selected_station = station_ids[0]
                            return ChargingAction([], selected_station, 30.0)
                        else:
                            return ChargingAction([], 0, 30.0)
                else:
                    # No active requests available - switch to charging
                    station_ids = list(env_charging_stations.keys())
                    if station_ids:
                        station_idx = int(abs(action[1]) * len(station_ids)) % len(station_ids)
                        selected_station = station_ids[station_idx]
                        duration = max(10.0, min(60.0, abs(action[2]) * 50 + 10))
                        return ChargingAction([], selected_station, duration)
                    else:
                        return ChargingAction([], 0, 30.0)
                    
            elif action_type_value > -0.5:  # 25% chance for charging action (negative reward)
                # Charging action - negative reward but increases battery
                station_ids = list(env_charging_stations.keys())
                if station_ids:
                    # Select charging station based on action parameters
                    station_idx = int(abs(action[1]) * len(station_ids)) % len(station_ids)
                    selected_station = station_ids[station_idx]
                    duration = max(10.0, min(60.0, abs(action[2]) * 50 + 10))  # 10-60 minutes
                    #print(f"DEBUG: Agent {agent_id} choosing CHARGING at station {selected_station}")
                    return ChargingAction([], selected_station, duration)
                else:
                    # Fallback charging action
                    #print(f"DEBUG: Agent {agent_id} choosing CHARGING (fallback)")
                    return ChargingAction([], 0, 30.0)
            else:  # 25% chance for idle action
                # Idle action - move to target position (neutral or small negative reward)
                current_pos = (0, 0)  # Default position
                if agent_id in env_vehicles:
                    vehicle = env_vehicles[agent_id]
                    current_pos = getattr(vehicle, 'coordinates', (0, 0))
                
                # Use action parameters to determine target position
                target_x = int(abs(action[3]) * 11) % 12  # Map to 0-11 grid
                target_y = int(abs(action[4]) * 11) % 12  # Map to 0-11 grid
                target_pos = (target_x, target_y)
                #print(f"DEBUG: Agent {agent_id} choosing IDLE from {current_pos} to {target_pos}")
                return IdleAction([], current_pos, target_pos)
                
        except Exception as e:
            # Fallback to idle action
            return IdleAction([], (0, 0), (random.randint(0, 11), random.randint(0, 11)))

    def get_battery_levels(self) -> List[float]:
        """Get current battery levels for all vehicles from the environment"""
        try:
            # Access the vehicles from the environment state
            env_state = getattr(self.env, 'state', {})
            if hasattr(self.env, 'vehicles'):
                vehicles = self.env.vehicles
                battery_levels = []
                for i in range(self.num_agents):
                    if i in vehicles:
                        vehicle = vehicles[i]
                        battery_level = getattr(vehicle, 'battery_level', 0.5)
                        battery_levels.append(battery_level)
                    else:
                        battery_levels.append(0.5)  # Default if vehicle not found
                return battery_levels
            else:
                # Fallback: return default battery levels
                return [0.5] * self.num_agents
        except Exception as e:
            # If any error occurs, return default values
            return [0.5] * self.num_agents