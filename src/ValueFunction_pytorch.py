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
    
    def update(self, *args, **kwargs):
        """No learning required for this value function"""
        pass
    
    def remember(self, *args, **kwargs):
        """No experience storage required"""
        pass


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
        
        # State embedding layers
        state_input_dim = lstm_hidden + embedding_dim + 2  # path + time + other_agents + num_requests
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
                num_requests: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network
        
        Args:
            path_locations: [batch_size, seq_len] - Location IDs in path
            path_delays: [batch_size, seq_len, 1] - Delay information
            current_time: [batch_size, 1] - Current time
            other_agents: [batch_size, 1] - Number of other agents nearby
            num_requests: [batch_size, 1] - Number of current requests
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
        
        # Combine all features
        combined_features = torch.cat([
            path_representation,
            time_embed,
            other_agents,
            num_requests
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
