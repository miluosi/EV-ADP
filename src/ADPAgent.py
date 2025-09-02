import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import time
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class ADPAgent:
    """Enhanced ADP Agent for decision making"""
    
    def __init__(self, state_dim, action_dim, learning_rate=0.001, epsilon=0.1, gamma=0.95):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon  # Exploration rate
        self.gamma = gamma      # Discount factor
        self.learning_rate = learning_rate
        
        # Neural network for Q-value approximation
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.q_network = self._build_network().to(self.device)
        self.target_network = self._build_network().to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay buffer
        self.memory = deque(maxlen=10000)
        self.batch_size = 32
        self.update_target_freq = 100
        self.step_count = 0
        
        # Training statistics
        self.training_losses = []
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        
        print(f"✓ Initialized ADP Agent with device: {self.device}")
    
    def _build_network(self):
        """Build the Q-network"""
        return nn.Sequential(
            nn.Linear(self.state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, self.action_dim)
        )
    
    def get_action_values(self, state):
        """Get Q-values for all actions given a state"""
        if isinstance(state, np.ndarray):
            state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            q_values = self.q_network(state)
        
        return q_values.cpu().numpy().flatten()
    
    def select_action(self, state, available_actions=None):
        """Select action using epsilon-greedy policy with ADP guidance"""
        if random.random() < self.epsilon:
            # Exploration: random action
            if available_actions:
                return random.choice(available_actions), "random"
            else:
                return random.randint(0, self.action_dim - 1), "random"
        else:
            # Exploitation: ADP-guided action
            q_values = self.get_action_values(state)
            
            if available_actions:
                # Filter Q-values for available actions only
                available_q_values = [(i, q_values[i]) for i in available_actions]
                best_action = max(available_q_values, key=lambda x: x[1])[0]
            else:
                best_action = np.argmax(q_values)
            
            return best_action, "adp"
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def train(self):
        """Train the Q-network using experience replay"""
        if len(self.memory) < self.batch_size:
            return 0.0
        
        # Sample batch from memory
        batch = random.sample(self.memory, self.batch_size)
        states = torch.FloatTensor([e[0] for e in batch]).to(self.device)
        actions = torch.LongTensor([e[1] for e in batch]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in batch]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in batch]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in batch]).to(self.device)
        
        # Current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q-values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # Update target network
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        
        loss_value = loss.item()
        self.training_losses.append(loss_value)
        return loss_value
    
    def save_model(self, path):
        """Save the trained model"""
        torch.save({
            'q_network': self.q_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_losses': self.training_losses
        }, path)
    
    def load_model(self, path):
        """Load a trained model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network'])
        self.target_network.load_state_dict(checkpoint['target_network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.epsilon = checkpoint['epsilon']
        self.training_losses = checkpoint.get('training_losses', [])

