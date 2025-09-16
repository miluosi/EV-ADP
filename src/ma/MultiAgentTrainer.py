"""
Multi-Agent Trainer for MADDPG in EV-ADP System
Integrates MADDPG algorithm with the existing EV charging coordination framework
"""

import torch
import numpy as np
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
from datetime import datetime
import pandas as pd

# Add current directory to path for imports
current_dir = Path(__file__).parent
sys.path.append(str(current_dir))

try:
    from MultiAgentMADDPG import MultiAgentMADDPG, MultiAgentEnvironmentWrapper
    from Environment import ChargingIntegratedEnvironment
except ImportError:
    # Fallback for relative imports when run as module
    try:
        from .MultiAgentMADDPG import MultiAgentMADDPG, MultiAgentEnvironmentWrapper
        from .Environment import ChargingIntegratedEnvironment
    except ImportError:
        # Final fallback - try absolute imports
        import MultiAgentMADDPG
        import Environment
        MultiAgentMADDPG = MultiAgentMADDPG.MultiAgentMADDPG
        MultiAgentEnvironmentWrapper = MultiAgentMADDPG.MultiAgentEnvironmentWrapper
        ChargingIntegratedEnvironment = Environment.ChargingIntegratedEnvironment


class MultiAgentTrainer:
    """
    Trainer for multi-agent MADDPG in EV charging coordination
    """

    def __init__(self, num_vehicles: int = 10, num_stations: int = 6,
                 obs_dim: int = 20, action_dim: int = 5,
                 hidden_dim: int = 256, lr_actor: float = 1e-4, lr_critic: float = 1e-3,
                 gamma: float = 0.95, tau: float = 0.01, buffer_size: int = 100000,
                 batch_size: int = 256, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):

        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = 'cuda'

        # Create multi-agent MADDPG
        self.maddpg = MultiAgentMADDPG(
            num_agents=num_vehicles,
            obs_dim=obs_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            lr_actor=lr_actor,
            lr_critic=lr_critic,
            gamma=gamma,
            tau=tau,
            buffer_size=buffer_size,
            batch_size=batch_size,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Create environment wrapper
        self.env_wrapper = None
        self.env = None

        # Training statistics
        self.training_stats = {
            'episodes': [],
            'rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'charging_events': [],
            'battery_levels': []
        }

    def initialize_environment(self, grid_size: int = 12, use_intense_requests: bool = True):
        """Initialize the multi-agent environment"""
        # Create base environment
        self.env = ChargingIntegratedEnvironment(
            num_vehicles=self.num_vehicles,
            num_stations=self.num_stations,
            grid_size=grid_size,
            use_intense_requests=use_intense_requests
        )

        # Create multi-agent wrapper
        self.env_wrapper = MultiAgentEnvironmentWrapper(self.env, self.num_vehicles)

        print(f"✓ Initialized multi-agent environment with {self.num_vehicles} vehicles and {self.num_stations} stations")

    def train(self, num_episodes: int = 1000, max_steps: int = 1000,
              noise_std_start: float = 0.3, noise_std_end: float = 0.05,
              warmup_steps: int = 1000, train_frequency: int = 10,
              save_frequency: int = 100, save_path: str = "models/maddpg"):
        """
        Train the multi-agent MADDPG system

        Args:
            num_episodes: Number of training episodes
            max_steps: Maximum steps per episode
            noise_std_start: Initial exploration noise
            noise_std_end: Final exploration noise
            warmup_steps: Steps before training starts
            train_frequency: Train every N steps
            save_frequency: Save models every N episodes
            save_path: Path to save models
        """

        if self.env_wrapper is None:
            raise ValueError("Environment not initialized. Call initialize_environment() first.")

        # Create save directory
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)

        print(f"🚀 Starting MADDPG training for {num_episodes} episodes")
        print(f"   - Device: {self.device}")
        print(f"   - Exploration: {noise_std_start:.3f} → {noise_std_end:.3f}")
        print(f"   - Training starts after {warmup_steps} warmup steps")

        total_steps = 0

        for episode in range(num_episodes):
            # Linear noise decay
            noise_std = max(noise_std_end,
                          noise_std_start - (noise_std_start - noise_std_end) * episode / num_episodes)

            # Reset environment
            obs = self.env_wrapper.reset()
            episode_reward = 0
            episode_actor_loss = 0
            episode_critic_loss = 0
            episode_charging_events = 0
            episode_battery_levels = []

            for step in range(max_steps):
                total_steps += 1

                # Select actions
                actions = self.maddpg.select_actions(obs, noise_std)

                # Execute actions
                next_obs, rewards, dones, info = self.env_wrapper.step(actions)

                # Store transition
                self.maddpg.store_transition(obs, actions, rewards, next_obs, dones)

                # Train if warmup is complete
                if len(self.maddpg.replay_buffer) >= warmup_steps and total_steps % train_frequency == 0:
                    train_stats = self.maddpg.train_step()
                    episode_actor_loss += train_stats['actor_loss']
                    episode_critic_loss += train_stats['critic_loss']

                # Update observations and accumulate rewards
                obs = next_obs
                episode_reward += sum(rewards)

                # Track statistics
                episode_charging_events += len(info.get('charging_events', []))
                episode_battery_levels.extend([self.env.vehicles[i]['battery'] for i in range(self.num_vehicles)])

                # Check if episode is done
                if any(dones) or step >= max_steps - 1:
                    break

            # Record episode statistics
            avg_actor_loss = episode_actor_loss / max(1, step // train_frequency)
            avg_critic_loss = episode_critic_loss / max(1, step // train_frequency)
            avg_battery = np.mean(episode_battery_levels) if episode_battery_levels else 0

            self.training_stats['episodes'].append(episode + 1)
            self.training_stats['rewards'].append(episode_reward)
            self.training_stats['actor_losses'].append(avg_actor_loss)
            self.training_stats['critic_losses'].append(avg_critic_loss)
            self.training_stats['charging_events'].append(episode_charging_events)
            self.training_stats['battery_levels'].append(avg_battery)

            # Print progress
            if (episode + 1) % 10 == 0:
                print(f"Episode {episode + 1:4d}: Reward={episode_reward:8.2f}, "
                      f"Actor Loss={avg_actor_loss:.4f}, Critic Loss={avg_critic_loss:.4f}, "
                      f"Battery={avg_battery:.3f}, Charging={episode_charging_events}")

            # Save models periodically
            if (episode + 1) % save_frequency == 0:
                model_path = save_path / f"episode_{episode + 1}"
                model_path.mkdir(exist_ok=True)
                self.maddpg.save_models(str(model_path))
                print(f"💾 Models saved to {model_path}")

        print("✅ Training completed!")
        self._save_training_results(save_path)

    def evaluate(self, num_episodes: int = 10, max_steps: int = 1000,
                 render: bool = False) -> Dict[str, float]:
        """Evaluate the trained policy"""
        if self.env_wrapper is None:
            raise ValueError("Environment not initialized. Call initialize_environment() first.")

        print(f"🔍 Evaluating MADDPG policy for {num_episodes} episodes")

        evaluation_stats = {
            'rewards': [],
            'charging_events': [],
            'battery_levels': [],
            'completed_requests': []
        }

        for episode in range(num_episodes):
            obs = self.env_wrapper.reset()
            episode_reward = 0
            episode_charging_events = 0
            episode_battery_levels = []

            for step in range(max_steps):
                # Select actions (no exploration)
                actions = self.maddpg.select_actions(obs, noise_std=0.0)

                # Execute actions
                next_obs, rewards, dones, info = self.env_wrapper.step(actions)

                # Update statistics
                obs = next_obs
                episode_reward += sum(rewards)
                episode_charging_events += len(info.get('charging_events', []))
                episode_battery_levels.extend([self.env.vehicles[i]['battery'] for i in range(self.num_vehicles)])

                if any(dones) or step >= max_steps - 1:
                    break

            # Record episode statistics
            evaluation_stats['rewards'].append(episode_reward)
            evaluation_stats['charging_events'].append(episode_charging_events)
            evaluation_stats['battery_levels'].append(np.mean(episode_battery_levels))
            evaluation_stats['completed_requests'].append(self.env.get_stats()['completed_requests'])

            print(f"Eval Episode {episode + 1}: Reward={episode_reward:.2f}, "
                  f"Charging={episode_charging_events}, Battery={np.mean(episode_battery_levels):.3f}")

        # Compute averages
        results = {
            'avg_reward': np.mean(evaluation_stats['rewards']),
            'std_reward': np.std(evaluation_stats['rewards']),
            'avg_charging_events': np.mean(evaluation_stats['charging_events']),
            'avg_battery_level': np.mean(evaluation_stats['battery_levels']),
            'avg_completed_requests': np.mean(evaluation_stats['completed_requests'])
        }

        print("📊 Evaluation Results:")
        for key, value in results.items():
            print(f"   {key}: {value:.3f}")

        return results

    def _save_training_results(self, save_path: Path):
        """Save training statistics and plots"""
        # Save statistics to CSV
        df = pd.DataFrame(self.training_stats)
        csv_path = save_path / "training_stats.csv"
        df.to_csv(csv_path, index=False)

        # Create plots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('MADDPG Training Results')

        # Plot rewards
        axes[0, 0].plot(self.training_stats['episodes'], self.training_stats['rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Total Reward')

        # Plot actor losses
        axes[0, 1].plot(self.training_stats['episodes'], self.training_stats['actor_losses'])
        axes[0, 1].set_title('Actor Loss')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Loss')

        # Plot critic losses
        axes[0, 2].plot(self.training_stats['episodes'], self.training_stats['critic_losses'])
        axes[0, 2].set_title('Critic Loss')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Loss')

        # Plot charging events
        axes[1, 0].plot(self.training_stats['episodes'], self.training_stats['charging_events'])
        axes[1, 0].set_title('Charging Events per Episode')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Count')

        # Plot battery levels
        axes[1, 1].plot(self.training_stats['episodes'], self.training_stats['battery_levels'])
        axes[1, 1].set_title('Average Battery Level')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Battery Level')

        # Plot moving averages
        window = 50
        if len(self.training_stats['rewards']) >= window:
            moving_avg = pd.Series(self.training_stats['rewards']).rolling(window=window).mean()
            axes[1, 2].plot(self.training_stats['episodes'], moving_avg)
            axes[1, 2].set_title(f'Moving Average Reward (window={window})')
            axes[1, 2].set_xlabel('Episode')
            axes[1, 2].set_ylabel('Reward')

        plt.tight_layout()
        plot_path = save_path / "training_plots.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"📊 Training results saved to {save_path}")

    def load_models(self, model_path: str):
        """Load trained models"""
        self.maddpg.load_models(model_path)
        print(f"📁 Models loaded from {model_path}")


def run_multi_agent_training(num_vehicles: int = 10, num_stations: int = 6,
                           num_episodes: int = 500, save_path: str = "models/maddpg"):
    """Run multi-agent MADDPG training"""

    # Create trainer
    trainer = MultiAgentTrainer(
        num_vehicles=num_vehicles,
        num_stations=num_stations,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )

    # Initialize environment
    trainer.initialize_environment()

    # Train the model
    trainer.train(
        num_episodes=num_episodes,
        save_frequency=50,
        save_path=save_path
    )

    # Evaluate the trained policy
    eval_results = trainer.evaluate(num_episodes=20)

    return trainer, eval_results


if __name__ == "__main__":
    # Example usage
    trainer, results = run_multi_agent_training(
        num_vehicles=8,
        num_stations=4,
        num_episodes=200
    )