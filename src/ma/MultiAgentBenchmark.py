"""
Unified Multi-Agent Testing Framework
for comparing different algorithms in EV-ADP charging coordination.

This framework allows for fair comparison between MADDPG, QMIX, MAPPO, and IQL
using consistent evaluation metrics and experimental settings.
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import json

# Import all multi-agent algorithms
from MultiAgentMADDPG import MultiAgentMADDPG, MultiAgentEnvironmentWrapper as MADDPGWrapper
from MultiAgentQMIX import QMIX, QMIXEnvironmentWrapper
from MultiAgentMAPPO import MAPPO, MAPPOEnvironmentWrapper  
from MultiAgentIQL import IQL, IQLEnvironmentWrapper
from Environment import ChargingIntegratedEnvironment


class MultiAgentBenchmark:
    """Comprehensive benchmarking suite for multi-agent algorithms"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.results = defaultdict(list)
        self.algorithms = {}
        self.wrappers = {}
        
        # Environment parameters
        self.num_vehicles = config.get('num_vehicles', 4)
        self.num_stations = config.get('num_stations', 3)
        self.grid_size = config.get('grid_size', 8)
        self.device = config.get('device', 'cpu')
        
        # Training parameters
        self.training_episodes = config.get('training_episodes', 100)
        self.evaluation_episodes = config.get('evaluation_episodes', 20)
        self.evaluation_frequency = config.get('evaluation_frequency', 10)
        
        # Observation and action spaces
        self.obs_dim = 6
        self.action_dim_continuous = 5  # For MADDPG
        self.action_dim_discrete = 3    # For QMIX, MAPPO, IQL
        self.state_dim = self.num_vehicles * 6 + 10
        
        # Results directory
        self.results_dir = Path(config.get('results_dir', 'benchmark_results'))
        self.results_dir.mkdir(exist_ok=True)
        
        self.setup_algorithms()

    def setup_algorithms(self):
        """Initialize all algorithms with consistent parameters"""
        
        # MADDPG
        if 'MADDPG' in self.config.get('algorithms', ['MADDPG']):
            self.algorithms['MADDPG'] = MultiAgentMADDPG(
                num_agents=self.num_vehicles,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim_continuous,
                hidden_dim=self.config.get('hidden_dim', 64),
                lr_actor=self.config.get('lr_actor', 1e-4),
                lr_critic=self.config.get('lr_critic', 1e-3),
                gamma=self.config.get('gamma', 0.95),
                tau=self.config.get('tau', 0.01),
                buffer_size=self.config.get('buffer_size', 50000),
                batch_size=self.config.get('batch_size', 64),
                device=self.device
            )
        
        # QMIX
        if 'QMIX' in self.config.get('algorithms', ['QMIX']):
            self.algorithms['QMIX'] = QMIX(
                num_agents=self.num_vehicles,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim_discrete,
                state_dim=self.state_dim,
                hidden_dim=self.config.get('hidden_dim', 64),
                lr=self.config.get('lr_qmix', 5e-4),
                gamma=self.config.get('gamma', 0.95),
                tau=self.config.get('tau', 0.01),
                buffer_size=self.config.get('buffer_size', 50000),
                batch_size=self.config.get('batch_size', 32),
                device=self.device
            )
        
        # MAPPO
        if 'MAPPO' in self.config.get('algorithms', ['MAPPO']):
            self.algorithms['MAPPO'] = MAPPO(
                num_agents=self.num_vehicles,
                obs_dim=self.obs_dim,
                state_dim=self.state_dim,
                action_dim=self.action_dim_discrete,
                hidden_dim=self.config.get('hidden_dim', 64),
                lr_actor=self.config.get('lr_actor', 3e-4),
                lr_critic=self.config.get('lr_critic', 3e-4),
                gamma=self.config.get('gamma', 0.95),
                device=self.device
            )
        
        # IQL
        if 'IQL' in self.config.get('algorithms', ['IQL']):
            self.algorithms['IQL'] = IQL(
                num_agents=self.num_vehicles,
                obs_dim=self.obs_dim,
                action_dim=self.action_dim_discrete,
                hidden_dim=self.config.get('hidden_dim', 64),
                lr=self.config.get('lr_iql', 5e-4),
                gamma=self.config.get('gamma', 0.95),
                tau=self.config.get('tau', 0.01),
                buffer_size=self.config.get('buffer_size', 50000),
                batch_size=self.config.get('batch_size', 32),
                device=self.device
            )

    def create_environment_wrapper(self, algorithm_name: str):
        """Create appropriate environment wrapper for algorithm"""
        base_env = ChargingIntegratedEnvironment(
            num_vehicles=self.num_vehicles,
            num_stations=self.num_stations,
            grid_size=self.grid_size,
            use_intense_requests=True
        )
        
        if algorithm_name == 'MADDPG':
            return MADDPGWrapper(base_env, self.num_vehicles)
        elif algorithm_name == 'QMIX':
            return QMIXEnvironmentWrapper(base_env, self.num_vehicles)
        elif algorithm_name == 'MAPPO':
            return MAPPOEnvironmentWrapper(base_env, self.num_vehicles)
        elif algorithm_name == 'IQL':
            return IQLEnvironmentWrapper(base_env, self.num_vehicles)
        else:
            raise ValueError(f"Unknown algorithm: {algorithm_name}")

    def train_algorithm(self, algorithm_name: str) -> Dict[str, List[float]]:
        """Train a specific algorithm and return training metrics"""
        print(f"\n🚀 Training {algorithm_name}...")
        
        algorithm = self.algorithms[algorithm_name]
        env_wrapper = self.create_environment_wrapper(algorithm_name)
        
        training_metrics = {
            'episode_rewards': [],
            'actor_losses': [],
            'critic_losses': [],
            'charging_events': [],
            'battery_levels': [],
            'episode_times': []
        }
        
        for episode in range(self.training_episodes):
            episode_start_time = time.time()
            episode_reward = 0
            episode_charging_events = 0
            episode_battery_levels = []
            
            # Reset environment
            if algorithm_name == 'MADDPG':
                obs = env_wrapper.reset()
            elif algorithm_name in ['QMIX', 'MAPPO']:
                state, obs = env_wrapper.reset()
            else:  # IQL
                obs = env_wrapper.reset()
            
            done = False
            step_count = 0
            max_steps = self.config.get('max_episode_steps', 200)
            
            while not done and step_count < max_steps:
                # Select actions based on algorithm
                if algorithm_name == 'MADDPG':
                    noise_std = max(0.1, 0.3 * (1 - episode / self.training_episodes))
                    actions = algorithm.select_actions(obs, noise_std)
                    next_obs, rewards, dones, info = env_wrapper.step(actions)
                    
                    algorithm.store_transition(obs, actions, rewards, next_obs, dones)
                    obs = next_obs
                    done = any(dones)
                    
                elif algorithm_name == 'QMIX':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions = algorithm.select_actions(obs, avail_actions)
                    next_state, next_obs, rewards, dones, info = env_wrapper.step(actions)
                    
                    algorithm.store_transition(state, obs, actions, rewards, next_state, next_obs, dones, avail_actions)
                    state, obs = next_state, next_obs
                    done = any(dones)
                    
                elif algorithm_name == 'MAPPO':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions, log_probs, value = algorithm.select_actions(obs, state, avail_actions)
                    next_state, next_obs, rewards, done, info = env_wrapper.step(actions)
                    
                    algorithm.store_transition(obs, state, actions, log_probs, rewards, value, done, avail_actions)
                    state, obs = next_state, next_obs
                    
                elif algorithm_name == 'IQL':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions = algorithm.select_actions(obs, avail_actions)
                    next_obs, rewards, dones, info = env_wrapper.step(actions)
                    
                    algorithm.store_transition(obs, actions, rewards, next_obs, dones, avail_actions)
                    obs = next_obs
                    done = any(dones)
                
                # Collect metrics
                episode_reward += sum(rewards)
                episode_charging_events += info.get('charging_events', 0)
                
                # Collect battery levels
                for ob in obs:
                    episode_battery_levels.append(ob[1])  # Battery is second feature
                
                step_count += 1
            
            # Train algorithm
            if algorithm_name == 'MADDPG':
                if len(algorithm.replay_buffer) > algorithm.batch_size:
                    losses = algorithm.train_step()
                    training_metrics['actor_losses'].append(losses.get('actor_loss', 0))
                    training_metrics['critic_losses'].append(losses.get('critic_loss', 0))
                else:
                    training_metrics['actor_losses'].append(0)
                    training_metrics['critic_losses'].append(0)
                    
            elif algorithm_name == 'QMIX':
                algorithm.update_epsilon()
                if len(algorithm.replay_buffer) > algorithm.batch_size:
                    losses = algorithm.train_step()
                    training_metrics['critic_losses'].append(losses.get('loss', 0))
                    training_metrics['actor_losses'].append(0)  # QMIX doesn't have separate actor
                else:
                    training_metrics['actor_losses'].append(0)
                    training_metrics['critic_losses'].append(0)
                    
            elif algorithm_name == 'MAPPO':
                if episode % 10 == 9:  # Update every 10 episodes
                    if algorithm.buffer.size > 0:
                        losses = algorithm.update(state)
                        training_metrics['actor_losses'].append(losses.get('actor_loss', 0))
                        training_metrics['critic_losses'].append(losses.get('critic_loss', 0))
                    else:
                        training_metrics['actor_losses'].append(0)
                        training_metrics['critic_losses'].append(0)
                else:
                    training_metrics['actor_losses'].append(0)
                    training_metrics['critic_losses'].append(0)
                    
            elif algorithm_name == 'IQL':
                algorithm.update_epsilon()
                losses = algorithm.train_step()
                training_metrics['critic_losses'].append(losses.get('loss', 0))
                training_metrics['actor_losses'].append(0)  # IQL doesn't have separate actor
            
            # Record episode metrics
            training_metrics['episode_rewards'].append(episode_reward)
            training_metrics['charging_events'].append(episode_charging_events)
            training_metrics['battery_levels'].append(np.mean(episode_battery_levels) if episode_battery_levels else 0.5)
            training_metrics['episode_times'].append(time.time() - episode_start_time)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(training_metrics['episode_rewards'][-10:])
                avg_charging = np.mean(training_metrics['charging_events'][-10:])
                avg_battery = np.mean(training_metrics['battery_levels'][-10:])
                print(f"  Episode {episode + 1:3d}: Reward={avg_reward:6.2f}, Charging={avg_charging:.1f}, Battery={avg_battery:.3f}")
        
        print(f"✅ {algorithm_name} training completed!")
        return training_metrics

    def evaluate_algorithm(self, algorithm_name: str, num_episodes: int = None) -> Dict[str, float]:
        """Evaluate a trained algorithm"""
        if num_episodes is None:
            num_episodes = self.evaluation_episodes
            
        print(f"🔍 Evaluating {algorithm_name} for {num_episodes} episodes...")
        
        algorithm = self.algorithms[algorithm_name]
        env_wrapper = self.create_environment_wrapper(algorithm_name)
        
        episode_rewards = []
        charging_events = []
        battery_levels = []
        completed_requests = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            episode_charging = 0
            episode_battery_levels = []
            episode_requests = 0
            
            # Reset environment
            if algorithm_name == 'MADDPG':
                obs = env_wrapper.reset()
            elif algorithm_name in ['QMIX', 'MAPPO']:
                state, obs = env_wrapper.reset()
            else:  # IQL
                obs = env_wrapper.reset()
            
            done = False
            step_count = 0
            max_steps = self.config.get('max_episode_steps', 200)
            
            while not done and step_count < max_steps:
                # Select actions (deterministic for evaluation)
                if algorithm_name == 'MADDPG':
                    actions = algorithm.select_actions(obs, noise_std=0.0)
                    next_obs, rewards, dones, info = env_wrapper.step(actions)
                    obs = next_obs
                    done = any(dones)
                    
                elif algorithm_name == 'QMIX':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions = algorithm.select_actions(obs, avail_actions, test_mode=True)
                    next_state, next_obs, rewards, dones, info = env_wrapper.step(actions)
                    state, obs = next_state, next_obs
                    done = any(dones)
                    
                elif algorithm_name == 'MAPPO':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions, _, _ = algorithm.select_actions(obs, state, avail_actions, deterministic=True)
                    next_state, next_obs, rewards, done, info = env_wrapper.step(actions)
                    state, obs = next_state, next_obs
                    
                elif algorithm_name == 'IQL':
                    avail_actions = env_wrapper.get_avail_actions(obs)
                    actions = algorithm.select_actions(obs, avail_actions, test_mode=True)
                    next_obs, rewards, dones, info = env_wrapper.step(actions)
                    obs = next_obs
                    done = any(dones)
                
                # Collect metrics
                episode_reward += sum(rewards)
                episode_charging += info.get('charging_events', 0)
                episode_requests += info.get('completed_requests', 0)
                
                # Collect battery levels
                for ob in obs:
                    episode_battery_levels.append(ob[1])
                
                step_count += 1
            
            episode_rewards.append(episode_reward)
            charging_events.append(episode_charging)
            battery_levels.append(np.mean(episode_battery_levels) if episode_battery_levels else 0.5)
            completed_requests.append(episode_requests)
            episode_lengths.append(step_count)
        
        # Compute evaluation metrics
        eval_metrics = {
            'avg_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'avg_charging_events': np.mean(charging_events),
            'std_charging_events': np.std(charging_events),
            'avg_battery_level': np.mean(battery_levels),
            'std_battery_level': np.std(battery_levels),
            'avg_completed_requests': np.mean(completed_requests),
            'std_completed_requests': np.std(completed_requests),
            'avg_episode_length': np.mean(episode_lengths),
            'std_episode_length': np.std(episode_lengths)
        }
        
        print(f"📊 {algorithm_name} Evaluation Results:")
        print(f"   Avg Reward: {eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}")
        print(f"   Avg Charging: {eval_metrics['avg_charging_events']:.1f} ± {eval_metrics['std_charging_events']:.1f}")
        print(f"   Avg Battery: {eval_metrics['avg_battery_level']:.3f} ± {eval_metrics['std_battery_level']:.3f}")
        
        return eval_metrics

    def run_benchmark(self) -> Dict[str, Any]:
        """Run complete benchmark comparing all algorithms"""
        print("🏁 Starting Multi-Agent Algorithm Benchmark")
        print(f"Environment: {self.num_vehicles} vehicles, {self.num_stations} stations, {self.grid_size}x{self.grid_size} grid")
        print(f"Algorithms: {list(self.algorithms.keys())}")
        print(f"Training episodes: {self.training_episodes}, Evaluation episodes: {self.evaluation_episodes}")
        
        benchmark_results = {}
        
        for algorithm_name in self.algorithms.keys():
            print(f"\n{'='*60}")
            print(f"BENCHMARKING {algorithm_name}")
            print(f"{'='*60}")
            
            start_time = time.time()
            
            # Train algorithm
            training_metrics = self.train_algorithm(algorithm_name)
            
            # Evaluate algorithm
            eval_metrics = self.evaluate_algorithm(algorithm_name)
            
            # Save model
            model_path = self.results_dir / f"{algorithm_name}_model.pth"
            self.algorithms[algorithm_name].save_models(str(model_path))
            
            training_time = time.time() - start_time
            
            benchmark_results[algorithm_name] = {
                'training_metrics': training_metrics,
                'evaluation_metrics': eval_metrics,
                'training_time': training_time
            }
        
        # Save results
        self.save_results(benchmark_results)
        
        # Generate comparison plots
        self.plot_comparison(benchmark_results)
        
        # Print summary
        self.print_summary(benchmark_results)
        
        return benchmark_results

    def save_results(self, results: Dict[str, Any]):
        """Save benchmark results to files"""
        # Save raw results as JSON
        json_results = {}
        for alg, data in results.items():
            json_results[alg] = {
                'evaluation_metrics': data['evaluation_metrics'],
                'training_time': data['training_time']
            }
        
        with open(self.results_dir / 'benchmark_results.json', 'w') as f:
            json.dump(json_results, f, indent=2)
        
        # Save training curves as CSV
        for alg, data in results.items():
            df = pd.DataFrame(data['training_metrics'])
            df.to_csv(self.results_dir / f'{alg}_training_curves.csv', index=False)

    def plot_comparison(self, results: Dict[str, Any]):
        """Generate comparison plots"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Multi-Agent Algorithm Comparison', fontsize=16)
        
        algorithms = list(results.keys())
        
        # Plot 1: Training curves (rewards)
        ax1 = axes[0, 0]
        for alg in algorithms:
            rewards = results[alg]['training_metrics']['episode_rewards']
            smoothed_rewards = pd.Series(rewards).rolling(10, min_periods=1).mean()
            ax1.plot(smoothed_rewards, label=alg, linewidth=2)
        ax1.set_title('Training Rewards')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Reward')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Evaluation metrics comparison
        ax2 = axes[0, 1]
        metrics = ['avg_reward', 'avg_charging_events', 'avg_battery_level']
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(algorithms)
        
        for i, alg in enumerate(algorithms):
            values = [results[alg]['evaluation_metrics'][m] for m in metrics]
            ax2.bar(x_pos + i * width, values, width, label=alg)
        
        ax2.set_title('Evaluation Metrics')
        ax2.set_xlabel('Metrics')
        ax2.set_xticks(x_pos + width * (len(algorithms) - 1) / 2)
        ax2.set_xticklabels(['Reward', 'Charging', 'Battery'])
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Battery levels over training
        ax3 = axes[1, 0]
        for alg in algorithms:
            battery_levels = results[alg]['training_metrics']['battery_levels']
            smoothed_battery = pd.Series(battery_levels).rolling(10, min_periods=1).mean()
            ax3.plot(smoothed_battery, label=alg, linewidth=2)
        ax3.set_title('Battery Levels During Training')
        ax3.set_xlabel('Episode')
        ax3.set_ylabel('Average Battery Level')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Training time comparison
        ax4 = axes[1, 1]
        training_times = [results[alg]['training_time'] for alg in algorithms]
        bars = ax4.bar(algorithms, training_times)
        ax4.set_title('Training Time Comparison')
        ax4.set_ylabel('Time (seconds)')
        
        # Add value labels on bars
        for bar, time_val in zip(bars, training_times):
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2., height,
                    f'{time_val:.1f}s', ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig(self.results_dir / 'algorithm_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_summary(self, results: Dict[str, Any]):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print("BENCHMARK SUMMARY")
        print(f"{'='*80}")
        
        # Create summary table
        summary_data = []
        for alg in results.keys():
            eval_metrics = results[alg]['evaluation_metrics']
            training_time = results[alg]['training_time']
            
            summary_data.append({
                'Algorithm': alg,
                'Avg Reward': f"{eval_metrics['avg_reward']:.3f} ± {eval_metrics['std_reward']:.3f}",
                'Charging Events': f"{eval_metrics['avg_charging_events']:.1f} ± {eval_metrics['std_charging_events']:.1f}",
                'Battery Level': f"{eval_metrics['avg_battery_level']:.3f} ± {eval_metrics['std_battery_level']:.3f}",
                'Training Time': f"{training_time:.1f}s"
            })
        
        df_summary = pd.DataFrame(summary_data)
        print(df_summary.to_string(index=False))
        
        # Save summary
        df_summary.to_csv(self.results_dir / 'benchmark_summary.csv', index=False)
        
        # Find best algorithm
        best_reward_alg = max(results.keys(), key=lambda x: results[x]['evaluation_metrics']['avg_reward'])
        best_charging_alg = max(results.keys(), key=lambda x: results[x]['evaluation_metrics']['avg_charging_events'])
        fastest_alg = min(results.keys(), key=lambda x: results[x]['training_time'])
        
        print(f"\n🏆 Best Performance:")
        print(f"   Highest Reward: {best_reward_alg} ({results[best_reward_alg]['evaluation_metrics']['avg_reward']:.3f})")
        print(f"   Most Charging: {best_charging_alg} ({results[best_charging_alg]['evaluation_metrics']['avg_charging_events']:.1f})")
        print(f"   Fastest Training: {fastest_alg} ({results[fastest_alg]['training_time']:.1f}s)")
        
        print(f"\n📁 Results saved to: {self.results_dir}")


# Example usage configuration
DEFAULT_BENCHMARK_CONFIG = {
    'algorithms': ['MADDPG', 'QMIX', 'MAPPO', 'IQL'],
    'num_vehicles': 4,
    'num_stations': 3,
    'grid_size': 8,
    'training_episodes': 100,
    'evaluation_episodes': 20,
    'max_episode_steps': 200,
    'hidden_dim': 64,
    'batch_size': 32,
    'buffer_size': 50000,
    'gamma': 0.95,
    'tau': 0.01,
    'lr_actor': 3e-4,
    'lr_critic': 3e-4,
    'lr_qmix': 5e-4,
    'lr_iql': 5e-4,
    'device': 'cpu',
    'results_dir': 'benchmark_results'
}


def run_benchmark_demo():
    """Run a quick demo of the benchmarking framework"""
    print("🚀 Running Multi-Agent Benchmark Demo")
    
    # Quick demo config
    demo_config = DEFAULT_BENCHMARK_CONFIG.copy()
    demo_config.update({
        'algorithms': ['MADDPG', 'IQL'],  # Just two algorithms for quick demo
        'training_episodes': 20,
        'evaluation_episodes': 5,
        'results_dir': 'demo_benchmark_results'
    })
    
    benchmark = MultiAgentBenchmark(demo_config)
    results = benchmark.run_benchmark()
    
    return results


if __name__ == "__main__":
    # Run demo
    run_benchmark_demo()