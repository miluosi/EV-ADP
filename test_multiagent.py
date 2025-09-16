"""
Multi-Agent Reinforcement Learning Performance Test
Test various multi-agent algorithms (MADDPG, QMIX, MAPPO, IQL) on EV charging coordination
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
from datetime import datetime

# Add src to path for imports
sys.path.append('src')
sys.path.append('src/ma')

# Import multi-agent algorithms from src
try:
    from src.ma.MultiAgentMADDPG import MultiAgentMADDPG, MultiAgentEnvironmentWrapper
    from src.ma.MultiAgentQMIX import QMIX, QMIXEnvironmentWrapper
    from src.ma.MultiAgentMAPPO import MAPPO, MAPPOEnvironmentWrapper
    from src.ma.MultiAgentIQL import IQL, IQLEnvironmentWrapper
    from src.Environment import ChargingIntegratedEnvironment
    print("✅ Successfully imported all multi-agent algorithms from src")
    
    # Debug: Print what we actually imported
    print(f"MADDPG class: {MultiAgentMADDPG}")
    print(f"QMIX class: {QMIX}")
    print(f"MAPPO class: {MAPPO}")
    print(f"IQL class: {IQL}")
    
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

# Set matplotlib fonts for better display
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

class MultiAgentTestSuite:
    """Comprehensive test suite for multi-agent algorithms in EV charging environment"""
    
    def __init__(self, num_vehicles=10, num_stations=10, grid_size=12):
        """Initialize test suite with environment configuration"""
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.grid_size = grid_size
        
        # Create the charging environment
        self.env = ChargingIntegratedEnvironment(
            num_vehicles=num_vehicles,
            num_stations=num_stations,
            grid_size=grid_size
        )
        
        # Algorithm configurations
        self.algorithms = {
            'MADDPG': {
                'class': MultiAgentMADDPG,
                'wrapper_class': MultiAgentEnvironmentWrapper,
                'params': {
                    'num_agents': num_vehicles,
                    'obs_dim': 6,
                    'action_dim': 5,
                    'lr_actor': 0.001,
                    'lr_critic': 0.002,
                    'gamma': 0.99,
                    'tau': 0.01,
                    'buffer_size': 100000
                }
            },
            'QMIX': {
                'class': QMIX,
                'wrapper_class': QMIXEnvironmentWrapper,
                'params': {
                    'num_agents': num_vehicles,
                    'obs_dim': 6,
                    'action_dim': 5,
                    'state_dim': 6 * num_vehicles,  # Combined state dimension
                    'lr': 0.001,
                    'gamma': 0.99,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.1,
                    'epsilon_decay': 50000,
                    'buffer_size': 100000
                }
            },
            'MAPPO': {
                'class': MAPPO,
                'wrapper_class': MAPPOEnvironmentWrapper,
                'params': {
                    'num_agents': num_vehicles,
                    'obs_dim': 6,
                    'state_dim': 6 * num_vehicles,  # Combined state dimension
                    'action_dim': 5,
                    'lr_actor': 0.0003,
                    'lr_critic': 0.0003,
                    'gamma': 0.99,
                    'gae_lambda': 0.95,
                    'clip_param': 0.2,
                    'value_loss_coef': 0.5,
                    'entropy_coef': 0.01
                }
            },
            'IQL': {
                'class': IQL,
                'wrapper_class': IQLEnvironmentWrapper,
                'params': {
                    'num_agents': num_vehicles,
                    'obs_dim': 6,
                    'action_dim': 5,
                    'lr': 0.001,
                    'gamma': 0.99,
                    'epsilon_start': 1.0,
                    'epsilon_end': 0.1,
                    'epsilon_decay': 50000,
                    'buffer_size': 100000
                }
            }
        }
        
        # Results storage
        self.results = {}
    
    def train_algorithm(self, algorithm_name, num_episodes=200, evaluation_interval=50):
        """Train a specific multi-agent algorithm"""
        print(f"\n{'='*60}")
        print(f"Training {algorithm_name} with {self.num_vehicles} vehicles and {self.num_stations} stations")
        print(f"{'='*60}")
        
        # Get algorithm configuration
        config = self.algorithms[algorithm_name]
        
        # Create wrapped environment
        wrapped_env = config['wrapper_class'](self.env, self.num_vehicles)
        
        # Initialize algorithm
        print(f"Initializing {algorithm_name} with params: {config['params']}")
        try:
            agent = config['class'](**config['params'])
        except Exception as e:
            print(f"Error creating {algorithm_name}: {e}")
            import traceback
            traceback.print_exc()
            raise e
        
        # Training metrics
        episode_rewards = []
        charging_events = []
        battery_levels = []
        training_losses = []
        evaluation_rewards = []
        
        start_time = time.time()
        
        for episode in range(num_episodes):
            # Reset environment and get initial observations
            try:
                observations = wrapped_env.reset()
            except Exception as e:
                print(f"Error in reset: {e}")
                import traceback
                traceback.print_exc()
                raise e
                
            episode_reward = 0
            episode_charging = 0
            done = False
            step_count = 0
            episode_loss = []
            
            while not done and step_count < 500:  # Max steps per episode
                # Get actions from agent (handle different API signatures)
                if algorithm_name == 'MAPPO':
                    # MAPPO needs state as well as observations
                    state = np.concatenate([np.array(obs).flatten() for obs in observations])
                    actions, _, _ = agent.select_actions(observations, state)
                else:
                    # MADDPG, QMIX, IQL use just observations
                    actions = agent.select_actions(observations)
                
                # Take step in environment
                next_observations, rewards, done, info = wrapped_env.step(actions)
                
                # Store experience (handle different storage methods)
                dones = [done] * self.num_vehicles if not isinstance(done, list) else done
                
                if algorithm_name == 'IQL':
                    # IQL stores transitions per agent
                    agent.store_transition(observations, actions, rewards, next_observations, dones)
                else:
                    # MADDPG, QMIX, MAPPO use unified storage
                    agent.store_transition(observations, actions, rewards, next_observations, dones)
                
                # Train the agent (handle different training methods and buffer checks)
                if algorithm_name == 'IQL':
                    if len(agent.replay_buffers[0]) > agent.batch_size:
                        loss_dict = agent.train_step()
                        if loss_dict and 'total_loss' in loss_dict:
                            episode_loss.append(loss_dict['total_loss'])
                elif algorithm_name == 'MAPPO':
                    # MAPPO trains at the end of episodes, not every step
                    pass
                else:
                    # MADDPG, QMIX
                    if len(agent.replay_buffer) > agent.batch_size:
                        loss_dict = agent.train_step()
                        if loss_dict and 'total_loss' in loss_dict:
                            episode_loss.append(loss_dict['total_loss'])
                
                # Update for next step
                observations = next_observations
                episode_reward += sum(rewards) if isinstance(rewards, list) else rewards
                
                # Count charging events
                if isinstance(info, dict) and 'charging_events' in info:
                    episode_charging += len(info['charging_events'])
                
                step_count += 1
            
            # Record episode metrics
            episode_rewards.append(episode_reward)
            charging_events.append(episode_charging)
            
            # Get environment stats
            if hasattr(wrapped_env, 'get_battery_levels'):
                battery_levels.append(np.mean(wrapped_env.get_battery_levels()))
            else:
                battery_levels.append(0.5)  # Default if not available
                
            if episode_loss:
                training_losses.append(np.mean(episode_loss))
            
            # Print progress
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                avg_charging = np.mean(charging_events[-20:])
                avg_battery = np.mean(battery_levels[-20:]) if battery_levels else 0
                print(f"Episode {episode + 1:3d}: Reward={avg_reward:6.2f}, "
                      f"Charging={avg_charging:4.1f}, Battery={avg_battery:5.2f}")
            
            # Evaluation
            if (episode + 1) % evaluation_interval == 0:
                eval_reward = self.evaluate_algorithm(agent, wrapped_env, num_eval_episodes=5, algorithm_name=algorithm_name)
                evaluation_rewards.append(eval_reward)
                print(f"Evaluation at episode {episode + 1}: {eval_reward:.2f}")
        
        training_time = time.time() - start_time
        
        # Store results
        self.results[algorithm_name] = {
            'episode_rewards': episode_rewards,
            'charging_events': charging_events,
            'battery_levels': battery_levels,
            'training_losses': training_losses,
            'evaluation_rewards': evaluation_rewards,
            'training_time': training_time,
            'final_reward': np.mean(episode_rewards[-10:]),
            'agent': agent,
            'wrapped_env': wrapped_env
        }
        
        print(f"\n{algorithm_name} training completed in {training_time:.1f}s")
        print(f"Final average reward: {self.results[algorithm_name]['final_reward']:.2f}")
        
        return agent, wrapped_env
    
    def evaluate_algorithm(self, agent, wrapped_env, num_eval_episodes=10, algorithm_name=''):
        """Evaluate an algorithm's performance"""
        eval_rewards = []
        
        for episode in range(num_eval_episodes):
            observations = wrapped_env.reset()
            episode_reward = 0
            done = False
            step_count = 0
            
            while not done and step_count < 500:
                # Get actions without exploration (handle different API signatures)
                if algorithm_name == 'MAPPO':
                    # MAPPO needs state as well as observations
                    state = np.concatenate([np.array(obs).flatten() for obs in observations])
                    actions, _, _ = agent.select_actions(observations, state, deterministic=True)
                elif algorithm_name in ['QMIX', 'IQL']:
                    # QMIX and IQL support test_mode
                    actions = agent.select_actions(observations, test_mode=True)
                else:
                    # MADDPG
                    actions = agent.select_actions(observations)
                    
                observations, rewards, done, _ = wrapped_env.step(actions)
                episode_reward += sum(rewards) if isinstance(rewards, list) else rewards
                step_count += 1
            
            eval_rewards.append(episode_reward)
        
        return np.mean(eval_rewards)
    
    def run_all_algorithms(self, num_episodes=200):
        """Train and evaluate all algorithms"""
        print(f"\n🚗 Starting Multi-Agent EV Charging Test Suite 🔋")
        print(f"Configuration: {self.num_vehicles} vehicles, {self.num_stations} stations")
        print(f"Grid size: {self.grid_size}x{self.grid_size}")
        
        for algorithm_name in ['MADDPG']:  # Test only MADDPG for debugging
            try:
                self.train_algorithm(algorithm_name, num_episodes)
            except Exception as e:
                print(f"❌ Error training {algorithm_name}: {e}")
                continue
        
        # Generate comparison report
        self.generate_comparison_report()
        self.plot_results()
        
        return self.results
    
    def generate_comparison_report(self):
        """Generate a comparison report of all algorithms"""
        print(f"\n{'='*80}")
        print("MULTI-AGENT ALGORITHM PERFORMANCE COMPARISON")
        print(f"{'='*80}")
        
        if not self.results:
            print("No results to compare")
            return
        
        # Create comparison table
        comparison_data = []
        for algo_name, results in self.results.items():
            comparison_data.append({
                'Algorithm': algo_name,
                'Final Reward': f"{results['final_reward']:.2f}",
                'Training Time (s)': f"{results['training_time']:.1f}",
                'Avg Charging Events': f"{np.mean(results['charging_events']):.1f}",
                'Avg Battery Level': f"{np.mean(results['battery_levels']) * 100:.1f}%"
            })
        
        df = pd.DataFrame(comparison_data)
        print(df.to_string(index=False))
        
        # Find best performing algorithm
        best_algo = max(self.results.keys(), key=lambda x: self.results[x]['final_reward'])
        print(f"\n🏆 Best performing algorithm: {best_algo}")
        print(f"   Final reward: {self.results[best_algo]['final_reward']:.2f}")
        
        return df
    
    def plot_results(self):
        """Plot training curves and comparison charts"""
        if not self.results:
            print("No results to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Training rewards
        axes[0, 0].set_title('Training Rewards')
        for algo_name, results in self.results.items():
            episodes = range(len(results['episode_rewards']))
            axes[0, 0].plot(episodes, results['episode_rewards'], label=algo_name, alpha=0.7)
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Episode Reward')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Charging events
        axes[0, 1].set_title('Charging Events per Episode')
        for algo_name, results in self.results.items():
            episodes = range(len(results['charging_events']))
            axes[0, 1].plot(episodes, results['charging_events'], label=algo_name, alpha=0.7)
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Charging Events')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Battery levels
        axes[1, 0].set_title('Average Battery Levels')
        for algo_name, results in self.results.items():
            episodes = range(len(results['battery_levels']))
            axes[1, 0].plot(episodes, results['battery_levels'], label=algo_name, alpha=0.7)
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Battery Level')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Final performance comparison
        algorithms = list(self.results.keys())
        final_rewards = [self.results[algo]['final_reward'] for algo in algorithms]
        
        bars = axes[1, 1].bar(algorithms, final_rewards)
        axes[1, 1].set_title('Final Performance Comparison')
        axes[1, 1].set_ylabel('Average Final Reward')
        axes[1, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, reward in zip(bars, final_rewards):
            axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{reward:.2f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save plot
        results_dir = Path("results/multi_agent_comparison")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = results_dir / f"multi_agent_comparison_{timestamp}.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        
        print(f"\n📊 Results plot saved to: {plot_path}")
        plt.show()
        
        return plot_path
    
    def save_detailed_results(self):
        """Save detailed results to Excel file"""
        if not self.results:
            print("No results to save")
            return
        
        results_dir = Path("results/multi_agent_detailed")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        excel_path = results_dir / f"multi_agent_results_{timestamp}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Summary sheet
            summary_data = []
            for algo_name, results in self.results.items():
                summary_data.append({
                    'Algorithm': algo_name,
                    'Final_Reward': results['final_reward'],
                    'Training_Time_s': results['training_time'],
                    'Total_Episodes': len(results['episode_rewards']),
                    'Avg_Charging_Events': np.mean(results['charging_events']),
                    'Avg_Battery_Level': np.mean(results['battery_levels']),
                    'Final_10_Avg_Reward': np.mean(results['episode_rewards'][-10:])
                })
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Individual algorithm results
            for algo_name, results in self.results.items():
                algo_df = pd.DataFrame({
                    'Episode': range(len(results['episode_rewards'])),
                    'Reward': results['episode_rewards'],
                    'Charging_Events': results['charging_events'],
                    'Battery_Level': results['battery_levels']
                })
                
                if results['training_losses']:
                    algo_df['Training_Loss'] = results['training_losses'] + [np.nan] * (len(results['episode_rewards']) - len(results['training_losses']))
                
                algo_df.to_excel(writer, sheet_name=algo_name, index=False)
        
        print(f"📋 Detailed results saved to: {excel_path}")
        return excel_path


def main():
    """Main function to run the multi-agent test suite"""
    print("🚀 Multi-Agent Reinforcement Learning Test Suite")
    print("Testing MADDPG, QMIX, MAPPO, and IQL algorithms")
    
    # Configuration
    num_vehicles = 10
    num_stations = 10
    num_episodes = 500  # Short test for debugging
    
    # Create test suite
    test_suite = MultiAgentTestSuite(
        num_vehicles=num_vehicles,
        num_stations=num_stations,
        grid_size=12
    )
    
    # Run all algorithms
    start_time = time.time()
    results = test_suite.run_all_algorithms(num_episodes=num_episodes)
    total_time = time.time() - start_time
    
    # Save detailed results
    test_suite.save_detailed_results()
    
    print(f"\n✅ Complete test suite finished in {total_time:.1f}s")
    print(f"Tested {len(results)} algorithms with {num_vehicles} vehicles and {num_stations} stations")
    
    return test_suite, results


if __name__ == "__main__":
    # Set random seeds for reproducibility
    np.random.seed(42)
    
    # Run the test suite
    test_suite, results = main()