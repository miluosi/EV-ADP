"""
DQN vs ILP-ADP Benchmark Comparison
==================================

This script demonstrates how to use the DQN implementation as a benchmark
for comparison with the existing ILP-ADP approach in vehicle dispatch optimization.

The DQN implementation provides an alternative decision-making strategy that can be
directly compared against the ILP-ADP method in terms of:
- Request completion rate
- Vehicle utilization
- Energy efficiency (for EVs)
- Response time
- Overall system performance
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

try:
    from src.ValueFunction_pytorch import DQNAgent, create_dqn_state_features
    from src.Environment import Environment
    from src.Request import Request
    print("Successfully imported DQN components")
except ImportError as e:
    print(f"Warning: Could not import some components: {e}")
    print("Please ensure all required files are available")


class DQNBenchmarkRunner:
    """
    Benchmark runner for comparing DQN and ILP-ADP approaches
    """
    
    def __init__(self, num_vehicles=20, num_locations=50, simulation_time=100):
        self.num_vehicles = num_vehicles
        self.num_locations = num_locations
        self.simulation_time = simulation_time
        
        # Initialize DQN agent
        self.dqn_agent = DQNAgent(
            state_dim=64,
            action_dim=32,
            lr=1e-4,
            gamma=0.99,
            epsilon_start=1.0,
            epsilon_end=0.1,
            epsilon_decay=1000,
            device='cuda' if self._check_cuda() else 'cpu'
        )
        
        # Results storage
        self.dqn_results = []
        self.ilp_results = []
        
    def _check_cuda(self):
        """Check if CUDA is available"""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def create_mock_environment(self):
        """
        Create a mock environment for testing
        Replace this with your actual environment initialization
        """
        class MockEnvironment:
            def __init__(self, num_locations, num_vehicles):
                self.NUM_LOCATIONS = num_locations
                self.MAX_CAPACITY = 4
                self.current_time = 0.0
                
                # Initialize vehicles
                self.vehicles = {}
                for i in range(num_vehicles):
                    self.vehicles[i] = {
                        'id': i,
                        'type': 1 if i % 2 == 0 else 2,  # Alternating EV/AEV
                        'location': i % num_locations,
                        'battery': 1.0,
                        'idle': True,
                        'assigned_request': None,
                        'passenger_onboard': None,
                        'charging_station': None,
                        'rejected_requests': 0
                    }
                
                # Initialize requests
                self.active_requests = []
                self.completed_requests = []
                
            def simulate_motion_dqn(self, dqn_agent=None, current_requests=None, training=True):
                """Mock DQN simulation"""
                # This would call the actual implementation in your Environment class
                results = {
                    'total_reward': np.random.normal(50, 10),
                    'actions_taken': [],
                    'vehicle_utilization': np.random.uniform(0.6, 0.9),
                    'request_completion_rate': np.random.uniform(0.7, 0.95),
                    'average_battery_level': np.random.uniform(0.4, 0.8),
                    'action_distribution': {
                        'assign': 0.4, 'rebalance': 0.2, 'charge': 0.15, 'wait': 0.1, 'idle': 0.15
                    }
                }
                return results
        
        return MockEnvironment(self.num_locations, self.num_vehicles)
    
    def generate_test_requests(self, num_requests=10):
        """Generate mock requests for testing"""
        requests = []
        for i in range(num_requests):
            # Mock request - replace with actual Request class
            request = {
                'id': i,
                'pickup_location': np.random.randint(0, self.num_locations),
                'dropoff_location': np.random.randint(0, self.num_locations),
                'time': self.simulation_time * np.random.random(),
                'value': np.random.uniform(10, 50)
            }
            requests.append(request)
        return requests
    
    def run_dqn_simulation(self, environment, requests, num_episodes=100):
        """
        Run DQN-based simulation
        
        Args:
            environment: Environment instance
            requests: List of requests to process
            num_episodes: Number of training episodes
        
        Returns:
            dict: DQN simulation results
        """
        print("Running DQN simulation...")
        episode_rewards = []
        episode_metrics = []
        
        for episode in range(num_episodes):
            # Training phase
            if hasattr(environment, 'simulate_motion_dqn'):
                results = environment.simulate_motion_dqn(
                    dqn_agent=self.dqn_agent,
                    current_requests=requests,
                    training=True
                )
            else:
                # Fallback for mock environment
                results = environment.simulate_motion_dqn(
                    dqn_agent=self.dqn_agent,
                    current_requests=requests,
                    training=True
                )
            
            if results:
                episode_rewards.append(results['total_reward'])
                episode_metrics.append({
                    'vehicle_utilization': results.get('vehicle_utilization', 0),
                    'completion_rate': results.get('request_completion_rate', 0),
                    'battery_level': results.get('average_battery_level', 1.0)
                })
            
            # Print progress
            if (episode + 1) % 20 == 0:
                avg_reward = np.mean(episode_rewards[-20:])
                print(f"Episode {episode + 1}/{num_episodes}, Avg Reward: {avg_reward:.2f}")
        
        # Evaluation phase (no exploration)
        print("Running DQN evaluation...")
        eval_results = []
        for _ in range(10):  # 10 evaluation episodes
            if hasattr(environment, 'simulate_motion_dqn'):
                result = environment.simulate_motion_dqn(
                    dqn_agent=self.dqn_agent,
                    current_requests=requests,
                    training=False  # No exploration
                )
            else:
                result = environment.simulate_motion_dqn(
                    dqn_agent=self.dqn_agent,
                    current_requests=requests,
                    training=False
                )
            
            if result:
                eval_results.append(result)
        
        # Aggregate results
        final_results = {
            'training_rewards': episode_rewards,
            'training_metrics': episode_metrics,
            'evaluation_results': eval_results,
            'avg_training_reward': np.mean(episode_rewards),
            'avg_eval_reward': np.mean([r['total_reward'] for r in eval_results]),
            'avg_vehicle_utilization': np.mean([r['vehicle_utilization'] for r in eval_results]),
            'avg_completion_rate': np.mean([r['request_completion_rate'] for r in eval_results]),
            'avg_battery_level': np.mean([r.get('average_battery_level', 1.0) for r in eval_results])
        }
        
        return final_results
    
    def run_ilp_adp_simulation(self, environment, requests):
        """
        Run ILP-ADP simulation for comparison
        
        Args:
            environment: Environment instance
            requests: List of requests to process
        
        Returns:
            dict: ILP-ADP simulation results
        """
        print("Running ILP-ADP simulation...")
        
        # Mock ILP-ADP results - replace with actual ILP-ADP implementation
        # This should call your existing ILP-ADP optimization
        
        # Simulate ILP-ADP performance (replace with actual implementation)
        ilp_results = {
            'total_reward': np.random.normal(60, 8),  # Typically higher but less consistent
            'vehicle_utilization': np.random.uniform(0.8, 0.95),
            'request_completion_rate': np.random.uniform(0.85, 0.98),
            'average_battery_level': np.random.uniform(0.5, 0.9),
            'computation_time': np.random.uniform(0.5, 2.0),  # ILP typically slower
            'optimality_gap': np.random.uniform(0.01, 0.05)
        }
        
        print("ILP-ADP simulation completed")
        return ilp_results
    
    def compare_approaches(self, dqn_results, ilp_results):
        """
        Compare DQN and ILP-ADP results
        
        Args:
            dqn_results: Results from DQN simulation
            ilp_results: Results from ILP-ADP simulation
        
        Returns:
            dict: Comparison analysis
        """
        comparison = {
            'performance_metrics': {
                'reward': {
                    'dqn': dqn_results['avg_eval_reward'],
                    'ilp_adp': ilp_results['total_reward'],
                    'winner': 'DQN' if dqn_results['avg_eval_reward'] > ilp_results['total_reward'] else 'ILP-ADP'
                },
                'vehicle_utilization': {
                    'dqn': dqn_results['avg_vehicle_utilization'],
                    'ilp_adp': ilp_results['vehicle_utilization'],
                    'winner': 'DQN' if dqn_results['avg_vehicle_utilization'] > ilp_results['vehicle_utilization'] else 'ILP-ADP'
                },
                'completion_rate': {
                    'dqn': dqn_results['avg_completion_rate'],
                    'ilp_adp': ilp_results['request_completion_rate'],
                    'winner': 'DQN' if dqn_results['avg_completion_rate'] > ilp_results['request_completion_rate'] else 'ILP-ADP'
                },
                'battery_efficiency': {
                    'dqn': dqn_results['avg_battery_level'],
                    'ilp_adp': ilp_results['average_battery_level'],
                    'winner': 'DQN' if dqn_results['avg_battery_level'] > ilp_results['average_battery_level'] else 'ILP-ADP'
                }
            },
            'computational_aspects': {
                'dqn_training_time': 'Variable (one-time cost)',
                'dqn_inference_time': 'Fast (neural network forward pass)',
                'ilp_computation_time': ilp_results.get('computation_time', 'N/A'),
                'scalability': 'DQN scales better with problem size'
            }
        }
        
        return comparison
    
    def visualize_results(self, dqn_results, ilp_results, comparison):
        """
        Create visualizations of the comparison results
        """
        try:
            import matplotlib.pyplot as plt
            
            # Create subplots for different metrics
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('DQN vs ILP-ADP Performance Comparison', fontsize=16)
            
            # Performance metrics comparison
            metrics = ['reward', 'vehicle_utilization', 'completion_rate', 'battery_efficiency']
            dqn_values = []
            ilp_values = []
            
            for metric in metrics:
                dqn_values.append(comparison['performance_metrics'][metric]['dqn'])
                ilp_values.append(comparison['performance_metrics'][metric]['ilp_adp'])
            
            # Bar chart comparison
            x = np.arange(len(metrics))
            width = 0.35
            
            axes[0, 0].bar(x - width/2, dqn_values, width, label='DQN', alpha=0.8)
            axes[0, 0].bar(x + width/2, ilp_values, width, label='ILP-ADP', alpha=0.8)
            axes[0, 0].set_title('Performance Metrics Comparison')
            axes[0, 0].set_xticks(x)
            axes[0, 0].set_xticklabels(metrics, rotation=45)
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Training progress (DQN only)
            if 'training_rewards' in dqn_results:
                axes[0, 1].plot(dqn_results['training_rewards'])
                axes[0, 1].set_title('DQN Training Progress')
                axes[0, 1].set_xlabel('Episode')
                axes[0, 1].set_ylabel('Reward')
                axes[0, 1].grid(True, alpha=0.3)
            
            # Radar chart for normalized metrics
            angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False)
            angles = np.concatenate((angles, [angles[0]]))
            
            # Normalize values for radar chart
            max_vals = [max(d, i) for d, i in zip(dqn_values, ilp_values)]
            dqn_norm = [d / m for d, m in zip(dqn_values, max_vals)] + [dqn_values[0] / max_vals[0]]
            ilp_norm = [i / m for i, m in zip(ilp_values, max_vals)] + [ilp_values[0] / max_vals[0]]
            
            axes[1, 0].plot(angles, dqn_norm, 'o-', linewidth=2, label='DQN')
            axes[1, 0].fill(angles, dqn_norm, alpha=0.25)
            axes[1, 0].plot(angles, ilp_norm, 'o-', linewidth=2, label='ILP-ADP')
            axes[1, 0].fill(angles, ilp_norm, alpha=0.25)
            axes[1, 0].set_xticks(angles[:-1])
            axes[1, 0].set_xticklabels(metrics)
            axes[1, 0].set_title('Normalized Performance Radar Chart')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
            
            # Winner summary
            winners = [comparison['performance_metrics'][m]['winner'] for m in metrics]
            dqn_wins = winners.count('DQN')
            ilp_wins = winners.count('ILP-ADP')
            
            axes[1, 1].pie([dqn_wins, ilp_wins], labels=['DQN Wins', 'ILP-ADP Wins'], 
                          autopct='%1.1f%%', startangle=90)
            axes[1, 1].set_title('Performance Metrics Won')
            
            plt.tight_layout()
            plt.savefig('dqn_vs_ilp_comparison.png', dpi=300, bbox_inches='tight')
            plt.show()
            
            print("Visualization saved as 'dqn_vs_ilp_comparison.png'")
            
        except ImportError:
            print("Matplotlib not available. Skipping visualization.")
    
    def run_full_benchmark(self):
        """
        Run complete benchmark comparison between DQN and ILP-ADP
        """
        print("=" * 60)
        print("DQN vs ILP-ADP Benchmark Comparison")
        print("=" * 60)
        
        # Create environment and requests
        environment = self.create_mock_environment()
        requests = self.generate_test_requests(20)
        
        print(f"Environment: {self.num_vehicles} vehicles, {self.num_locations} locations")
        print(f"Test requests: {len(requests)}")
        print()
        
        # Run DQN simulation
        start_time = time.time()
        dqn_results = self.run_dqn_simulation(environment, requests, num_episodes=50)
        dqn_time = time.time() - start_time
        
        print(f"DQN simulation completed in {dqn_time:.2f} seconds")
        print()
        
        # Run ILP-ADP simulation
        start_time = time.time()
        ilp_results = self.run_ilp_adp_simulation(environment, requests)
        ilp_time = time.time() - start_time
        
        print(f"ILP-ADP simulation completed in {ilp_time:.2f} seconds")
        print()
        
        # Compare results
        comparison = self.compare_approaches(dqn_results, ilp_results)
        
        # Print results
        print("BENCHMARK RESULTS:")
        print("-" * 40)
        for metric, values in comparison['performance_metrics'].items():
            print(f"{metric.replace('_', ' ').title()}:")
            print(f"  DQN: {values['dqn']:.3f}")
            print(f"  ILP-ADP: {values['ilp_adp']:.3f}")
            print(f"  Winner: {values['winner']}")
            print()
        
        print("COMPUTATIONAL ANALYSIS:")
        print("-" * 40)
        for aspect, value in comparison['computational_aspects'].items():
            print(f"{aspect.replace('_', ' ').title()}: {value}")
        print()
        
        # Visualize results
        self.visualize_results(dqn_results, ilp_results, comparison)
        
        return {
            'dqn_results': dqn_results,
            'ilp_results': ilp_results,
            'comparison': comparison,
            'timing': {
                'dqn_time': dqn_time,
                'ilp_time': ilp_time
            }
        }


def main():
    """
    Main function to run the DQN vs ILP-ADP benchmark
    """
    print("Initializing DQN vs ILP-ADP Benchmark...")
    
    # Create benchmark runner
    benchmark = DQNBenchmarkRunner(
        num_vehicles=20,
        num_locations=50,
        simulation_time=100
    )
    
    # Run full benchmark
    results = benchmark.run_full_benchmark()
    
    print("\nBenchmark completed successfully!")
    print("Check 'dqn_vs_ilp_comparison.png' for detailed visualizations.")
    
    return results


if __name__ == "__main__":
    main()