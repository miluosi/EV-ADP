"""
Example usage of the new state-based value function training framework
Following src2's approach but integrated with the existing EV-ADP system
"""

import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Environment import ChargingIntegratedEnvironment
from src.ValueFunction_state import PyTorchChargingValueFunction_state
from src.GurobiOptimizer import GurobiOptimizer
import torch
import numpy as np

def main():
    print("=== State-Based Value Function Training Example ===")
    
    # 1. Initialize environment
    num_vehicles = 10
    num_stations = 8
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles, 
        num_stations=num_stations, 
        random_seed=42
    )
    
    # 2. Initialize state-based value function (src2-style)
    value_function_state = PyTorchChargingValueFunction_state(
        grid_size=env.grid_size,
        num_vehicles=num_vehicles,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        episode_length=env.episode_length,
        max_requests=1000
    )
    
    # 3. Set the state-based value function in environment
    env.set_value_function_state(value_function_state)
    
    # 4. Initialize Gurobi optimizer
    gurobi_optimizer = GurobiOptimizer(env)
    
    print(f"✓ Initialized state-based training framework")
    print(f"   - Environment: {num_vehicles} vehicles, {num_stations} stations")
    print(f"   - Device: {value_function_state.device}")
    print(f"   - Network parameters: {sum(p.numel() for p in value_function_state.network.parameters())}")
    
    # 5. Training loop (simplified example)
    num_episodes = 5
    training_frequency = 10
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # Reset environment
        env.reset()
        
        # Simulate episode steps
        for step in range(50):  # Shorter episodes for demo
            # Get current vehicles that need rebalancing
            available_vehicles = list(env.vehicles.keys())[:5]  # Take first 5 for demo
            
            if len(available_vehicles) > 0:
                # Use enhanced Gurobi optimization with state-based value function
                try:
                    assignments, vehicle_rewards = gurobi_optimizer._gurobi_vehicle_rebalancing_knownreject_state_enhanced(
                        vehicle_ids=available_vehicles,
                        available_requests=list(env.active_requests.values()) if env.active_requests else [],
                        charging_stations=list(env.charging_manager.stations.values()) if hasattr(env, 'charging_manager') else []
                    )
                    
                    print(f"Step {step}: Optimized {len(available_vehicles)} vehicles")
                    print(f"   Assignments: {len(assignments)} vehicles assigned")
                    print(f"   Vehicle rewards (y_ei): {[f'{r:.3f}' for r in vehicle_rewards[:3]]}...")
                    
                    # Store experiences and train (src2-style supervised learning)
                    if len(vehicle_rewards) > 0:
                        training_loss = gurobi_optimizer.store_and_train_state_experiences(
                            vehicle_ids=available_vehicles,
                            vehicle_rewards=vehicle_rewards,
                            batch_size=32
                        )
                        
                        if training_loss > 0:
                            print(f"   Training loss: {training_loss:.4f}")
                    
                except Exception as e:
                    print(f"   Optimization failed: {e}")
                    continue
            
            # Simulate environment step
            try:
                env.step({}, [])  # Simplified step
            except Exception as e:
                print(f"   Environment step failed: {e}")
        
        # Episode summary
        if hasattr(value_function_state, 'experience_buffer_state'):
            buffer_size = len(value_function_state.experience_buffer_state)
            print(f"Episode {episode + 1} completed. Experience buffer size: {buffer_size}")
            
            if hasattr(value_function_state, 'training_losses_state') and value_function_state.training_losses_state:
                avg_loss = np.mean(value_function_state.training_losses_state[-10:])
                print(f"   Recent average training loss: {avg_loss:.4f}")

    print("\n=== Training Framework Demo Completed ===")
    print("Key Features Demonstrated:")
    print("✓ State-based value function evaluation")
    print("✓ Gurobi optimization with state values as coefficients")
    print("✓ Experience storage with y_ei targets from optimization")
    print("✓ Supervised learning training (MSE loss)")
    print("✓ Integration with existing EV-ADP framework")


if __name__ == "__main__":
    main()