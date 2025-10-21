"""
Integrated Test: Vehicle Charging Behavior Integration Test using src folder components
"""
from datetime import datetime
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import time
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
from src.ChargingIntegrationVisualization import ChargingIntegrationVisualization
# Set matplotlib Chinese font support
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue

# 导入配置管理器
from config.config_manager import ConfigManager, get_config, get_training_config, get_sampling_config

from src.Environment import Environment
from src.LearningAgent import LearningAgent
from src.Action import Action, ChargingAction, ServiceAction
from src.Request import Request
from src.charging_station import ChargingStationManager, ChargingStation
from src.CentralAgent import CentralAgent
from src.ValueFunction_pytorch import PyTorchChargingValueFunction
from src.Environment import ChargingIntegratedEnvironment
from src.SpatialVisualization import SpatialVisualization
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
print("✓ Successfully imported core components from src folder")
USE_SRC_COMPONENTS = True

def set_random_seeds(seed=42):
    """
    设置所有随机数生成器的种子，确保实验的可重复性
    
    Args:
        seed (int): 随机数种子，默认为42
    """
    # Python内置random模块
    random.seed(seed)
    
    # NumPy随机数生成器
    np.random.seed(seed)
    
    # PyTorch随机数生成器  
    torch.manual_seed(seed)
    
    # 如果使用CUDA，设置CUDA随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 确保CUDA操作的确定性（可能会影响性能）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"✓ Random seeds set to {seed} for all generators (Python, NumPy, PyTorch)")


def run_charging_integration_test(adpvalue,num_episodes,use_intense_requests,assignmentgurobi,batch_size=256, num_vehicles = 10):
    """Run charging integration test with EV/AEV analysis"""
    print("=== Starting Enhanced Charging Behavior Integration Test ===")
    
    # 设置全局随机数种子，确保车辆初始化一致
    set_random_seeds(seed=42)
    
    # Create environment with significantly more complexity for better learning
    num_vehicles = num_vehicles
    num_stations = 4
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles, 
        num_stations=num_stations, 
        random_seed=42  # 传入种子确保环境初始化的一致性
    )
    
    print("✓ Fixed initial state setup: Vehicle positions and battery levels will be identical across all episodes")
    print("✓ Request generation will vary by episode for learning progression")
    
    # Initialize neural network-based ValueFunction for decision making only if needed
    # Use PyTorchChargingValueFunction with neural network only when ADP > 0 and assignmentgurobi is True
    use_neural_network = adpvalue > 0 and assignmentgurobi
    
    if use_neural_network:
        value_function = PyTorchChargingValueFunction(
            grid_size=env.grid_size, 
            num_vehicles=num_vehicles,
            device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
            episode_length=env.episode_length,  # 传递正确的episode长度
            max_requests=10000,  # 设置合理的最大请求数
            env=env  # 传递环境引用
        )
        # Set the value function in the environment for Q-value calculation
        env.set_value_function(value_function)
    else:
        value_function = None
        
    env.adp_value = adpvalue
    env.use_intense_requests = use_intense_requests
    env.assignmentgurobi = assignmentgurobi
    # Exploration parameters for enhanced learning with complex environment
    exploration_episodes = max(1, num_episodes // 2)  # Half episodes for exploration  
    epsilon_start = 0.4  # Higher exploration for complex environment
    epsilon_end = 0.1   # End with 10% random actions
    epsilon_decay = (epsilon_start - epsilon_end) / exploration_episodes
    
    # Enhanced training parameters for complex environment
    training_frequency = 2
    warmup_steps = 100     # Increased warmup for complex environment
    
    print(f"✓ Initialized environment with {num_vehicles} vehicles and {num_stations} charging stations")
    if use_neural_network:
        print(f"✓ Initialized PyTorchChargingValueFunction with neural network")
        print(f"   - Network parameters: {sum(p.numel() for p in value_function.network.parameters())}")
        print(f"✓ Enhanced exploration strategy: {exploration_episodes} episodes with epsilon {epsilon_start:.2f} → {epsilon_end:.2f}")
        print(f"   - Training frequency: every {training_frequency} steps after {warmup_steps} warmup steps")
        print(f"   - Using device: {value_function.device}")
    else:
        print(f"✓ Neural network training disabled (ADP={adpvalue}, AssignmentGurobi={assignmentgurobi})")
        print(f"   - Running without neural network training")
    
    # Display vehicle type distribution
    ev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'EV')
    aev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'AEV')
    print(f"✓ Vehicle distribution: {ev_count} EV vehicles, {aev_count} AEV vehicles")
    
    # Test parameters
    num_episodes = num_episodes
    results = {
        'Idle_average': [],
        'episode_rewards': [],
        'charging_events': [],
        'episode_detailed_stats': [],  # New: detailed stats for each episode
        'vehicle_visit_stats': [],     # New: vehicle visit patterns for each episode
        'battery_levels': [],
        'environment_stats': [],
        'value_function_losses': [],
        'qvalue_losses': []  # Added: to store all training losses
    }
    
    for episode in range(num_episodes):
        # 为每个episode设置请求生成专用的种子，确保请求序列的多样性
        # 但保持不同ADP值下相同episode的请求序列一致
        episode_seed = 32 + episode  # 基础种子42加上episode编号
        env.set_request_generation_seed(episode_seed)
        print(f"Episode {episode + 1}: Request generation seed set to {episode_seed}")
        
        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        use_exploration = False
        
        # Reset environment
        states = env.reset()
        episode_reward = 0
        episode_charging_events = []
        episode_losses = []
        
        Idle_list = []
        for step in range(env.episode_length):
            # Generate actions using ValueFunction
            actions = {}
            states_for_training = []
            actions_for_training = []
            current_requests = list(env.active_requests.values())
            actions, storeactions = env.simulate_motion(agents=[], current_requests=current_requests, rebalance=True)
            next_states, rewards, dur_rewards, done, info = env.step(actions,storeactions)
            # Debug: Output step statistics every 100 steps
            if step % 25 == 0:
                stats = env.get_stats()
                active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                assigned_vehicles = len([v for v in env.vehicles.values() if v['assigned_request'] is not None])
                charging_vehicles = len([v for v in env.vehicles.values() if v['charging_station'] is not None])
                onboard = len([v for v in env.vehicles.values() if v['passenger_onboard'] is not None])
                idlecar = len([v for v in env.vehicles.values() if  v.get('idle_target') is not None ])
                waitcar = len([v for v in env.vehicles.values() if  v.get('is_stationary') is True ])
                movecharge = len([v for v in env.vehicles.values() if v.get('charging_target') is not None])
                target_location_v = len([v for v in env.vehicles.values() if v.get('target_location') is not None])
                idle_vehicles = len([v for v in env.vehicles.values() 
                                   if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None])
                step_reward = sum(rewards.values())
                print(f"Step {step}: Active requests: {active_requests}, Assigned: {assigned_vehicles}, Onboard: {onboard}, Charging: {charging_vehicles}, Idle: {idlecar}, waitcar: {waitcar}, movecharge: {movecharge}, Idle Vehicles: {idle_vehicles}, Step reward: {step_reward:.2f}")
                Idle_list.append(idle_vehicles)
                # Neural network monitoring (if using neural network)
                if use_neural_network and hasattr(value_function, 'training_losses') and value_function.training_losses:
                    recent_loss = value_function.training_losses[-1] if value_function.training_losses else 0.0
                    buffer_size = len(value_function.experience_buffer)
                    training_step = value_function.training_step
                    
                    # Sample some Q-values to show the actual raw values used by Gurobi
                    if buffer_size > 0:
                        # Get a sample Q-value to demonstrate what Gurobi actually uses
                        sample_vehicle_id = list(env.vehicles.keys())[0] if env.vehicles else 0
                        sample_location = list(env.vehicles.values())[0]['location'] if env.vehicles else 0
                        sample_battery = list(env.vehicles.values())[0]['battery'] if env.vehicles else 1.0
                        
                        try:
                            # Test different action types - these are the raw Q-values Gurobi uses
                            idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, current_time=step)
                            assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, current_time=step, battery_level=sample_battery)
                            charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, current_time=step, battery_level=sample_battery)
                            
                            print(f"  Neural Network Status:")
                            print(f"    Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                            print(f"    Raw Q-values (no normalization): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
                            print(f"    Note: Gurobi uses these raw Q-values directly in optimization objective")
                            
                            # 添加经验数据分析
                            if step > 100 and step % 100 == 0:  # 每100步分析一次
                                exp_analysis = value_function.analyze_experience_data()
                                if exp_analysis:
                                    reward_stats = exp_analysis['reward_stats']
                                    action_stats = exp_analysis['action_stats']
                                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']:.1%} | 0{reward_stats['neutral_ratio']:.1%} | -{reward_stats['negative_ratio']:.1%}")
                                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']:.2f}, Assign={action_stats['assign_mean_reward']:.2f}, Charge={action_stats['charge_mean_reward']:.2f}, Idle={action_stats['idle_mean_reward']:.2f}")
                                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                                    
                        except Exception as e:
                            print(f"  Neural Network Status: Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                            print(f"    Error getting sample Q-values: {e}")
                else:
                    print(f"  Neural Network: {'Not training yet' if use_neural_network else 'Disabled'}")
            
            # Note: Q-learning experience storage is now handled automatically in env.step()
            # This ensures consistency between traditional Q-table and neural network training
            
            # Enhanced training: much more frequent training for better learning (only if using neural network)
            if use_neural_network and len(value_function.experience_buffer) >= warmup_steps:
                # Train more frequently based on our new parameters
                if step % training_frequency == 0:
                    training_loss = value_function.train_step(batch_size=batch_size)  # Larger batch
                    if training_loss > 0:
                        episode_losses.append(training_loss)
                
            episode_reward += sum(rewards.values())
            episode_charging_events.extend(info.get('charging_events', []))
            
            if done:
                break
        results['Idle_average'].append(sum(Idle_list)/len(Idle_list) if Idle_list else 0)
        results['episode_rewards'].append(episode_reward)
        results['charging_events'].extend(episode_charging_events)
        results['value_function_losses'].append(np.mean(episode_losses) if episode_losses else 0.0)
        results['qvalue_losses'].extend(episode_losses)  # Fixed: extend instead of assign
        # Record environment statistics
        stats = env.get_stats()
        results['active_requests'] = stats['active_requests']
        results['environment_stats'].append(stats)
        results['battery_levels'].append(stats['average_battery'])
        results['completed_requests'] = stats['completed_requests']
        results['avg_requestvalue'] = stats['completed_orders_req']
        # Collect detailed episode statistics
        episode_stats = env.get_episode_stats()
        episode_stats['episode_number'] = episode + 1
        episode_stats['episode_reward'] = episode_reward
        episode_stats['charging_events_count'] = len(episode_charging_events)
        
        # Output rebalancing assignment statistics
        rebalancing_calls = episode_stats.get('total_rebalancing_calls', 0)
        total_assignments = episode_stats.get('total_rebalancing_assignments', 0)
        avg_assignments = episode_stats.get('avg_rebalancing_assignments_per_call', 0)
        avg_whole = episode_stats.get('avg_rebalancing_assignments_per_whole', 0)
        print(f"Episode {episode + 1} Completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Orders: Total={episode_stats['total_orders']}, Accepted={episode_stats['accepted_orders']}, Completed={episode_stats['completed_orders']}, Rejected={episode_stats['rejected_orders']}")
        print(f"  Battery: {episode_stats['avg_battery_level']:.2f}")
        print(f"  Rebalancing: Calls={rebalancing_calls}, Total Assignments={total_assignments}, Avg Assignments={avg_assignments:.2f}, Avg Rebalance Whole={avg_whole:.2f}")

        # Add neural network Q-value summary
        if use_neural_network:
            idle_q = episode_stats.get('sample_idle_q_value', 0.0)
            assign_q = episode_stats.get('sample_assign_q_value', 0.0)
            charge_q = episode_stats.get('sample_charge_q_value', 0.0)
            nn_loss = episode_stats.get('neural_network_loss', 0.0)
            print(f"  Neural Network: Loss={nn_loss:.4f}, Q-values(Gurobi): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
        # Only record neural network metrics if using neural network
        if use_neural_network:
            episode_stats['neural_network_loss'] = np.mean(episode_losses) if episode_losses else 0.0
            episode_stats['neural_network_loss_std'] = np.std(episode_losses) if episode_losses else 0.0
            episode_stats['training_steps_in_episode'] = len(episode_losses)
            
            # Sample Q-values for different action types (actual values used by Gurobi)
            if len(value_function.experience_buffer) > 0:
                try:
                    sample_vehicle_id = list(env.vehicles.keys())[0] if env.vehicles else 0
                    sample_location = list(env.vehicles.values())[0]['location'] if env.vehicles else 0
                    sample_battery = list(env.vehicles.values())[0]['battery'] if env.vehicles else 1.0
                    
                    # Get sample Q-values for statistics
                    idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, current_time=env.current_time)
                    assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, current_time=env.current_time, battery_level=sample_battery)
                    charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, current_time=env.current_time, battery_level=sample_battery)
                    
                    episode_stats['sample_idle_q_value'] = idle_q
                    episode_stats['sample_assign_q_value'] = assign_q
                    episode_stats['sample_charge_q_value'] = charge_q
                    
                except Exception as e:
                    episode_stats['sample_idle_q_value'] = 0.0
                    episode_stats['sample_assign_q_value'] = 0.0
                    episode_stats['sample_charge_q_value'] = 0.0
            else:
                episode_stats['sample_idle_q_value'] = 0.0
                episode_stats['sample_assign_q_value'] = 0.0
                episode_stats['sample_charge_q_value'] = 0.0
        else:
            episode_stats['neural_network_loss'] = 0.0
            episode_stats['neural_network_loss_std'] = 0.0
            episode_stats['training_steps_in_episode'] = 0
            episode_stats['sample_idle_q_value'] = 0.0
            episode_stats['sample_assign_q_value'] = 0.0
            episode_stats['sample_charge_q_value'] = 0.0
        results['episode_detailed_stats'].append(episode_stats)
        
        # Analyze charging usage history for this episode
        if 'charging_usage_history' in episode_stats and episode_stats['charging_usage_history']:
            charging_history = episode_stats['charging_usage_history']
            avg_usage = sum(h['vehicles_per_station'] for h in charging_history) / len(charging_history)
            max_usage = max(h['vehicles_per_station'] for h in charging_history)
            min_usage = min(h['vehicles_per_station'] for h in charging_history)
            print(f"  Charging History: {len(charging_history)} time steps, Avg: {avg_usage:.2f}, Max: {max_usage:.2f}, Min: {min_usage:.2f} vehicles/station")
        
        # Analyze vehicle visit patterns for this episode
        vehicle_visit_stats = analyze_vehicle_visit_patterns(env)
        results['vehicle_visit_stats'].append(vehicle_visit_stats)
        

    print("\n=== Integration Test Complete ===")
    if use_neural_network:
        print(f"✓ Neural Network ValueFunction trained over {num_episodes} episodes")
        print(f"✓ Final average training loss: {np.mean(results['value_function_losses']):.4f}")
        print(f"✓ Neural network has {sum(p.numel() for p in value_function.network.parameters())} parameters")
    else:
        print(f"✓ Test completed without neural network training")
        print(f"✓ Used traditional Q-table approach")
    
    # Create results directory for analysis - choose directory based on assignmentgurobi
    if assignmentgurobi:
        results_dir = Path("results/integrated_tests")
    else:
        results_dir = Path("results/integrated_tests_h")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results will be saved to: {results_dir}")
    
    # Save detailed episode statistics to Excel including vehicle visit patterns
    excel_path, spatial_path = save_episode_stats_to_excel(env, results['episode_detailed_stats'], results_dir, results.get('vehicle_visit_stats'))
    
    # Store file paths in results for reference
    results['excel_path'] = excel_path
    results['spatial_image_path'] = spatial_path
    
    return results, env




def run_charging_integration_test_threshold(adpvalue,num_episodes,use_intense_requests,assignmentgurobi,batch_size=256,heuristic_battery_threshold = 0.5, num_vehicles = 10):
    """Run charging integration test with EV/AEV analysis"""
    print("=== Starting Enhanced Charging Behavior Integration Test ===")
    
    # 设置全局随机数种子，确保车辆初始化一致
    set_random_seeds(seed=42)
    
    # Create environment with significantly more complexity for better learning
    num_vehicles = num_vehicles
    num_stations = 4
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles, 
        num_stations=num_stations, 
        random_seed=42  # 传入种子确保环境初始化的一致性
    )
    
    print("✓ Fixed initial state setup: Vehicle positions and battery levels will be identical across all episodes")
    print("✓ Request generation will vary by episode for learning progression")
    
    # Initialize neural network-based ValueFunction for decision making only if needed
    # Use PyTorchChargingValueFunction with neural network only when ADP > 0 and assignmentgurobi is True
    use_neural_network = adpvalue > 0 and assignmentgurobi
    
    if use_neural_network:
        value_function = PyTorchChargingValueFunction(
            grid_size=env.grid_size, 
            num_vehicles=num_vehicles,
            device='cuda' if torch.cuda.is_available() else 'cpu',  # Use GPU if available
            episode_length=env.episode_length,  # 传递正确的episode长度
            max_requests=10000,  # 设置合理的最大请求数
            env=env  # 传递环境引用
        )
        # Set the value function in the environment for Q-value calculation
        env.set_value_function(value_function)
    else:
        value_function = None
        
    env.adp_value = adpvalue
    env.use_intense_requests = use_intense_requests
    env.assignmentgurobi = assignmentgurobi
    env.heuristic_battery_threshold = heuristic_battery_threshold
    # Exploration parameters for enhanced learning with complex environment
    exploration_episodes = max(1, num_episodes // 2)  # Half episodes for exploration  
    epsilon_start = 0.4  # Higher exploration for complex environment
    epsilon_end = 0.1   # End with 10% random actions
    epsilon_decay = (epsilon_start - epsilon_end) / exploration_episodes
    
    # Enhanced training parameters for complex environment
    training_frequency = 2
    warmup_steps = 100     # Increased warmup for complex environment
    
    print(f"✓ Initialized environment with {num_vehicles} vehicles and {num_stations} charging stations")
    if use_neural_network:
        print(f"✓ Initialized PyTorchChargingValueFunction with neural network")
        print(f"   - Network parameters: {sum(p.numel() for p in value_function.network.parameters())}")
        print(f"✓ Enhanced exploration strategy: {exploration_episodes} episodes with epsilon {epsilon_start:.2f} → {epsilon_end:.2f}")
        print(f"   - Training frequency: every {training_frequency} steps after {warmup_steps} warmup steps")
        print(f"   - Using device: {value_function.device}")
    else:
        print(f"✓ Neural network training disabled (ADP={adpvalue}, AssignmentGurobi={assignmentgurobi})")
        print(f"   - Running without neural network training")
    
    # Display vehicle type distribution
    ev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'EV')
    aev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'AEV')
    print(f"✓ Vehicle distribution: {ev_count} EV vehicles, {aev_count} AEV vehicles")
    
    # Test parameters
    num_episodes = num_episodes
    results = {
        'Idle_average': [],
        'episode_rewards': [],
        'charging_events': [],
        'episode_detailed_stats': [],  # New: detailed stats for each episode
        'vehicle_visit_stats': [],     # New: vehicle visit patterns for each episode
        'battery_levels': [],
        'environment_stats': [],
        'value_function_losses': [],
        'qvalue_losses': []  # Added: to store all training losses
    }
    
    for episode in range(num_episodes):
        # 为每个episode设置请求生成专用的种子，确保请求序列的多样性
        # 但保持不同ADP值下相同episode的请求序列一致
        episode_seed = 32 + episode  # 基础种子42加上episode编号
        env.set_request_generation_seed(episode_seed)
        print(f"Episode {episode + 1}: Request generation seed set to {episode_seed}")
        
        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        use_exploration = False
        
        # Reset environment
        states = env.reset()
        episode_reward = 0
        episode_charging_events = []
        episode_losses = []
        
        Idle_list = []
        for step in range(env.episode_length):
            # Generate actions using ValueFunction
            actions = {}
            states_for_training = []
            actions_for_training = []
            current_requests = list(env.active_requests.values())
            actions, storeactions = env.simulate_motion(agents=[], current_requests=current_requests, rebalance=True)
            next_states, rewards, dur_rewards, done, info = env.step(actions,storeactions)
            # Debug: Output step statistics every 100 steps
            if step % 25 == 0:
                stats = env.get_stats()
                active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                assigned_vehicles = len([v for v in env.vehicles.values() if v['assigned_request'] is not None])
                charging_vehicles = len([v for v in env.vehicles.values() if v['charging_station'] is not None])
                onboard = len([v for v in env.vehicles.values() if v['passenger_onboard'] is not None])
                idlecar = len([v for v in env.vehicles.values() if  v.get('idle_target') is not None ])
                waitcar = len([v for v in env.vehicles.values() if  v.get('is_stationary') is True ])
                movecharge = len([v for v in env.vehicles.values() if v.get('charging_target') is not None])
                target_location_v = len([v for v in env.vehicles.values() if v.get('target_location') is not None])
                idle_vehicles = len([v for v in env.vehicles.values() 
                                   if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None])
                step_reward = sum(rewards.values())
                print(f"Step {step}: Active requests: {active_requests}, Assigned: {assigned_vehicles}, Onboard: {onboard}, Charging: {charging_vehicles}, Idle: {idlecar}, waitcar: {waitcar}, movecharge: {movecharge}, Idle Vehicles: {idle_vehicles}, Step reward: {step_reward:.2f}")
                Idle_list.append(idle_vehicles)
                # Neural network monitoring (if using neural network)
                if use_neural_network and hasattr(value_function, 'training_losses') and value_function.training_losses:
                    recent_loss = value_function.training_losses[-1] if value_function.training_losses else 0.0
                    buffer_size = len(value_function.experience_buffer)
                    training_step = value_function.training_step
                    
                    # Sample some Q-values to show the actual raw values used by Gurobi
                    if buffer_size > 0:
                        # Get a sample Q-value to demonstrate what Gurobi actually uses
                        sample_vehicle_id = list(env.vehicles.keys())[0] if env.vehicles else 0
                        sample_location = list(env.vehicles.values())[0]['location'] if env.vehicles else 0
                        sample_battery = list(env.vehicles.values())[0]['battery'] if env.vehicles else 1.0
                        
                        try:
                            # Test different action types - these are the raw Q-values Gurobi uses
                            idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, current_time=step)
                            assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, current_time=step, battery_level=sample_battery)
                            charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, current_time=step, battery_level=sample_battery)
                            
                            print(f"  Neural Network Status:")
                            print(f"    Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                            print(f"    Raw Q-values (no normalization): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
                            print(f"    Note: Gurobi uses these raw Q-values directly in optimization objective")
                            
                            # 添加经验数据分析
                            if step > 100 and step % 100 == 0:  # 每100步分析一次
                                exp_analysis = value_function.analyze_experience_data()
                                if exp_analysis:
                                    reward_stats = exp_analysis['reward_stats']
                                    action_stats = exp_analysis['action_stats']
                                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']:.1%} | 0{reward_stats['neutral_ratio']:.1%} | -{reward_stats['negative_ratio']:.1%}")
                                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']:.2f}, Assign={action_stats['assign_mean_reward']:.2f}, Charge={action_stats['charge_mean_reward']:.2f}, Idle={action_stats['idle_mean_reward']:.2f}")
                                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                                    
                        except Exception as e:
                            print(f"  Neural Network Status: Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                            print(f"    Error getting sample Q-values: {e}")
                else:
                    print(f"  Neural Network: {'Not training yet' if use_neural_network else 'Disabled'}")
            
            # Note: Q-learning experience storage is now handled automatically in env.step()
            # This ensures consistency between traditional Q-table and neural network training
            
            # Enhanced training: much more frequent training for better learning (only if using neural network)
            if use_neural_network and len(value_function.experience_buffer) >= warmup_steps:
                # Train more frequently based on our new parameters
                if step % training_frequency == 0:
                    training_loss = value_function.train_step(batch_size=batch_size)  # Larger batch
                    if training_loss > 0:
                        episode_losses.append(training_loss)
                
            episode_reward += sum(rewards.values())
            episode_charging_events.extend(info.get('charging_events', []))
            
            if done:
                break
        results['Idle_average'].append(sum(Idle_list)/len(Idle_list) if Idle_list else 0)
        results['episode_rewards'].append(episode_reward)
        results['charging_events'].extend(episode_charging_events)
        results['value_function_losses'].append(np.mean(episode_losses) if episode_losses else 0.0)
        results['qvalue_losses'].extend(episode_losses)  # Fixed: extend instead of assign
        # Record environment statistics
        stats = env.get_stats()
        results['active_requests'] = stats['active_requests']
        results['environment_stats'].append(stats)
        results['battery_levels'].append(stats['average_battery'])
        results['completed_requests'] = stats['completed_requests']
        # Collect detailed episode statistics
        episode_stats = env.get_episode_stats()
        episode_stats['episode_number'] = episode + 1
        episode_stats['episode_reward'] = episode_reward
        episode_stats['charging_events_count'] = len(episode_charging_events)
        
        # Output rebalancing assignment statistics
        rebalancing_calls = episode_stats.get('total_rebalancing_calls', 0)
        total_assignments = episode_stats.get('total_rebalancing_assignments', 0)
        avg_assignments = episode_stats.get('avg_rebalancing_assignments_per_call', 0)
        avg_whole = episode_stats.get('avg_rebalancing_assignments_per_whole', 0)
        print(f"Episode {episode + 1} Completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Orders: Total={episode_stats['total_orders']}, Accepted={episode_stats['accepted_orders']}, Completed={episode_stats['completed_orders']}, Rejected={episode_stats['rejected_orders']}")
        print(f"  Battery: {episode_stats['avg_battery_level']:.2f}")
        print(f"  Rebalancing: Calls={rebalancing_calls}, Total Assignments={total_assignments}, Avg Assignments={avg_assignments:.2f}, Avg Rebalance Whole={avg_whole:.2f}")

        # Add neural network Q-value summary
        if use_neural_network:
            idle_q = episode_stats.get('sample_idle_q_value', 0.0)
            assign_q = episode_stats.get('sample_assign_q_value', 0.0)
            charge_q = episode_stats.get('sample_charge_q_value', 0.0)
            nn_loss = episode_stats.get('neural_network_loss', 0.0)
            print(f"  Neural Network: Loss={nn_loss:.4f}, Q-values(Gurobi): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
        # Only record neural network metrics if using neural network
        if use_neural_network:
            episode_stats['neural_network_loss'] = np.mean(episode_losses) if episode_losses else 0.0
            episode_stats['neural_network_loss_std'] = np.std(episode_losses) if episode_losses else 0.0
            episode_stats['training_steps_in_episode'] = len(episode_losses)
            
            # Sample Q-values for different action types (actual values used by Gurobi)
            if len(value_function.experience_buffer) > 0:
                try:
                    sample_vehicle_id = list(env.vehicles.keys())[0] if env.vehicles else 0
                    sample_location = list(env.vehicles.values())[0]['location'] if env.vehicles else 0
                    sample_battery = list(env.vehicles.values())[0]['battery'] if env.vehicles else 1.0
                    
                    # Get sample Q-values for statistics
                    idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, current_time=env.current_time)
                    assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, current_time=env.current_time, battery_level=sample_battery)
                    charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, current_time=env.current_time, battery_level=sample_battery)
                    
                    episode_stats['sample_idle_q_value'] = idle_q
                    episode_stats['sample_assign_q_value'] = assign_q
                    episode_stats['sample_charge_q_value'] = charge_q
                    
                except Exception as e:
                    episode_stats['sample_idle_q_value'] = 0.0
                    episode_stats['sample_assign_q_value'] = 0.0
                    episode_stats['sample_charge_q_value'] = 0.0
            else:
                episode_stats['sample_idle_q_value'] = 0.0
                episode_stats['sample_assign_q_value'] = 0.0
                episode_stats['sample_charge_q_value'] = 0.0
        else:
            episode_stats['neural_network_loss'] = 0.0
            episode_stats['neural_network_loss_std'] = 0.0
            episode_stats['training_steps_in_episode'] = 0
            episode_stats['sample_idle_q_value'] = 0.0
            episode_stats['sample_assign_q_value'] = 0.0
            episode_stats['sample_charge_q_value'] = 0.0
        results['episode_detailed_stats'].append(episode_stats)
        
        # Analyze charging usage history for this episode
        if 'charging_usage_history' in episode_stats and episode_stats['charging_usage_history']:
            charging_history = episode_stats['charging_usage_history']
            avg_usage = sum(h['vehicles_per_station'] for h in charging_history) / len(charging_history)
            max_usage = max(h['vehicles_per_station'] for h in charging_history)
            min_usage = min(h['vehicles_per_station'] for h in charging_history)
            print(f"  Charging History: {len(charging_history)} time steps, Avg: {avg_usage:.2f}, Max: {max_usage:.2f}, Min: {min_usage:.2f} vehicles/station")
        
        # Analyze vehicle visit patterns for this episode
        vehicle_visit_stats = analyze_vehicle_visit_patterns(env)
        results['vehicle_visit_stats'].append(vehicle_visit_stats)
        

    print("\n=== Integration Test Complete ===")
    if use_neural_network:
        print(f"✓ Neural Network ValueFunction trained over {num_episodes} episodes")
        print(f"✓ Final average training loss: {np.mean(results['value_function_losses']):.4f}")
        print(f"✓ Neural network has {sum(p.numel() for p in value_function.network.parameters())} parameters")
    else:
        print(f"✓ Test completed without neural network training")
        print(f"✓ Used traditional Q-table approach")
    
    # Create results directory for analysis - choose directory based on assignmentgurobi
    if assignmentgurobi:
        results_dir = Path("results/integrated_tests")
    else:
        results_dir = Path("results/integrated_tests_h")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results will be saved to: {results_dir}")
    
    # Save detailed episode statistics to Excel including vehicle visit patterns
    excel_path, spatial_path = save_episode_stats_to_excel(env, results['episode_detailed_stats'], results_dir, results.get('vehicle_visit_stats'))
    
    # Store file paths in results for reference
    results['excel_path'] = excel_path
    results['spatial_image_path'] = spatial_path
    
    return results, env








def analyze_vehicle_visit_patterns(env):
    """Analyze vehicle visit patterns and identify most frequently visited locations"""
    vehicle_visit_stats = {}
    
    # Define hotspot locations for reference
    hotspots = [
        (env.grid_size // 4, env.grid_size // 4),           # Bottom-left hotspot
        (3 * env.grid_size // 4, env.grid_size // 4),       # Bottom-right hotspot
        (env.grid_size // 2, 3 * env.grid_size // 4)        # Top-center hotspot
    ]
    
    for vehicle_id, vehicle in env.vehicles.items():
        # Get position history for this vehicle
        position_history = env.vehicle_position_history.get(vehicle_id, [])
        
        if not position_history:
            # If no history, use current position
            current_coords = vehicle['coordinates']
            location_counts = {str(current_coords): 1}
        else:
            # Count visits to each location
            location_counts = {}
            for entry in position_history:
                coords_str = str(entry['coords'])
                location_counts[coords_str] = location_counts.get(coords_str, 0) + 1
        
        # Find most visited location
        if location_counts:
            most_visited_location = max(location_counts, key=location_counts.get)
            most_visited_coords = eval(most_visited_location)
            visit_count = location_counts[most_visited_location]
            
            # Calculate location diversity (number of unique locations visited)
            unique_locations = len(location_counts)
            total_visits = sum(location_counts.values())
            diversity_score = unique_locations / total_visits if total_visits > 0 else 0
            
            # Calculate average distance from hotspots
            avg_distance_from_hotspots = 0
            hotspot_visits = 0
            for location_str, count in location_counts.items():
                coords = eval(location_str)
                min_distance_to_hotspot = min(
                    abs(coords[0] - hx) + abs(coords[1] - hy) 
                    for hx, hy in hotspots
                )
                avg_distance_from_hotspots += min_distance_to_hotspot * count
                
                # Check if in hotspot area (within 2 grid units)
                if min_distance_to_hotspot <= 2:
                    hotspot_visits += count
            
            avg_distance_from_hotspots = avg_distance_from_hotspots / total_visits if total_visits > 0 else 0
            hotspot_time_percentage = (hotspot_visits / total_visits * 100) if total_visits > 0 else 0
            
            # Get top 3 most visited locations
            sorted_locations = sorted(location_counts.items(), key=lambda x: x[1], reverse=True)
            top_3_locations = [f"{loc}({count})" for loc, count in sorted_locations[:3]]
            
            vehicle_visit_stats[vehicle_id] = {
                'vehicle_type': vehicle['type'],
                'most_visited_location': most_visited_location,
                'most_visited_coords': most_visited_coords,
                'visit_count': visit_count,
                'unique_locations': unique_locations,
                'diversity_score': round(diversity_score, 3),
                'avg_distance_from_hotspots': round(avg_distance_from_hotspots, 2),
                'hotspot_time_percentage': round(hotspot_time_percentage, 1),
                'top_3_locations': ', '.join(top_3_locations),
                'location_counts': location_counts
            }
        else:
            # Fallback for vehicles with no data
            vehicle_visit_stats[vehicle_id] = {
                'vehicle_type': vehicle['type'],
                'most_visited_location': 'N/A',
                'most_visited_coords': vehicle['coordinates'],
                'visit_count': 0,
                'unique_locations': 0,
                'diversity_score': 0.0,
                'avg_distance_from_hotspots': 0.0,
                'hotspot_time_percentage': 0.0,
                'top_3_locations': 'N/A',
                'location_counts': {}
            }
    
    return vehicle_visit_stats


def print_vehicle_visit_summary(vehicle_visit_stats_list):
    """Print summary of vehicle visit patterns across all episodes"""
    if not vehicle_visit_stats_list:
        print("⚠ No vehicle visit data available")
        return
    
    print("\n" + "="*60)
    print("🚗 车辆访问模式总结")
    print("="*60)
    
    # Aggregate statistics across all episodes
    all_vehicles_data = {}
    location_popularity = {}
    
    for episode_visits in vehicle_visit_stats_list:
        for vehicle_id, visit_info in episode_visits.items():
            if vehicle_id not in all_vehicles_data:
                all_vehicles_data[vehicle_id] = {
                    'vehicle_type': visit_info['vehicle_type'],
                    'total_visits': 0,
                    'total_unique_locations': 0,
                    'total_hotspot_time': 0,
                    'episodes_count': 0
                }
            
            data = all_vehicles_data[vehicle_id]
            data['total_visits'] += visit_info['visit_count']
            data['total_unique_locations'] += visit_info['unique_locations']
            data['total_hotspot_time'] += visit_info['hotspot_time_percentage']
            data['episodes_count'] += 1
            
            # Track location popularity
            for location, count in visit_info.get('location_counts', {}).items():
                if location not in location_popularity:
                    location_popularity[location] = 0
                location_popularity[location] += count
    
    # Vehicle type analysis
    ev_vehicles = {vid: data for vid, data in all_vehicles_data.items() if data['vehicle_type'] == 'EV'}
    aev_vehicles = {vid: data for vid, data in all_vehicles_data.items() if data['vehicle_type'] == 'AEV'}
    
    print(f"📈 车辆类型统计:")
    print(f"   EV车辆数量: {len(ev_vehicles)}")
    print(f"   AEV车辆数量: {len(aev_vehicles)}")
    
    # Calculate averages
    if ev_vehicles:
        avg_ev_hotspot_time = np.mean([data['total_hotspot_time'] / data['episodes_count'] 
                                      for data in ev_vehicles.values()])
        print(f"   EV平均热点区域时间: {avg_ev_hotspot_time:.1f}%")
    
    if aev_vehicles:
        avg_aev_hotspot_time = np.mean([data['total_hotspot_time'] / data['episodes_count'] 
                                       for data in aev_vehicles.values()])
        print(f"   AEV平均热点区域时间: {avg_aev_hotspot_time:.1f}%")
    
    # Most popular locations
    if location_popularity:
        print(f"\n📍 最受欢迎的位置 (前10名):")
        sorted_locations = sorted(location_popularity.items(), key=lambda x: x[1], reverse=True)
        for i, (location, visits) in enumerate(sorted_locations[:10], 1):
            coords = eval(location) if isinstance(location, str) and '(' in location else location
            print(f"   {i:2d}. {coords}: {visits} 次访问")
    
    # Vehicle mobility analysis
    print(f"\n🚛 车辆移动性分析:")
    if all_vehicles_data:
        avg_unique_locations = np.mean([data['total_unique_locations'] / data['episodes_count'] 
                                       for data in all_vehicles_data.values()])
        avg_visits_per_episode = np.mean([data['total_visits'] / data['episodes_count'] 
                                         for data in all_vehicles_data.values()])
        
        print(f"   平均每episode访问的不同位置数: {avg_unique_locations:.1f}")
        print(f"   平均每episode总访问次数: {avg_visits_per_episode:.1f}")
        
        # Identify most and least mobile vehicles
        mobility_scores = {vid: data['total_unique_locations'] / data['episodes_count'] 
                          for vid, data in all_vehicles_data.items()}
        
        most_mobile = max(mobility_scores, key=mobility_scores.get)
        least_mobile = min(mobility_scores, key=mobility_scores.get)
        
        print(f"   最活跃车辆: Vehicle {most_mobile} ({mobility_scores[most_mobile]:.1f} 个不同位置/episode)")
        print(f"   最不活跃车辆: Vehicle {least_mobile} ({mobility_scores[least_mobile]:.1f} 个不同位置/episode)")


def save_episode_stats_to_excel(env, episode_stats, results_dir, vehicle_visit_stats=None):
    """Save detailed episode statistics to Excel file including vehicle visit patterns, ADP values, and spatial analysis"""
    if not episode_stats:
        print("⚠ No episode statistics to save")
        return
    
    # Create DataFrame from episode statistics
    df = pd.DataFrame(episode_stats)
    
    # Extract ADP value and demand pattern information
    adpvalue = getattr(env, 'adp_value', 1.0)
    demand_pattern = "intense" if getattr(env, 'use_intense_requests', True) else "random"
    charging_penalty = getattr(env, 'charging_penalty', 2.0)
    unserved_penalty = getattr(env, 'unserved_penalty', 1.5)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"episode_statistics_adp{adpvalue}_demand{demand_pattern}_{env.heuristic_battery_threshold}_{timestamp}.xlsx"
    excel_path = results_dir / excel_filename
    
    # Generate spatial visualization
    spatial_image_path = results_dir / f"spatial_analysis_adp{adpvalue}_demand{demand_pattern}_{timestamp}.png"
    
    try:
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main statistics sheet
            df.to_excel(writer, sheet_name='Episode_Statistics', index=False)
            
            # ADP Configuration sheet
            adp_config_data = {
                'Parameter': [
                    'ADP_Value',
                    'Demand_Pattern',
                    'Charging_Penalty',
                    'Unserved_Penalty',
                    'Grid_Size',
                    'Number_of_Vehicles',
                    'Number_of_Stations',
                    'Episode_Length',
                    'Request_Generation_Rate',
                    'Vehicle_Types',
                    'Hotspot_Configuration'
                ],
                'Value': [
                    adpvalue,
                    demand_pattern,
                    charging_penalty,
                    unserved_penalty,
                    env.grid_size,
                    env.num_vehicles,
                    env.num_stations,
                    env.episode_length,
                    env.request_generation_rate,
                    f"EV: {sum(1 for v in env.vehicles.values() if v['type'] == 'EV')}, AEV: {sum(1 for v in env.vehicles.values() if v['type'] == 'AEV')}",
                    "3 hotspots with weights [0.6, 0.3, 0.1]" if demand_pattern == "intense" else "Random distribution"
                ],
                'Description': [
                    'Weight for Q-value contribution in optimization',
                    'Request generation pattern (intense=hotspots, random=uniform)',
                    'Penalty coefficient for charging actions',
                    'Penalty coefficient for unserved requests',
                    'Size of the simulation grid',
                    'Total number of vehicles in simulation',
                    'Total number of charging stations',
                    'Length of each episode in time steps',
                    'Probability of generating new request each step',
                    'Distribution of vehicle types',
                    'Spatial distribution pattern for request generation'
                ]
            }
            
            adp_config_df = pd.DataFrame(adp_config_data)
            adp_config_df.to_excel(writer, sheet_name='ADP_Configuration', index=False)
            
            # Demand Pattern Analysis sheet
            if hasattr(env, 'request_generation_history') and env.request_generation_history:
                demand_data = []
                hotspot_counts = {0: 0, 1: 0, 2: 0}  # Track requests per hotspot
                
                for req_info in env.request_generation_history:
                    hotspot_idx = req_info.get('hotspot_idx', -1)
                    if hotspot_idx in hotspot_counts:
                        hotspot_counts[hotspot_idx] += 1
                    
                    demand_data.append({
                        'Pickup_X': req_info['pickup_coords'][0],
                        'Pickup_Y': req_info['pickup_coords'][1],
                        'Dropoff_X': req_info['dropoff_coords'][0],
                        'Dropoff_Y': req_info['dropoff_coords'][1],
                        'Hotspot_Index': hotspot_idx,
                        'Generation_Time': req_info['time']
                    })
                
                if demand_data:
                    demand_df = pd.DataFrame(demand_data)
                    demand_df.to_excel(writer, sheet_name='Demand_Pattern', index=False)
                    
                    # Hotspot statistics
                    total_requests = len(demand_data)
                    hotspot_stats = []
                    for hotspot_id, count in hotspot_counts.items():
                        percentage = (count / total_requests * 100) if total_requests > 0 else 0
                        hotspot_stats.append({
                            'Hotspot_ID': hotspot_id,
                            'Request_Count': count,
                            'Percentage': f"{percentage:.1f}%",
                            'Expected_Percentage': ['60%', '30%', '10%'][hotspot_id] if hotspot_id < 3 else 'N/A'
                        })
                    
                    hotspot_stats_df = pd.DataFrame(hotspot_stats)
                    hotspot_stats_df.to_excel(writer, sheet_name='Hotspot_Statistics', index=False)
            
            # Vehicle Visit Patterns sheet
            if vehicle_visit_stats:
                visit_data = []
                for episode_idx, episode_visits in enumerate(vehicle_visit_stats):
                    for vehicle_id, visit_info in episode_visits.items():
                        visit_data.append({
                            'Episode': episode_idx + 1,
                            'Vehicle_ID': vehicle_id,
                            'Vehicle_Type': visit_info.get('vehicle_type', 'Unknown'),
                            'Most_Visited_Location': visit_info.get('most_visited_location', 'N/A'),
                            'Most_Visited_Coords': visit_info.get('most_visited_coords', 'N/A'),
                            'Visit_Count': visit_info.get('visit_count', 0),
                            'Total_Unique_Locations': visit_info.get('unique_locations', 0),
                            'Location_Diversity_Score': visit_info.get('diversity_score', 0.0),
                            'Average_Distance_from_Hotspots': visit_info.get('avg_distance_from_hotspots', 0.0),
                            'Time_in_Hotspot_Areas_%': visit_info.get('hotspot_time_percentage', 0.0),
                            'Top_3_Visited_Locations': visit_info.get('top_3_locations', 'N/A')
                        })
                
                if visit_data:
                    visit_df = pd.DataFrame(visit_data)
                    visit_df.to_excel(writer, sheet_name='Vehicle_Visit_Patterns', index=False)
                
                # Location Heatmap Summary
                location_summary = {}
                for episode_visits in vehicle_visit_stats:
                    for vehicle_id, visit_info in episode_visits.items():
                        for location, count in visit_info.get('location_counts', {}).items():
                            if location not in location_summary:
                                location_summary[location] = {'total_visits': 0, 'vehicles_visited': set()}
                            location_summary[location]['total_visits'] += count
                            location_summary[location]['vehicles_visited'].add(vehicle_id)
                
                if location_summary:
                    heatmap_data = []
                    for location, info in location_summary.items():
                        coords = eval(location) if isinstance(location, str) and '(' in location else location
                        heatmap_data.append({
                            'Location_Coords': coords,
                            'Total_Visits': info['total_visits'],
                            'Unique_Vehicles_Visited': len(info['vehicles_visited']),
                            'Average_Visits_per_Vehicle': info['total_visits'] / len(info['vehicles_visited']) if info['vehicles_visited'] else 0
                        })
                    
                    heatmap_df = pd.DataFrame(heatmap_data)
                    heatmap_df = heatmap_df.sort_values('Total_Visits', ascending=False)
                    heatmap_df.to_excel(writer, sheet_name='Location_Heatmap', index=False)
            
            # Summary statistics sheet
            summary_stats = {
                'Metric': [
                    'Total Episodes',
                    'Average Orders per Episode',
                    'Average Accepted Orders per Episode',
                    'Average Rejected Orders per Episode',
                    'Overall Rejection Rate (%)',
                    'Average Battery Level',
                    'Total EV Vehicles',
                    'Total AEV Vehicles',
                    'EV Rejection Rate (%)',
                    'AEV Rejection Rate (%)',
                    'Average Neural Network Loss',
                    'Neural Network Loss Std Dev',
                    'Average Training Steps per Episode'
                ],
                'Value': [
                    len(df),
                    df['total_orders'].mean(),
                    df['accepted_orders'].mean(),
                    df['rejected_orders'].mean(),
                    (df['rejected_orders'].sum() / df['total_orders'].sum() * 100) if df['total_orders'].sum() > 0 else 0,
                    df['avg_battery_level'].mean(),
                    df['ev_count'].iloc[0] if not df.empty else 0,
                    df['aev_count'].iloc[0] if not df.empty else 0,
                    (df['ev_rejected'].sum() / (df['accepted_orders'].sum() + df['ev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['ev_rejected'].sum()) > 0 else 0,
                    (df['aev_rejected'].sum() / (df['accepted_orders'].sum() + df['aev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['aev_rejected'].sum()) > 0 else 0,
                    df['neural_network_loss'].mean() if 'neural_network_loss' in df.columns else 0,
                    df['neural_network_loss_std'].mean() if 'neural_network_loss_std' in df.columns else 0,
                    df['training_steps_in_episode'].mean() if 'training_steps_in_episode' in df.columns else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_stats)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Vehicle type comparison sheet
            if not df.empty:
                vehicle_comparison = pd.DataFrame({
                    'Vehicle_Type': ['EV', 'AEV'],
                    'Count': [df['ev_count'].iloc[0], df['aev_count'].iloc[0]],
                    'Total_Rejected_Orders': [df['ev_rejected'].sum(), df['aev_rejected'].sum()],
                    'Rejection_Rate_%': [
                        (df['ev_rejected'].sum() / (df['accepted_orders'].sum() + df['ev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['ev_rejected'].sum()) > 0 else 0,
                        (df['aev_rejected'].sum() / (df['accepted_orders'].sum() + df['aev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['aev_rejected'].sum()) > 0 else 0
                    ]
                })
                vehicle_comparison.to_excel(writer, sheet_name='Vehicle_Comparison', index=False)
        
        # Generate and save spatial visualization
        try:
            spatial_viz = SpatialVisualization(env.grid_size)
            success = spatial_viz.create_comprehensive_spatial_plot(
                env=env, 
                save_path=spatial_image_path,
                adpvalue=adpvalue,
                demand_pattern=demand_pattern
            )
            
            if success:
                print(f"✓ Spatial visualization saved: {spatial_image_path}")
            else:
                print(f"⚠ Failed to generate spatial visualization")
            
        except Exception as e:
            print(f"⚠ Error generating spatial visualization: {e}")
        
        print(f"✓ Episode statistics saved to Excel: {excel_path}")
        print(f"  - Episode_Statistics: Detailed data for each episode")
        print(f"  - ADP_Configuration: System parameters and settings")
        print(f"  - Demand_Pattern: Request generation analysis")
        print(f"  - Hotspot_Statistics: Hotspot performance metrics")
        print(f"  - Summary: Overall performance metrics")
        print(f"  - Vehicle_Comparison: EV vs AEV performance comparison")
        if vehicle_visit_stats:
            print(f"  - Vehicle_Visit_Patterns: Individual vehicle movement analysis")
            print(f"  - Location_Heatmap: Aggregated location popularity")
        
        return excel_path, spatial_image_path
        print(f"  - Summary: Overall performance metrics")
        print(f"  - Vehicle_Comparison: EV vs AEV performance comparison")
        
    except Exception as e:
        print(f"❌ Error saving Excel file: {e}")
        # Save as CSV as backup
        csv_path = results_dir / f"episode_statistics_{timestamp}.csv"
        df.to_csv(csv_path, index=False)
        print(f"✓ Backup saved as CSV: {csv_path}")


def analyze_results(results):
    """Analyze test results including EV/AEV behavior"""
    print("\n=== Enhanced Results Analysis ===")
    
    # Basic statistics
    total_episodes = len(results['episode_rewards'])
    avg_reward = np.mean(results['episode_rewards'])
    total_charging = len(results['charging_events'])
    avg_battery = np.mean(results['battery_levels'])
    
    print(f"Total episodes: {total_episodes}")
    print(f"Average reward: {avg_reward:.2f}")
    print(f"Total charging events: {total_charging}")
    print(f"Average battery level: {avg_battery:.2f}")
    
    # Vehicle type analysis
    if 'environment_stats' in results and results['environment_stats']:
        latest_stats = results['environment_stats'][-1]
        ev_count = latest_stats.get('ev_count', 0)
        aev_count = latest_stats.get('aev_count', 0)
        total_rejected = latest_stats.get('total_rejected_requests', 0)
        ev_rejected = latest_stats.get('ev_rejected_requests', 0)
        aev_rejected = latest_stats.get('aev_rejected_requests', 0)
        
        print(f"\nVehicle Type Analysis:")
        print(f"  EV vehicles: {ev_count}")
        print(f"  AEV vehicles: {aev_count}")
        print(f"  Total rejected requests: {total_rejected}")
        print(f"  EV rejected requests: {ev_rejected}")
        print(f"  AEV rejected requests: {aev_rejected}")
        
        if ev_count > 0:
            ev_rejection_rate = ev_rejected / max(1, ev_rejected + latest_stats.get('completed_requests', 0))
            print(f"  EV rejection rate: {ev_rejection_rate:.2%}")
        
        if aev_count > 0:
            aev_rejection_rate = aev_rejected / max(1, aev_rejected + latest_stats.get('completed_requests', 0))
            print(f"  AEV rejection rate: {aev_rejection_rate:.2%}")
    
    # Request fulfillment analysis
    if 'environment_stats' in results and results['environment_stats']:
        completed_requests = sum(stats.get('completed_requests', 0) for stats in results['environment_stats'])
        total_earnings = sum(stats.get('total_earnings', 0) for stats in results['environment_stats'])
        avg_fulfillment = np.mean([stats.get('request_fulfillment_rate', 0) for stats in results['environment_stats']])
        
        print(f"\nRequest Fulfillment Analysis:")
        print(f"  Total completed requests: {completed_requests}")
        print(f"  Total earnings: {total_earnings:.2f}")
        print(f"  Average fulfillment rate: {avg_fulfillment:.2%}")
    
    # Charging behavior analysis
    if results['charging_events']:
        station_usage = defaultdict(int)
        duration_stats = []
        
        for event in results['charging_events']:
            station_usage[event['station_id']] += 1
            duration_stats.append(event['duration'])
        
        print(f"\nCharging Station Usage Statistics:")
        for station_id, count in station_usage.items():
            print(f"  Station {station_id}: {count} times")
        
        print(f"Average charging duration: {np.mean(duration_stats):.1f}")
        print(f"Max charging duration: {max(duration_stats)}")
        print(f"Min charging duration: {min(duration_stats)}")
    
    # Analyze charging usage history across all episodes
    if 'episode_detailed_stats' in results:
        all_usage_data = []
        for episode_stats in results['episode_detailed_stats']:
            if 'charging_usage_history' in episode_stats and episode_stats['charging_usage_history']:
                for usage_point in episode_stats['charging_usage_history']:
                    all_usage_data.append(usage_point['vehicles_per_station'])
        
        if all_usage_data:
            print(f"\nOverall Charging Station Usage Analysis:")
            print(f"  Total time steps recorded: {len(all_usage_data)}")
            print(f"  Average vehicles per station: {np.mean(all_usage_data):.3f}")
            print(f"  Maximum vehicles per station: {max(all_usage_data):.3f}")
            print(f"  Minimum vehicles per station: {min(all_usage_data):.3f}")
            print(f"  Standard deviation: {np.std(all_usage_data):.3f}")
        else:
            print(f"\nNo charging usage history data found across episodes")
    
    # Learning curve analysis
    improvement = 0
    if len(results['episode_rewards']) > 10:
        early_rewards = results['episode_rewards'][:10]
        late_rewards = results['episode_rewards'][-10:]
        improvement = np.mean(late_rewards) - np.mean(early_rewards)
        print(f"\nReward improvement: {improvement:.2f}")
        
        if improvement > 0:
            print("✓ Shows learning improvement trend")
        else:
            print("⚠ Learning effectiveness needs improvement")
    
    # Battery management assessment
    if results['battery_levels']:
        min_battery = min(results['battery_levels'])
        max_battery = max(results['battery_levels'])
        battery_stability = np.std(results['battery_levels'])
        
        print(f"\nBattery Management Analysis:")
        print(f"  Lowest average battery: {min_battery:.2f}")
        print(f"  Highest average battery: {max_battery:.2f}")
        print(f"  Battery stability (std dev): {battery_stability:.3f}")
        
        if min_battery > 0.2:
            print("✓ Good battery management, no severe low battery issues")
        else:
            print("⚠ Risk of critically low battery levels")
    
    return {
        'avg_reward': avg_reward,
        'total_charging': total_charging,
        'avg_battery': avg_battery,
        'improvement': improvement,
        'min_battery': min_battery if results['battery_levels'] else 0,
        'battery_stability': battery_stability if results['battery_levels'] else 0
    }


def visualize_integrated_results(env,results, assignmentgurobi=True):
    """可视化集成测试结果"""
    print("\n=== 生成可视化图表 ===")
    
    try:
        # 创建可视化器
        visualizer = ChargingIntegrationVisualization(figsize=(15, 10))
        
        # 保存路径 - 根据assignmentgurobi选择目录
        if assignmentgurobi:
            results_dir = Path("results/integrated_tests")
        else:
            results_dir = Path("results/integrated_tests_h")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        adpvalue = getattr(env, 'adp_value', 1.0)
        plot_path = results_dir / f"integrated_charging_results_{adpvalue}.png"
        fig1 = visualizer.plot_integrated_results(results,  save_path=str(plot_path))

        # 生成策略分析图表
        strategy_plot_path = results_dir / f"charging_strategy_analysis_{adpvalue}.png"
        fig2 = visualizer.plot_charging_strategy_analysis(results, save_path=str(strategy_plot_path))

        print(f"✓ 主要结果图表已保存至: {plot_path}")
        print(f"✓ 策略分析图表已保存至: {strategy_plot_path}")
        
        # 关闭图表以释放内存
        plt.close(fig1)
        plt.close(fig2)
        
        return True
            
    except Exception as e:
        print(f"❌ 可视化失败: {e}")
        return False


def generate_integration_report(results, analysis, assignmentgurobi=True):
    """生成集成测试报告"""
    print("\n=== 生成测试报告 ===")
    
    # 根据assignmentgurobi选择目录
    if assignmentgurobi:
        results_dir = Path("results/integrated_tests")
    else:
        results_dir = Path("results/integrated_tests_h")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    report_path = results_dir / "integration_report.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("充电行为集成测试报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("测试概况:\n")
        f.write(f"- 总回合数: {len(results['episode_rewards'])}\n")
        f.write(f"- 平均奖励: {analysis['avg_reward']:.2f}\n")
        f.write(f"- 总充电次数: {analysis['total_charging']}\n")
        f.write(f"- 平均电池电量: {analysis['avg_battery']:.2f}\n")
        f.write(f"- 奖励改进: {analysis['improvement']:.2f}\n")

        f.write(f"- 最低电量: {analysis['min_battery']:.2f}\n")
        f.write(f"- 电量稳定性: {analysis['battery_stability']:.3f}\n\n")
        
        f.write("充电行为评估:\n")
        if analysis['total_charging'] > 0:
            f.write("✓ 充电功能正常工作\n")
            avg_charging_per_episode = analysis['total_charging'] / len(results['episode_rewards'])
            f.write(f"✓ 平均每回合 {avg_charging_per_episode:.1f} 次充电\n")
            
            if avg_charging_per_episode > 3:
                f.write("✓ 充电频率合理\n")
            else:
                f.write("⚠ 充电频率可能偏低\n")
        else:
            f.write("❌ 未检测到充电行为\n")
        
        if analysis['avg_battery'] > 0.4:
            f.write("✓ 电池管理优秀\n")
        elif analysis['avg_battery'] > 0.25:
            f.write("✓ 电池管理良好\n")
        else:
            f.write("⚠ 电池管理需要改进\n")
        
        if analysis['improvement'] > 5:
            f.write("✓ 学习效果显著\n")
        elif analysis['improvement'] > 0:
            f.write("✓ 学习效果良好\n")
        else:
            f.write("⚠ 学习效果待改进\n")
        
        f.write(f"\n充电策略质量评估:\n")
        if analysis['min_battery'] > 0.15:
            f.write("✓ 很好地避免了电量危机\n")
        else:
            f.write("⚠ 存在电量管理风险\n")
            
        if analysis['battery_stability'] < 0.1:
            f.write("✓ 电量管理稳定\n")
        else:
            f.write("⚠ 电量波动较大\n")
    
    print(f"✓ 测试报告已保存至: {report_path}")


def testtimeperformance(carnumlist):
    import time
    num_episodes = 10
    use_intense_requests = False
    config_manager = ConfigManager()
    print("📋 加载配置参数...")
    config_manager.print_config('training')
    config_manager.print_config('environment')
    
    # 从配置获取参数
    training_config = get_training_config()
    env_config = config_manager.get_environment_config()
    
    heuristictime = []
    ILPtimelist = []
    ADPtimelist = []
    for carnum in carnumlist:
        adpvalue = 0
        assignmentgurobi =False
        start_time = time.time()
        results, env = run_charging_integration_test_threshold(adpvalue,num_episodes,use_intense_requests,assignmentgurobi,batch_size=256, num_vehicles = carnum)
        end_time = time.time()
        
        heuristictime.append(end_time - start_time)
    np.savetxt("heuristictime.txt",heuristictime)
    assignmentgurobi =True
    for carnum in carnumlist:
        start_time = time.time()
        results, env = run_charging_integration_test(0, num_episodes=num_episodes, use_intense_requests=use_intense_requests, assignmentgurobi=assignmentgurobi, num_vehicles = carnum)
        end_time = time.time()
        ILPtimelist.append(end_time - start_time)
    np.savetxt("ILPtimelist.txt",ILPtimelist)   
    for carnum in carnumlist:
        start_time = time.time()
        results, env = run_charging_integration_test(0.1, num_episodes=num_episodes, use_intense_requests=use_intense_requests, assignmentgurobi=assignmentgurobi, num_vehicles = carnum)
        end_time = time.time()
        ADPtimelist.append(end_time - start_time)
    np.savetxt("ADPtimelist.txt",ADPtimelist)   
        
def main():

    carnumlist = [i*5 for i in range(1,6)]
    
    print("🚗⚡ 充电行为集成测试程序")
    print("使用src文件夹中的Environment和充电组件")
    print("-" * 60)

    # 加载配置
    config_manager = ConfigManager()
    print("📋 加载配置参数...")
    config_manager.print_config('training')
    config_manager.print_config('environment')
    
    # 从配置获取参数
    training_config = get_training_config()
    env_config = config_manager.get_environment_config()
    
    use_intense_requests = False
    try:
        # 从配置获取训练参数
        num_episodes = 100
        # print(f"📊 使用配置参数: episodes={num_episodes}")
        
        # batch_size = training_config.get('batch_size', 256)
        # adpvalue = 0
        # assignmentgurobi =False
        # # for charge_th in charge_threshold:
        # results, env = run_charging_integration_test_threshold(adpvalue,num_episodes,use_intense_requests,assignmentgurobi,batch_size=256)

        #     # 分析结果
        # analysis = analyze_results(results)
        
        # # 生成可视化
        # success = visualize_integrated_results(env,results, assignmentgurobi=assignmentgurobi)
        
        # # 空间分布可视化已在Excel导出中生成
        # print(f"\n🗺️  空间分布分析已完成，图像路径: {results.get('spatial_image_path', 'N/A')}")
        
        # # 生成传统的空间分布分析（用于兼容性）
        # spatial_viz = SpatialVisualization(env.grid_size)
        # spatial_analysis = spatial_viz.analyze_spatial_patterns(env)
        # spatial_viz.print_spatial_analysis(spatial_analysis)
        
    
        # generate_integration_report(results, analysis, assignmentgurobi=assignmentgurobi)
        
        # # 输出车辆访问模式总结
        # print_vehicle_visit_summary(results.get('vehicle_visit_stats', []))
        
        # print("\n" + "="*60)
        # assignment_type = "Gurobi" if assignmentgurobi else "Heuristic"
        # print(f"🎉 集成测试完成! (ADP={adpvalue}, {assignment_type})")
        # print("📊 结果摘要:")
        # print(f"   - 平均奖励: {analysis['avg_reward']:.2f}")
        # print(f"   - 充电次数: {analysis['total_charging']}")
        # print(f"   - 平均电量: {analysis['avg_battery']:.2f}")
        # print(f"   - 奖励改进: {analysis['improvement']:.2f}")
        

        print("\n" + "="*60)
        assignmentgurobi =True
        results_folder = "results/integrated_tests/" if assignmentgurobi else "results/integrated_tests_h/"
        print(f"📁 请检查 {results_folder} 文件夹中的详细结果")
        print("="*60)
        # adplist = [0]
        adplist = [0, 0.1]
        #adplist = [0.1]
        for adpvalue in adplist:
            assignment_type = "Gurobi" if assignmentgurobi else "Heuristic"
            print(f"\n⚡ 开始集成测试 (ADP={adpvalue}, Assignment={assignment_type})")
            results, env = run_charging_integration_test(adpvalue, num_episodes=num_episodes, use_intense_requests=use_intense_requests, assignmentgurobi=assignmentgurobi)

            # 分析结果
            analysis = analyze_results(results)
            
            # 生成可视化
            success = visualize_integrated_results(env, results, assignmentgurobi=assignmentgurobi)
            
            # 空间分布可视化已在Excel导出中生成
            print(f"\n🗺️  空间分布分析已完成，图像路径: {results.get('spatial_image_path', 'N/A')}")
            
            # 生成传统的空间分布分析（用于兼容性）
            spatial_viz = SpatialVisualization(env.grid_size)
            spatial_analysis = spatial_viz.analyze_spatial_patterns(env, adp_value=adpvalue)
            spatial_viz.print_spatial_analysis(spatial_analysis)
            
            # 生成报告
            generate_integration_report(results, analysis, assignmentgurobi=assignmentgurobi)
            
            # 输出车辆访问模式总结
            print_vehicle_visit_summary(results.get('vehicle_visit_stats', []))
            
            print("\n" + "="*60)
            print(f"🎉 集成测试完成! (ADP={adpvalue}, {assignment_type})")
            print("📊 结果摘要:")
            print(f"   - 平均奖励: {analysis['avg_reward']:.2f}")
            print(f"   - 充电次数: {analysis['total_charging']}")
            print(f"   - 平均电量: {analysis['avg_battery']:.2f}")
            print(f"   - 奖励改进: {analysis['improvement']:.2f}")
            
            if success:
                print("📈 可视化图表生成成功")
            
            results_folder = "results/integrated_tests/" if assignmentgurobi else "results/integrated_tests_h/"
            print(f"📁 请检查 {results_folder} 文件夹中的详细结果")
            print("="*60)
            
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()



if __name__ == "__main__":
    main()