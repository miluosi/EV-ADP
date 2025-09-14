"""
ADP Training Module - 电动车充电优化训练器
"""
import os
import sys
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
from datetime import datetime
from src.ChargingIntegrationVisualization import ChargingIntegrationVisualization
# 导入配置管理器
from config.config_manager import ConfigManager, get_config, get_training_config, get_sampling_config

# 导入核心组件
from .Environment import ChargingIntegratedEnvironment
from .ValueFunction_pytorch import PyTorchChargingValueFunction
from .Action import Action, ChargingAction, ServiceAction
from .Request import Request
from .charging_station import ChargingStationManager, ChargingStation
from .CentralAgent import CentralAgent
from .SpatialVisualization import SpatialVisualization


class ADPTrainer:
    """ADP训练器类 - 负责电动车充电优化的强化学习训练"""
    
    def __init__(self, config_manager=None):
        """
        初始化训练器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager or ConfigManager()
        self.training_config = self.config_manager.get_training_config()
        self.env_config = self.config_manager.get_environment_config()
        self.sampling_config = self.config_manager.get_sampling_config()
        self.adp_value = self.training_config.get('adp_value', 0)
        self.assignmentgurobi = self.training_config.get('assignmentgurobi', True)
        # 训练状态  
        self.env = None
        self.batch_size = self.training_config.get('batch_size', 256)
        self.value_function = None
        self.training_history = {
            'episode_rewards': [],
            'training_losses': [],
            'q_values': [],
            'exploration_rates': []
        }
        
        print("🚀 ADPTrainer初始化完成")
        print(f"   - 配置加载: {self.config_manager.config_path}")
    
    def setup_environment(self, num_vehicles=None, num_stations=None):
        """
        设置训练环境
        
        Args:
            num_vehicles: 车辆数量，默认从配置获取
            num_stations: 充电站数量，默认从配置获取
        """
        num_vehicles = num_vehicles or self.env_config.get('max_vehicles', 40)
        num_stations = num_stations or self.env_config.get('max_charging_stations', 12)
        
        self.env = ChargingIntegratedEnvironment(
            num_vehicles=num_vehicles, 
            num_stations=num_stations
        )
        
        print(f"✓ 环境设置完成: {num_vehicles}辆车, {num_stations}个充电站")
        return self.env
    
    def setup_value_function(self):
        """
        设置价值函数
        
        Args:
            adp_value: ADP参数值
            use_neural_network: 是否使用神经网络
        """
        use_neural_network = self.adp_value > 0 and self.assignmentgurobi
        if use_neural_network and self.adp_value > 0:
            network_config = self.config_manager.get_network_config()
            
            self.value_function = PyTorchChargingValueFunction(
                grid_size=self.env.grid_size,
                num_vehicles=self.env.num_vehicles,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                episode_length=self.env.episode_length,
                max_requests=1000,
            )
            
            # 设置价值函数到环境
            self.env.set_value_function(self.value_function)
            
            print(f"✓ 神经网络价值函数初始化完成")
            print(f"   - 网络参数数量: {sum(p.numel() for p in self.value_function.network.parameters())}")
            print(f"   - 设备: {self.value_function.device}")
        else:
            self.value_function = None
            print(f"✓ 不使用神经网络 (ADP={self.adp_value})")

        return self.value_function
    
    def run_charging_integration_test(self, num_episodes,use_intense_requests,):
        """Run charging integration test with EV/AEV analysis"""
        print("=== Starting Enhanced Charging Behavior Integration Test ===")
        
        # Create environment with significantly more complexity for better learning
        num_vehicles = 50  # Doubled vehicles for more interaction
        num_stations = 12
        env = self.env or self.setup_environment(num_vehicles=num_vehicles, num_stations=num_stations)
        assignmentgurobi = self.assignmentgurobi
        # Initialize neural network-based ValueFunction for decision making only if needed
        # Use PyTorchChargingValueFunction with neural network only when ADP > 0 and assignmentgurobi is True
        use_neural_network = self.adp_value > 0 and assignmentgurobi
        batch_size = self.batch_size 
        if use_neural_network:
            value_function = self.value_function
        else:
            value_function = None
  

        
        # Enhanced training parameters for complex environment
        training_frequency = 2  # Train every 2 steps for much more frequent learning
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


            
            # Reset environment
            states = env.reset()
            episode_reward = 0
            episode_charging_events = []
            episode_losses = []
            
            # Run one episode
            for step in range(env.episode_length):
                # Generate actions using ValueFunction
                actions = {}
                states_for_training = []
                actions_for_training = []


                current_requests = list(env.active_requests.values())
                actions = env.simulate_motion(agents=[], current_requests=current_requests, rebalance=True)
                next_states, rewards, done, info = env.step(actions)

                # Debug: Output step statistics every 100 steps
                if step % 100 == 0:
                    stats = env.get_stats()
                    active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                    assigned_vehicles = len([v for v in env.vehicles.values() if v['assigned_request'] is not None])
                    charging_vehicles = len([v for v in env.vehicles.values() if v['charging_station'] is not None])
                    idle_vehicles = len([v for v in env.vehicles.values() 
                                    if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
                    step_reward = sum(rewards.values())
                    print(f"Step {step}: Active requests: {active_requests}, Assigned: {assigned_vehicles}, Charging: {charging_vehicles}, Idle: {idle_vehicles}, Step reward: {step_reward:.2f}")
                    
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
                                idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, step)
                                assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, step)
                                charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, step)
                                
                                print(f"  Neural Network Status:")
                                print(f"    Training step: {training_step}, Buffer: {buffer_size}, Recent loss: {recent_loss:.4f}")
                                print(f"    Raw Q-values (no normalization): Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
                                print(f"    Note: Gurobi uses these raw Q-values directly in optimization objective")
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
            episode_stats['charging_events_count'] = len(episode_charging_events)/env.episode_length  # Average per step
            
            # Output rebalancing assignment statistics
            rebalancing_calls = episode_stats.get('total_rebalancing_calls', 0)
            total_assignments = episode_stats.get('total_rebalancing_assignments', 0)
            avg_assignments = episode_stats.get('avg_rebalancing_assignments_per_call', 0)
            
            print(f"Episode {episode + 1} Completed:")
            print(f"  Reward: {episode_reward:.2f}")
            print(f"  Orders: Total={episode_stats['total_orders']}, Accepted={episode_stats['accepted_orders']}, Completed={episode_stats['completed_orders']}, Rejected={episode_stats['rejected_orders']}")
            print(f"  Battery: {episode_stats['avg_battery_level']:.2f}")
            print(f"  Station Usage: {episode_stats['avg_vehicles_per_station']:.1f} vehicles/station")
            print(f"  Rebalancing: {rebalancing_calls} calls, {total_assignments} total assignments, {avg_assignments:.1f} avg assignments/call")
            
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
                        idle_q = value_function.get_idle_q_value(sample_vehicle_id, sample_location, sample_battery, env.current_time)
                        assign_q = value_function.get_q_value(sample_vehicle_id, "assign_1", sample_location, sample_location+1, env.current_time, battery_level=sample_battery)
                        charge_q = value_function.get_q_value(sample_vehicle_id, "charge_1", sample_location, sample_location+5, env.current_time, battery_level=sample_battery)
                        
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
        excel_path, spatial_path = self.save_episode_stats_to_excel(env, results['episode_detailed_stats'], results_dir, results.get('vehicle_visit_stats'))
        
        # Store file paths in results for reference
        results['excel_path'] = excel_path
        results['spatial_image_path'] = spatial_path
        
        return results

    def save_episode_stats_to_excel(env, episode_stats, results_dir, vehicle_visit_stats=None):
        """Save detailed episode statistics to Excel file including vehicle visit patterns, ADP values, and spatial analysis"""
        env = self.env 
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
        excel_filename = f"episode_statistics_adp{adpvalue}_demand{demand_pattern}_{timestamp}.xlsx"
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
                        'Average Vehicles per Station',
                        'Average Station Utilization Rate (%)',
                        'Total EV Vehicles',
                        'Total AEV Vehicles',
                        'EV Rejection Rate (%)',
                        'AEV Rejection Rate (%)',
                        'Total Earnings',
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
                        df['avg_vehicles_per_station'].mean(),
                        df['station_utilization_rate'].mean() * 100,
                        df['ev_count'].iloc[0] if not df.empty else 0,
                        df['aev_count'].iloc[0] if not df.empty else 0,
                        (df['ev_rejected'].sum() / (df['accepted_orders'].sum() + df['ev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['ev_rejected'].sum()) > 0 else 0,
                        (df['aev_rejected'].sum() / (df['accepted_orders'].sum() + df['aev_rejected'].sum()) * 100) if (df['accepted_orders'].sum() + df['aev_rejected'].sum()) > 0 else 0,
                        df['total_earnings'].sum(),
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
                print(f"🗺️ Generating spatial visualization...")
                spatial_viz = SpatialVisualization(env.grid_size)
                
                # Create comprehensive spatial plot
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Request generation heatmap
                if hasattr(env, 'request_generation_history') and env.request_generation_history:
                    pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
                    pickup_x = [coord[0] for coord in pickup_coords]
                    pickup_y = [coord[1] for coord in pickup_coords]
                    
                    ax1.hist2d(pickup_x, pickup_y, bins=env.grid_size//2, alpha=0.7, cmap='Reds')
                    ax1.set_title(f'Request Generation Heatmap (ADP={adpvalue})\n(Pattern: {demand_pattern})')
                    ax1.set_xlabel('X Coordinate')
                    ax1.set_ylabel('Y Coordinate')
                    ax1.grid(True, alpha=0.3)
                
                # Plot 2: Vehicle position distribution
                vehicle_positions = [v['coordinates'] for v in env.vehicles.values()]
                if vehicle_positions:
                    veh_x = [pos[0] for pos in vehicle_positions]
                    veh_y = [pos[1] for pos in vehicle_positions]
                    
                    # Color by vehicle type
                    ev_x = [pos[0] for v_id, v in env.vehicles.items() if v['type'] == 'EV' for pos in [v['coordinates']]]
                    ev_y = [pos[1] for v_id, v in env.vehicles.items() if v['type'] == 'EV' for pos in [v['coordinates']]]
                    aev_x = [pos[0] for v_id, v in env.vehicles.items() if v['type'] == 'AEV' for pos in [v['coordinates']]]
                    aev_y = [pos[1] for v_id, v in env.vehicles.items() if v['type'] == 'AEV' for pos in [v['coordinates']]]
                    
                    ax2.scatter(ev_x, ev_y, c='blue', alpha=0.6, label='EV', s=50)
                    ax2.scatter(aev_x, aev_y, c='green', alpha=0.6, label='AEV', s=50)
                    ax2.set_title(f'Final Vehicle Distribution (ADP={adpvalue})')
                    ax2.set_xlabel('X Coordinate')
                    ax2.set_ylabel('Y Coordinate')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                
                # Plot 3: Charging station utilization
                if hasattr(env, 'charging_manager') and env.charging_manager.stations:
                    station_data = []
                    for station_id, station in env.charging_manager.stations.items():
                        station_x = station.location % env.grid_size
                        station_y = station.location // env.grid_size
                        utilization = len(station.current_vehicles) / station.max_capacity
                        station_data.append((station_x, station_y, utilization))
                    
                    if station_data:
                        station_x, station_y, utilizations = zip(*station_data)
                        scatter = ax3.scatter(station_x, station_y, c=utilizations, s=200, 
                                            cmap='YlOrRd', alpha=0.8, edgecolor='black')
                        plt.colorbar(scatter, ax=ax3, label='Utilization Rate')
                        ax3.set_title(f'Charging Station Utilization (ADP={adpvalue})')
                        ax3.set_xlabel('X Coordinate')
                        ax3.set_ylabel('Y Coordinate')
                        ax3.grid(True, alpha=0.3)
                
                # Plot 4: Performance summary
                ax4.axis('off')
                performance_text = f"""
    Performance Summary
    ADP Value: {adpvalue}
    Demand Pattern: {demand_pattern}
    Charging Penalty: {charging_penalty}
    Unserved Penalty: {unserved_penalty}

    Episodes: {len(df)}
    Avg Battery Level: {df['avg_battery_level'].mean():.2f}
    Total Orders: {df['total_orders'].sum()}
    Accepted Orders: {df['accepted_orders'].sum()}
    Rejection Rate: {((df['rejected_orders'].sum() / df['total_orders'].sum()) * 100) if df['total_orders'].sum() > 0 else 0:.1f}%

    EV Vehicles: {df['ev_count'].iloc[0] if not df.empty else 0}
    AEV Vehicles: {df['aev_count'].iloc[0] if not df.empty else 0}
                """
                ax4.text(0.1, 0.9, performance_text, transform=ax4.transAxes, 
                        fontsize=11, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
                
                plt.tight_layout()
                plt.savefig(spatial_image_path, dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"✓ Spatial visualization saved: {spatial_image_path}")
                
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

    def visualize_integrated_results(results):
        """可视化集成测试结果"""
        print("\n=== 生成可视化图表 ===")
        assignmentgurobi = self.assignmentgurobi
        try:
            # 创建可视化器
            visualizer = ChargingIntegrationVisualization(figsize=(15, 10))
            
            # 保存路径 - 根据assignmentgurobi选择目录
            if assignmentgurobi:
                results_dir = Path("results/integrated_tests")
            else:
                results_dir = Path("results/integrated_tests_h")
            results_dir.mkdir(parents=True, exist_ok=True)
            
            # 生成主要结果图表
            plot_path = results_dir / "integrated_charging_results.png"
            fig1 = visualizer.plot_integrated_results(results, save_path=str(plot_path))
            
            # 生成策略分析图表
            strategy_plot_path = results_dir / "charging_strategy_analysis.png"
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

   