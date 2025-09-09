"""
Integrated Test: Vehicle Charging Behavior Integration Test using src folder components
"""

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



def run_charging_integration_test(adpvalue,num_episodes,use_intense_requests,assignmentgurobi):
    """Run charging integration test with EV/AEV analysis"""
    print("=== Starting Enhanced Charging Behavior Integration Test ===")
    
    # Create environment with significantly more complexity for better learning
    num_vehicles = 40  # Doubled vehicles for more interaction
    num_stations = 12  # More stations for complex charging decisions
    env = ChargingIntegratedEnvironment(num_vehicles=num_vehicles, num_stations=num_stations)
    
    # Initialize neural network-based ValueFunction for decision making
    # Use PyTorchChargingValueFunction with neural network
    value_function = PyTorchChargingValueFunction(
        grid_size=env.grid_size, 
        num_vehicles=num_vehicles,
        device='cuda' if torch.cuda.is_available() else 'cpu'  # Use GPU if available
    )
    
    # Set the value function in the environment for Q-value calculation
    env.set_value_function(value_function)
    env.adp_value = adpvalue
    env.use_intense_requests = use_intense_requests
    env.assignmentgurobi = assignmentgurobi
    # Exploration parameters for enhanced learning with complex environment
    exploration_episodes = max(1, num_episodes // 2)  # Half episodes for exploration  
    epsilon_start = 0.4  # Higher exploration for complex environment
    epsilon_end = 0.1   # End with 10% random actions
    epsilon_decay = (epsilon_start - epsilon_end) / exploration_episodes
    
    # Enhanced training parameters for complex environment
    training_frequency = 2  # Train every 2 steps for much more frequent learning
    warmup_steps = 100     # Increased warmup for complex environment
    
    print(f"✓ Initialized environment with {num_vehicles} vehicles and {num_stations} charging stations")
    print(f"✓ Initialized PyTorchChargingValueFunction with neural network")
    print(f"   - Network parameters: {sum(p.numel() for p in value_function.network.parameters())}")
    print(f"✓ Enhanced exploration strategy: {exploration_episodes} episodes with epsilon {epsilon_start:.2f} → {epsilon_end:.2f}")
    print(f"   - Training frequency: every {training_frequency} steps after {warmup_steps} warmup steps")
    print(f"   - Using device: {value_function.device}")
    
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

        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        use_exploration = episode < exploration_episodes and random.random() < current_epsilon
            
        
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
            
            for vehicle_id in env.vehicles:
                vehicle = env.vehicles[vehicle_id]
                current_state = env._get_vehicle_state(vehicle_id)
                
                # # Apply exploration: randomly choose actions during exploration phase
                # if use_exploration:
                #     # Random action selection for exploration
                #     action_type = random.choice(['move', 'service', 'charge'])
                    
                #     if action_type == 'service' and env.active_requests:
                #         # Random service assignment
                #         available_requests = list(env.active_requests.values())
                #         random_request = random.choice(available_requests)
                #         actions[vehicle_id] = ServiceAction([], random_request.request_id)
                #         action_idx = 5
                #         print(f"🎲 Vehicle {vehicle_id} exploring: random service {random_request.request_id}")
                #     elif action_type == 'charge' and hasattr(env, 'charging_manager'):
                #         # Random charging assignment
                #         available_stations = [s for s in env.charging_manager.stations.values() 
                #                             if len(s.current_vehicles) < s.max_capacity]
                #         if available_stations:
                #             random_station = random.choice(available_stations)
                #             charge_duration = random.randint(2, 5)
                #             actions[vehicle_id] = ChargingAction([], random_station.id, charge_duration)
                #             action_idx = 4
                #             print(f"🎲 Vehicle {vehicle_id} exploring: random charging at station {random_station.id}")
                #         else:
                #             actions[vehicle_id] = Action([])
                #             action_idx = random.randint(0, 3)
                #     else:
                #         # Random movement
                #         actions[vehicle_id] = Action([])
                #         action_idx = random.randint(0, 3)
                #         print(f"🎲 Vehicle {vehicle_id} exploring: random movement")
                    
                #     action_chosen = True
                # else:
                #     # Normal decision logic
                #     # Enhanced strategy: prioritize passenger requests, then charging, then movement
                action_chosen = False
                
                # 1. First priority: Check for passenger requests if vehicle is available
                if (vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None and 
                    vehicle['charging_station'] is None and vehicle['battery'] > 0.2):
                    # Look for nearby passenger requests
                    available_requests = [req for req in env.active_requests.values()]
                    if available_requests:
                        # Choose closest request
                        best_request = None
                        min_distance = float('inf')
                        vehicle_coords = vehicle['coordinates']
                        
                        for request in available_requests:
                            pickup_coords = (request.pickup // env.grid_size, request.pickup % env.grid_size)
                            distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                            if distance < min_distance:
                                min_distance = distance
                                best_request = request
                        
                        if best_request and min_distance <= 5:  # Increased from 3 to 5 for better order acceptance
                            actions[vehicle_id] = ServiceAction([], best_request.request_id)
                            action_idx = 5  # Service action index
                            action_chosen = True
                            print(f"Vehicle {vehicle_id} accepting request {best_request.request_id} (distance: {min_distance})")
                
                # 2. Second priority: Continue with assigned passenger service
                if not action_chosen and (vehicle['assigned_request'] is not None or vehicle['passenger_onboard'] is not None):
                    actions[vehicle_id] = ServiceAction([], 0) # Continue service action
                    action_idx = 5
                    action_chosen = True
                
                # 3. Third priority: Charging if battery is low
                if not action_chosen and vehicle['battery'] < 0.4 and vehicle['charging_station'] is None:
                    # Find nearest available charging station
                    best_station = None
                    min_distance = float('inf')
                    
                    for station_id, station in env.charging_manager.stations.items():
                        if len(station.current_vehicles) < station.max_capacity:
                            # Calculate distance using coordinates
                            coords = vehicle['coordinates']
                            station_coords = ((station.location // env.grid_size), (station.location % env.grid_size))
                            distance = abs(coords[0] - station_coords[0]) + abs(coords[1] - station_coords[1])
                            if distance < min_distance:
                                min_distance = distance
                                best_station = station_id
                    
                    if best_station:
                        # Smarter charging duration based on battery level
                        if vehicle['battery'] < 0.2:
                            charge_duration = 6  # Longer charge for critical battery
                        elif vehicle['battery'] < 0.3:
                            charge_duration = 4  # Medium charge for low battery
                        else:
                            charge_duration = 3  # Quick top-up for moderate battery
                        actions[vehicle_id] = ChargingAction([], best_station, charge_duration)
                        action_idx = 4  # Charging action index
                        action_chosen = True
                
                # 4. Default: Movement action
                if not action_chosen:
                    if vehicle['charging_station'] is None:
                        actions[vehicle_id] = Action([])  # Move when no other priority
                        action_idx = random.randint(0, 3)  # Random move action
                    else:
                        action_idx = 4  # Stay charging
                
                # Store for training
                states_for_training.append(current_state)
                actions_for_training.append(action_idx)
            
            # Execute actions
            next_states, rewards, done, info = env.step(actions)
            
            # Store experiences for neural network training
            for vehicle_id, action in actions.items():
                if vehicle_id in rewards:
                    current_location = env.vehicles[vehicle_id]['location']
                    next_location = current_location  # Simplified
                    
                    if isinstance(action, ServiceAction):
                        action_type = f"assign_{action.request_id}"
                        target_location = action.request_id % env.grid_size**2  # Simplified
                    elif isinstance(action, ChargingAction):
                        action_type = f"charge_{action.charging_station_id}"
                        target_location = action.charging_station_id * 10  # Simplified
                    else:
                        action_type = "move"
                        target_location = current_location
                    
                    # Store experience for training
                    env.store_q_learning_experience(
                        vehicle_id=int(vehicle_id.split('_')[-1]) if '_' in str(vehicle_id) else vehicle_id,
                        action_type=action_type,
                        vehicle_location=current_location,
                        target_location=target_location,
                        reward=rewards[vehicle_id],
                        next_vehicle_location=next_location
                    )
            
            # Enhanced training: much more frequent training for better learning
            if len(value_function.experience_buffer) >= warmup_steps:
                # Train more frequently based on our new parameters
                if step % training_frequency == 0:
                    training_loss = value_function.train_step(batch_size=64)  # Larger batch
                    if training_loss > 0:
                        episode_losses.append(training_loss)
                
                # Additional mid-episode training for exploration episodes
                if use_exploration and step % (training_frequency * 2) == 0:
                    training_loss = value_function.train_step(batch_size=32)
                    if training_loss > 0:
                        episode_losses.append(training_loss)
            
            # Intensive training at episode end
            if step == env.episode_length - 1 and len(value_function.experience_buffer) >= warmup_steps:
                # Multiple training steps at episode end
                for _ in range(5):  # More training steps
                    training_loss = value_function.train_step(batch_size=64)
                    if training_loss > 0:
                        episode_losses.append(training_loss)
            
            # Update results
            episode_reward += sum(rewards.values())
            episode_charging_events.extend(info.get('charging_events', []))
            
            if done:
                break
        
        # Record episode results
        results['episode_rewards'].append(episode_reward)
        results['charging_events'].extend(episode_charging_events)
        results['value_function_losses'].append(np.mean(episode_losses) if episode_losses else 0.0)
        results['qvalue_losses'].extend(episode_losses)  # Fixed: extend instead of assign
        # Record environment statistics
        stats = env.get_stats()
        results['environment_stats'].append(stats)
        results['battery_levels'].append(stats['average_battery'])
        
        # Collect detailed episode statistics
        episode_stats = env.get_episode_stats()
        episode_stats['episode_number'] = episode + 1
        episode_stats['episode_reward'] = episode_reward
        episode_stats['charging_events_count'] = len(episode_charging_events)
        episode_stats['neural_network_loss'] = np.mean(episode_losses) if episode_losses else 0.0
        episode_stats['neural_network_loss_std'] = np.std(episode_losses) if episode_losses else 0.0
        episode_stats['training_steps_in_episode'] = len(episode_losses)
        results['episode_detailed_stats'].append(episode_stats)
        
        # Analyze vehicle visit patterns for this episode
        vehicle_visit_stats = analyze_vehicle_visit_patterns(env)
        results['vehicle_visit_stats'].append(vehicle_visit_stats)
        
        print(f"Episode {episode + 1} Completed:")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Orders: Total={episode_stats['total_orders']}, Accepted={episode_stats['accepted_orders']}, Rejected={episode_stats['rejected_orders']}")
        print(f"  Battery: {episode_stats['avg_battery_level']:.2f}")
        print(f"  Station Usage: {episode_stats['avg_vehicles_per_station']:.1f} vehicles/station")
    
    print("\n=== Integration Test Complete ===")
    print(f"✓ Neural Network ValueFunction trained over {num_episodes} episodes")
    print(f"✓ Final average training loss: {np.mean(results['value_function_losses']):.4f}")
    print(f"✓ Neural network has {sum(p.numel() for p in value_function.network.parameters())} parameters")
    
    # Create results directory for analysis
    results_dir = Path("results/integrated_tests")
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
                ax1.set_title(f'Request Generation Heatmap\n(Pattern: {demand_pattern})')
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
                ax2.set_title('Final Vehicle Distribution')
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
                    ax3.set_title('Charging Station Utilization')
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


def visualize_integrated_results(results):
    """可视化集成测试结果"""
    print("\n=== 生成可视化图表 ===")
    
    try:
        # 创建可视化器
        visualizer = ChargingIntegrationVisualization(figsize=(15, 10))
        
        # 保存路径
        results_dir = Path("results/integrated_tests")
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


def generate_integration_report(results, analysis):
    """生成集成测试报告"""
    print("\n=== 生成测试报告 ===")
    
    results_dir = Path("results/integrated_tests")
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


def main():
    """主函数"""
    print("🚗⚡ 充电行为集成测试程序")
    print("使用src文件夹中的Environment和充电组件")
    print("-" * 60)
    
    num_episodes = 100
    adpvalue = 0
    print(f"\n⚡ 开始集成测试 (ADP={adpvalue})")
    results, env = run_charging_integration_test(adpvalue, num_episodes=num_episodes, use_intense_requests=True,assignmentgurobi=False)

    # 分析结果
    analysis = analyze_results(results)
    
    # 生成可视化
    success = visualize_integrated_results(results)
    
    # 空间分布可视化已在Excel导出中生成
    print(f"\n🗺️  空间分布分析已完成，图像路径: {results.get('spatial_image_path', 'N/A')}")
    
    # 生成传统的空间分布分析（用于兼容性）
    spatial_viz = SpatialVisualization(env.grid_size)
    spatial_analysis = spatial_viz.analyze_spatial_patterns(env)
    spatial_viz.print_spatial_analysis(spatial_analysis)
    
    # 生成报告
    generate_integration_report(results, analysis)
    
    # 输出车辆访问模式总结
    print_vehicle_visit_summary(results.get('vehicle_visit_stats', []))
    
    print("\n" + "="*60)
    print("🎉 集成测试完成!")
    print("📊 结果摘要:")
    print(f"   - 平均奖励: {analysis['avg_reward']:.2f}")
    print(f"   - 充电次数: {analysis['total_charging']}")
    print(f"   - 平均电量: {analysis['avg_battery']:.2f}")
    print(f"   - 奖励改进: {analysis['improvement']:.2f}")
    
    if success:
        print("📈 可视化图表生成成功")
    
    print("📁 请检查 results/integrated_tests/ 文件夹中的详细结果")
    print("="*60)
            
    
    
    print(f"\n⚡ 开始集成测试 (ADP={adpvalue})")
    results, env = run_charging_integration_test(adpvalue, num_episodes=num_episodes, use_intense_requests=False,assignmentgurobi=False)

    # 分析结果
    analysis = analyze_results(results)
    
    # 生成可视化
    success = visualize_integrated_results(results)
    
    # 空间分布可视化已在Excel导出中生成
    print(f"\n🗺️  空间分布分析已完成，图像路径: {results.get('spatial_image_path', 'N/A')}")
    
    # 生成传统的空间分布分析（用于兼容性）
    spatial_viz = SpatialVisualization(env.grid_size)
    spatial_analysis = spatial_viz.analyze_spatial_patterns(env)
    spatial_viz.print_spatial_analysis(spatial_analysis)
    
    # 生成报告
    generate_integration_report(results, analysis)
    
    # 输出车辆访问模式总结
    print_vehicle_visit_summary(results.get('vehicle_visit_stats', []))
    
    print("\n" + "="*60)
    print("🎉 集成测试完成!")
    print("📊 结果摘要:")
    print(f"   - 平均奖励: {analysis['avg_reward']:.2f}")
    print(f"   - 充电次数: {analysis['total_charging']}")
    print(f"   - 平均电量: {analysis['avg_battery']:.2f}")
    print(f"   - 奖励改进: {analysis['improvement']:.2f}")
    
    if success:
        print("📈 可视化图表生成成功")
    
    print("📁 请检查 results/integrated_tests/ 文件夹中的详细结果")
    print("="*60)
    
      
            
    # try:
        
        
    #     num_episodes = 100
    #     adplist = [0, 0.5, 1]
    #     for adpvalue in adplist:
    #         print(f"\n⚡ 开始集成测试 (ADP={adpvalue})")
    #         results, env = run_charging_integration_test(adpvalue, num_episodes=num_episodes)

    #         # 分析结果
    #         analysis = analyze_results(results)
            
    #         # 生成可视化
    #         success = visualize_integrated_results(results)
            
    #         # 空间分布可视化已在Excel导出中生成
    #         print(f"\n🗺️  空间分布分析已完成，图像路径: {results.get('spatial_image_path', 'N/A')}")
            
    #         # 生成传统的空间分布分析（用于兼容性）
    #         spatial_viz = SpatialVisualization(env.grid_size)
    #         spatial_analysis = spatial_viz.analyze_spatial_patterns(env)
    #         spatial_viz.print_spatial_analysis(spatial_analysis)
            
    #         # 生成报告
    #         generate_integration_report(results, analysis)
            
    #         # 输出车辆访问模式总结
    #         print_vehicle_visit_summary(results.get('vehicle_visit_stats', []))
            
    #         print("\n" + "="*60)
    #         print("🎉 集成测试完成!")
    #         print("📊 结果摘要:")
    #         print(f"   - 平均奖励: {analysis['avg_reward']:.2f}")
    #         print(f"   - 充电次数: {analysis['total_charging']}")
    #         print(f"   - 平均电量: {analysis['avg_battery']:.2f}")
    #         print(f"   - 奖励改进: {analysis['improvement']:.2f}")
            
    #         if success:
    #             print("📈 可视化图表生成成功")
            
    #         print("📁 请检查 results/integrated_tests/ 文件夹中的详细结果")
    #         print("="*60)
            
    # except Exception as e:
    #     print(f"\n❌ 测试失败: {e}")
    #     import traceback
    #     traceback.print_exc()


def run_heuristic_vs_gurobi_test(num_episodes=10):
    """运行启发式策略与Gurobi优化的对比测试"""
    print("=== 启发式策略 vs Gurobi优化对比测试 ===")
    
    from src.GurobiOptimizer import GurobiOptimizer
    from src.Heuristic import HeuristicPolicy
    
    # 创建环境
    num_vehicles = 20
    num_stations = 6
    env = ChargingIntegratedEnvironment(num_vehicles=num_vehicles, num_stations=num_stations)
    
    # 创建优化器和启发式策略
    gurobi_optimizer = GurobiOptimizer(env)
    heuristic_policy = HeuristicPolicy(battery_threshold=0.5, max_service_distance=8)
    
    # 测试结果
    gurobi_results = []
    heuristic_results = []
    
    for episode in range(num_episodes):
        print(f"\n--- Episode {episode + 1}/{num_episodes} ---")
        
        # 测试Gurobi优化
        env.reset()
        gurobi_reward, gurobi_stats = run_single_episode_with_optimizer(env, gurobi_optimizer, "Gurobi")
        gurobi_results.append({'reward': gurobi_reward, 'stats': gurobi_stats})
        
        # 测试启发式策略
        env.reset()
        heuristic_reward, heuristic_stats = run_single_episode_with_heuristic(env, heuristic_policy, "Heuristic")
        heuristic_results.append({'reward': heuristic_reward, 'stats': heuristic_stats})
        
        print(f"Gurobi奖励: {gurobi_reward:.2f}, 启发式奖励: {heuristic_reward:.2f}")
    
    # 分析结果
    analyze_comparison_results(gurobi_results, heuristic_results)
    
    return gurobi_results, heuristic_results


def run_single_episode_with_optimizer(env, optimizer, strategy_name):
    """使用Gurobi优化器运行单个episode"""
    total_reward = 0
    charging_events = 0
    served_requests = 0
    rejected_requests = 0
    
    for step in range(200):  # 运行200步
        # 获取可用车辆和请求
        available_vehicles = [
            vehicle_id for vehicle_id, vehicle in env.vehicles.items()
            if (vehicle['assigned_request'] is None and 
                vehicle['passenger_onboard'] is None and 
                vehicle['charging_station'] is None)
        ]
        
        available_requests = [
            req for req in env.active_requests.values()
            if not any(v.get('assigned_request') == req.request_id or 
                      v.get('passenger_onboard') == req.request_id 
                      for v in env.vehicles.values())
        ]
        
        charging_stations = list(env.charging_manager.stations.values())
        
        # 使用Gurobi优化器获取分配
        if available_vehicles:
            assignments = optimizer.optimize_vehicle_rebalancing_reject(available_vehicles)
            
            # 将分配转换为动作
            actions = {}
            for vehicle_id in env.vehicles:
                if vehicle_id in assignments:
                    assignment = assignments[vehicle_id]
                    if isinstance(assignment, str) and assignment.startswith("charge_"):
                        # 充电分配
                        station_id = assignment.replace("charge_", "")
                        actions[vehicle_id] = ChargingAction([], station_id, 4)
                        charging_events += 1
                    elif hasattr(assignment, 'request_id'):
                        # 服务分配
                        actions[vehicle_id] = ServiceAction([], assignment.request_id)
                        served_requests += 1
                    else:
                        actions[vehicle_id] = Action([])
                else:
                    actions[vehicle_id] = Action([])
            
            # 执行动作
            next_states, step_rewards, done, info = env.step(actions)
            step_reward = sum(step_rewards.values()) if isinstance(step_rewards, dict) else step_rewards
            total_reward += step_reward
        else:
            # 如果没有可用车辆，执行空动作
            actions = {vehicle_id: Action([]) for vehicle_id in env.vehicles}
            next_states, step_rewards, done, info = env.step(actions)
            step_reward = sum(step_rewards.values()) if isinstance(step_rewards, dict) else step_rewards
            total_reward += step_reward
    
    stats = {
        'charging_events': charging_events,
        'served_requests': served_requests,
        'avg_battery': np.mean([v['battery'] for v in env.vehicles.values()]),
        'strategy': strategy_name
    }
    
    return total_reward, stats


def run_single_episode_with_heuristic(env, heuristic_policy, strategy_name):
    """使用启发式策略运行单个episode"""
    total_reward = 0
    charging_events = 0
    served_requests = 0
    rejected_requests = 0
    
    for step in range(200):  # 运行200步
        # 获取可用车辆和请求
        available_vehicles = [
            vehicle_id for vehicle_id, vehicle in env.vehicles.items()
            if (vehicle['assigned_request'] is None and 
                vehicle['passenger_onboard'] is None and 
                vehicle['charging_station'] is None)
        ]
        
        available_requests = [
            req for req in env.active_requests.values()
            if not any(v.get('assigned_request') == req.request_id or 
                      v.get('passenger_onboard') == req.request_id 
                      for v in env.vehicles.values())
        ]
        
        charging_stations = list(env.charging_manager.stations.values())
        
        # 使用启发式策略获取分配
        if available_vehicles:
            assignments = heuristic_policy.get_assignments(
                env, available_vehicles, available_requests, charging_stations
            )
            
            # 将分配转换为动作
            actions = {}
            for vehicle_id in env.vehicles:
                if vehicle_id in assignments:
                    assignment = assignments[vehicle_id]
                    if isinstance(assignment, str) and assignment.startswith("charge_"):
                        # 充电分配
                        station_id = assignment.replace("charge_", "")
                        actions[vehicle_id] = ChargingAction([], station_id, 4)
                        charging_events += 1
                    elif hasattr(assignment, 'request_id'):
                        # 服务分配
                        actions[vehicle_id] = ServiceAction([], assignment.request_id)
                        served_requests += 1
                    else:
                        actions[vehicle_id] = Action([])
                else:
                    actions[vehicle_id] = Action([])
            
            # 执行动作
            next_states, step_rewards, done, info = env.step(actions)
            step_reward = sum(step_rewards.values()) if isinstance(step_rewards, dict) else step_rewards
            total_reward += step_reward
        else:
            # 如果没有可用车辆，执行空动作
            actions = {vehicle_id: Action([]) for vehicle_id in env.vehicles}
            next_states, step_rewards, done, info = env.step(actions)
            step_reward = sum(step_rewards.values()) if isinstance(step_rewards, dict) else step_rewards
            total_reward += step_reward
    
    stats = {
        'charging_events': charging_events,
        'served_requests': served_requests,
        'avg_battery': np.mean([v['battery'] for v in env.vehicles.values()]),
        'strategy': strategy_name
    }
    
    return total_reward, stats


def analyze_comparison_results(gurobi_results, heuristic_results):
    """分析对比结果"""
    print("\n=== 对比分析结果 ===")
    
    # 计算平均值
    gurobi_avg_reward = np.mean([r['reward'] for r in gurobi_results])
    heuristic_avg_reward = np.mean([r['reward'] for r in heuristic_results])
    
    gurobi_avg_charging = np.mean([r['stats']['charging_events'] for r in gurobi_results])
    heuristic_avg_charging = np.mean([r['stats']['charging_events'] for r in heuristic_results])
    
    gurobi_avg_served = np.mean([r['stats']['served_requests'] for r in gurobi_results])
    heuristic_avg_served = np.mean([r['stats']['served_requests'] for r in heuristic_results])
    
    gurobi_avg_battery = np.mean([r['stats']['avg_battery'] for r in gurobi_results])
    heuristic_avg_battery = np.mean([r['stats']['avg_battery'] for r in heuristic_results])
    
    print(f"Gurobi优化:")
    print(f"  - 平均奖励: {gurobi_avg_reward:.2f}")
    print(f"  - 平均充电事件: {gurobi_avg_charging:.2f}")
    print(f"  - 平均服务请求: {gurobi_avg_served:.2f}")
    print(f"  - 平均电池水平: {gurobi_avg_battery:.2f}")
    
    print(f"\n启发式策略:")
    print(f"  - 平均奖励: {heuristic_avg_reward:.2f}")
    print(f"  - 平均充电事件: {heuristic_avg_charging:.2f}")
    print(f"  - 平均服务请求: {heuristic_avg_served:.2f}")
    print(f"  - 平均电池水平: {heuristic_avg_battery:.2f}")
    
    # 计算改进百分比
    reward_improvement = ((heuristic_avg_reward - gurobi_avg_reward) / abs(gurobi_avg_reward)) * 100
    print(f"\n性能对比:")
    print(f"  - 启发式策略奖励相对改进: {reward_improvement:.2f}%")
    
    if reward_improvement > 0:
        print("  ✅ 启发式策略表现更好")
    else:
        print("  ❌ Gurobi优化表现更好")
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建DataFrame
    all_results = []
    for i, (g_result, h_result) in enumerate(zip(gurobi_results, heuristic_results)):
        all_results.append({
            'episode': i + 1,
            'strategy': 'Gurobi',
            'reward': g_result['reward'],
            'charging_events': g_result['stats']['charging_events'],
            'served_requests': g_result['stats']['served_requests'],
            'avg_battery': g_result['stats']['avg_battery']
        })
        all_results.append({
            'episode': i + 1,
            'strategy': 'Heuristic',
            'reward': h_result['reward'],
            'charging_events': h_result['stats']['charging_events'],
            'served_requests': h_result['stats']['served_requests'],
            'avg_battery': h_result['stats']['avg_battery']
        })
    
    df = pd.DataFrame(all_results)
    df.to_csv(f'results/heuristic_vs_gurobi_comparison_{timestamp}.csv', index=False)
    print(f"\n📊 详细结果已保存到: results/heuristic_vs_gurobi_comparison_{timestamp}.csv")


if __name__ == "__main__":
    main()