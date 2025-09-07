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
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
print("✓ Successfully imported core components from src folder")
USE_SRC_COMPONENTS = True






def run_charging_integration_test(num_episodes):
    """Run charging integration test with EV/AEV analysis"""
    print("=== Starting Enhanced Charging Behavior Integration Test ===")
    
    # Create environment with significantly more complexity for better learning
    num_vehicles = 30  # Doubled vehicles for more interaction
    num_stations = 10  # More stations for complex charging decisions
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
    
    # Save detailed episode statistics to Excel
    save_episode_stats_to_excel(results['episode_detailed_stats'], results_dir)
    
    return results


def save_episode_stats_to_excel(episode_stats, results_dir):
    """Save detailed episode statistics to Excel file"""
    if not episode_stats:
        print("⚠ No episode statistics to save")
        return
    
    # Create DataFrame from episode statistics
    df = pd.DataFrame(episode_stats)
    
    # Add timestamp to filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    excel_filename = f"episode_statistics_{timestamp}.xlsx"
    excel_path = results_dir / excel_filename
    
    try:
        # Create Excel writer with multiple sheets
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Main statistics sheet
            df.to_excel(writer, sheet_name='Episode_Statistics', index=False)
            
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
        
        print(f"✓ Episode statistics saved to Excel: {excel_path}")
        print(f"  - Episode_Statistics: Detailed data for each episode")
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
    
    try:
        
        
        num_episodes = 50
        results = run_charging_integration_test(num_episodes=num_episodes)

        # 分析结果
        analysis = analyze_results(results)
        
        # 生成可视化
        success = visualize_integrated_results(results)
        
        # 生成报告
        generate_integration_report(results, analysis)
        
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
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()