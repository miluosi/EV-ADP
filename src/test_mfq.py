"""
Integrated Test: Vehicle Charging Behavior Integration Test using src folder components
Mean Field Reinforcement Learning Integration Test
"""
from datetime import datetime
import os
import sys

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

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
import math

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
from src.ValueFunction_pytorch_mf import (
    MeanFieldAgent, MeanFieldQNetwork, MeanFieldExperienceReplay,
    MeanFieldTrainer, create_mf_state_features, map_mf_action_to_environment_action
)
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
print("✓ Successfully imported core components from src folder")
print("✓ Successfully imported Mean Field RL components")
USE_SRC_COMPONENTS = True


def load_q_network_checkpoint(value_function, checkpoint_path):
    """
    加载已保存的Q-network检查点
    
    Args:
        value_function: PyTorchChargingValueFunction实例
        checkpoint_path: 检查点文件路径
    
    Returns:
        bool: 是否成功加载
    """
    try:
        if not os.path.exists(checkpoint_path):
            print(f"❌ 检查点文件不存在: {checkpoint_path}")
            return False
        
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=value_function.device)
        
        # 恢复网络参数
        if 'network_state_dict' in checkpoint:
            value_function.network.load_state_dict(checkpoint['network_state_dict'])
        
        if 'target_network_state_dict' in checkpoint:
            value_function.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        
        # 恢复优化器状态
        if 'optimizer_state_dict' in checkpoint:
            value_function.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # 恢复学习率调度器状态
        if 'scheduler_state_dict' in checkpoint:
            value_function.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # 恢复训练步数
        if 'training_step' in checkpoint:
            value_function.training_step = checkpoint['training_step']
        
        # 恢复损失历史
        if 'training_losses' in checkpoint:
            value_function.training_losses = checkpoint['training_losses']
        
        # 恢复Q值历史
        if 'q_values_history' in checkpoint:
            value_function.q_values_history = checkpoint['q_values_history']
        
        episode = checkpoint.get('episode', 0)
        buffer_size = checkpoint.get('experience_buffer_size', 0)
        
        print(f"✓ 成功加载检查点: {checkpoint_path}")
        print(f"  - Episode: {episode}")
        print(f"  - Training step: {value_function.training_step}")
        print(f"  - Experience buffer size: {buffer_size}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载检查点失败: {e}")
        return False


def save_q_network_checkpoint(value_function, episode, checkpoint_dir="checkpoints/q_networks"):
    """
    保存Q-network检查点的通用函数
    
    Args:
        value_function: PyTorchChargingValueFunction实例
        episode: 当前episode数
        checkpoint_dir: 保存目录
    
    Returns:
        dict: 保存的文件路径
    """
    import os
    
    if value_function is None:
        print("❌ Value function为空，无法保存")
        return {}
    
    # 创建保存目录
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 定义保存路径
    paths = {
        'main_network': os.path.join(checkpoint_dir, f"q_network_episode_{episode}.pth"),
        'target_network': os.path.join(checkpoint_dir, f"target_network_episode_{episode}.pth"),
        'full_state': os.path.join(checkpoint_dir, f"full_state_episode_{episode}.pth")
    }
    
    try:
        # 保存主Q-network参数
        torch.save(value_function.network.state_dict(), paths['main_network'])
        
        # 保存target network参数
        torch.save(value_function.target_network.state_dict(), paths['target_network'])
        
        # 保存完整状态
        full_state = {
            'episode': episode,
            'training_step': value_function.training_step,
            'network_state_dict': value_function.network.state_dict(),
            'target_network_state_dict': value_function.target_network.state_dict(),
            'optimizer_state_dict': value_function.optimizer.state_dict(),
            'scheduler_state_dict': value_function.scheduler.state_dict(),
            'experience_buffer_size': len(value_function.experience_buffer),
            'training_losses': value_function.training_losses[-100:] if value_function.training_losses else [],
            'q_values_history': value_function.q_values_history[-100:] if value_function.q_values_history else []
        }
        torch.save(full_state, paths['full_state'])
        
        print(f"✓ Episode {episode}: 成功保存Q-network检查点")
        print(f"  - Q-network: {paths['main_network']}")
        print(f"  - Target network: {paths['target_network']}")
        print(f"  - Full state: {paths['full_state']}")
        print(f"  - Training step: {value_function.training_step}")
        print(f"  - Experience buffer size: {len(value_function.experience_buffer)}")
        
        return paths
        
    except Exception as e:
        print(f"❌ Episode {episode}: 保存网络失败: {e}")
        return {}


def list_available_checkpoints(checkpoint_dir="checkpoints/q_networks"):
    """
    列出可用的检查点文件
    
    Args:
        checkpoint_dir: 检查点目录
        
    Returns:
        list: 可用的完整状态检查点列表
    """
    import os
    import glob
    
    if not os.path.exists(checkpoint_dir):
        print(f"❌ 检查点目录不存在: {checkpoint_dir}")
        return []
    
    # 寻找完整状态文件
    pattern = os.path.join(checkpoint_dir, "full_state_episode_*.pth")
    checkpoints = glob.glob(pattern)
    
    # 提取episode编号并排序
    checkpoint_info = []
    for checkpoint in checkpoints:
        try:
            basename = os.path.basename(checkpoint)
            episode_str = basename.replace("full_state_episode_", "").replace(".pth", "")
            episode = int(episode_str)
            checkpoint_info.append((episode, checkpoint))
        except ValueError:
            continue
    
    # 按episode排序
    checkpoint_info.sort(key=lambda x: x[0])
    
    if checkpoint_info:
        print(f"✓ 找到 {len(checkpoint_info)} 个检查点:")
        for episode, path in checkpoint_info:
            print(f"  - Episode {episode}: {path}")
    else:
        print(f"❌ 未找到检查点文件在: {checkpoint_dir}")
    
    return checkpoint_info

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


def run_charging_integration_test_mf(adpvalue, num_episodes, use_intense_requests, assignmentgurobi, batch_size=256, num_vehicles=10):
    """
    Run Mean Field Reinforcement Learning (MFRL) integration test
    
    Mean Field Q-Learning assumes each agent knows the action distribution of neighboring agents.
    Q(s, a, μ) where μ is the mean action distribution of neighbors.
    
    Args:
        adpvalue: ADP coefficient (>0 to use neural network training)
        num_episodes: Number of episodes to run
        use_intense_requests: Whether to use intense request generation
        assignmentgurobi: Whether to use Gurobi for assignment optimization  
        batch_size: Training batch size
        num_vehicles: Number of vehicles in the fleet
    
    Returns:
        results: Dictionary containing training results
        env: Final environment instance
    """
    print("=== Starting Mean Field Reinforcement Learning Integration Test ===")
    print("=== MFRL assumes agents know neighboring agents' action distributions ===")
    
    # 设置全局随机数种子，确保车辆初始化一致
    set_random_seeds(seed=42)
    
    # Import Mean Field components
    from src.ValueFunction_pytorch_mf import (
        MeanFieldAgent, MeanFieldTrainer, MeanFieldQNetwork,
        create_mf_state_features, map_mf_action_to_environment_action
    )
    import math
    
    # Create environment with complexity for learning
    num_stations = 4
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles,
        num_stations=num_stations,
        random_seed=42
    )
    
    print("✓ Fixed initial state setup: Vehicle positions and battery levels will be identical across episodes")
    print("✓ Request generation will vary by episode for learning progression")
    
    # Determine device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"✓ Using device: {device}")
    
    # Mean Field specific parameters
    action_dim = 32  # Same action space: assign(10) + rebalance(10) + charge(5) + wait(4) + idle(3)
    neighbor_radius = 3.0  # Radius to determine neighboring agents (in grid units)
    boltzmann_temp = 1.0   # Temperature for action distribution
    
    # Initialize Mean Field components only if needed
    use_neural_network = adpvalue > 0 and assignmentgurobi
    
    if use_neural_network:
        # Create Mean Field Trainer that manages all agents
        mf_trainer = MeanFieldTrainer(
            environment=env,
            num_agents=num_vehicles,
            action_dim=action_dim,
            neighbor_radius=neighbor_radius,
            device=device
        )
        
        # 创建 PyTorchChargingValueFunction 作为环境的 value_function
        # 这是 GurobiOptimizer 评估选项所必需的
        value_function = PyTorchChargingValueFunction(
            grid_size=env.grid_size,
            num_vehicles=num_vehicles,
            log_dir='logs/mf_charging_nn',
            device=device,
            env=env
        )
        env.set_value_function(value_function)
        
        print(f"✓ Initialized Mean Field Trainer with:")
        print(f"   - {num_vehicles} agents")
        print(f"   - Action dimension: {action_dim}")
        print(f"   - Neighbor radius: {neighbor_radius}")
        print(f"   - Network parameters: {sum(p.numel() for p in mf_trainer.mf_agent.policy_net.parameters())}")
        print(f"✓ Value function set for environment (required by GurobiOptimizer)")
    else:
        mf_trainer = None
        # 即使不使用 Mean Field，也需要设置 value_function 用于 Gurobi 优化
        value_function = PyTorchChargingValueFunction(
            grid_size=env.grid_size,
            num_vehicles=num_vehicles,
            log_dir='logs/mf_charging_nn',
            device=device,
            env=env
        )
        env.set_value_function(value_function)
        print(f"✓ Mean Field training disabled (ADP={adpvalue}, AssignmentGurobi={assignmentgurobi})")
        print(f"✓ Value function set for environment (required by GurobiOptimizer)")
    
    env.adp_value = adpvalue
    env.use_intense_requests = use_intense_requests
    env.assignmentgurobi = assignmentgurobi
    
    # Exploration parameters
    exploration_episodes = max(1, num_episodes // 2)
    epsilon_start = 0.4
    epsilon_end = 0.1
    epsilon_decay = (epsilon_start - epsilon_end) / exploration_episodes
    
    # Training parameters
    training_frequency = 2
    warmup_steps = 100
    
    print(f"✓ Initialized environment with {num_vehicles} vehicles and {num_stations} charging stations")
    if use_neural_network:
        print(f"✓ Enhanced exploration: {exploration_episodes} episodes with epsilon {epsilon_start:.2f} → {epsilon_end:.2f}")
        print(f"   - Training frequency: every {training_frequency} steps after {warmup_steps} warmup steps")
    
    # Display vehicle type distribution
    ev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'EV')
    aev_count = sum(1 for v in env.vehicles.values() if v['type'] == 'AEV')
    print(f"✓ Vehicle distribution: {ev_count} EV vehicles, {aev_count} AEV vehicles")
    
    # Results storage
    results = {
        'Idle_average': [],
        'episode_rewards': [],
        'charging_events': [],
        'episode_detailed_stats': [],
        'vehicle_visit_stats': [],
        'battery_levels': [],
        'environment_stats': [],
        'value_function_losses': [],
        'qvalue_losses': [],
        'mean_field_entropy': [],  # Track mean field entropy over training
        'neighbor_action_diversity': []  # Track how diverse neighbor actions are
    }
    
    for episode in range(num_episodes):
        # Set episode-specific request generation seed
        episode_seed = 32 + episode
        env.set_request_generation_seed(episode_seed)
        print(f"\nEpisode {episode + 1}: Request generation seed set to {episode_seed}")
        
        current_epsilon = max(epsilon_end, epsilon_start - episode * epsilon_decay)
        
        # Reset environment
        states = env.reset()
        episode_reward = 0
        episode_charging_events = []
        episode_losses = []
        episode_mf_entropy = []
        
        Idle_list = []
        
        # Save checkpoint periodically
        if episode % 10 == 0 and use_neural_network:
            checkpoint_dir = "checkpoints/mf_networks"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"mf_agent_episode_{episode}.pth")
            mf_trainer.save_model(checkpoint_path)
            print(f"✓ Saved Mean Field checkpoint: {checkpoint_path}")
        
        for step in range(env.episode_length):
            # Get current requests
            current_requests = list(env.active_requests.values())
            
            # ========================================
            # MEAN FIELD ACTION SELECTION
            # ========================================
            if use_neural_network and mf_trainer is not None:
                # Update environment reference in trainer
                mf_trainer.environment = env
                
                # Compute mean field (neighboring agents' action distributions) for all agents
                mean_fields = mf_trainer.compute_all_mean_fields()
                
                # Calculate mean field entropy (measure of action diversity)
                mf_entropy_values = []
                for vid, mf in mean_fields.items():
                    # Entropy = -Σ p(a) log p(a)
                    entropy = -torch.sum(mf * torch.log(mf + 1e-10)).item()
                    mf_entropy_values.append(entropy)
                
                if mf_entropy_values:
                    avg_mf_entropy = sum(mf_entropy_values) / len(mf_entropy_values)
                    episode_mf_entropy.append(avg_mf_entropy)
                
                # Select actions using mean field Q-learning
                mf_actions, q_values_all, action_probs_all = mf_trainer.select_actions_for_all_agents(
                    current_time=step,
                    training=True
                )
                
                # Store pre-step mean fields for experience
                pre_step_mean_fields = {k: v.clone() for k, v in mean_fields.items()}
            
            # Use environment's simulate_motion for actual action execution
            # This handles the complex assignment logic including Gurobi optimization
            actions, storeactions = env.simulate_motion(
                agents=[], 
                current_requests=current_requests, 
                rebalance=True
            )
            
            # Execute step
            next_states, rewards, dur_rewards, done, info = env.step(actions, storeactions)
            
            # ========================================
            # STORE MEAN FIELD EXPERIENCES
            # ========================================
            if use_neural_network and mf_trainer is not None:
                # Compute next mean fields after step
                next_mean_fields = mf_trainer.compute_all_mean_fields()
                
                # Store transitions with mean field information
                dones = {vid: done for vid in env.vehicles.keys()}
                mf_trainer.store_transitions(rewards, next_mean_fields, dones)
            
            # Debug output every 25 steps
            if step % 25 == 0:
                stats = env.get_stats()
                active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                assigned_vehicles = len([v for v in env.vehicles.values() if v['assigned_request'] is not None])
                charging_vehicles = len([v for v in env.vehicles.values() if v['charging_station'] is not None])
                onboard = len([v for v in env.vehicles.values() if v['passenger_onboard'] is not None])
                idlecar = len([v for v in env.vehicles.values() if v.get('idle_target') is not None])
                waitcar = len([v for v in env.vehicles.values() if v.get('is_stationary') is True])
                movecharge = len([v for v in env.vehicles.values() if v.get('charging_target') is not None])
                idle_vehicles = len([v for v in env.vehicles.values()
                                   if v['assigned_request'] is None and v['passenger_onboard'] is None 
                                   and v['charging_station'] is None and v['target_location'] is None])
                step_reward = sum(rewards.values())
                
                print(f"Step {step}: Active requests: {active_requests}, Assigned: {assigned_vehicles}, "
                      f"Onboard: {onboard}, Charging: {charging_vehicles}, Idle: {idlecar}, "
                      f"Wait: {waitcar}, MoveCharge: {movecharge}, Idle Vehicles: {idle_vehicles}, "
                      f"Step reward: {step_reward:.2f}")
                
                Idle_list.append(idle_vehicles)
                
                # Mean Field specific monitoring
                if use_neural_network and mf_trainer is not None:
                    mf_stats = mf_trainer.mf_agent.training_stats
                    recent_loss = mf_stats['losses'][-1] if mf_stats['losses'] else 0.0
                    buffer_size = len(mf_trainer.mf_agent.memory)
                    training_step = mf_trainer.mf_agent.steps_done
                    
                    # Get sample Q-values with mean field
                    if buffer_size > 0 and len(mean_fields) > 0:
                        sample_vid = list(env.vehicles.keys())[0]
                        sample_state = create_mf_state_features(env, sample_vid, step)
                        sample_mf = mean_fields.get(sample_vid, torch.ones(action_dim) / action_dim)
                        
                        try:
                            with torch.no_grad():
                                vehicle_t = sample_state['vehicle'].unsqueeze(0).to(device)
                                request_t = sample_state['request'].unsqueeze(0).to(device)
                                global_t = sample_state['global'].unsqueeze(0).to(device)
                                mf_t = sample_mf.unsqueeze(0).to(device)
                                
                                q_vals = mf_trainer.mf_agent.policy_net.forward_dueling(
                                    vehicle_t, request_t, global_t, mf_t
                                ).squeeze(0)
                                
                                # Get Q-values for different action types
                                assign_q = q_vals[:10].mean().item()  # Assign actions
                                charge_q = q_vals[20:25].mean().item()  # Charge actions
                                idle_q = q_vals[29:].mean().item()  # Idle actions
                                
                            print(f"  Mean Field Status:")
                            print(f"    Training step: {training_step}, Buffer: {buffer_size}, Loss: {recent_loss:.4f}")
                            print(f"    Mean Field Q-values: Assign={assign_q:.3f}, Charge={charge_q:.3f}, Idle={idle_q:.3f}")
                            
                            if episode_mf_entropy:
                                print(f"    Mean Field Entropy (neighbor action diversity): {episode_mf_entropy[-1]:.3f}")
                        except Exception as e:
                            print(f"  Mean Field Status: Error getting Q-values: {e}")
                else:
                    print(f"  Mean Field: {'Not training yet' if use_neural_network else 'Disabled'}")
            
            # Training step (Mean Field Q-Learning)
            if use_neural_network and len(mf_trainer.mf_agent.memory) >= warmup_steps:
                if step % training_frequency == 0:
                    training_loss = mf_trainer.train_step(batch_size=batch_size)
                    if training_loss is not None and training_loss > 0:
                        episode_losses.append(training_loss)
            
            episode_reward += sum(rewards.values())
            episode_charging_events.extend(info.get('charging_events', []))
            
            if done:
                break
        
        # Episode statistics
        results['Idle_average'].append(sum(Idle_list) / len(Idle_list) if Idle_list else 0)
        results['episode_rewards'].append(episode_reward)
        results['charging_events'].extend(episode_charging_events)
        results['value_function_losses'].append(np.mean(episode_losses) if episode_losses else 0.0)
        results['qvalue_losses'].extend(episode_losses)
        results['mean_field_entropy'].append(np.mean(episode_mf_entropy) if episode_mf_entropy else 0.0)
        
        # Environment statistics
        stats = env.get_stats()
        results['active_requests'] = stats['active_requests']
        results['environment_stats'].append(stats)
        results['battery_levels'].append(stats['average_battery'])
        results['completed_requests'] = stats['completed_requests']
        
        # Detailed episode statistics
        episode_stats = env.get_episode_stats()
        episode_stats['episode_number'] = episode + 1
        episode_stats['episode_reward'] = episode_reward
        episode_stats['charging_events_count'] = len(episode_charging_events)
        
        # Mean Field specific stats
        if use_neural_network:
            episode_stats['mean_field_entropy'] = np.mean(episode_mf_entropy) if episode_mf_entropy else 0.0
            episode_stats['neural_network_loss'] = np.mean(episode_losses) if episode_losses else 0.0
            episode_stats['neural_network_loss_std'] = np.std(episode_losses) if episode_losses else 0.0
            episode_stats['training_steps_in_episode'] = len(episode_losses)
            
            # Sample Q-values
            if len(mf_trainer.mf_agent.memory) > 0:
                try:
                    sample_vid = list(env.vehicles.keys())[0]
                    sample_state = create_mf_state_features(env, sample_vid, env.current_time)
                    sample_mf = torch.ones(action_dim, device=device) / action_dim
                    
                    with torch.no_grad():
                        vehicle_t = sample_state['vehicle'].unsqueeze(0).to(device)
                        request_t = sample_state['request'].unsqueeze(0).to(device)
                        global_t = sample_state['global'].unsqueeze(0).to(device)
                        mf_t = sample_mf.unsqueeze(0).to(device)
                        
                        q_vals = mf_trainer.mf_agent.policy_net.forward_dueling(
                            vehicle_t, request_t, global_t, mf_t
                        ).squeeze(0)
                        
                        episode_stats['sample_idle_q_value'] = q_vals[29:].mean().item()
                        episode_stats['sample_assign_q_value'] = q_vals[:10].mean().item()
                        episode_stats['sample_charge_q_value'] = q_vals[20:25].mean().item()
                except:
                    episode_stats['sample_idle_q_value'] = 0.0
                    episode_stats['sample_assign_q_value'] = 0.0
                    episode_stats['sample_charge_q_value'] = 0.0
            else:
                episode_stats['sample_idle_q_value'] = 0.0
                episode_stats['sample_assign_q_value'] = 0.0
                episode_stats['sample_charge_q_value'] = 0.0
        else:
            episode_stats['mean_field_entropy'] = 0.0
            episode_stats['neural_network_loss'] = 0.0
            episode_stats['neural_network_loss_std'] = 0.0
            episode_stats['training_steps_in_episode'] = 0
            episode_stats['sample_idle_q_value'] = 0.0
            episode_stats['sample_assign_q_value'] = 0.0
            episode_stats['sample_charge_q_value'] = 0.0
        
        results['episode_detailed_stats'].append(episode_stats)
        
        # Print episode summary
        rebalancing_calls = episode_stats.get('total_rebalancing_calls', 0)
        total_assignments = episode_stats.get('total_rebalancing_assignments', 0)
        avg_assignments = episode_stats.get('avg_rebalancing_assignments_per_call', 0)
        avg_whole = episode_stats.get('avg_rebalancing_assignments_per_whole', 0)
        
        print(f"\nEpisode {episode + 1} Completed (Mean Field RL):")
        print(f"  Reward: {episode_reward:.2f}")
        print(f"  Orders: Total={episode_stats['total_orders']}, Accepted={episode_stats['accepted_orders']}, "
              f"Completed={episode_stats['completed_orders']}, Rejected={episode_stats['rejected_orders']}")
        print(f"  Battery: {episode_stats['avg_battery_level']:.2f}")
        print(f"  Rebalancing: Calls={rebalancing_calls}, Total Assignments={total_assignments}, "
              f"Avg Assignments={avg_assignments:.2f}")
        
        if use_neural_network:
            mf_entropy = episode_stats.get('mean_field_entropy', 0.0)
            nn_loss = episode_stats.get('neural_network_loss', 0.0)
            idle_q = episode_stats.get('sample_idle_q_value', 0.0)
            assign_q = episode_stats.get('sample_assign_q_value', 0.0)
            charge_q = episode_stats.get('sample_charge_q_value', 0.0)
            print(f"  Mean Field: Entropy={mf_entropy:.3f}, Loss={nn_loss:.4f}")
            print(f"  Q-values: Idle={idle_q:.3f}, Assign={assign_q:.3f}, Charge={charge_q:.3f}")
    
    # Test complete
    print("\n=== Mean Field Integration Test Complete ===")
    if use_neural_network:
        print(f"✓ Mean Field Q-Network trained over {num_episodes} episodes")
        print(f"✓ Final average training loss: {np.mean(results['value_function_losses']):.4f}")
        print(f"✓ Final average mean field entropy: {np.mean(results['mean_field_entropy']):.4f}")
        print(f"✓ Network parameters: {sum(p.numel() for p in mf_trainer.mf_agent.policy_net.parameters())}")
    else:
        print(f"✓ Test completed without Mean Field training")
    
    # Save results
    if assignmentgurobi:
        results_dir = Path("results/integrated_tests_mf")
    else:
        results_dir = Path("results/integrated_tests_mf_h")
    results_dir.mkdir(parents=True, exist_ok=True)
    print(f"✓ Results will be saved to: {results_dir}")
    
    # Save episode statistics to Excel
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        demand_str = "demandintense" if use_intense_requests else "demandnormal"
        excel_filename = f"episode_statistics_mfrl_{adpvalue}_{demand_str}_{timestamp}.xlsx"
        excel_path = results_dir / excel_filename
        
        df = pd.DataFrame(results['episode_detailed_stats'])
        df.to_excel(excel_path, index=False)
        print(f"✓ Saved episode statistics to: {excel_path}")
        results['excel_path'] = str(excel_path)
    except Exception as e:
        print(f"❌ Failed to save Excel file: {e}")
        results['excel_path'] = None
    
    return results, env


def analyze_vehicle_visit_patterns_mf(env):
    """Analyze vehicle visit patterns for Mean Field RL test"""
    vehicle_visit_stats = {}
    
    hotspots = [
        (env.grid_size // 4, env.grid_size // 4),
        (3 * env.grid_size // 4, env.grid_size // 4),
        (env.grid_size // 2, 3 * env.grid_size // 4)
    ]
    
    for vehicle_id, vehicle in env.vehicles.items():
        position_history = env.vehicle_position_history.get(vehicle_id, [])
        
        if not position_history:
            current_coords = vehicle['coordinates']
            location_counts = {str(current_coords): 1}
        else:
            location_counts = {}
            for entry in position_history:
                coords_str = str(entry['coords'])
                location_counts[coords_str] = location_counts.get(coords_str, 0) + 1
        
        if location_counts:
            most_visited_location = max(location_counts, key=location_counts.get)
            unique_locations = len(location_counts)
            total_visits = sum(location_counts.values())
            diversity_score = unique_locations / total_visits if total_visits > 0 else 0
            
            vehicle_visit_stats[vehicle_id] = {
                'most_visited_location': most_visited_location,
                'unique_locations': unique_locations,
                'total_visits': total_visits,
                'diversity_score': diversity_score,
                'vehicle_type': vehicle['type']
            }
    
    return vehicle_visit_stats


def compare_mfrl_with_baseline(num_episodes=5, num_vehicles=10):
    """
    Compare Mean Field RL performance with baseline (no RL) approach
    
    Args:
        num_episodes: Number of episodes to run
        num_vehicles: Number of vehicles
    """
    print("=" * 60)
    print("MEAN FIELD RL VS BASELINE COMPARISON")
    print("=" * 60)
    
    results_comparison = {}
    
    # Run baseline (no ADP/neural network)
    print("\n[1/2] Running Baseline (Heuristic only)...")
    baseline_results, _ = run_charging_integration_test_mf(
        adpvalue=0,
        num_episodes=num_episodes,
        use_intense_requests=True,
        assignmentgurobi=True,
        batch_size=256,
        num_vehicles=num_vehicles
    )
    results_comparison['baseline'] = baseline_results
    
    # Run Mean Field RL
    print("\n[2/2] Running Mean Field RL...")
    mfrl_results, _ = run_charging_integration_test_mf(
        adpvalue=0.1,
        num_episodes=num_episodes,
        use_intense_requests=True,
        assignmentgurobi=True,
        batch_size=256,
        num_vehicles=num_vehicles
    )
    results_comparison['mfrl'] = mfrl_results
    
    # Print comparison summary
    print("\n" + "=" * 60)
    print("COMPARISON RESULTS")
    print("=" * 60)
    
    baseline_rewards = results_comparison['baseline']['episode_rewards']
    mfrl_rewards = results_comparison['mfrl']['episode_rewards']
    
    print(f"\nBaseline (Heuristic only):")
    print(f"  Average Episode Reward: {np.mean(baseline_rewards):.2f}")
    print(f"  Final Episode Reward: {baseline_rewards[-1]:.2f}")
    print(f"  Completed Orders: {results_comparison['baseline'].get('completed_requests', 'N/A')}")
    
    print(f"\nMean Field RL:")
    print(f"  Average Episode Reward: {np.mean(mfrl_rewards):.2f}")
    print(f"  Final Episode Reward: {mfrl_rewards[-1]:.2f}")
    print(f"  Completed Orders: {results_comparison['mfrl'].get('completed_requests', 'N/A')}")
    print(f"  Average MF Entropy: {np.mean(results_comparison['mfrl']['mean_field_entropy']):.3f}")
    print(f"  Average Training Loss: {np.mean(results_comparison['mfrl']['value_function_losses']):.4f}")
    
    improvement = (np.mean(mfrl_rewards) - np.mean(baseline_rewards)) / abs(np.mean(baseline_rewards)) * 100
    print(f"\nMFRL Improvement over Baseline: {improvement:+.1f}%")
    
    return results_comparison


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Mean Field RL Integration Test')
    parser.add_argument('--episodes', type=int, default=5, help='Number of episodes')
    parser.add_argument('--vehicles', type=int, default=10, help='Number of vehicles')
    parser.add_argument('--adp', type=float, default=0.1, help='ADP coefficient')
    parser.add_argument('--batch_size', type=int, default=256, help='Training batch size')
    parser.add_argument('--compare', action='store_true', help='Run comparison with baseline')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        results = compare_mfrl_with_baseline(
            num_episodes=args.episodes,
            num_vehicles=args.vehicles
        )
    else:
        # Run single Mean Field RL test
        results, env = run_charging_integration_test_mf(
            adpvalue=args.adp,
            num_episodes=args.episodes,
            use_intense_requests=True,
            assignmentgurobi=True,
            batch_size=args.batch_size,
            num_vehicles=args.vehicles
        )
        
        print("\n" + "=" * 60)
        print("FINAL SUMMARY")
        print("=" * 60)
        print(f"Total Episodes: {args.episodes}")
        print(f"Average Reward: {np.mean(results['episode_rewards']):.2f}")
        print(f"Completed Requests: {results.get('completed_requests', 'N/A')}")
        if results['mean_field_entropy']:
            print(f"Average Mean Field Entropy: {np.mean(results['mean_field_entropy']):.3f}")