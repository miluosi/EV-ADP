"""
测试已保存的Checkpoint在新随机环境下的推理性能
不进行训练，只评估已训练模型的表现
"""
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import numpy as np
import random
import torch
from pathlib import Path
from datetime import datetime
import pandas as pd
import glob
import re

from src.Environment import ChargingIntegratedEnvironment
from src.ValueFunction_pytorch import PyTorchChargingValueFunction


def find_latest_checkpoint(checkpoint_dir, by_time=True):
    """
    自动找到目录下最新的checkpoint
    
    Args:
        checkpoint_dir: checkpoint目录路径
        by_time: True=按修改时间找最新, False=按episode编号找最大
        
    Returns:
        int: 最新的episode编号，如果没有找到返回None
    """
    if not os.path.exists(checkpoint_dir):
        print(f"❌ Checkpoint目录不存在: {checkpoint_dir}")
        return None
    
    # 查找所有full_state_episode_*.pth文件
    pattern = os.path.join(checkpoint_dir, "full_state_episode_*.pth")
    checkpoint_files = glob.glob(pattern)
    
    if not checkpoint_files:
        print(f"❌ 未找到checkpoint文件在: {checkpoint_dir}")
        return None
    
    if by_time:
        # 按修改时间排序，找最新的
        checkpoint_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        latest_file = checkpoint_files[0]
        filename = os.path.basename(latest_file)
        match = re.search(r'full_state_episode_(\d+)\.pth', filename)
        if match:
            latest_episode = int(match.group(1))
            modification_time = datetime.fromtimestamp(os.path.getmtime(latest_file))
            print(f"✓ 找到 {len(checkpoint_files)} 个checkpoints")
            print(f"  最新修改: episode {latest_episode} ({modification_time.strftime('%Y-%m-%d %H:%M:%S')})")
            return latest_episode
    else:
        # 按episode编号排序，找最大的
        episodes = []
        for filepath in checkpoint_files:
            filename = os.path.basename(filepath)
            match = re.search(r'full_state_episode_(\d+)\.pth', filename)
            if match:
                episodes.append(int(match.group(1)))
        
        if episodes:
            latest_episode = max(episodes)
            print(f"✓ 找到 {len(episodes)} 个checkpoints，最大episode: {latest_episode}")
            return latest_episode
    
    return None


def set_random_seeds(seed=42):
    """设置所有随机数生成器的种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seeds set to {seed}")


def load_checkpoint_for_inference(value_function, checkpoint_path):
    """
    加载checkpoint用于推理（只加载target_network权重）
    
    Args:
        value_function: PyTorchChargingValueFunction实例
        checkpoint_path: checkpoint文件路径
    
    Returns:
        bool: 是否成功加载
    """
    try:
        if not os.path.exists(checkpoint_path):
            print(f"❌ Checkpoint文件不存在: {checkpoint_path}")
            return False
        
        print(f"📂 加载checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=value_function.device)
        
        # 加载target_network的权重到主网络（用于推理）
        if 'target_network_state_dict' in checkpoint:
            value_function.network.load_state_dict(checkpoint['target_network_state_dict'])
            print(f"✓ 成功加载target_network权重")
        elif 'network_state_dict' in checkpoint:
            value_function.network.load_state_dict(checkpoint['network_state_dict'])
            print(f"✓ 成功加载network权重")
        else:
            print(f"❌ Checkpoint中没有找到网络权重")
            return False
        
        # 设置为评估模式
        value_function.network.eval()
        
        # 显示checkpoint信息
        if 'episode' in checkpoint:
            print(f"  - Episode: {checkpoint['episode']}")
        if 'training_step' in checkpoint:
            print(f"  - Training step: {checkpoint['training_step']}")
        if 'buffer_size' in checkpoint:
            print(f"  - Buffer size: {checkpoint['buffer_size']}")
        
        return True
        
    except Exception as e:
        print(f"❌ 加载checkpoint失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_checkpoint_inference(
    checkpoint_path_aev,
    checkpoint_path_ev,
    num_episodes=5,
    use_intense_requests=True,
    batch_size=256,
    num_vehicles=10,
    num_ev=5,
    transportation_mode='integrated',
    test_seed=9999,  # 使用不同的种子测试泛化能力
    use_heuristic=False  ,
    onlyilp = False
):
    """
    测试已保存的checkpoint在新环境下的推理性能
    
    Args:
        checkpoint_path_aev: AEV的checkpoint路径
        checkpoint_path_ev: EV的checkpoint路径
        num_episodes: 测试episode数量
        test_seed: 测试用的随机数种子（不同于训练时的种子）
        use_heuristic: 是否使用启发式方法（True=启发式，False=Gurobi ILP）
    """
    assignment_method = "启发式" if use_heuristic else "Gurobi ILP"
    print("=== 开始Checkpoint推理测试 ===")
    print(f"📊 测试配置:")
    print(f"  - 测试episodes: {num_episodes}")
    print(f"  - 测试种子: {test_seed} (不同于训练种子)")
    print(f"  - 车辆配置: {num_vehicles} 总车辆, {num_ev} EV")
    print(f"  - 模式: {transportation_mode}")
    print(f"  - 分配方法: {assignment_method}")
    
    # 设置测试用的随机种子
    set_random_seeds(seed=test_seed)
    
    # 初始化环境
    num_stations = 4
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles,
        num_stations=num_stations,
        ev_num_vehicles=num_ev,
        random_seed=test_seed,  # 使用测试种子
        use_intense_requests=use_intense_requests
    )
    
    env.adp_value = 1  # 使用ADP
    env.assignmentgurobi = not use_heuristic  # True=使用Gurobi, False=使用启发式
    
    if use_heuristic:
        print(f"✓ 使用启发式分配方法（不使用Gurobi优化器）")
    else:
        print(f"✓ 使用Gurobi ILP优化分配")
    
    if not onlyilp:
        value_function = PyTorchChargingValueFunction(
            grid_size=env.grid_size,
            num_vehicles=num_vehicles,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            episode_length=env.episode_length,
            max_requests=10000,
            env=env
        )
        
        value_function_ev = PyTorchChargingValueFunction(
            grid_size=env.grid_size,
            num_vehicles=num_vehicles,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            episode_length=env.episode_length,
            max_requests=10000,
            env=env
        )
        
        # 加载checkpoint
        print("\n📦 加载AEV checkpoint...")
        if not load_checkpoint_for_inference(value_function, checkpoint_path_aev):
            print("❌ 无法加载AEV checkpoint，测试终止")
            return None, None
        
        print("\n📦 加载EV checkpoint...")
        if not load_checkpoint_for_inference(value_function_ev, checkpoint_path_ev):
            print("❌ 无法加载EV checkpoint，测试终止")
            return None, None
        
        # 设置value function到环境
        env.set_value_function(value_function)
        env.set_value_function_ev(value_function_ev)
    else:
        value_function = None
        value_function_ev = None
        env.adp_value = 0
    print("\n✓ Checkpoint加载完成，开始推理测试...")
    print(f"✓ 使用设备: {value_function.device if value_function else 'N/A'}")
    
    # 测试结果记录
    results = {
        'episode_rewards': [],
        'episode_detailed_stats': [],
        'charging_events': [],
        'battery_levels': [],
        'environment_stats': [],
        'Idle_average': []
    }
    
    # 运行测试episodes
    for episode in range(num_episodes):
        # 为每个episode设置不同的请求生成种子
        episode_seed = test_seed + 1000 + episode
        env.set_request_generation_seed(episode_seed)
        print(f"\n{'='*70}")
        print(f"Episode {episode + 1}/{num_episodes}: 请求生成种子 = {episode_seed}")
        print(f"{'='*70}")
        
        # Reset环境
        states = env.reset()
        episode_reward = 0
        episode_charging_events = []
        Idle_list = []
        
        # 运行episode
        for step in range(env.episode_length):
            current_requests = list(env.active_requests.values())
            
            # 生成动作（使用已训练的网络，不训练）
            if transportation_mode == 'integrated':
                actions, storeactions, storeactions_ev = env.simulate_motion(
                    agents=[], 
                    current_requests=current_requests, 
                    rebalance=True
                )
            elif transportation_mode == 'evfirst':
                actions, storeactions, storeactions_ev = env.simulate_motion_evfirst(
                    agents=[], 
                    current_requests=current_requests, 
                    rebalance=True
                )
            elif transportation_mode == 'aevfirst':
                actions, storeactions, storeactions_ev = env.simulate_motion_aevfirst(
                    agents=[], 
                    current_requests=current_requests, 
                    rebalance=True
                )
            elif transportation_mode == 'mode_onlyadp':
                actions, storeactions, storeactions_ev = env.simulate_motion(
                    agents=[], 
                    current_requests=current_requests, 
                    rebalance=True
                )
            else:
                actions, storeactions, storeactions_ev = env.simulate_motion(
                    agents=[], 
                    current_requests=current_requests, 
                    rebalance=True
                )
                
                
            # 执行动作
            next_states, rewards, dur_rewards, done, info = env.step(
                actions, storeactions, storeactions_ev
            )
            
            episode_reward += sum(rewards.values())
            episode_charging_events.extend(info.get('charging_events', []))
            
            # 每50步输出统计
            if step % 50 == 0:
                stats = env.get_stats()
                active_requests = len(env.active_requests) if hasattr(env, 'active_requests') else 0
                
                # 统计车辆状态
                vehicle_status_count = {
                    'charging': 0,
                    'onboard': 0,
                    'to_pickup': 0,
                    'to_charge': 0,
                    'idle_moving': 0,
                    'fully_idle': 0
                }
                
                for vid, v in env.vehicles.items():
                    if v['charging_station'] is not None:
                        status = 'charging'
                    elif v['passenger_onboard'] is not None:
                        status = 'onboard'
                    elif v['assigned_request'] is not None:
                        status = 'to_pickup'
                    elif v.get('charging_target') is not None:
                        status = 'to_charge'
                    elif v.get('idle_target') is not None or v.get('target_location') is not None:
                        status = 'idle_moving'
                    else:
                        status = 'fully_idle'
                    vehicle_status_count[status] += 1
                
                step_reward = sum(rewards.values())
                idle_vehicles = vehicle_status_count['fully_idle']
                Idle_list.append(idle_vehicles)
                
                print(f"  Step {step}: Requests={active_requests}, Reward={step_reward:.2f}")
                print(f"    Status: Charging={vehicle_status_count['charging']}, "
                      f"Onboard={vehicle_status_count['onboard']}, "
                      f"To_pickup={vehicle_status_count['to_pickup']}, "
                      f"Idle={idle_vehicles}")
        
        # Episode结束统计
        results['episode_rewards'].append(episode_reward)
        results['charging_events'].extend(episode_charging_events)
        results['Idle_average'].append(sum(Idle_list)/len(Idle_list) if Idle_list else 0)
        
        stats = env.get_stats()
        results['environment_stats'].append(stats)
        results['battery_levels'].append(stats['average_battery'])
        
        episode_stats = env.get_episode_stats()
        episode_stats['episode_number'] = episode + 1
        episode_stats['episode_reward'] = episode_reward
        episode_stats['charging_events_count'] = len(episode_charging_events)
        results['episode_detailed_stats'].append(episode_stats)
        
        # 输出episode总结
        print(f"\n📊 Episode {episode + 1} 完成:")
        print(f"  总奖励: {episode_reward:.2f}")
        print(f"  订单统计: 总={episode_stats['total_orders']}, "
              f"接受={episode_stats['accepted_orders']}, "
              f"完成={episode_stats['completed_orders']}, "
              f"拒绝={episode_stats['rejected_orders']}")
        print(f"  平均电量: {episode_stats['avg_battery_level']:.2f}")
        print(f"  平均空闲车辆: {results['Idle_average'][-1]:.2f}")
    
    # 保存测试结果
    print("\n" + "="*70)
    print("保存测试结果...")
    results_dir = Path("results/checkpoint_inference")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    method_suffix = "heuristic" if use_heuristic else "gurobi"
    result_file = results_dir / f"inference_test_{use_intense_requests}_{transportation_mode}_{method_suffix}_seed{test_seed}_{timestamp}.xlsx"
    
    # 创建汇总DataFrame
    summary_data = {
        'Episode': range(1, num_episodes + 1),
        'Total_Reward': results['episode_rewards'],
        'Avg_Idle_Vehicles': results['Idle_average'],
        'Avg_Battery': results['battery_levels'],
    }
    
    for i, stats in enumerate(results['episode_detailed_stats']):
        summary_data.setdefault('Total_Orders', []).append(stats['total_orders'])
        summary_data.setdefault('Accepted_Orders', []).append(stats['accepted_orders'])
        summary_data.setdefault('Completed_Orders', []).append(stats['completed_orders'])
        summary_data.setdefault('Rejected_Orders', []).append(stats['rejected_orders'])
    
    summary_df = pd.DataFrame(summary_data)
    
    with pd.ExcelWriter(result_file, engine='openpyxl') as writer:
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 计算平均值
        avg_data = {
            'Metric': ['Avg_Reward', 'Avg_Idle', 'Avg_Battery', 
                      'Avg_Total_Orders', 'Avg_Accepted', 'Avg_Completed', 'Avg_Rejected'],
            'Value': [
                np.mean(results['episode_rewards']),
                np.mean(results['Idle_average']),
                np.mean(results['battery_levels']),
                np.mean([s['total_orders'] for s in results['episode_detailed_stats']]),
                np.mean([s['accepted_orders'] for s in results['episode_detailed_stats']]),
                np.mean([s['completed_orders'] for s in results['episode_detailed_stats']]),
                np.mean([s['rejected_orders'] for s in results['episode_detailed_stats']])
            ]
        }
        avg_df = pd.DataFrame(avg_data)
        avg_df.to_excel(writer, sheet_name='Averages', index=False)
    
    print(f"✓ 结果已保存到: {result_file}")
    
    # 打印最终统计
    print("\n" + "="*70)
    print("📈 测试总结:")
    print(f"  平均奖励: {np.mean(results['episode_rewards']):.2f} ± {np.std(results['episode_rewards']):.2f}")
    print(f"  平均空闲车辆: {np.mean(results['Idle_average']):.2f}")
    print(f"  平均电量: {np.mean(results['battery_levels']):.2f}")
    print(f"  平均订单完成: {np.mean([s['completed_orders'] for s in results['episode_detailed_stats']]):.2f}")
    print(f"  平均拒单率: {np.mean([s['rejected_orders']/(s['total_orders']+1e-6) for s in results['episode_detailed_stats']])*100:.2f}%")
    print("="*70)
    
    return results, env


def main():
    """主函数：配置并运行checkpoint推理测试"""
    
    # 配置checkpoint路径
    checkpoint_dir = "checkpoints"
    transportation_mode = "integrated"
    num_ev = 3  # 🔧 EV数量
    num_vehicles = 6  # 🔧 总车辆数 = 6 (3 EV + 3 AEV)，必须匹配训练时的配置！
    use_intense_requests = True

    
    # 🆕 自动查找最新checkpoint
    # aev_checkpoint_dir = f"{checkpoint_dir}/q_networks_{transportation_mode}_{num_ev}_{use_intense_requests}_aev"
    # ev_checkpoint_dir = f"{checkpoint_dir}/q_networks_{transportation_mode}_{num_ev}_{use_intense_requests}_ev"
    # print("🔍 查找最新的checkpoint（按修改时间）...")
    # latest_episode_aev = find_latest_checkpoint(aev_checkpoint_dir, by_time=True)  # True=按修改时间
    # latest_episode_ev = find_latest_checkpoint(ev_checkpoint_dir, by_time=True)
    # if latest_episode_aev is None or latest_episode_ev is None:
    #     print("❌ 无法找到checkpoint，测试终止")
    #     return
    # episode = min(latest_episode_aev, latest_episode_ev)
    # print(f"📌 将使用 episode {episode} 的checkpoint")
    # checkpoint_path_aev = f"{aev_checkpoint_dir}/full_state_episode_{episode}.pth"
    # checkpoint_path_ev = f"{ev_checkpoint_dir}/full_state_episode_{episode}.pth"
    # if not os.path.exists(checkpoint_path_aev):
    #     print(f"❌ AEV checkpoint不存在: {checkpoint_path_aev}")
    #     return
    # if not os.path.exists(checkpoint_path_ev):
    #     print(f"❌ EV checkpoint不存在: {checkpoint_path_ev}")
    #     return
    # use_heuristic = False  # True=启发式(不用Gurobi), False=Gurobi ILP
    # assignment_method = "启发式" if use_heuristic else "Gurobi ILP"
    # print(f"\n📋 测试配置:")
    # modelist = ['integrated', 'evfirst', 'aevfirst']
    # for mode in modelist:
    #     results, env = test_checkpoint_inference(
    #         checkpoint_path_aev=checkpoint_path_aev,
    #         checkpoint_path_ev=checkpoint_path_ev,
    #         num_episodes=50,  # 测试5个episodes
    #         use_intense_requests=use_intense_requests,
    #         batch_size=256,
    #         num_vehicles=num_vehicles,
    #         num_ev=num_ev,
    #         transportation_mode=mode,
    #         test_seed=128,
    #         use_heuristic=use_heuristic  # 传递启发式配置
    #     )
    
    
    
    
    results, env = test_checkpoint_inference(
            checkpoint_path_aev=None,
            checkpoint_path_ev=None,
            num_episodes=50,  # 测试5个episodes
            use_intense_requests=use_intense_requests,
            batch_size=256,
            num_vehicles=num_vehicles,
            num_ev=num_ev,
            transportation_mode="mode_onlyadp_unknown",
            test_seed=128,
            use_heuristic=False,  # 传递启发式配置
            onlyilp = True
        )
    # results, env = test_checkpoint_inference(
    #         checkpoint_path_aev=None,
    #         checkpoint_path_ev=None,
    #         num_episodes=50,  # 测试5个episodes
    #         use_intense_requests=use_intense_requests,
    #         batch_size=256,
    #         num_vehicles=num_vehicles,
    #         num_ev=num_ev,
    #         transportation_mode="mode_heuristic",
    #         test_seed=128,
    #         use_heuristic=True,  # 传递启发式配置
    #         onlyilp = True
    #     )
    
    print("\n✅ 测试完成!")


if __name__ == "__main__":
    main()
