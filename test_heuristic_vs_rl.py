"""
启发式策略 vs 强化学习策略性能对比测试
"""

import os
import sys
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 添加路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Environment import ChargingIntegratedEnvironment
from src.LearningAgent import LearningAgent
from src.Heuristic import HeuristicPolicy
from src.ChargingIntegrationVisualization import ChargingIntegrationVisualization


class PolicyComparison:
    """策略对比测试类"""
    
    def __init__(self, config=None):
        self.config = config or self._get_default_config()
        self.results = {}
        
    def _get_default_config(self):
        """获取默认测试配置"""
        return {
            'grid_size': 10,
            'num_vehicles': 12,
            'num_stations': 3,
            'use_intense_requests': True,
            'episode_length': 200,
            'num_test_episodes': 5,  # 减少测试轮次以加快测试
            'value_function_path': './logs/charging_nn/PyTorchChargingValueFunction/value_function_20241214_190439.pth'
        }
    
    def run_comparison(self):
        """运行完整的对比测试"""
        print("🚀 开始策略对比测试...")
        
        # 测试启发式策略
        print("\n📊 测试启发式策略...")
        heuristic_results = self._test_heuristic_policy()
        
        # 测试强化学习策略
        print("\n🧠 测试强化学习策略...")
        rl_results = self._test_rl_policy()
        
        # 保存结果
        self.results = {
            'heuristic': heuristic_results,
            'rl': rl_results,
            'config': self.config
        }
        
        # 生成对比报告
        self._generate_comparison_report()
        
        return self.results
    
    def _test_heuristic_policy(self):
        """测试启发式策略"""
        policy = HeuristicPolicy(battery_threshold=0.5, max_service_distance=8)
        return self._run_policy_test(policy, "Heuristic")
    
    def _test_rl_policy(self):
        """测试强化学习策略"""
        # 创建环境
        env = ChargingIntegratedEnvironment(
            grid_size=self.config['grid_size'],
            num_vehicles=self.config['num_vehicles'],
            num_stations=self.config['num_stations'],
            use_intense_requests=self.config['use_intense_requests']
        )
        
        # 创建简单的RL策略代理
        rl_agent = SimpleRLAgent(env)
        
        # 加载训练好的价值函数
        if os.path.exists(self.config['value_function_path']):
            from src.ValueFunction_pytorch import PyTorchValueFunction
            value_function = PyTorchValueFunction(env)
            value_function.load_model(self.config['value_function_path'])
            env.set_value_function(value_function)
            print(f"✅ 已加载价值函数: {self.config['value_function_path']}")
        else:
            print(f"⚠️  价值函数文件不存在: {self.config['value_function_path']}")
            print("使用随机策略作为RL基准")
        
        return self._run_rl_test(rl_agent, env)
    
    def _run_policy_test(self, policy, policy_name):
        """运行策略测试"""
        episode_results = []
        
        for episode in range(self.config['num_test_episodes']):
            print(f"  Episode {episode + 1}/{self.config['num_test_episodes']}")
            
            # 创建环境
            env = ChargingIntegratedEnvironment(
                grid_size=self.config['grid_size'],
                num_vehicles=self.config['num_vehicles'],
                num_stations=self.config['num_stations'],
                use_intense_requests=self.config['use_intense_requests']
            )
            
            # 运行单个episode
            episode_result = self._run_single_episode_heuristic(env, policy)
            episode_result['episode'] = episode
            episode_result['policy'] = policy_name
            episode_results.append(episode_result)
        
        return episode_results
    
    def _run_rl_test(self, agent, env):
        """运行强化学习测试"""
        episode_results = []
        
        for episode in range(self.config['num_test_episodes']):
            print(f"  Episode {episode + 1}/{self.config['num_test_episodes']}")
            
            # 重置环境
            env.reset()
            
            # 运行单个episode
            episode_result = self._run_single_episode_rl(env, agent)
            episode_result['episode'] = episode
            episode_result['policy'] = 'RL'
            episode_results.append(episode_result)
        
        return episode_results
    
    def _run_single_episode_heuristic(self, env, policy):
        """运行单个episode - 启发式策略"""
        env.reset()
        
        total_reward = 0
        charging_events = 0
        
        step_rewards = []
        battery_levels = []
        utilization_rates = []
        
        # 跟踪请求状态
        initial_completed = len(env.completed_requests)
        initial_rejected = len(env.rejected_requests)
        
        for step in range(self.config['episode_length']):
            # 记录状态
            avg_battery = np.mean([v['battery'] for v in env.vehicles.values()])
            battery_levels.append(avg_battery)
            
            # 获取启发式动作
            actions = policy.get_actions(env)
            
            # 执行动作
            next_states, step_rewards_dict, done, info = env.step(actions)
            
            # 计算总步骤奖励
            step_reward = sum(step_rewards_dict.values()) if isinstance(step_rewards_dict, dict) else step_rewards_dict
            total_reward += step_reward
            step_rewards.append(step_reward)
            
            # 统计充电事件
            charging_actions = sum(1 for action in actions.values() 
                                 if hasattr(action, 'charging_station_id') and action.charging_station_id is not None)
            charging_events += charging_actions
            
            # 计算利用率
            if env.charging_manager.stations:
                total_capacity = sum(station.max_capacity for station in env.charging_manager.stations.values())
                used_capacity = sum(len(station.current_vehicles) for station in env.charging_manager.stations.values())
                utilization = used_capacity / total_capacity if total_capacity > 0 else 0
            else:
                utilization = 0
            utilization_rates.append(utilization)
        
        # 计算最终统计
        total_served = len(env.completed_requests) - initial_completed
        total_rejected = len(env.rejected_requests) - initial_rejected
        
        return {
            'total_reward': total_reward,
            'total_served': total_served,
            'total_rejected': total_rejected,
            'total_charging_cost': 0,  # 暂时设为0，稍后可以添加计算
            'total_service_reward': 0,  # 暂时设为0，稍后可以添加计算
            'charging_events': charging_events,
            'avg_battery': np.mean(battery_levels),
            'min_battery': np.min(battery_levels),
            'avg_utilization': np.mean(utilization_rates),
            'service_rate': total_served / (total_served + total_rejected) if (total_served + total_rejected) > 0 else 0,
            'step_rewards': step_rewards,
            'battery_levels': battery_levels,
            'utilization_rates': utilization_rates
        }
    
    def _run_single_episode_rl(self, env, agent):
        """运行单个episode - 强化学习策略"""
        total_reward = 0
        charging_events = 0
        
        step_rewards = []
        battery_levels = []
        utilization_rates = []
        
        # 跟踪请求状态
        initial_completed = len(env.completed_requests)
        initial_rejected = len(env.rejected_requests)
        
        for step in range(self.config['episode_length']):
            # 记录状态
            avg_battery = np.mean([v['battery'] for v in env.vehicles.values()])
            battery_levels.append(avg_battery)
            
            # 获取RL动作
            actions = agent.get_actions(env)
            
            # 执行动作
            next_states, step_rewards_dict, done, info = env.step(actions)
            
            # 计算总步骤奖励
            step_reward = sum(step_rewards_dict.values()) if isinstance(step_rewards_dict, dict) else step_rewards_dict
            total_reward += step_reward
            step_rewards.append(step_reward)
            
            # 统计充电事件
            charging_actions = sum(1 for action in actions.values() 
                                 if hasattr(action, 'charging_station_id') and action.charging_station_id is not None)
            charging_events += charging_actions
            
            # 计算利用率
            if env.charging_manager.stations:
                total_capacity = sum(station.max_capacity for station in env.charging_manager.stations.values())
                used_capacity = sum(len(station.current_vehicles) for station in env.charging_manager.stations.values())
                utilization = used_capacity / total_capacity if total_capacity > 0 else 0
            else:
                utilization = 0
            utilization_rates.append(utilization)
        
        # 计算最终统计
        total_served = len(env.completed_requests) - initial_completed
        total_rejected = len(env.rejected_requests) - initial_rejected
        
        return {
            'total_reward': total_reward,
            'total_served': total_served,
            'total_rejected': total_rejected,
            'total_charging_cost': 0,  # 暂时设为0，稍后可以添加计算
            'total_service_reward': 0,  # 暂时设为0，稍后可以添加计算
            'charging_events': charging_events,
            'avg_battery': np.mean(battery_levels),
            'min_battery': np.min(battery_levels),
            'avg_utilization': np.mean(utilization_rates),
            'service_rate': total_served / (total_served + total_rejected) if (total_served + total_rejected) > 0 else 0,
            'step_rewards': step_rewards,
            'battery_levels': battery_levels,
            'utilization_rates': utilization_rates
        }
    
    def _generate_comparison_report(self):
        """生成对比报告"""
        print("\n📈 生成对比报告...")
        
        # 创建DataFrame
        all_results = []
        for policy_name, results in [('Heuristic', self.results['heuristic']), 
                                   ('RL', self.results['rl'])]:
            for result in results:
                result_copy = result.copy()
                result_copy['policy'] = policy_name
                all_results.append(result_copy)
        
        df = pd.DataFrame(all_results)
        
        # 计算统计汇总
        summary_stats = df.groupby('policy').agg({
            'total_reward': ['mean', 'std'],
            'total_served': ['mean', 'std'],
            'total_rejected': ['mean', 'std'],
            'service_rate': ['mean', 'std'],
            'charging_events': ['mean', 'std'],
            'avg_battery': ['mean', 'std'],
            'avg_utilization': ['mean', 'std']
        }).round(3)
        
        # 保存统计结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        df.to_csv(f'results/heuristic_vs_rl_detailed_{timestamp}.csv', index=False)
        
        # 保存汇总统计
        with open(f'results/heuristic_vs_rl_summary_{timestamp}.txt', 'w', encoding='utf-8') as f:
            f.write("策略对比汇总统计\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"测试配置:\n")
            f.write(f"- 网格大小: {self.config['grid_size']}x{self.config['grid_size']}\n")
            f.write(f"- 车辆数量: {self.config['num_vehicles']}\n")
            f.write(f"- 充电站数量: {self.config['num_stations']}\n")
            f.write(f"- 密集请求模式: {self.config['use_intense_requests']}\n")
            f.write(f"- 测试轮次: {self.config['num_test_episodes']}\n")
            f.write(f"- Episode长度: {self.config['episode_length']}\n\n")
            
            f.write("统计结果:\n")
            f.write(str(summary_stats))
            f.write("\n\n")
            
            # 添加性能对比
            heuristic_reward = summary_stats.loc['Heuristic', ('total_reward', 'mean')]
            rl_reward = summary_stats.loc['RL', ('total_reward', 'mean')]
            improvement = ((rl_reward - heuristic_reward) / abs(heuristic_reward)) * 100
            
            f.write(f"性能对比:\n")
            f.write(f"- 启发式策略平均奖励: {heuristic_reward:.2f}\n")
            f.write(f"- 强化学习策略平均奖励: {rl_reward:.2f}\n")
            f.write(f"- 强化学习相对改进: {improvement:.2f}%\n")
        
        # 生成可视化图表
        self._generate_comparison_plots(df, timestamp)
        
        print(f"✅ 报告已保存:")
        print(f"   详细结果: results/heuristic_vs_rl_detailed_{timestamp}.csv")
        print(f"   汇总统计: results/heuristic_vs_rl_summary_{timestamp}.txt")
        print(f"   可视化图表: results/policy_comparison_plots_{timestamp}.png")
        
        # 打印关键结果
        print(f"\n🎯 关键对比结果:")
        print(f"启发式策略 - 平均奖励: {heuristic_reward:.2f}")
        print(f"强化学习策略 - 平均奖励: {rl_reward:.2f}")
        print(f"强化学习相对改进: {improvement:.2f}%")
    
    def _generate_comparison_plots(self, df, timestamp):
        """生成对比可视化图表"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('策略性能对比分析', fontsize=16, fontweight='bold')
        
        # 1. 总奖励对比
        sns.boxplot(data=df, x='policy', y='total_reward', ax=axes[0, 0])
        axes[0, 0].set_title('总奖励对比')
        axes[0, 0].set_ylabel('总奖励')
        
        # 2. 服务请求数对比
        sns.boxplot(data=df, x='policy', y='total_served', ax=axes[0, 1])
        axes[0, 1].set_title('服务请求数对比')
        axes[0, 1].set_ylabel('服务请求数')
        
        # 3. 服务率对比
        sns.boxplot(data=df, x='policy', y='service_rate', ax=axes[0, 2])
        axes[0, 2].set_title('服务率对比')
        axes[0, 2].set_ylabel('服务率')
        
        # 4. 充电事件对比
        sns.boxplot(data=df, x='policy', y='charging_events', ax=axes[1, 0])
        axes[1, 0].set_title('充电事件数对比')
        axes[1, 0].set_ylabel('充电事件数')
        
        # 5. 平均电池水平对比
        sns.boxplot(data=df, x='policy', y='avg_battery', ax=axes[1, 1])
        axes[1, 1].set_title('平均电池水平对比')
        axes[1, 1].set_ylabel('平均电池水平')
        
        # 6. 充电站利用率对比
        sns.boxplot(data=df, x='policy', y='avg_utilization', ax=axes[1, 2])
        axes[1, 2].set_title('充电站平均利用率对比')
        axes[1, 2].set_ylabel('平均利用率')
        
        plt.tight_layout()
        plt.savefig(f'results/policy_comparison_plots_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 生成时间序列对比图
        self._generate_time_series_plots(timestamp)
    
    def _generate_time_series_plots(self, timestamp):
        """生成时间序列对比图"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('策略性能时间序列对比', fontsize=16, fontweight='bold')
        
        # 获取第一个episode的时间序列数据作为示例
        heuristic_sample = self.results['heuristic'][0]
        rl_sample = self.results['rl'][0]
        
        steps = range(len(heuristic_sample['step_rewards']))
        
        # 1. 步骤奖励对比
        axes[0, 0].plot(steps, heuristic_sample['step_rewards'], 
                       label='Heuristic', alpha=0.7, linewidth=1)
        axes[0, 0].plot(steps, rl_sample['step_rewards'], 
                       label='RL', alpha=0.7, linewidth=1)
        axes[0, 0].set_title('步骤奖励对比 (示例Episode)')
        axes[0, 0].set_xlabel('时间步')
        axes[0, 0].set_ylabel('步骤奖励')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 电池水平对比
        axes[0, 1].plot(steps, heuristic_sample['battery_levels'], 
                       label='Heuristic', alpha=0.7, linewidth=1)
        axes[0, 1].plot(steps, rl_sample['battery_levels'], 
                       label='RL', alpha=0.7, linewidth=1)
        axes[0, 1].set_title('平均电池水平对比')
        axes[0, 1].set_xlabel('时间步')
        axes[0, 1].set_ylabel('平均电池水平')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. 充电站利用率对比
        axes[1, 0].plot(steps, heuristic_sample['utilization_rates'], 
                       label='Heuristic', alpha=0.7, linewidth=1)
        axes[1, 0].plot(steps, rl_sample['utilization_rates'], 
                       label='RL', alpha=0.7, linewidth=1)
        axes[1, 0].set_title('充电站利用率对比')
        axes[1, 0].set_xlabel('时间步')
        axes[1, 0].set_ylabel('利用率')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. 累积奖励对比
        heuristic_cumulative = np.cumsum(heuristic_sample['step_rewards'])
        rl_cumulative = np.cumsum(rl_sample['step_rewards'])
        
        axes[1, 1].plot(steps, heuristic_cumulative, 
                       label='Heuristic', alpha=0.7, linewidth=2)
        axes[1, 1].plot(steps, rl_cumulative, 
                       label='RL', alpha=0.7, linewidth=2)
        axes[1, 1].set_title('累积奖励对比')
        axes[1, 1].set_xlabel('时间步')
        axes[1, 1].set_ylabel('累积奖励')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'results/time_series_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
        plt.close()


def main():
    """主测试函数"""
    print("🎯 启发式策略 vs 强化学习策略性能对比测试")
    print("=" * 60)
    
    # 确保结果目录存在
    os.makedirs('results', exist_ok=True)
    
    # 创建对比测试实例
    comparison = PolicyComparison()
    
    # 运行对比测试
    start_time = time.time()
    results = comparison.run_comparison()
    end_time = time.time()
    
    print(f"\n⏱️  总测试时间: {end_time - start_time:.2f} 秒")
    print("🎉 测试完成！")


if __name__ == "__main__":
    main()
