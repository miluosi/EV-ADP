#!/usr/bin/env python3
"""
Q-Value Analysis Tool
分析Q-network训练过程中的experience数据和Q-value问题
导出experience数据为CSV格式便于查看和分析
"""

import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from datetime import datetime
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

# 添加src目录到路径
sys.path.append('src')

class QValueAnalyzer:
    def __init__(self):
        self.results_dir = "results/q_value_analysis"
        os.makedirs(self.results_dir, exist_ok=True)
        
    def load_experience_from_running_env(self, value_function):
        """从正在运行的环境中加载experience数据"""
        if not hasattr(value_function, 'experience_buffer'):
            print("❌ No experience_buffer found in value_function")
            return None
        
        experiences = list(value_function.experience_buffer)
        print(f"📊 Loaded {len(experiences)} experiences from running environment")
        return experiences
    
    def load_experience_from_file(self, file_path):
        """从文件加载experience数据"""
        try:
            if file_path.endswith('.pkl'):
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    experiences = data.get('experiences', [])
            elif file_path.endswith('.json'):
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    experiences = data.get('experiences', [])
            else:
                print(f"❌ Unsupported file format: {file_path}")
                return None
                
            print(f"📊 Loaded {len(experiences)} experiences from {file_path}")
            return experiences
        except Exception as e:
            print(f"❌ Error loading file {file_path}: {e}")
            return None
    
    def export_experience_to_csv(self, experiences, filename=None):
        """导出experience数据到CSV文件"""
        if not experiences:
            print("❌ No experiences to export")
            return None
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"experience_data_{timestamp}.csv"
        
        # 转换experience数据为DataFrame
        rows = []
        for i, exp in enumerate(experiences):
            row = {
                'experience_id': i,
                'vehicle_id': exp.get('vehicle_id', 0),
                'vehicle_type': exp.get('vehicle_type', 1),  # 1=EV, 2=AEV
                'action_type': exp.get('action_type', ''),
                'vehicle_location': exp.get('vehicle_location', 0),
                'target_location': exp.get('target_location', 0),
                'battery_level': exp.get('battery_level', 1.0),
                'current_time': exp.get('current_time', 0.0),
                'reward': exp.get('reward', 0.0),
                'next_vehicle_location': exp.get('next_vehicle_location', 0),
                'next_battery_level': exp.get('next_battery_level', 1.0),
                'next_action_type': exp.get('next_action_type', ''),
                'other_vehicles': exp.get('other_vehicles', 0),
                'num_requests': exp.get('num_requests', 0),
                'request_value': exp.get('request_value', 0.0),
                'next_request_value': exp.get('next_request_value', 0.0),
                'is_idle': exp.get('is_idle', False),
                'is_rejection': exp.get('is_rejection', False),
                'rejection_reason': exp.get('rejection_reason', ''),
                'rejection_distance': exp.get('rejection_distance', 0.0)
            }
            
            # 计算距离
            if 'vehicle_location' in exp and 'target_location' in exp:
                v_loc = exp['vehicle_location']
                t_loc = exp['target_location']
                # 假设grid_size为10（需要根据实际情况调整）
                grid_size = 10
                vx, vy = v_loc % grid_size, v_loc // grid_size
                tx, ty = t_loc % grid_size, t_loc // grid_size
                row['manhattan_distance'] = abs(vx - tx) + abs(vy - ty)
            else:
                row['manhattan_distance'] = 0
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        csv_path = os.path.join(self.results_dir, filename)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"✅ Experience data exported to CSV:")
        print(f"   📄 File: {csv_path}")
        print(f"   📊 Rows: {len(df)}")
        print(f"   📋 Columns: {list(df.columns)}")
        
        return df, csv_path
    
    def analyze_q_value_problem(self, experiences, value_function=None):
        """深度分析Q-value问题"""
        print("\n🔍 DEEP Q-VALUE ANALYSIS")
        print("=" * 50)
        
        if not experiences:
            print("❌ No experiences to analyze")
            return
        
        df = pd.DataFrame(experiences)
        
        # 基础统计
        print(f"\n📊 Dataset Overview:")
        print(f"   Total experiences: {len(df)}")
        print(f"   Time range: {df['current_time'].min():.0f} - {df['current_time'].max():.0f}")
        print(f"   Unique vehicles: {df['vehicle_id'].nunique()}")
        
        # 动作类型分析
        print(f"\n🎯 Action Type Distribution:")
        action_counts = df['action_type'].value_counts()
        for action, count in action_counts.head(10).items():
            percentage = count / len(df) * 100
            print(f"   {action}: {count} ({percentage:.1f}%)")
        
        # 奖励分析按动作类型
        print(f"\n💰 Reward Analysis by Action Type:")
        
        # 简化动作类型分类
        df['action_category'] = df['action_type'].apply(self._categorize_action)
        
        reward_stats = df.groupby('action_category')['reward'].agg([
            'count', 'mean', 'std', 'min', 'max',
            lambda x: (x > 0).sum(),  # positive count
            lambda x: (x > 0).mean()  # positive ratio
        ]).round(3)
        reward_stats.columns = ['count', 'mean', 'std', 'min', 'max', 'positive_count', 'positive_ratio']
        
        print(reward_stats)
        
        # Q-value问题分析
        print(f"\n⚠️  Q-VALUE PROBLEM ANALYSIS:")
        
        # 分析assign vs idle的奖励差异
        assign_rewards = df[df['action_category'] == 'assign']['reward']
        idle_rewards = df[df['action_category'] == 'idle']['reward']
        
        if len(assign_rewards) > 0 and len(idle_rewards) > 0:
            print(f"\n🔄 Assign vs Idle Comparison:")
            print(f"   Assign: mean={assign_rewards.mean():.3f}, std={assign_rewards.std():.3f}, pos_ratio={(assign_rewards > 0).mean():.1%}")
            print(f"   Idle:   mean={idle_rewards.mean():.3f}, std={idle_rewards.std():.3f}, pos_ratio={(idle_rewards > 0).mean():.1%}")
            print(f"   Difference: {assign_rewards.mean() - idle_rewards.mean():.3f}")
            
            if assign_rewards.mean() < idle_rewards.mean():
                print(f"   🚨 PROBLEM: Assign actions have lower average reward than idle!")
                
                # 分析原因
                print(f"\n🔍 Root Cause Analysis:")
                
                # 1. 距离分析
                assign_df = df[df['action_category'] == 'assign'].copy()
                if len(assign_df) > 0:
                    # 计算距离
                    assign_df['distance'] = assign_df.apply(
                        lambda row: self._calculate_distance(row['vehicle_location'], row['target_location']), 
                        axis=1
                    )
                    
                    # 距离vs奖励关系
                    distance_reward_corr = assign_df['distance'].corr(assign_df['reward'])
                    print(f"   Distance-Reward correlation: {distance_reward_corr:.3f}")
                    
                    # 按距离分组分析
                    try:
                        assign_df['distance_bin'] = pd.qcut(assign_df['distance'], q=5, labels=['Very Close', 'Close', 'Medium', 'Far', 'Very Far'], duplicates='drop')
                        distance_analysis = assign_df.groupby('distance_bin')['reward'].agg(['mean', 'count'])
                        print(f"\n   Reward by Distance:")
                        print(distance_analysis)
                    except ValueError:
                        # 如果qcut失败，使用简单分组
                        max_dist = assign_df['distance'].max()
                        if max_dist > 0:
                            assign_df['distance_bin'] = pd.cut(assign_df['distance'], bins=3, labels=['Close', 'Medium', 'Far'])
                            distance_analysis = assign_df.groupby('distance_bin')['reward'].agg(['mean', 'count'])
                            print(f"\n   Reward by Distance:")
                            print(distance_analysis)
                        else:
                            print(f"   All distances are the same ({max_dist}) - no distance analysis possible")
                
                # 2. 时间分析
                print(f"\n   Reward trends over time:")
                try:
                    time_bins = pd.qcut(df['current_time'], q=5, labels=['Early', 'Mid-Early', 'Middle', 'Mid-Late', 'Late'], duplicates='drop')
                    time_analysis = df.groupby([time_bins, 'action_category'])['reward'].mean().unstack(fill_value=0)
                    print(time_analysis)
                except ValueError:
                    # 如果qcut失败，使用简单的时间范围分析
                    time_range = df['current_time'].max() - df['current_time'].min()
                    if time_range > 0:
                        df_copy = df.copy()
                        df_copy['time_period'] = pd.cut(df_copy['current_time'], bins=3, labels=['Early', 'Middle', 'Late'])
                        time_analysis = df_copy.groupby(['time_period', 'action_category'])['reward'].mean().unstack(fill_value=0)
                        print(time_analysis)
                    else:
                        print("   All data from same time period - no trend analysis possible")
        
        # 保存分析结果
        self._save_analysis_results(df, reward_stats)
        
        # 生成可视化图表
        self._create_visualizations(df)
        
        return df
    
    def _categorize_action(self, action_type):
        """简化动作类型分类"""
        if action_type == 'idle':
            return 'idle'
        elif action_type.startswith('assign'):
            return 'assign'
        elif action_type.startswith('charge'):
            return 'charge'
        else:
            return 'other'
    
    def _calculate_distance(self, loc1, loc2, grid_size=10):
        """计算曼哈顿距离"""
        x1, y1 = loc1 % grid_size, loc1 // grid_size
        x2, y2 = loc2 % grid_size, loc2 // grid_size
        return abs(x1 - x2) + abs(y1 - y2)
    
    def _save_analysis_results(self, df, reward_stats):
        """保存分析结果"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细分析结果
        analysis_file = os.path.join(self.results_dir, f"q_value_analysis_{timestamp}.txt")
        with open(analysis_file, 'w', encoding='utf-8') as f:
            f.write("Q-Value Problem Analysis Report\n")
            f.write("=" * 40 + "\n\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Dataset size: {len(df)} experiences\n\n")
            
            f.write("Reward Statistics by Action Category:\n")
            f.write(reward_stats.to_string())
            f.write("\n\n")
            
            # 详细问题分析
            f.write("Identified Issues:\n")
            assign_mean = df[df['action_category'] == 'assign']['reward'].mean()
            idle_mean = df[df['action_category'] == 'idle']['reward'].mean()
            
            if assign_mean < idle_mean:
                f.write(f"1. Assign actions have lower reward than idle ({assign_mean:.3f} vs {idle_mean:.3f})\n")
                f.write("   This causes Q-network to prefer idle over productive actions\n")
                f.write("   Possible solutions:\n")
                f.write("   - Increase reward for successful assignments\n")
                f.write("   - Reduce penalty for distance in assign actions\n")
                f.write("   - Add exploration bonus for assign actions\n")
                f.write("   - Rebalance reward function\n\n")
        
        print(f"📊 Analysis results saved to: {analysis_file}")
    
    def _create_visualizations(self, df):
        """创建可视化图表"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        plt.style.use('default')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Q-Value Problem Analysis Dashboard', fontsize=16, fontweight='bold')
        
        # 1. 奖励分布按动作类型
        ax1 = axes[0, 0]
        action_categories = df['action_category'].unique()
        for category in action_categories:
            rewards = df[df['action_category'] == category]['reward']
            ax1.hist(rewards, alpha=0.7, label=category, bins=30)
        ax1.set_xlabel('Reward')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Reward Distribution by Action Type')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. 奖励均值比较
        ax2 = axes[0, 1]
        reward_means = df.groupby('action_category')['reward'].mean()
        bars = ax2.bar(reward_means.index, reward_means.values)
        ax2.set_ylabel('Mean Reward')
        ax2.set_title('Mean Reward by Action Type')
        ax2.grid(True, alpha=0.3)
        
        # 标注数值
        for bar, value in zip(bars, reward_means.values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{value:.3f}', ha='center', va='bottom')
        
        # 3. 时间序列奖励趋势
        ax3 = axes[1, 0]
        try:
            time_bins = pd.qcut(df['current_time'], q=20, duplicates='drop')
            time_reward_trend = df.groupby([time_bins, 'action_category'])['reward'].mean().unstack(fill_value=0)
            
            for category in time_reward_trend.columns:
                ax3.plot(range(len(time_reward_trend)), time_reward_trend[category], 
                        marker='o', label=category, linewidth=2)
        except (ValueError, KeyError):
            # 如果时间分箱失败，使用简单的时间窗口
            time_window = 20
            df_sorted = df.sort_values('current_time')
            windows = []
            window_rewards = {cat: [] for cat in df['action_category'].unique()}
            
            for i in range(0, len(df_sorted), time_window):
                window_data = df_sorted.iloc[i:i+time_window]
                for category in window_rewards.keys():
                    cat_data = window_data[window_data['action_category'] == category]['reward']
                    mean_reward = cat_data.mean() if len(cat_data) > 0 else 0
                    window_rewards[category].append(mean_reward)
                windows.append(i // time_window)
            
            for category, rewards in window_rewards.items():
                if len(rewards) > 0:
                    ax3.plot(windows[:len(rewards)], rewards, marker='o', label=category, linewidth=2)
        
        ax3.set_xlabel('Time Progress')
        ax3.set_ylabel('Mean Reward')
        ax3.set_title('Reward Trends Over Time')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. 距离vs奖励散点图(仅assign动作)
        ax4 = axes[1, 1]
        assign_df = df[df['action_category'] == 'assign'].copy()
        if len(assign_df) > 0:
            assign_df['distance'] = assign_df.apply(
                lambda row: self._calculate_distance(row['vehicle_location'], row['target_location']), 
                axis=1
            )
            scatter = ax4.scatter(assign_df['distance'], assign_df['reward'], 
                                alpha=0.6, c=assign_df['reward'], cmap='RdYlBu_r')
            ax4.set_xlabel('Manhattan Distance')
            ax4.set_ylabel('Reward')
            ax4.set_title('Distance vs Reward (Assign Actions)')
            plt.colorbar(scatter, ax=ax4, label='Reward')
            ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # 保存图表
        viz_file = os.path.join(self.results_dir, f"q_value_visualization_{timestamp}.png")
        plt.savefig(viz_file, dpi=300, bbox_inches='tight')
        print(f"📈 Visualizations saved to: {viz_file}")
        
        plt.show()
    
    def run_analysis_from_file(self, file_path):
        """从文件运行完整分析"""
        print(f"🚀 Starting Q-Value Analysis from file: {file_path}")
        
        # 加载数据
        experiences = self.load_experience_from_file(file_path)
        if experiences is None:
            return
        
        # 导出为CSV
        df, csv_path = self.export_experience_to_csv(experiences)
        
        # 分析Q-value问题
        self.analyze_q_value_problem(experiences)
        
        print(f"\n✅ Analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")
        return df
    
    def run_analysis_from_environment(self, value_function):
        """从运行中的环境分析"""
        print(f"🚀 Starting Q-Value Analysis from running environment")
        
        # 加载数据
        experiences = self.load_experience_from_running_env(value_function)
        if experiences is None:
            return
        
        # 导出为CSV
        df, csv_path = self.export_experience_to_csv(experiences)
        
        # 分析Q-value问题
        self.analyze_q_value_problem(experiences, value_function)
        
        print(f"\n✅ Analysis completed!")
        print(f"📁 Results saved in: {self.results_dir}")
        return df

def main():
    """主函数 - 可以单独运行分析"""
    analyzer = QValueAnalyzer()
    
    # 检查是否有保存的dataset文件
    dataset_dir = "results/training_datasets"
    if os.path.exists(dataset_dir):
        dataset_files = [f for f in os.listdir(dataset_dir) if f.endswith(('.pkl', '.json'))]
        if dataset_files:
            latest_file = max(dataset_files, key=lambda x: os.path.getctime(os.path.join(dataset_dir, x)))
            file_path = os.path.join(dataset_dir, latest_file)
            print(f"Found dataset file: {file_path}")
            analyzer.run_analysis_from_file(file_path)
        else:
            print("No dataset files found in results/training_datasets")
    else:
        print("Training datasets directory not found")

if __name__ == "__main__":
    main()