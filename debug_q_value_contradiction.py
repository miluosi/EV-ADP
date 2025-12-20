"""
调试Q值与奖励矛盾问题的专门分析工具
分析为什么assign奖励高但Q值有时低于idle
"""

import pandas as pd
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import seaborn as sns

def analyze_q_value_contradiction(value_function, recent_experiences_count=500):
    """
    专门分析Q值与实际奖励矛盾的问题
    """
    print("🔍 Q值与奖励矛盾分析")
    print("=" * 60)
    
    if len(value_function.experience_buffer) < 100:
        print("❌ 经验数据不足，无法分析")
        return
    
    # 1. 获取最近的经验数据
    experiences = list(value_function.experience_buffer)[-recent_experiences_count:]
    
    # 2. 按动作类型分组分析
    assign_exps = [exp for exp in experiences if exp['action_type'].startswith('assign')]
    idle_exps = [exp for exp in experiences if exp['action_type'] == 'idle']
    charge_exps = [exp for exp in experiences if exp['action_type'].startswith('charge')]
    
    print(f"📊 最近 {len(experiences)} 条经验分析:")
    print(f"   Assign经验: {len(assign_exps)}")
    print(f"   Idle经验:   {len(idle_exps)}")
    print(f"   Charge经验: {len(charge_exps)}")
    print()
    
    # 3. 奖励统计分析
    if assign_exps:
        assign_rewards = [exp['reward'] for exp in assign_exps]
        print(f"🎯 Assign动作奖励分析:")
        print(f"   平均奖励: {np.mean(assign_rewards):.3f}")
        print(f"   中位奖励: {np.median(assign_rewards):.3f}")
        print(f"   奖励范围: [{np.min(assign_rewards):.3f}, {np.max(assign_rewards):.3f}]")
        print(f"   正奖励比例: {len([r for r in assign_rewards if r > 0]) / len(assign_rewards):.1%}")
        print()
    
    if idle_exps:
        idle_rewards = [exp['reward'] for exp in idle_exps]
        print(f"💤 Idle动作奖励分析:")
        print(f"   平均奖励: {np.mean(idle_rewards):.3f}")
        print(f"   中位奖励: {np.median(idle_rewards):.3f}")
        print(f"   奖励范围: [{np.min(idle_rewards):.3f}, {np.max(idle_rewards):.3f}]")
        print(f"   正奖励比例: {len([r for r in idle_rewards if r > 0]) / len(idle_rewards):.1%}")
        print()
    
    # 4. 网络预测Q值分析
    print("🧠 当前网络Q值预测分析:")
    try:
        # 随机抽取一些经验进行Q值预测
        sample_size = min(50, len(experiences))
        sample_exps = np.random.choice(experiences, sample_size, replace=False)
        
        assign_q_values = []
        idle_q_values = []
        actual_assign_rewards = []
        actual_idle_rewards = []
        
        for exp in sample_exps:
            # 获取Q值预测
            q_value = value_function.get_q_value(
                vehicle_id=exp['vehicle_id'],
                action_type=exp['action_type'],
                vehicle_location=exp['vehicle_location'],
                target_location=exp['target_location'],
                current_time=exp['current_time'],
                other_vehicles=exp['other_vehicles'],
                num_requests=exp['num_requests'],
                battery_level=exp.get('battery_level', 1.0),
                request_value=exp.get('request_value', 0.0)
            )
            
            if exp['action_type'].startswith('assign'):
                assign_q_values.append(q_value)
                actual_assign_rewards.append(exp['reward'])
            elif exp['action_type'] == 'idle':
                idle_q_values.append(q_value)
                actual_idle_rewards.append(exp['reward'])
        
        if assign_q_values:
            print(f"   Assign Q值: 平均={np.mean(assign_q_values):.3f}, 范围=[{np.min(assign_q_values):.3f}, {np.max(assign_q_values):.3f}]")
            print(f"   对应实际奖励: 平均={np.mean(actual_assign_rewards):.3f}")
        
        if idle_q_values:
            print(f"   Idle Q值:   平均={np.mean(idle_q_values):.3f}, 范围=[{np.min(idle_q_values):.3f}, {np.max(idle_q_values):.3f}]")
            print(f"   对应实际奖励: 平均={np.mean(actual_idle_rewards):.3f}")
        
        # Q值与奖励的相关性分析
        if assign_q_values and len(assign_q_values) > 3:
            assign_correlation = np.corrcoef(assign_q_values, actual_assign_rewards)[0, 1]
            print(f"   Assign Q值与奖励相关性: {assign_correlation:.3f}")
        
        if idle_q_values and len(idle_q_values) > 3:
            idle_correlation = np.corrcoef(idle_q_values, actual_idle_rewards)[0, 1]
            print(f"   Idle Q值与奖励相关性: {idle_correlation:.3f}")
        
        print()
        
    except Exception as e:
        print(f"   ❌ Q值分析失败: {e}")
    
    # 5. 训练采样分析
    print("🎲 训练采样分析:")
    try:
        # 模拟一次训练采样
        sample_batch = value_function._action_balanced_sample(64)
        sample_assign_count = len([exp for exp in sample_batch if exp['action_type'].startswith('assign')])
        sample_idle_count = len([exp for exp in sample_batch if exp['action_type'] == 'idle'])
        sample_charge_count = len([exp for exp in sample_batch if exp['action_type'].startswith('charge')])
        
        print(f"   最新训练批次构成: Assign={sample_assign_count}, Idle={sample_idle_count}, Charge={sample_charge_count}")
        
        # 分析训练批次中的奖励分布
        sample_assign_rewards = [exp['reward'] for exp in sample_batch if exp['action_type'].startswith('assign')]
        sample_idle_rewards = [exp['reward'] for exp in sample_batch if exp['action_type'] == 'idle']
        
        if sample_assign_rewards:
            print(f"   训练批次Assign平均奖励: {np.mean(sample_assign_rewards):.3f}")
        if sample_idle_rewards:
            print(f"   训练批次Idle平均奖励: {np.mean(sample_idle_rewards):.3f}")
        
    except Exception as e:
        print(f"   ❌ 采样分析失败: {e}")
    
    print()
    
    # 6. 问题诊断建议
    print("🔧 问题诊断与建议:")
    
    # 检查奖励差异
    if assign_exps and idle_exps:
        avg_assign_reward = np.mean([exp['reward'] for exp in assign_exps])
        avg_idle_reward = np.mean([exp['reward'] for exp in idle_exps])
        
        if avg_assign_reward > avg_idle_reward:
            print(f"   ✅ 奖励逻辑正确: Assign({avg_assign_reward:.2f}) > Idle({avg_idle_reward:.2f})")
            
            # 如果奖励逻辑正确但Q值不对，可能的原因：
            print("   可能的Q值矛盾原因:")
            print("   1. 🔄 训练尚未收敛，需要更多训练步骤")
            print("   2. 📊 样本不均衡，idle样本过多影响网络学习")
            print("   3. 🎯 目标Q值计算有误，检查TD target计算")
            print("   4. 🏗️ 网络容量不足，无法学习复杂的状态-动作映射")
            print("   5. 📈 学习率过高或过低，影响收敛")
            
            # 具体建议
            print("\n   🚀 改进建议:")
            print("   1. 增强action-balanced采样权重")
            print("   2. 调整学习率和训练频率")
            print("   3. 添加assign动作的额外奖励bonus")
            print("   4. 增加网络深度或宽度")
            print("   5. 使用prioritized experience replay")
        
        else:
            print(f"   ❌ 奖励逻辑异常: Assign({avg_assign_reward:.2f}) <= Idle({avg_idle_reward:.2f})")
            print("   建议检查奖励函数设计和环境状态转换逻辑")

if __name__ == "__main__":
    print("Q值矛盾分析工具")
    print("使用方法: analyze_q_value_contradiction(value_function)")