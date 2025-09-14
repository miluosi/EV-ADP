#!/usr/bin/env python3
"""
分析assign、charge、idle三种action类型在神经网络输入层面的区分度
检查神经网络是否有足够的信息来学习不同action的模式
"""

import sys
import torch
import numpy as np
sys.path.append('src')

from ValueFunction_pytorch import PyTorchChargingValueFunction
from Environment import ChargingIntegratedEnvironment

def analyze_action_input_patterns():
    """分析不同action类型的输入模式"""
    print("=" * 80)
    print("分析Action类型在神经网络输入层面的区分度")
    print("=" * 80)
    
    # 创建值函数
    vf = PyTorchChargingValueFunction(grid_size=10, num_vehicles=5)
    
    # 测试参数
    vehicle_location = 45  # (4, 5)
    target_location_assign = 67  # (6, 7) - 假设请求位置
    target_location_charge = 23  # (2, 3) - 充电站位置
    current_time = 150.0
    other_vehicles = 2
    num_requests = 8
    battery_level = 0.6
    request_value = 25.0
    
    print(f"测试参数:")
    print(f"  车辆位置: {vehicle_location} (坐标: {(vehicle_location//10, vehicle_location%10)})")
    print(f"  请求位置: {target_location_assign} (坐标: {(target_location_assign//10, target_location_assign%10)})")
    print(f"  充电站位置: {target_location_charge} (坐标: {(target_location_charge//10, target_location_charge%10)})")
    print(f"  当前时间: {current_time}")
    print(f"  其他车辆: {other_vehicles}")
    print(f"  请求数量: {num_requests}")
    print(f"  电池电量: {battery_level}")
    print(f"  请求价值: {request_value}")
    print()
    
    # 1. 分析ASSIGN动作的输入
    print("1. ASSIGN动作输入分析:")
    print("-" * 40)
    assign_inputs = vf._prepare_network_input_with_battery(
        vehicle_location, target_location_assign, current_time, 
        other_vehicles, num_requests, "assign_1", battery_level, request_value
    )
    
    if len(assign_inputs) == 7:
        path_locs, path_delays, time_t, others_t, requests_t, battery_t, value_t = assign_inputs
        
        print(f"路径位置序列: {path_locs.squeeze().cpu().numpy()}")
        print(f"路径延迟序列: {path_delays.squeeze().cpu().numpy().flatten()}")
        print(f"时间特征: {time_t.item():.4f}")
        print(f"其他车辆特征: {others_t.item():.4f}")
        print(f"请求数量特征: {requests_t.item():.4f}")
        print(f"电池电量特征: {battery_t.item():.4f}")
        print(f"请求价值特征: {value_t.item():.4f}")
        print()
    
    # 2. 分析CHARGE动作的输入
    print("2. CHARGE动作输入分析:")
    print("-" * 40)
    charge_inputs = vf._prepare_network_input_with_battery(
        vehicle_location, target_location_charge, current_time, 
        other_vehicles, num_requests, "charge_0", battery_level, 0.0
    )
    
    if len(charge_inputs) == 7:
        path_locs, path_delays, time_t, others_t, requests_t, battery_t, value_t = charge_inputs
        
        print(f"路径位置序列: {path_locs.squeeze().cpu().numpy()}")
        print(f"路径延迟序列: {path_delays.squeeze().cpu().numpy().flatten()}")
        print(f"时间特征: {time_t.item():.4f}")
        print(f"其他车辆特征: {others_t.item():.4f}")
        print(f"请求数量特征: {requests_t.item():.4f}")
        print(f"电池电量特征: {battery_t.item():.4f}")
        print(f"请求价值特征: {value_t.item():.4f}")
        print()
    
    # 3. 分析IDLE动作的输入
    print("3. IDLE动作输入分析:")
    print("-" * 40)
    idle_inputs = vf._prepare_network_input_with_battery(
        vehicle_location, vehicle_location, current_time, 
        other_vehicles, num_requests, "idle", battery_level, 0.0
    )
    
    if len(idle_inputs) == 7:
        path_locs, path_delays, time_t, others_t, requests_t, battery_t, value_t = idle_inputs
        
        print(f"路径位置序列: {path_locs.squeeze().cpu().numpy()}")
        print(f"路径延迟序列: {path_delays.squeeze().cpu().numpy().flatten()}")
        print(f"时间特征: {time_t.item():.4f}")
        print(f"其他车辆特征: {others_t.item():.4f}")
        print(f"请求数量特征: {requests_t.item():.4f}")
        print(f"电池电量特征: {battery_t.item():.4f}")
        print(f"请求价值特征: {value_t.item():.4f}")
        print()

def analyze_action_type_identification():
    """分析action类型识别的关键特征"""
    print("=" * 80)
    print("Action类型识别的关键特征分析")
    print("=" * 80)
    
    print("当前区分三种action类型的关键特征:")
    print()
    
    print("1. 路径模式区分:")
    print("   - ASSIGN: 车辆位置 -> 请求位置 -> 结束")
    print("   - CHARGE: 车辆位置 -> 充电站位置 -> 结束") 
    print("   - IDLE:   车辆位置 -> 车辆位置 -> 结束 (原地停留)")
    print()
    
    print("2. 延迟模式区分:")
    print("   - ASSIGN: [0.0, 服务延迟(0.2), 0.0]")
    print("   - CHARGE: [0.0, 充电延迟(0.5), 0.0]")
    print("   - IDLE:   [0.0, 等待延迟(0.05), 0.0]")
    print()
    
    print("3. 请求价值区分:")
    print("   - ASSIGN: 实际请求价值 (归一化到0-1)")
    print("   - CHARGE: 0.0 (无请求价值)")
    print("   - IDLE:   0.0 (无请求价值)")
    print()
    
    print("4. 目标位置区分:")
    print("   - ASSIGN: 目标位置 != 当前位置")
    print("   - CHARGE: 目标位置 != 当前位置 (充电站)")
    print("   - IDLE:   目标位置 == 当前位置")
    print()

def analyze_potential_issues():
    """分析潜在的区分度问题"""
    print("=" * 80)
    print("潜在的Action区分度问题分析")
    print("=" * 80)
    
    print("可能的问题:")
    print()
    
    print("1. ASSIGN vs CHARGE区分问题:")
    print("   - 两者都是移动到不同位置")
    print("   - 主要区别在延迟模式和请求价值")
    print("   - 如果神经网络没有充分学习延迟模式，可能混淆")
    print()
    
    print("2. 特征表示不够丰富:")
    print("   - 没有显式的action_type embedding")
    print("   - 完全依赖路径和延迟模式来区分")
    print("   - 可能需要更明确的action类型标识")
    print()
    
    print("3. 训练数据不平衡:")
    print("   - 如果某种action类型的样本过少")
    print("   - 神经网络可能无法学习到该类型的模式")
    print()
    
    print("4. 奖励信号不够清晰:")
    print("   - 不同action类型的奖励分布可能重叠")
    print("   - 神经网络难以区分哪种action更优")
    print()

def suggest_improvements():
    """建议改进方案"""
    print("=" * 80)
    print("改进Action区分度的建议")
    print("=" * 80)
    
    print("建议的改进方案:")
    print()
    
    print("1. 添加显式的Action Type Embedding:")
    print("   - 为每种action类型创建独立的embedding")
    print("   - assign_embedding, charge_embedding, idle_embedding")
    print("   - 与现有特征concatenate")
    print()
    
    print("2. 增强延迟模式区分:")
    print("   - 使用更显著的延迟差异")
    print("   - assign: 0.1-0.3, charge: 0.4-0.7, idle: 0.01-0.05")
    print()
    
    print("3. 添加更多上下文特征:")
    print("   - 车辆到充电站的距离")
    print("   - 请求紧急程度")
    print("   - 车辆当前任务状态")
    print()
    
    print("4. 使用Multi-Head Attention:")
    print("   - 让神经网络关注不同特征组合")
    print("   - 更好地学习action-specific patterns")
    print()
    
    print("5. 改进奖励设计:")
    print("   - 确保不同action类型有清晰的奖励区分")
    print("   - 避免奖励信号冲突")
    print()

def test_current_q_values():
    """测试当前的Q值计算"""
    print("=" * 80)
    print("当前Q值计算测试")
    print("=" * 80)
    
    # 创建环境和值函数
    env = ChargingIntegratedEnvironment(num_vehicles=3, num_stations=2)
    vf = PyTorchChargingValueFunction()
    env.set_value_function(vf)
    
    # 测试参数
    vehicle_id = 0
    vehicle_location = 45
    target_location_assign = 67
    target_location_charge = 23
    current_time = 150.0
    other_vehicles = 2
    num_requests = 8
    battery_level = 0.6
    request_value = 25.0
    
    print("计算不同action的Q值:")
    print()
    
    # 计算Q值
    q_assign = vf.get_assignment_q_value(
        vehicle_id, 1, vehicle_location, target_location_assign, 
        current_time, other_vehicles, num_requests, battery_level, request_value
    )
    
    q_charge = vf.get_charging_q_value(
        vehicle_id, 0, vehicle_location, target_location_charge,
        current_time, other_vehicles, num_requests, battery_level
    )
    
    q_idle = vf.get_idle_q_value(
        vehicle_id, vehicle_location, battery_level, 
        current_time, other_vehicles, num_requests
    )
    
    print(f"Q-values:")
    print(f"  Assign: {q_assign:.6f}")
    print(f"  Charge: {q_charge:.6f}")
    print(f"  Idle:   {q_idle:.6f}")
    print()
    
    print(f"Q-value ranking:")
    q_values = [("Assign", q_assign), ("Charge", q_charge), ("Idle", q_idle)]
    q_values.sort(key=lambda x: x[1], reverse=True)
    for i, (action, q_val) in enumerate(q_values):
        print(f"  {i+1}. {action}: {q_val:.6f}")
    print()

if __name__ == "__main__":
    try:
        # 分析输入模式
        analyze_action_input_patterns()
        
        # 分析类型识别特征
        analyze_action_type_identification()
        
        # 分析潜在问题
        analyze_potential_issues()
        
        # 建议改进方案
        suggest_improvements()
        
        # 测试当前Q值
        test_current_q_values()
        
    except Exception as e:
        print(f"分析过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
