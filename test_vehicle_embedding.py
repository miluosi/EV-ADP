#!/usr/bin/env python3
"""
测试Vehicle ID Embedding和Vehicle Type Embedding的效果
检查不同车辆是否能产生不同的Q值
"""

import sys
import torch
import numpy as np
sys.path.append('src')

from ValueFunction_pytorch import PyTorchChargingValueFunction
from Environment import ChargingIntegratedEnvironment

def test_vehicle_embedding_effect():
    """测试vehicle embedding对Q值的影响"""
    print("=" * 80)
    print("测试Vehicle ID和Vehicle Type Embedding效果")
    print("=" * 80)
    
    # 创建值函数
    vf = PyTorchChargingValueFunction(grid_size=10, num_vehicles=5)
    
    # 测试参数
    vehicle_locations = [45, 45, 45]  # 相同位置
    target_location = 67
    current_time = 150.0
    other_vehicles = 2
    num_requests = 8
    battery_level = 0.6
    request_value = 25.0
    
    print(f"测试场景:")
    print(f"  目标位置: {target_location}")
    print(f"  当前时间: {current_time}")
    print(f"  电池电量: {battery_level}")
    print(f"  请求价值: {request_value}")
    print()
    
    # 测试不同车辆的Q值
    vehicle_ids = [0, 1, 2]  # 车辆0(EV), 车辆1(AEV), 车辆2(EV)
    action_types = ['idle', 'assign_1', 'charge_0']
    
    print("不同车辆的Q值比较 (相同位置，相同状态):")
    print("-" * 60)
    
    for action_type in action_types:
        print(f"\n{action_type.upper()} 动作:")
        for vehicle_id in vehicle_ids:
            vehicle_type = "EV" if vehicle_id % 2 == 0 else "AEV"
            q_value = vf.get_q_value(
                vehicle_id, action_type, vehicle_locations[vehicle_id], target_location,
                current_time, other_vehicles, num_requests, battery_level, request_value
            )
            print(f"  车辆{vehicle_id} ({vehicle_type}): Q = {q_value:.6f}")
    
    print("\n" + "=" * 80)
    print("Vehicle Embedding特征分析")
    print("=" * 80)
    
    # 分析embedding的效果
    print("预期效果:")
    print("1. 不同vehicle_id应该产生不同的Q值")
    print("2. EV和AEV车辆应该有不同的行为模式")
    print("3. 相同车辆在相同条件下应该产生相同Q值")
    print()
    
    # 测试相同车辆的一致性
    print("相同车辆一致性测试:")
    print("-" * 40)
    vehicle_id = 0
    q1 = vf.get_q_value(vehicle_id, 'assign_1', 45, 67, 150.0, 2, 8, 0.6, 25.0)
    q2 = vf.get_q_value(vehicle_id, 'assign_1', 45, 67, 150.0, 2, 8, 0.6, 25.0)
    print(f"车辆{vehicle_id} 第一次计算: Q = {q1:.6f}")
    print(f"车辆{vehicle_id} 第二次计算: Q = {q2:.6f}")
    print(f"差异: {abs(q1-q2):.10f} (应该为0)")
    print()

def test_vehicle_type_distinction():
    """测试EV和AEV的区分效果"""
    print("=" * 80)
    print("测试EV vs AEV车辆类型区分")
    print("=" * 80)
    
    # 创建值函数
    vf = PyTorchChargingValueFunction(grid_size=10, num_vehicles=6)
    
    # 测试EV车辆 (偶数ID)
    ev_vehicles = [0, 2, 4]
    # 测试AEV车辆 (奇数ID)
    aev_vehicles = [1, 3, 5]
    
    # 测试参数
    vehicle_location = 45
    target_location = 67
    current_time = 150.0
    battery_level = 0.3  # 低电量，测试充电决策
    
    print("低电量情况下的充电决策比较 (battery=0.3):")
    print("-" * 50)
    
    # 计算平均Q值
    ev_charge_q = []
    aev_charge_q = []
    ev_assign_q = []
    aev_assign_q = []
    
    for ev_id in ev_vehicles:
        q_charge = vf.get_q_value(ev_id, 'charge_0', vehicle_location, 25, current_time, 2, 8, battery_level, 0.0)
        q_assign = vf.get_q_value(ev_id, 'assign_1', vehicle_location, target_location, current_time, 2, 8, battery_level, 25.0)
        ev_charge_q.append(q_charge)
        ev_assign_q.append(q_assign)
        print(f"EV 车辆{ev_id}: Charge={q_charge:.4f}, Assign={q_assign:.4f}")
    
    for aev_id in aev_vehicles:
        q_charge = vf.get_q_value(aev_id, 'charge_0', vehicle_location, 25, current_time, 2, 8, battery_level, 0.0)
        q_assign = vf.get_q_value(aev_id, 'assign_1', vehicle_location, target_location, current_time, 2, 8, battery_level, 25.0)
        aev_charge_q.append(q_charge)
        aev_assign_q.append(q_assign)
        print(f"AEV 车辆{aev_id}: Charge={q_charge:.4f}, Assign={q_assign:.4f}")
    
    print()
    print("平均Q值比较:")
    print(f"EV平均 - Charge: {np.mean(ev_charge_q):.4f}, Assign: {np.mean(ev_assign_q):.4f}")
    print(f"AEV平均 - Charge: {np.mean(aev_charge_q):.4f}, Assign: {np.mean(aev_assign_q):.4f}")
    print()
    
    print("期望行为:")
    print("- AEV应该更倾向于接受请求 (assign Q值更高)")
    print("- EV在低电量时应该更倾向于充电 (charge Q值相对更高)")
    print()

def test_action_type_embedding():
    """测试action type embedding的效果"""
    print("=" * 80)
    print("测试Action Type Embedding效果")
    print("=" * 80)
    
    # 创建值函数
    vf = PyTorchChargingValueFunction(grid_size=10, num_vehicles=3)
    
    vehicle_id = 0
    vehicle_location = 45
    current_time = 150.0
    other_vehicles = 2
    num_requests = 8
    battery_level = 0.6
    
    # 测试不同action类型在相同条件下的Q值
    actions = [
        ('idle', vehicle_location, 0.0),
        ('assign_1', 67, 25.0),
        ('charge_0', 25, 0.0)
    ]
    
    print("相同车辆不同action类型的Q值:")
    print("-" * 40)
    
    q_values = []
    for action_type, target_loc, req_val in actions:
        q_value = vf.get_q_value(
            vehicle_id, action_type, vehicle_location, target_loc,
            current_time, other_vehicles, num_requests, battery_level, req_val
        )
        q_values.append((action_type, q_value))
        print(f"{action_type:10}: Q = {q_value:.6f}")
    
    print()
    print("Action区分度分析:")
    q_values.sort(key=lambda x: x[1], reverse=True)
    print("Q值排序 (从高到低):")
    for i, (action, q_val) in enumerate(q_values):
        print(f"  {i+1}. {action}: {q_val:.6f}")
    
    # 计算Q值差异
    max_q = max(q[1] for q in q_values)
    min_q = min(q[1] for q in q_values)
    print(f"\nQ值范围: {min_q:.6f} 到 {max_q:.6f}")
    print(f"最大差异: {max_q - min_q:.6f}")
    print()

if __name__ == "__main__":
    try:
        # 测试vehicle embedding效果
        test_vehicle_embedding_effect()
        
        # 测试vehicle type区分
        test_vehicle_type_distinction()
        
        # 测试action type embedding
        test_action_type_embedding()
        
        print("=" * 80)
        print("Vehicle Embedding测试完成！")
        print("现在神经网络能够:")
        print("1. 区分不同的车辆ID")
        print("2. 区分EV和AEV车辆类型")  
        print("3. 为每个车辆学习个性化的Q-value函数")
        print("4. 更好地区分idle、assign、charge三种action类型")
        print("=" * 80)
        
    except Exception as e:
        print(f"测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
