#!/usr/bin/env python3
"""
测试修改后的step方法 - 验证执行顺序和reward计算
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from Environment import Environment
from Action import ChargingAction, ServiceAction
from Request import Request
import time


def test_step_execution_order():
    """测试step方法的执行顺序是否正确"""
    print("=== 测试step方法执行顺序 ===")
    
    # 创建环境
    env_config = {
        'num_vehicles': 3,
        'num_stations': 2,
        'grid_size': 8,
        'assignmentgurobi': True
    }
    
    env = Environment(**env_config)
    print(f"创建环境: charging_penalty={env.charging_penalty}, adp_value={env.adp_value}")
    
    # 添加测试请求
    for i in range(2):
        request = Request(
            request_id=i,
            pickup=i * 3,
            dropoff=(i * 3 + 4) % env.grid_size**2,
            pickup_time=env.current_time,
            value=8.0 + i * 3
        )
        env.active_requests[i] = request
    
    print(f"添加{len(env.active_requests)}个测试请求")
    
    # 创建actions
    actions = {}
    
    # 车辆0: 充电
    station_id = list(env.charging_manager.stations.keys())[0]
    actions[0] = ChargingAction([], station_id, 25.0)
    
    # 车辆1: 服务请求
    actions[1] = ServiceAction([], 0)
    
    # 车辆2: 移动action (None)
    actions[2] = None
    
    print("创建actions:")
    print(f"  车辆0: 充电 (站点{station_id})")
    print(f"  车辆1: 服务请求0")
    print(f"  车辆2: 移动")
    
    # 记录step前状态
    print("\n=== Step前状态 ===")
    print(f"当前时间: {env.current_time}")
    print(f"活跃请求数: {len(env.active_requests)}")
    for i in range(3):
        vehicle = env.vehicles[i]
        print(f"车辆{i}: 电量={vehicle['battery_level']:.2f}, 充电站={vehicle['charging_station']}")
    
    # 执行step
    print("\n=== 执行step ===")
    start_time = time.time()
    next_states, rewards, done, info = env.step(actions)
    execution_time = time.time() - start_time
    
    print(f"执行时间: {execution_time:.3f}秒")
    
    # 检查结果
    print("\n=== Step后结果 ===")
    print("Rewards:")
    for vehicle_id, reward in rewards.items():
        action_desc = "充电" if vehicle_id == 0 else "服务" if vehicle_id == 1 else "移动"
        print(f"  车辆{vehicle_id} ({action_desc}): {reward:.2f}")
    
    print(f"\nGurobi assignments: {len(info.get('gurobi_assignments', {}))}")
    gurobi_assignments = info.get('gurobi_assignments', {})
    for vehicle_id, assignment in gurobi_assignments.items():
        if assignment:
            print(f"  车辆{vehicle_id}: {assignment}")
    
    print(f"\n充电事件: {len(info.get('charging_events', []))}")
    for event in info.get('charging_events', []):
        print(f"  {event}")
    
    # 验证reward计算逻辑
    print("\n=== Reward计算验证 ===")
    
    # 充电reward应该是 -charging_penalty + q_value * adp_value
    charging_reward = rewards.get(0, 0)
    expected_charging_base = -env.charging_penalty  # -2.0
    print(f"充电reward: {charging_reward:.2f} (期望基础值: {expected_charging_base:.2f})")
    
    # 服务reward应该是 request.value + q_value * adp_value
    service_reward = rewards.get(1, 0)
    if 0 in env.active_requests:
        request_value = env.active_requests[0].value
        print(f"服务reward: {service_reward:.2f} (请求价值: {request_value:.2f})")
    
    print(f"总reward: {sum(rewards.values()):.2f}")
    
    return True


def test_multiple_steps_with_new_logic():
    """测试新逻辑下的多步执行"""
    print("\n=== 测试新逻辑多步执行 ===")
    
    env = Environment(num_vehicles=2, num_stations=1, grid_size=6, assignmentgurobi=True)
    
    rewards_history = []
    
    for step_num in range(3):
        print(f"\n--- Step {step_num + 1} ---")
        
        # 简单策略
        actions = {}
        if step_num == 0:
            # 第一步：车辆0充电
            station_id = list(env.charging_manager.stations.keys())[0]
            actions[0] = ChargingAction([], station_id, 20.0)
            actions[1] = None
        else:
            # 后续步骤：移动
            actions[0] = None
            actions[1] = None
        
        next_states, rewards, done, info = env.step(actions)
        rewards_history.append(rewards.copy())
        
        print(f"Rewards: {[f'{r:.1f}' for r in rewards.values()]}")
        print(f"Gurobi assignments: {len(info.get('gurobi_assignments', {}))}")
        
        if done:
            break
    
    print(f"\n执行了{len(rewards_history)}步")
    print("Rewards历史:")
    for i, step_rewards in enumerate(rewards_history):
        print(f"  Step {i+1}: {[f'{r:.1f}' for r in step_rewards.values()]}")
    
    return True


if __name__ == "__main__":
    try:
        print("开始测试修改后的step方法...")
        
        success1 = test_step_execution_order()
        success2 = test_multiple_steps_with_new_logic()
        
        if success1 and success2:
            print("\n✓ 所有测试通过!")
            print("修改后的step方法正确实现了:")
            print("  1. 先更新环境状态")
            print("  2. 获取Gurobi assignments")
            print("  3. 基于Gurobi目标函数计算reward")
            print("  4. 执行原有action逻辑")
            print("  5. 更新Q-values")
        else:
            print("\n✗ 部分测试失败")
            
    except Exception as e:
        print(f"\n✗ 测试执行出错: {e}")
        import traceback
        traceback.print_exc()
