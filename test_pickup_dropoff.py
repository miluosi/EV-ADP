"""
测试车辆完整的pickup-dropoff流程
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Environment import ChargingIntegratedEnvironment
from src.Action import ServiceAction
import random

def test_pickup_dropoff_flow():
    """测试完整的pickup-dropoff流程"""
    
    # 设置随机种子
    random.seed(42)
    
    # 创建简单环境
    env = ChargingIntegratedEnvironment(
        grid_size=5,
        num_vehicles=1,
        num_stations=1,
        use_intense_requests=False
    )
    
    # 手动创建一个请求在固定位置
    from src.Request import Request
    
    # 车辆在(0,0)，创建请求从(1,1)到(3,3)
    vehicle_id = 0
    vehicle = env.vehicles[vehicle_id]
    vehicle['coordinates'] = (0, 0)
    vehicle['location'] = 0
    
    # 创建测试请求
    test_request = Request(
        request_id=999,
        source=1*5 + 1,  # (1,1) 
        destination=3*5 + 3,  # (3,3)
        current_time=0,
        travel_time=4,
        value=10.0
    )
    
    env.active_requests[999] = test_request
    print(f"创建测试请求: 从({1},{1})到({3},{3}), 价值:{test_request.value}")
    print(f"车辆初始位置: {vehicle['coordinates']}")
    
    step_count = 0
    total_reward = 0
    
    # 1. 尝试分配请求
    print(f"\n步骤 {step_count}: 分配请求")
    action = ServiceAction([], 999)
    actions = {vehicle_id: action}
    
    next_states, rewards, done, info = env.step(actions)
    step_reward = rewards[vehicle_id]
    total_reward += step_reward
    
    print(f"分配奖励: {step_reward}")
    print(f"车辆assigned_request: {vehicle['assigned_request']}")
    print(f"车辆passenger_onboard: {vehicle['passenger_onboard']}")
    print(f"车辆位置: {vehicle['coordinates']}")
    
    # 2. 移动到pickup位置
    max_steps = 20
    while vehicle['assigned_request'] is not None and step_count < max_steps:
        step_count += 1
        print(f"\n步骤 {step_count}: 移动到pickup")
        
        # 继续执行相同的服务动作（应该触发移动）
        actions = {vehicle_id: ServiceAction([], 999)}
        next_states, rewards, done, info = env.step(actions)
        step_reward = rewards[vehicle_id]
        total_reward += step_reward
        
        print(f"移动奖励: {step_reward}")
        print(f"车辆位置: {vehicle['coordinates']}")
        print(f"车辆assigned_request: {vehicle['assigned_request']}")
        print(f"车辆passenger_onboard: {vehicle['passenger_onboard']}")
        
        # 检查是否已经pickup
        if vehicle['passenger_onboard'] is not None:
            print("✓ 成功pickup乘客!")
            break
    
    # 3. 移动到dropoff位置
    while vehicle['passenger_onboard'] is not None and step_count < max_steps:
        step_count += 1
        print(f"\n步骤 {step_count}: 移动到dropoff")
        
        # 继续执行服务动作（应该移动到dropoff）
        actions = {vehicle_id: ServiceAction([], 999)}
        next_states, rewards, done, info = env.step(actions)
        step_reward = rewards[vehicle_id]
        total_reward += step_reward
        
        print(f"移动奖励: {step_reward}")
        print(f"车辆位置: {vehicle['coordinates']}")
        print(f"车辆passenger_onboard: {vehicle['passenger_onboard']}")
        
        # 检查是否完成
        if vehicle['passenger_onboard'] is None:
            print("✓ 成功完成订单!")
            break
    
    # 检查完成状态
    print(f"\n=== 最终状态 ===")
    print(f"总步数: {step_count}")
    print(f"总奖励: {total_reward}")
    print(f"活跃请求数: {len(env.active_requests)}")
    print(f"完成请求数: {len(env.completed_requests)}")
    print(f"车辆位置: {vehicle['coordinates']}")
    print(f"车辆assigned_request: {vehicle['assigned_request']}")
    print(f"车辆passenger_onboard: {vehicle['passenger_onboard']}")
    
    if len(env.completed_requests) > 0:
        print("✓ 测试成功：订单已完成!")
        return True
    else:
        print("❌ 测试失败：订单未完成")
        return False

if __name__ == "__main__":
    test_pickup_dropoff_flow()
