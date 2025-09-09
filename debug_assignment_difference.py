"""
调试启发式和Gurobi分配方法的差异
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Environment import ChargingIntegratedEnvironment
from src.GurobiOptimizer import GurobiOptimizer
import random

def test_assignment_difference():
    """测试两种分配方法的差异"""
    
    # 设置随机种子以便复现
    random.seed(42)
    
    # 创建环境
    env = ChargingIntegratedEnvironment(
        grid_size=10,
        num_vehicles=5,
        num_stations=3,
        use_intense_requests=True
    )
    
    # 设置adp_value
    env.adp_value = 1.0  # 确保Q值有影响
    
    # 打印车辆和充电站的详细信息
    print("=== 环境详细信息 ===")
    for vehicle_id, vehicle in env.vehicles.items():
        print(f"车辆 {vehicle_id}: 位置={vehicle['coordinates']}, 电池={vehicle['battery']:.2f}, 类型={vehicle['type']}")
    
    for station_id, station in env.charging_manager.stations.items():
        print(f"充电站 {station_id}: 位置=({station.location // env.grid_size}, {station.location % env.grid_size}), 容量={station.max_capacity}, 可用={station.available_slots}")
    
    print(f"最小电池要求: {env.min_battery_level}")
    print(f"充电惩罚: {env.charging_penalty}")
    print(f"未服务惩罚: {env.unserved_penalty}")
    
    # 创建优化器
    optimizer = GurobiOptimizer(env)
    
    # 手动生成一些测试请求来模拟批量需求
    env.current_time = 0
    generated_requests = env._generate_intense_requests()
    print(f"生成了 {len(generated_requests)} 个请求")
    
    # 获取可用车辆
    available_vehicles = []
    for vehicle_id, vehicle in env.vehicles.items():
        if (vehicle['assigned_request'] is None and 
            vehicle['passenger_onboard'] is None and 
            vehicle['charging_station'] is None):
            available_vehicles.append(vehicle_id)
    
    print(f"可用车辆: {len(available_vehicles)} 个")
    
    # 获取充电站
    available_charging_stations = [
        station for station in env.charging_manager.stations.values() 
        if station.available_slots > 0
    ]
    
    available_requests = list(env.active_requests.values())
    print(f"可用请求: {len(available_requests)} 个")
    
    # 打印前几个请求的详细信息
    print("\n前5个请求详情:")
    for i, req in enumerate(available_requests[:5]):
        pickup_coords = (req.pickup // env.grid_size, req.pickup % env.grid_size)
        dropoff_coords = (req.dropoff // env.grid_size, req.dropoff % env.grid_size)
        print(f"  请求 {req.request_id}: {pickup_coords} -> {dropoff_coords}, 价值: {req.value:.2f}, 距离: {abs(pickup_coords[0]-dropoff_coords[0]) + abs(pickup_coords[1]-dropoff_coords[1])}")
    
    # 测试启发式方法
    print("\n=== 启发式方法分配 ===")
    heuristic_assignments = optimizer._heuristic_assignment_with_reject(
        available_vehicles, available_requests, available_charging_stations
    )
    
    print("启发式分配结果:")
    heuristic_total_value = 0
    heuristic_request_count = 0
    for vehicle_id, assignment in heuristic_assignments.items():
        if isinstance(assignment, env.active_requests[list(env.active_requests.keys())[0]].__class__):
            print(f"  车辆 {vehicle_id} -> 请求 {assignment.request_id} (价值: {assignment.value:.2f})")
            heuristic_total_value += assignment.value
            heuristic_request_count += 1
        else:
            print(f"  车辆 {vehicle_id} -> {assignment}")
    
    print(f"启发式总价值: {heuristic_total_value:.2f}, 服务请求数: {heuristic_request_count}")
    
    # 测试Gurobi方法
    print("\n=== Gurobi方法分配 ===")
    gurobi_assignments = optimizer._gurobi_vehicle_rebalancing_knownreject(
        available_vehicles, available_requests, available_charging_stations
    )
    
    print("Gurobi分配结果:")
    gurobi_total_value = 0
    gurobi_request_count = 0
    gurobi_total_q_value = 0
    
    for vehicle_id, assignment in gurobi_assignments.items():
        if isinstance(assignment, env.active_requests[list(env.active_requests.keys())[0]].__class__):
            # 计算Q值
            vehicle = env.vehicles[vehicle_id]
            q_value = 0
            if hasattr(env, 'get_assignment_q_value'):
                q_value = env.get_assignment_q_value(
                    vehicle_id, assignment.request_id,
                    vehicle['location'], assignment.pickup
                )
            
            total_objective_value = assignment.value + q_value * env.adp_value
            print(f"  车辆 {vehicle_id} -> 请求 {assignment.request_id} (价值: {assignment.value:.2f}, Q值: {q_value:.2f}, 总目标: {total_objective_value:.2f})")
            gurobi_total_value += assignment.value
            gurobi_total_q_value += q_value
            gurobi_request_count += 1
        else:
            print(f"  车辆 {vehicle_id} -> {assignment}")
    
    print(f"Gurobi总价值: {gurobi_total_value:.2f}, 总Q值: {gurobi_total_q_value:.2f}, 服务请求数: {gurobi_request_count}")
    print(f"Gurobi总目标函数值: {gurobi_total_value + gurobi_total_q_value * env.adp_value:.2f}")
    
    # 分析差异
    print("\n=== 差异分析 ===")
    print(f"价值差异: {gurobi_total_value - heuristic_total_value:.2f}")
    print(f"服务请求数差异: {gurobi_request_count - heuristic_request_count}")
    print(f"Q值贡献: {gurobi_total_q_value * env.adp_value:.2f}")
    
    # 检查未分配的请求
    heuristic_assigned_request_ids = set()
    gurobi_assigned_request_ids = set()
    
    for assignment in heuristic_assignments.values():
        if hasattr(assignment, 'request_id'):
            heuristic_assigned_request_ids.add(assignment.request_id)
    
    for assignment in gurobi_assignments.values():
        if hasattr(assignment, 'request_id'):
            gurobi_assigned_request_ids.add(assignment.request_id)
    
    all_request_ids = set(req.request_id for req in available_requests)
    
    heuristic_unassigned = all_request_ids - heuristic_assigned_request_ids
    gurobi_unassigned = all_request_ids - gurobi_assigned_request_ids
    
    print(f"\n启发式未分配请求: {len(heuristic_unassigned)} 个")
    print(f"Gurobi未分配请求: {len(gurobi_unassigned)} 个")
    
    # 计算未分配请求的价值损失
    heuristic_lost_value = sum(req.value for req in available_requests if req.request_id in heuristic_unassigned)
    gurobi_lost_value = sum(req.value for req in available_requests if req.request_id in gurobi_unassigned)
    
    print(f"启发式损失价值: {heuristic_lost_value:.2f}")
    print(f"Gurobi损失价值: {gurobi_lost_value:.2f}")
    
    # 检查车辆分配情况
    print(f"\n车辆分配情况:")
    print(f"启发式分配车辆数: {len(heuristic_assignments)}")
    print(f"Gurobi分配车辆数: {len(gurobi_assignments)}")

if __name__ == "__main__":
    test_assignment_difference()
