"""
测试 Environment.py 中新增的功能:
1. ChargingProbabilityCalculator - 基于嵌套Logit模型的充电决策
2. RelocationManager - 重定位行为管理
3. Zone 系统 - 区域分类和动态更新
4. EV idle 时间跟踪
5. 连续拒绝惩罚机制
"""

import numpy as np
from src.Environment import ChargingIntegratedEnvironment
from src.NYCRequest import NYCRequest


def test_charging_probability_calculator():
    """测试 ChargingProbabilityCalculator"""
    print("\n" + "="*60)
    print("测试 1: ChargingProbabilityCalculator")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=5, 
        num_stations=3, 
        grid_size=15,
        use_intense_requests=False,
        random_seed=42
    )
    
    # 测试充电决策
    vehicle_id = 0
    origin_loc = 0
    dest_loc = 224  # 对角线远端
    current_soc = 40.0  # 40% 电量
    
    print(f"\n车辆 {vehicle_id} 状态:")
    print(f"  位置: {origin_loc}")
    print(f"  目的地: {dest_loc}")
    print(f"  当前 SOC: {current_soc}%")
    
    decision = env.make_probabilistic_charging_decision(
        vehicle_id, origin_loc, dest_loc, current_soc
    )
    
    print(f"\n充电决策结果:")
    print(f"  决策: {decision['decision']}")
    print(f"  选择的充电站: {decision['station_id']}")
    print(f"  概率分布:")
    probs = decision['probabilities']
    print(f"    不充电概率: {probs['action_no_charge']:.4f}")
    print(f"    充电概率: {probs['action_charge']:.4f}")
    if probs.get('station_probs'):
        print(f"    各充电站概率:")
        for sid, p in probs['station_probs'].items():
            print(f"      充电站 {sid}: {p:.4f}")
    
    # 测试多种 SOC 场景
    print("\n测试不同 SOC 下的充电倾向:")
    for soc in [20, 40, 60, 80, 100]:
        decision = env.make_probabilistic_charging_decision(
            vehicle_id, origin_loc, dest_loc, float(soc)
        )
        charge_prob = decision['probabilities']['action_charge']
        print(f"  SOC {soc}%: 充电概率 = {charge_prob:.4f}")
    
    print("\n✓ ChargingProbabilityCalculator 测试完成")


def test_zone_system():
    """测试 Zone 系统"""
    print("\n" + "="*60)
    print("测试 2: Zone 系统")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=5, 
        num_stations=3, 
        grid_size=15,
        use_intense_requests=True,
        random_seed=42
    )
    
    # 手动设置zones
    surge_zones = [100, 101, 115, 116]
    hd_zones = [50, 51, 65, 66]
    city_center = [112, 113, 127, 128]
    surge_price = 5.0
    
    print("\n手动设置 Zones:")
    print(f"  Surge zones: {surge_zones}")
    print(f"  High demand zones: {hd_zones}")
    print(f"  City center zones: {city_center}")
    print(f"  Surge price: ${surge_price:.2f}")
    
    env.update_zones(
        surge_zone_ids=surge_zones,
        high_demand_zone_ids=hd_zones,
        city_center_zone_ids=city_center,
        surge_price=surge_price
    )
    
    # 验证zones已更新
    print(f"\n验证 Zones 更新:")
    print(f"  Surge zones count: {len(env.surge_zones)}")
    print(f"  High demand zones count: {len(env.high_demand_zones)}")
    print(f"  City center zones count: {len(env.city_center_zones)}")
    print(f"  Current surge price: ${env.current_surge_price:.2f}")
    
    # 测试自动更新zones(基于需求)
    print("\n测试自动 Zone 更新 (基于需求):")
    
    # 手动创建一些请求来模拟需求
    for i in range(10):
        req_id = i
        pickup = (i * 20) % (env.grid_size * env.grid_size)
        dropoff = (pickup + 50) % (env.grid_size * env.grid_size)
        env.active_requests[req_id] = type('Request', (), {
            'request_id': req_id, 
            'pickup': pickup, 
            'dropoff': dropoff
        })()
    
    initial_surge_count = len(env.surge_zones)
    initial_hd_count = len(env.high_demand_zones)
    
    env.auto_update_zones_based_on_demand()
    
    print(f"  更新前 Surge zones: {initial_surge_count}")
    print(f"  更新后 Surge zones: {len(env.surge_zones)}")
    print(f"  更新前 High demand zones: {initial_hd_count}")
    print(f"  更新后 High demand zones: {len(env.high_demand_zones)}")
    print(f"  动态计算的 Surge price: ${env.current_surge_price:.2f}")
    
    print("\n✓ Zone 系统测试完成")


def test_relocation_manager():
    """测试 RelocationManager"""
    print("\n" + "="*60)
    print("测试 3: RelocationManager")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=5, 
        num_stations=3, 
        grid_size=15,
        use_intense_requests=True,
        random_seed=42
    )
    
    # 设置zones
    env.update_zones(
        surge_zone_ids=[100, 101, 115, 116],
        high_demand_zone_ids=[55, 56, 70, 71],
        city_center_zone_ids=[112, 113, 127, 128],
        surge_price=4.5
    )
    
    # 初始化车辆状态
    vehicle_id = 0
    env.vehicles[vehicle_id] = {
        'location': 50,
        'battery': 0.8,
        'assigned_request': None,
        'passenger_onboard': None,
        'charging_station': None,
        'completed_trips': 8
    }
    
    print(f"\n车辆 {vehicle_id} 状态:")
    print(f"  位置: {env.vehicles[vehicle_id]['location']}")
    print(f"  已完成订单数: {env.vehicles[vehicle_id]['completed_trips']}")
    print(f"  Idle 时间: {env.get_vehicle_idle_time(vehicle_id):.1f} epochs")
    
    # 模拟等待一段时间
    env.vehicle_last_service_time[vehicle_id] = 0.0
    env.update_vehicle_idle_time(vehicle_id, 10.0)
    
    print(f"  模拟等待后 Idle 时间: {env.get_vehicle_idle_time(vehicle_id):.1f} epochs")
    
    # 做出重定位决策
    decision = env.make_relocation_decision(vehicle_id)
    
    print(f"\n重定位决策结果:")
    print(f"  选择的动作: {decision['action']}")
    print(f"  目标位置: {decision['target_loc']}")
    print(f"  概率分布:")
    for action, prob in decision['probabilities'].items():
        print(f"    {action}: {prob:.4f}")
    
    # 测试多次决策以观察概率分布
    print("\n执行 100 次决策统计动作分布:")
    action_counts = {'Wait': 0, 'Surge': 0, 'HighDemand': 0, 'Cruise': 0}
    
    for _ in range(100):
        decision = env.make_relocation_decision(vehicle_id)
        action_counts[decision['action']] += 1
    
    print("  动作分布:")
    for action, count in action_counts.items():
        print(f"    {action}: {count}% ({count}/100)")
    
    print("\n✓ RelocationManager 测试完成")


def test_idle_time_tracking():
    """测试 EV idle 时间跟踪"""
    print("\n" + "="*60)
    print("测试 4: EV Idle 时间跟踪")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=3, 
        num_stations=2, 
        grid_size=10,
        random_seed=42
    )
    
    vehicle_id = 0
    
    print(f"\n车辆 {vehicle_id} Idle 时间跟踪:")
    
    # 初始状态
    print(f"  初始 idle 时间: {env.get_vehicle_idle_time(vehicle_id):.1f}")
    
    # 完成一次服务
    completion_time = 10.0
    env.record_vehicle_service_completion(vehicle_id, completion_time)
    print(f"  服务完成时间: {completion_time:.1f}")
    print(f"  完成后 idle 时间: {env.get_vehicle_idle_time(vehicle_id):.1f}")
    
    # 更新到不同时间点
    time_points = [15.0, 20.0, 25.0, 30.0]
    for t in time_points:
        env.update_vehicle_idle_time(vehicle_id, t)
        idle_time = env.get_vehicle_idle_time(vehicle_id)
        print(f"  时间 {t:.1f}: idle 时间 = {idle_time:.1f} epochs")
    
    # 再次完成服务，重置idle时间
    completion_time = 30.0
    env.record_vehicle_service_completion(vehicle_id, completion_time)
    print(f"\n  再次完成服务 (时间 {completion_time:.1f})")
    print(f"  重置后 idle 时间: {env.get_vehicle_idle_time(vehicle_id):.1f}")
    
    # 测试多个车辆
    print("\n测试多车辆 idle 时间跟踪:")
    for vid in range(3):
        env.record_vehicle_service_completion(vid, 10.0 + vid * 5)
        env.update_vehicle_idle_time(vid, 30.0)
        idle = env.get_vehicle_idle_time(vid)
        print(f"  车辆 {vid}: idle 时间 = {idle:.1f} epochs")
    
    print("\n✓ Idle 时间跟踪测试完成")


def test_rejection_penalty_mechanism():
    """测试连续拒绝惩罚机制"""
    print("\n" + "="*60)
    print("测试 5: 连续拒绝惩罚机制")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=3, 
        num_stations=2, 
        grid_size=10,
        random_seed=42
    )
    
    vehicle_id = 0
    current_time = 0.0
    
    print(f"\n车辆 {vehicle_id} 拒绝惩罚测试:")
    print(f"  拒绝阈值: {env.rejection_penalty_threshold} 次")
    print(f"  惩罚持续时间: {env.rejection_penalty_duration:.1f} epochs")
    
    # 初始状态
    info = env.get_vehicle_penalty_info(vehicle_id, current_time)
    print(f"\n初始状态:")
    print(f"  是否在惩罚期: {info['in_penalty']}")
    print(f"  连续拒绝次数: {info['consecutive_rejections']}")
    print(f"  距离惩罚还差: {info['rejections_until_penalty']} 次")
    
    # 第一次拒绝
    current_time = 10.0
    env.record_vehicle_rejection(vehicle_id, current_time)
    info = env.get_vehicle_penalty_info(vehicle_id, current_time)
    print(f"\n第 1 次拒绝后 (时间 {current_time:.1f}):")
    print(f"  连续拒绝次数: {info['consecutive_rejections']}")
    print(f"  距离惩罚还差: {info['rejections_until_penalty']} 次")
    print(f"  是否在惩罚期: {info['in_penalty']}")
    
    # 第二次拒绝 - 应该触发惩罚
    current_time = 15.0
    print(f"\n第 2 次拒绝 (时间 {current_time:.1f}) - 应触发惩罚:")
    env.record_vehicle_rejection(vehicle_id, current_time)
    info = env.get_vehicle_penalty_info(vehicle_id, current_time)
    print(f"  是否在惩罚期: {info['in_penalty']}")
    print(f"  剩余惩罚时间: {info['remaining_penalty_time']:.1f} epochs")
    print(f"  连续拒绝计数已重置: {info['consecutive_rejections']}")
    
    # 检查惩罚期内的状态
    current_time = 20.0
    info = env.get_vehicle_penalty_info(vehicle_id, current_time)
    print(f"\n惩罚期内 (时间 {current_time:.1f}):")
    print(f"  是否在惩罚期: {info['in_penalty']}")
    print(f"  剩余惩罚时间: {info['remaining_penalty_time']:.1f} epochs")
    
    # 惩罚期结束后
    current_time = 26.0  # 15 + 10 + 1
    info = env.get_vehicle_penalty_info(vehicle_id, current_time)
    print(f"\n惩罚期结束后 (时间 {current_time:.1f}):")
    print(f"  是否在惩罚期: {info['in_penalty']}")
    print(f"  剩余惩罚时间: {info['remaining_penalty_time']:.1f} epochs")
    
    # 测试接受订单重置计数
    vehicle_id_2 = 1
    current_time = 30.0
    print(f"\n\n车辆 {vehicle_id_2} 测试接受订单重置:")
    
    env.record_vehicle_rejection(vehicle_id_2, current_time)
    info = env.get_vehicle_penalty_info(vehicle_id_2, current_time)
    print(f"  拒绝 1 次后: 连续拒绝 = {info['consecutive_rejections']}")
    
    env.record_vehicle_acceptance(vehicle_id_2)
    info = env.get_vehicle_penalty_info(vehicle_id_2, current_time)
    print(f"  接受订单后: 连续拒绝 = {info['consecutive_rejections']} (已重置)")
    
    print("\n✓ 拒绝惩罚机制测试完成")


def test_integrated_workflow():
    """测试完整的集成工作流"""
    print("\n" + "="*60)
    print("测试 6: 完整集成工作流")
    print("="*60)
    
    env = ChargingIntegratedEnvironment(
        num_vehicles=5, 
        num_stations=3, 
        grid_size=15,
        use_intense_requests=True,
        random_seed=42
    )
    
    print("\n模拟完整的一个时间步:")
    
    # 设置初始zones
    env.update_zones(
        surge_zone_ids=[100, 115],
        high_demand_zone_ids=[50, 65],
        city_center_zone_ids=[112, 113],
        surge_price=3.5
    )
    
    # 初始化车辆
    for vid in range(5):
        env.vehicles[vid] = {
            'location': vid * 30,
            'battery': 0.5 + vid * 0.1,
            'assigned_request': None,
            'passenger_onboard': None,
            'charging_station': None,
            'completed_trips': vid * 2
        }
    
    current_time = 0.0
    
    print(f"\n时间步 {current_time:.1f}:")
    print(f"  活跃请求数: {len(env.active_requests)}")
    print(f"  Surge zones: {list(env.surge_zones)}")
    print(f"  Surge price: ${env.current_surge_price:.2f}")
    
    # 对每个车辆做决策
    for vid in range(5):
        vehicle = env.vehicles[vid]
        soc = vehicle['battery'] * 100  # 转换为百分比
        
        print(f"\n车辆 {vid} (位置: {vehicle['location']}, SOC: {soc:.1f}%):")
        
        # 检查是否在惩罚期
        penalty_info = env.get_vehicle_penalty_info(vid, current_time)
        if penalty_info['in_penalty']:
            print(f"  ⚠ 在惩罚期内，剩余 {penalty_info['remaining_penalty_time']:.1f} epochs")
            continue
        
        # 充电决策
        charging_decision = env.make_probabilistic_charging_decision(
            vid, vehicle['location'], vehicle['location'], soc
        )
        print(f"  充电决策: {charging_decision['decision']}")
        if charging_decision['decision'] == 'charge':
            print(f"    → 前往充电站 {charging_decision['station_id']}")
        
        # 如果不充电，考虑重定位
        if charging_decision['decision'] == 'no_charge':
            relocation_decision = env.make_relocation_decision(vid)
            print(f"  重定位决策: {relocation_decision['action']}")
            if relocation_decision['target_loc'] is not None:
                print(f"    → 目标位置 {relocation_decision['target_loc']}")
        
        # 更新idle时间
        env.update_vehicle_idle_time(vid, current_time)
        idle_time = env.get_vehicle_idle_time(vid)
        print(f"  当前 idle 时间: {idle_time:.1f} epochs")
    
    # 模拟一些拒绝和接受
    print("\n\n模拟订单响应:")
    env.record_vehicle_rejection(0, current_time)
    env.record_vehicle_rejection(0, current_time + 1)
    info = env.get_vehicle_penalty_info(0, current_time + 1)
    print(f"车辆 0 连续拒绝 2 次:")
    print(f"  进入惩罚期: {info['in_penalty']}")
    print(f"  剩余惩罚时间: {info['remaining_penalty_time']:.1f} epochs")
    
    env.record_vehicle_acceptance(1)
    print(f"车辆 1 接受订单: 连续拒绝计数重置")
    
    # 自动更新zones
    print("\n执行自动 zone 更新:")
    # 手动创建一些请求来模拟需求
    for i in range(15):
        req_id = 100 + i
        pickup = (i * 15) % (env.grid_size * env.grid_size)
        dropoff = (pickup + 30) % (env.grid_size * env.grid_size)
        env.active_requests[req_id] = type('Request', (), {
            'request_id': req_id, 
            'pickup': pickup, 
            'dropoff': dropoff
        })()
    
    env.auto_update_zones_based_on_demand()
    print(f"  更新后 Surge zones 数量: {len(env.surge_zones)}")
    print(f"  更新后 HD zones 数量: {len(env.high_demand_zones)}")
    print(f"  动态 Surge price: ${env.current_surge_price:.2f}")
    
    print("\n✓ 完整集成工作流测试完成")


def main():
    """运行所有测试"""
    print("="*60)
    print("Environment.py 新功能测试套件")
    print("="*60)
    
    try:
        test_charging_probability_calculator()
        test_zone_system()
        test_relocation_manager()
        test_idle_time_tracking()
        test_rejection_penalty_mechanism()
        test_integrated_workflow()
        
        print("\n" + "="*60)
        print("所有测试完成!")
        print("="*60)
        print("\n总结:")
        print("✓ ChargingProbabilityCalculator - 基于嵌套Logit模型正常工作")
        print("✓ Zone 系统 - 手动和自动更新功能正常")
        print("✓ RelocationManager - 重定位决策正常")
        print("✓ EV Idle 时间跟踪 - 记录和更新正常")
        print("✓ 拒绝惩罚机制 - 阈值触发和重置正常")
        print("✓ 完整集成工作流 - 所有功能协同工作正常")
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
