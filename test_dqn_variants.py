#!/usr/bin/env python3
"""
测试不同DQN变体的实现 - DQN、DDQN、Dueling DQN
Test script for different DQN variants implementation
"""

import torch
import numpy as np
from src.ValueFunction_pytorch import PyTorchChargingValueFunction

def test_dqn_variants():
    """测试三种DQN变体的实现"""
    print("🧪 Testing DQN Variants Implementation")
    print("=" * 60)
    
    # 测试参数
    grid_size = 5  # 较小的网格用于快速测试
    num_vehicles = 4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🖥️  Using device: {device}")
    
    # 测试三种模式
    modes = ["DQN", "DDQN", "DuelingDQN"]
    
    value_functions = {}
    
    for mode in modes:
        print(f"\n🔧 Initializing {mode} Value Function...")
        try:
            vf = PyTorchChargingValueFunction(
                grid_size=grid_size,
                num_vehicles=num_vehicles,
                device=device,
                network_mode=mode,
                log_dir=f"logs/test_{mode.lower()}"
            )
            value_functions[mode] = vf
            print(f"✅ {mode} initialization successful")
            
            # 打印网络类型信息
            network_type = type(vf.network).__name__
            target_network_type = type(vf.target_network).__name__
            print(f"   Main network: {network_type}")
            print(f"   Target network: {target_network_type}")
            
            # 计算参数数量
            main_params = sum(p.numel() for p in vf.network.parameters())
            target_params = sum(p.numel() for p in vf.target_network.parameters())
            print(f"   Parameters: Main={main_params:,}, Target={target_params:,}")
            
        except Exception as e:
            print(f"❌ {mode} initialization failed: {e}")
            continue
    
    print(f"\n🧠 Testing Q-value computation for each variant...")
    
    # 创建测试输入
    test_vehicle_id = 0
    test_vehicle_location = 5
    test_target_location = 10
    test_current_time = 50.0
    test_battery = 0.8
    test_request_value = 25.0
    
    for mode, vf in value_functions.items():
        print(f"\n🎯 Testing {mode} Q-value computation...")
        try:
            # 测试assignment Q-value
            assign_q = vf.get_assignment_q_value(
                vehicle_id=test_vehicle_id,
                target_id=1,
                vehicle_location=test_vehicle_location,
                target_reject=test_vehicle_location,
                target_location=test_target_location,
                current_time=test_current_time,
                battery_level=test_battery,
                request_value=test_request_value
            )
            print(f"   Assignment Q-value: {assign_q:.4f}")
            
            # 测试idle Q-value
            idle_q = vf.get_idle_q_value(
                vehicle_id=test_vehicle_id,
                vehicle_location=test_vehicle_location,
                battery_level=test_battery,
                current_time=test_current_time
            )
            print(f"   Idle Q-value: {idle_q:.4f}")
            
            # 测试charging Q-value
            charge_q = vf.get_charging_q_value(
                vehicle_id=test_vehicle_id,
                station_id=1,
                vehicle_location=test_vehicle_location,
                station_location=test_target_location,
                current_time=test_current_time,
                battery_level=test_battery
            )
            print(f"   Charging Q-value: {charge_q:.4f}")
            
            print(f"   ✅ {mode} Q-value computation successful")
            
        except Exception as e:
            print(f"   ❌ {mode} Q-value computation failed: {e}")
    
    print(f"\n🏋️  Testing training with different modes...")
    
    # 生成一些模拟训练数据
    for mode, vf in value_functions.items():
        print(f"\n🚂 Training {mode} for 5 steps...")
        
        try:
            # 添加一些模拟经验
            for i in range(10):
                vf.store_experience(
                    vehicle_id=i % num_vehicles,
                    action_type=f"assign_{i}",
                    vehicle_location=i % (grid_size * grid_size),
                    target_location=(i + 5) % (grid_size * grid_size),
                    current_time=i * 10.0,
                    reward=np.random.normal(10.0, 5.0),  # 随机奖励
                    next_vehicle_location=(i + 1) % (grid_size * grid_size),
                    battery_level=max(0.1, np.random.random()),
                    next_battery_level=max(0.1, np.random.random()),
                    request_value=np.random.uniform(0, 50),
                    next_request_value=np.random.uniform(0, 50),
                    dur_time=1.0
                )
            
            # 执行训练步骤，使用相应的模式
            total_loss = 0
            for step in range(5):
                loss = vf.train_step(batch_size=8, mode=mode)  # 传递mode参数
                total_loss += loss
                if step == 0:  # 只显示第一步的详细信息
                    print(f"   Step {step+1}: Loss={loss:.4f}")
            
            avg_loss = total_loss / 5
            print(f"   ✅ {mode} training successful, Average loss: {avg_loss:.4f}")
            
        except Exception as e:
            print(f"   ❌ {mode} training failed: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🔍 Comparing network architectures...")
    
    # 比较不同网络的架构差异
    if "DQN" in value_functions and "DuelingDQN" in value_functions:
        dqn_net = value_functions["DQN"].network
        dueling_net = value_functions["DuelingDQN"].network
        
        print(f"Standard DQN layers:")
        for name, module in dqn_net.named_children():
            print(f"   {name}: {module.__class__.__name__}")
            
        print(f"Dueling DQN layers:")
        for name, module in dueling_net.named_children():
            print(f"   {name}: {module.__class__.__name__}")
    
    print(f"\n🎉 DQN Variants Test Complete!")
    print("=" * 60)
    print("Summary:")
    for mode in modes:
        status = "✅ Success" if mode in value_functions else "❌ Failed"
        print(f"   {mode}: {status}")


if __name__ == "__main__":
    test_dqn_variants()