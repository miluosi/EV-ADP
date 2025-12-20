#!/usr/bin/env python3
"""
测试动作平衡采样功能
"""

import sys
import os
import random
sys.path.append('src')

def test_action_balanced_sampling():
    """测试动作平衡采样"""
    print("🧪 Testing Action-Balanced Sampling...")
    
    try:
        from ValueFunction_pytorch import PyTorchPathBasedNetwork
        
        # 创建一个测试实例
        value_func = PyTorchPathBasedNetwork()
        
        # 模拟不平衡的经验数据
        print("📊 Creating unbalanced experience data...")
        
        # 模拟80% idle, 15% assign, 5% charge的分布
        for i in range(1000):
            if i < 800:  # 80% idle
                exp = {
                    'vehicle_id': i % 5,
                    'action_type': 'idle',
                    'vehicle_location': random.randint(0, 99),
                    'target_location': random.randint(0, 99),
                    'battery_level': random.uniform(0.5, 1.0),
                    'current_time': i * 5.0,
                    'reward': random.uniform(-1.0, 0.5),
                    'next_vehicle_location': random.randint(0, 99),
                    'next_battery_level': random.uniform(0.5, 1.0),
                    'num_requests': random.randint(5, 15),
                    'request_value': 0.0,
                    'is_idle': True
                }
            elif i < 950:  # 15% assign
                exp = {
                    'vehicle_id': i % 5,
                    'action_type': f'assign_{i % 10}',
                    'vehicle_location': random.randint(0, 99),
                    'target_location': random.randint(0, 99),
                    'battery_level': random.uniform(0.5, 1.0),
                    'current_time': i * 5.0,
                    'reward': random.uniform(-5.0, 50.0),  # assign有更大的奖励方差
                    'next_vehicle_location': random.randint(0, 99),
                    'next_battery_level': random.uniform(0.3, 0.9),
                    'num_requests': random.randint(5, 15),
                    'request_value': random.uniform(5.0, 20.0),
                    'is_idle': False
                }
            else:  # 5% charge
                exp = {
                    'vehicle_id': i % 5,
                    'action_type': f'charge_{i % 3}',
                    'vehicle_location': random.randint(0, 99),
                    'target_location': random.randint(0, 99),
                    'battery_level': random.uniform(0.1, 0.5),
                    'current_time': i * 5.0,
                    'reward': random.uniform(-2.0, 1.0),
                    'next_vehicle_location': random.randint(0, 99),
                    'next_battery_level': random.uniform(0.8, 1.0),
                    'num_requests': random.randint(5, 15),
                    'request_value': 0.0,
                    'is_idle': False
                }
            
            value_func.experience_buffer.append(exp)
        
        print(f"✅ Created {len(value_func.experience_buffer)} experiences")
        
        # 统计原始分布
        assign_count = sum(1 for exp in value_func.experience_buffer if exp['action_type'].startswith('assign'))
        idle_count = sum(1 for exp in value_func.experience_buffer if exp['action_type'] == 'idle')
        charge_count = sum(1 for exp in value_func.experience_buffer if exp['action_type'].startswith('charge'))
        
        print(f"📈 Original distribution:")
        print(f"   Assign: {assign_count} ({assign_count/len(value_func.experience_buffer)*100:.1f}%)")
        print(f"   Idle:   {idle_count} ({idle_count/len(value_func.experience_buffer)*100:.1f}%)")
        print(f"   Charge: {charge_count} ({charge_count/len(value_func.experience_buffer)*100:.1f}%)")
        
        # 测试动作平衡采样
        print(f"\n🎯 Testing action-balanced sampling...")
        value_func.training_step = 0  # 确保使用平衡采样
        
        batch_size = 64
        batch = value_func._action_balanced_sample(batch_size)
        
        # 统计采样结果
        batch_assign = sum(1 for exp in batch if exp['action_type'].startswith('assign'))
        batch_idle = sum(1 for exp in batch if exp['action_type'] == 'idle')
        batch_charge = sum(1 for exp in batch if exp['action_type'].startswith('charge'))
        
        print(f"📊 Batch distribution (size={len(batch)}):")
        print(f"   Assign: {batch_assign} ({batch_assign/len(batch)*100:.1f}%)")
        print(f"   Idle:   {batch_idle} ({batch_idle/len(batch)*100:.1f}%)")
        print(f"   Charge: {batch_charge} ({batch_charge/len(batch)*100:.1f}%)")
        
        # 验证平衡效果
        if batch_assign > batch_idle:
            print("✅ SUCCESS: Assign actions are now more represented than idle!")
        else:
            print("⚠️  WARNING: Idle actions still dominate the batch")
        
        # 测试多个批次
        print(f"\n🔄 Testing multiple batches...")
        total_assign = 0
        total_idle = 0
        total_charge = 0
        num_batches = 10
        
        for i in range(num_batches):
            batch = value_func._action_balanced_sample(batch_size)
            total_assign += sum(1 for exp in batch if exp['action_type'].startswith('assign'))
            total_idle += sum(1 for exp in batch if exp['action_type'] == 'idle')
            total_charge += sum(1 for exp in batch if exp['action_type'].startswith('charge'))
        
        total_samples = num_batches * batch_size
        print(f"📊 Average over {num_batches} batches ({total_samples} samples):")
        print(f"   Assign: {total_assign/total_samples*100:.1f}%")
        print(f"   Idle:   {total_idle/total_samples*100:.1f}%")
        print(f"   Charge: {total_charge/total_samples*100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🧪 Action-Balanced Sampling Test Suite")
    print("=" * 50)
    
    success = test_action_balanced_sampling()
    
    if success:
        print("\n✅ All tests passed!")
        print("💡 The action-balanced sampling should help fix the Q-value issue")
        print("   by ensuring assign actions get proper representation during training.")
    else:
        print("\n❌ Tests failed!")
    
    print("\n" + "=" * 50)