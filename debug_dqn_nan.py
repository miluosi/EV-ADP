"""
诊断DQN训练中的NaN损失问题
"""
import torch
import torch.nn as nn
import numpy as np
from src.ValueFunction_pytorch import DQNAgent, DQNActionNetwork, create_dqn_state_features
from src.Environment import ChargingIntegratedEnvironment

def diagnose_dqn_nan_issue():
    """诊断DQN训练中的NaN损失问题"""
    print("=== DQN NaN损失诊断开始 ===")
    
    # 1. 检查网络初始化
    print("\n1. 检查网络初始化...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dqn_agent = DQNAgent(state_dim=64, action_dim=32, device=device)
    
    # 检查网络参数
    for name, param in dqn_agent.policy_net.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"❌ 发现异常参数在 {name}: NaN={torch.isnan(param).sum()}, Inf={torch.isinf(param).sum()}")
        else:
            print(f"✓ {name}: 正常 (范围: {param.min().item():.4f} ~ {param.max().item():.4f})")
    
    # 2. 测试前向传播
    print("\n2. 测试前向传播...")
    batch_size = 4
    vehicle_features = torch.randn(batch_size, 8).to(device)
    request_features = torch.randn(batch_size, 6).to(device) 
    global_features = torch.randn(batch_size, 4).to(device)
    
    print(f"输入特征范围:")
    print(f"  vehicle_features: {vehicle_features.min().item():.4f} ~ {vehicle_features.max().item():.4f}")
    print(f"  request_features: {request_features.min().item():.4f} ~ {request_features.max().item():.4f}")
    print(f"  global_features: {global_features.min().item():.4f} ~ {global_features.max().item():.4f}")
    
    try:
        with torch.no_grad():
            q_values = dqn_agent.policy_net(vehicle_features, request_features, global_features)
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                print(f"❌ 前向传播输出异常: NaN={torch.isnan(q_values).sum()}, Inf={torch.isinf(q_values).sum()}")
            else:
                print(f"✓ 前向传播正常: Q值范围 {q_values.min().item():.4f} ~ {q_values.max().item():.4f}")
    except Exception as e:
        print(f"❌ 前向传播失败: {e}")
    
    # 3. 测试实际环境数据
    print("\n3. 测试实际环境数据...")
    try:
        env = ChargingIntegratedEnvironment(num_vehicles=2, num_stations=2, random_seed=42)
        env.reset()
        
        # 生成真实的状态特征
        vehicle_id = 0
        real_state = create_dqn_state_features(env, vehicle_id, current_time=0.0)
        
        print(f"真实状态特征:")
        for key, tensor in real_state.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"❌ {key}: NaN={torch.isnan(tensor).sum()}, Inf={torch.isinf(tensor).sum()}")
            else:
                print(f"✓ {key}: 正常 (范围: {tensor.min().item():.4f} ~ {tensor.max().item():.4f})")
        
        # 测试网络输出
        with torch.no_grad():
            q_values = dqn_agent.policy_net(
                real_state['vehicle'].unsqueeze(0),
                real_state['request'].unsqueeze(0), 
                real_state['global'].unsqueeze(0)
            )
            if torch.isnan(q_values).any() or torch.isinf(q_values).any():
                print(f"❌ 真实数据前向传播异常: NaN={torch.isnan(q_values).sum()}, Inf={torch.isinf(q_values).sum()}")
            else:
                print(f"✓ 真实数据前向传播正常: Q值范围 {q_values.min().item():.4f} ~ {q_values.max().item():.4f}")
                
    except Exception as e:
        print(f"❌ 环境测试失败: {e}")
    
    # 4. 测试训练步骤
    print("\n4. 测试训练步骤...")
    
    # 向经验缓冲区添加一些假数据
    for i in range(50):  # 添加足够的经验
        state = {
            'vehicle': torch.randn(8).to(device),
            'request': torch.randn(6).to(device),
            'global': torch.randn(4).to(device)
        }
        next_state = {
            'vehicle': torch.randn(8).to(device),
            'request': torch.randn(6).to(device),
            'global': torch.randn(4).to(device)
        }
        action = np.random.randint(0, 32)
        reward = np.random.normal(0, 1)  # 正态分布奖励
        done = np.random.choice([True, False])
        
        dqn_agent.store_transition(state, action, reward, next_state, done)
    
    print(f"经验缓冲区大小: {len(dqn_agent.memory)}")
    
    # 尝试训练步骤
    try:
        loss = dqn_agent.train_step(batch_size=32)
        if loss is None:
            print("❌ train_step返回None")
        elif np.isnan(loss) or np.isinf(loss):
            print(f"❌ train_step返回异常损失: {loss}")
        else:
            print(f"✓ train_step正常: 损失 = {loss:.6f}")
    except Exception as e:
        print(f"❌ train_step失败: {e}")
        import traceback
        traceback.print_exc()
    
    # 5. 检查优化器状态
    print("\n5. 检查优化器状态...")
    for group in dqn_agent.optimizer.param_groups:
        print(f"  学习率: {group['lr']}")
        for param in group['params']:
            if param.grad is not None:
                grad_norm = param.grad.data.norm().item()
                if np.isnan(grad_norm) or np.isinf(grad_norm):
                    print(f"❌ 梯度异常: {grad_norm}")
                else:
                    print(f"✓ 梯度正常: norm={grad_norm:.6f}")
            else:
                print("  梯度为None")
    
    print("\n=== 诊断完成 ===")

if __name__ == "__main__":
    diagnose_dqn_nan_issue()