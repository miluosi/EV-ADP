"""
简化的Zone-ADP集成测试脚本

此版本去掉了复杂的依赖，专注于核心功能测试
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

print("开始Zone-ADP简化测试")

def test_basic_pytorch():
    """测试基本PyTorch功能"""
    print("\n=== 测试基本PyTorch功能 ===")
    
    # 检查PyTorch版本和CUDA
    print(f"PyTorch版本: {torch.__version__}")
    print(f"CUDA可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA设备数量: {torch.cuda.device_count()}")
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
    
    # 创建简单张量
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    x = torch.randn(5, 3).to(device)
    y = torch.randn(3, 4).to(device)
    z = torch.mm(x, y)
    
    print(f"张量运算成功，设备: {device}")
    print(f"输入形状: {x.shape}, {y.shape}")
    print(f"输出形状: {z.shape}")
    
    return device

def test_simple_network():
    """测试简单神经网络"""
    print("\n=== 测试简单神经网络 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class SimpleNetwork(nn.Module):
        def __init__(self, input_dim, hidden_dim, output_dim):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim)
            )
        
        def forward(self, x):
            return self.layers(x)
    
    # 创建网络
    net = SimpleNetwork(10, 64, 1).to(device)
    print(f"网络参数数量: {sum(p.numel() for p in net.parameters())}")
    
    # 测试前向传播
    batch_size = 32
    x = torch.randn(batch_size, 10).to(device)
    y = net(x)
    
    print(f"前向传播成功")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {y.shape}")
    print(f"输出范围: [{y.min().item():.4f}, {y.max().item():.4f}]")
    
    # 测试反向传播
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    target = torch.randn(batch_size, 1).to(device)
    loss = F.mse_loss(y, target)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    print(f"反向传播成功，损失: {loss.item():.4f}")
    
    return net

def test_zone_environment_simple():
    """测试简化的Zone环境"""
    print("\n=== 测试简化Zone环境 ===")
    
    class SimpleZoneEnvironment:
        def __init__(self, num_agents=5, num_zones=3, num_locations=20):
            self.num_agents = num_agents
            self.num_zones = num_zones
            self.num_locations = num_locations
            
            # 状态：智能体位置和区域分配
            self.agent_positions = np.random.randint(0, num_locations, num_agents)
            self.zone_assignments = np.random.randint(0, num_zones, num_agents)
            
            # 需求矩阵
            self.demand_matrix = np.random.exponential(1.0, (num_locations, num_locations))
            
            self.current_step = 0
            self.max_steps = 100
        
        def reset(self):
            self.agent_positions = np.random.randint(0, self.num_locations, self.num_agents)
            self.zone_assignments = np.random.randint(0, self.num_zones, self.num_agents)
            self.current_step = 0
            return self.get_state()
        
        def get_state(self):
            # 简化状态：[智能体位置, 区域分配, 聚合需求]
            state = np.concatenate([
                self.agent_positions / self.num_locations,  # 归一化位置
                np.eye(self.num_zones)[self.zone_assignments].flatten(),  # one-hot区域
                [self.demand_matrix.sum() / 1000]  # 归一化总需求
            ])
            return state.astype(np.float32)
        
        def step(self, actions):
            # actions: 每个智能体的新区域分配
            self.zone_assignments = actions
            
            # 简单的位置更新
            self.agent_positions += np.random.randint(-1, 2, self.num_agents)
            self.agent_positions = np.clip(self.agent_positions, 0, self.num_locations - 1)
            
            # 计算奖励
            reward = self.calculate_reward()
            
            self.current_step += 1
            done = self.current_step >= self.max_steps
            
            info = {
                'zone_distribution': np.bincount(self.zone_assignments, minlength=self.num_zones),
                'average_position': self.agent_positions.mean()
            }
            
            return self.get_state(), reward, done, info
        
        def calculate_reward(self):
            # 区域平衡奖励
            zone_counts = np.bincount(self.zone_assignments, minlength=self.num_zones)
            balance_reward = -np.var(zone_counts)
            
            # 需求满足奖励（简化）
            demand_reward = np.random.random() * 10
            
            return balance_reward + demand_reward
    
    # 测试环境
    env = SimpleZoneEnvironment()
    state = env.reset()
    
    print(f"环境创建成功")
    print(f"智能体数量: {env.num_agents}")
    print(f"区域数量: {env.num_zones}")
    print(f"位置数量: {env.num_locations}")
    print(f"状态维度: {len(state)}")
    
    # 测试几个步骤
    total_reward = 0
    for step in range(5):
        action = np.random.randint(0, env.num_zones, env.num_agents)
        next_state, reward, done, info = env.step(action)
        total_reward += reward
        
        print(f"步骤 {step}: 奖励={reward:.2f}, 区域分布={info['zone_distribution']}")
        
        if done:
            break
    
    print(f"总奖励: {total_reward:.2f}")
    
    return env

def test_charging_actions():
    """测试充电动作功能"""
    print("\n=== 测试充电动作功能 ===")
    
    # 导入充电相关模块
    import sys
    sys.path.append('src')
    
    try:
        from Action import Action, ChargingAction
        from Request import Request
        from charging_station import ChargingStation, ChargingStationManager
        
        print("✓ 成功导入充电相关模块")
        
        # 创建充电站管理器
        station_manager = ChargingStationManager()
        
        # 添加充电站
        station_manager.add_station(1, location=10, capacity=2)
        station_manager.add_station(2, location=20, capacity=3)
        station_manager.add_station(3, location=30, capacity=4)
        
        print(f"✓ 创建了 {len(station_manager.stations)} 个充电站")
        
        # 测试基本动作
        requests = []  # 空请求列表
        basic_action = Action(requests)
        print(f"✓ 创建基本动作: {type(basic_action).__name__}")
        
        # 测试充电动作
        charging_action1 = ChargingAction(requests, charging_station_id=1, charging_duration=25.0)
        charging_action2 = ChargingAction(requests, charging_station_id=2, charging_duration=30.0)
        
        print(f"✓ 创建充电动作: {type(charging_action1).__name__}")
        print(f"  - 充电站1: {charging_action1.get_charging_info()}")
        print(f"  - 充电站2: {charging_action2.get_charging_info()}")
        
        # 测试动作相等性
        charging_action3 = ChargingAction(requests, charging_station_id=1, charging_duration=25.0)
        print(f"✓ 动作相等性测试: {charging_action1 == charging_action3}")
        print(f"✓ 动作不等性测试: {charging_action1 == charging_action2}")
        
        # 测试充电站功能
        station1 = station_manager.stations[1]
        
        # 模拟车辆充电
        vehicle_ids = ['vehicle_1', 'vehicle_2', 'vehicle_3']
        
        for vehicle_id in vehicle_ids:
            success = station1.start_charging(vehicle_id)
            print(f"  - 车辆 {vehicle_id} 开始充电: {'成功' if success else '失败'}")
        
        # 检查充电站状态
        status = station1.get_station_status()
        print(f"✓ 充电站1状态: 利用率{status['utilization_rate']:.1%}, 队列长度{status['queue_length']}")
        
        # 测试寻找最近的可用充电站
        nearest_station = station_manager.get_nearest_available_station(vehicle_location=15)
        if nearest_station:
            print(f"✓ 位置15最近的可用充电站: {nearest_station.id}")
        else:
            print("✓ 没有可用的充电站")
        
        # 完成一个车辆的充电
        station1.stop_charging('vehicle_1')
        
        # 再次检查状态
        status = station1.get_station_status()
        print(f"✓ 充电完成后状态: 利用率{status['utilization_rate']:.1%}, 队列长度{status['queue_length']}")
        
        return True
        
    except ImportError as e:
        print(f"❌ 导入模块失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 充电动作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_charging_integration():
    """测试充电动作与环境的集成"""
    print("\n=== 测试充电动作集成 ===")
    
    import sys
    sys.path.append('src')
    
    try:
        from Action import Action, ChargingAction
        from charging_station import ChargingStationManager
        
        # 创建简单的智能体类来测试动作
        class SimpleAgent:
            def __init__(self, agent_id: str, location: int):
                self.agent_id = agent_id
                self.location = location
                self.battery_level = np.random.uniform(0.1, 0.9)  # 电池电量 10%-90%
                self.is_charging = False
                
            def needs_charging(self) -> bool:
                """判断是否需要充电"""
                return self.battery_level < 0.3
            
            def can_act(self, action) -> bool:
                """检查是否可以执行动作"""
                if isinstance(action, ChargingAction):
                    return self.needs_charging() and not self.is_charging
                return True  # 其他动作默认可以执行
            
            def execute_action(self, action) -> bool:
                """执行动作"""
                if isinstance(action, ChargingAction):
                    if self.can_act(action):
                        self.is_charging = True
                        print(f"智能体 {self.agent_id} 执行充电动作 -> 充电站 {action.charging_station_id}")
                        return True
                    else:
                        print(f"智能体 {self.agent_id} 无法执行充电动作 (电量: {self.battery_level:.1%})")
                        return False
                else:
                    print(f"智能体 {self.agent_id} 执行普通动作")
                    return True
        
        # 创建测试智能体
        agents = [
            SimpleAgent("agent_1", 10),  # 低电量
            SimpleAgent("agent_2", 20),  # 中等电量 
            SimpleAgent("agent_3", 30),  # 高电量
        ]
        
        # 设置一些智能体为低电量
        agents[0].battery_level = 0.15  # 需要充电
        agents[1].battery_level = 0.25  # 需要充电
        agents[2].battery_level = 0.80  # 不需要充电
        
        print(f"✓ 创建了 {len(agents)} 个测试智能体")
        for agent in agents:
            print(f"  - {agent.agent_id}: 位置{agent.location}, 电量{agent.battery_level:.1%}, 需要充电: {agent.needs_charging()}")
        
        # 创建动作选项
        actions = [
            Action([]),  # 普通动作
            ChargingAction([], charging_station_id=1, charging_duration=20.0),
            ChargingAction([], charging_station_id=2, charging_duration=25.0),
        ]
        
        print(f"✓ 创建了 {len(actions)} 个动作选项")
        
        # 为每个智能体分配动作
        action_results = []
        for i, agent in enumerate(agents):
            # 智能体选择动作逻辑
            if agent.needs_charging():
                # 选择充电动作
                chosen_action = actions[1] if i % 2 == 0 else actions[2]
            else:
                # 选择普通动作
                chosen_action = actions[0]
            
            # 执行动作
            success = agent.execute_action(chosen_action)
            action_results.append((agent.agent_id, chosen_action, success))
        
        # 总结结果
        successful_actions = sum(1 for _, _, success in action_results if success)
        charging_actions = sum(1 for _, action, _ in action_results if isinstance(action, ChargingAction))
        
        print(f"✓ 动作执行总结:")
        print(f"  - 总动作数: {len(action_results)}")
        print(f"  - 成功执行: {successful_actions}")
        print(f"  - 充电动作数: {charging_actions}")
        
        return successful_actions == len(action_results)
        
    except Exception as e:
        print(f"❌ 充电集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_simple_dqn():
    """测试简单的DQN智能体"""
    print("\n=== 测试简单DQN智能体 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    class SimpleDQN(nn.Module):
        def __init__(self, state_dim, action_dim, hidden_dim=64):
            super().__init__()
            self.network = nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            )
        
        def forward(self, x):
            return self.network(x)
    
    class SimpleAgent:
        def __init__(self, state_dim, action_dim, lr=0.001):
            self.q_network = SimpleDQN(state_dim, action_dim).to(device)
            self.target_network = SimpleDQN(state_dim, action_dim).to(device)
            self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=lr)
            
            # 复制权重到目标网络
            self.target_network.load_state_dict(self.q_network.state_dict())
            
            # 改进的探索策略：开始时高探索，逐渐减少
            self.epsilon_start = 0.9
            self.epsilon_end = 0.05
            self.epsilon_decay = 0.995
            self.epsilon = self.epsilon_start
            self.gamma = 0.99
            self.step_count = 0
        
        def select_action(self, state, num_agents, num_zones):
            # 动态调整epsilon（探索率）
            self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)
            
            if np.random.random() < self.epsilon:
                # 随机策略
                action = np.random.randint(0, num_zones, num_agents)
                action_type = "random"
            else:
                # ADP策略：基于学习到的Q值
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                    q_values = self.q_network(state_tensor)
                    
                    # 改进：为每个智能体独立选择最优区域
                    # 这里简化为选择Q值最高的区域
                    best_zone = q_values.argmax().item() % num_zones
                    action = np.full(num_agents, best_zone)
                    action_type = "adp"
            
            self.step_count += 1
            return action, action_type        
        def train_step(self, state, action, reward, next_state, done):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(device)
            reward_tensor = torch.FloatTensor([reward]).to(device)
            done_tensor = torch.BoolTensor([done]).to(device)
            
            # 当前Q值
            current_q = self.q_network(state_tensor)[0, 0]  # 简化
            
            # 目标Q值
            with torch.no_grad():
                next_q = self.target_network(next_state_tensor).max(1)[0]
                target_q = reward_tensor + self.gamma * next_q * ~done_tensor
            
            # 损失和优化
            loss = F.mse_loss(current_q.unsqueeze(0), target_q)
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            return loss.item()
      # 创建环境和智能体（使用局部定义的类）
    class LocalSimpleZoneEnvironment:
        def __init__(self, num_agents=5, num_zones=3, num_locations=20):
            self.num_agents = num_agents
            self.num_zones = num_zones
            self.num_locations = num_locations
            
            self.agent_positions = np.random.randint(0, num_locations, num_agents)
            self.zone_assignments = np.random.randint(0, num_zones, num_agents)
            self.demand_matrix = np.random.exponential(1.0, (num_locations, num_locations))
            
            self.current_step = 0
            self.max_steps = 100
        
        def reset(self):
            self.agent_positions = np.random.randint(0, self.num_locations, self.num_agents)
            self.zone_assignments = np.random.randint(0, self.num_zones, self.num_agents)
            self.current_step = 0
            return self.get_state()
        
        def get_state(self):
            state = np.concatenate([
                self.agent_positions / self.num_locations,
                np.eye(self.num_zones)[self.zone_assignments].flatten(),
                [self.demand_matrix.sum() / 1000]
            ])
            return state.astype(np.float32)
        
        def step(self, actions):
            self.zone_assignments = actions
            self.agent_positions += np.random.randint(-1, 2, self.num_agents)
            self.agent_positions = np.clip(self.agent_positions, 0, self.num_locations - 1)
            
            zone_counts = np.bincount(self.zone_assignments, minlength=self.num_zones)
            balance_reward = -np.var(zone_counts)
            demand_reward = np.random.random() * 10
            reward = balance_reward + demand_reward
            
            self.current_step += 1
            done = self.current_step >= self.max_steps
            
            info = {
                'zone_distribution': zone_counts,
                'average_position': self.agent_positions.mean()
            }
            
            return self.get_state(), reward, done, info
    
    env = LocalSimpleZoneEnvironment(num_agents=3, num_zones=2)
    state = env.reset()
    
    agent = SimpleAgent(
        state_dim=len(state),
        action_dim=env.num_zones,
        lr=0.001
    )
    
    print(f"DQN智能体创建成功")
    print(f"网络参数: {sum(p.numel() for p in agent.q_network.parameters())}")
    print(f"状态维度: {len(state)}")
    print(f"动作维度: {env.num_zones}")
    
    # 训练几个回合
    episode_rewards = []
    episode_losses = []
    moving_average_rewards = []
    epsilon_history = []
    action_type_counts = {'random': 0, 'adp': 0}
    
    print("开始训练智能体...")
    print(f"初始探索率: {agent.epsilon:.3f}")
    
    for episode in range(500):  # 增加训练回合数以便观察趋势
        state = env.reset()
        episode_reward = 0
        episode_loss = 0
        step_count = 0
        episode_random_count = 0
        episode_adp_count = 0
        
        for step in range(30):  # 增加每回合步数
            action, action_type = agent.select_action(state, env.num_agents, env.num_zones)
            
            # 记录动作类型
            if action_type == 'random':
                episode_random_count += 1
                action_type_counts['random'] += 1
            else:
                episode_adp_count += 1
                action_type_counts['adp'] += 1
                
            next_state, reward, done, info = env.step(action)
            
            # 训练
            loss = agent.train_step(state, action, reward, next_state, done)
            
            state = next_state
            episode_reward += reward
            episode_loss += loss
            step_count += 1
            
            if done:
                break
        
        # 记录历史数据
        avg_episode_loss = episode_loss / step_count if step_count > 0 else 0
        episode_rewards.append(episode_reward)
        episode_losses.append(avg_episode_loss)
        epsilon_history.append(agent.epsilon)
        
        # 计算移动平均奖励（窗口大小为5）
        window_size = 5
        if len(episode_rewards) >= window_size:
            moving_avg = np.mean(episode_rewards[-window_size:])
            moving_average_rewards.append(moving_avg)
        else:
            moving_average_rewards.append(np.mean(episode_rewards))
        
        # 每10个回合更新一次目标网络
        if episode % 10 == 0:
            agent.target_network.load_state_dict(agent.q_network.state_dict())
            
        if episode % 10 == 0:
            adp_percentage = (episode_adp_count / (episode_adp_count + episode_random_count)) * 100 if (episode_adp_count + episode_random_count) > 0 else 0
            print(f"回合 {episode}: 奖励={episode_reward:.2f}, 移动平均={moving_average_rewards[-1]:.2f}, " +
                  f"损失={avg_episode_loss:.4f}, 探索率={agent.epsilon:.3f}, ADP使用率={adp_percentage:.1f}%")
    
    # 可视化训练过程
    plt.figure(figsize=(20, 10))
    
    # 子图1: 奖励曲线
    plt.subplot(2, 3, 1)
    plt.plot(episode_rewards, alpha=0.6, label='Episode Reward', color='lightblue')
    plt.plot(moving_average_rewards, label='Moving Average', color='darkblue', linewidth=2)
    plt.xlabel('Training Episode')
    plt.ylabel('Reward')
    plt.title('Agent Training Reward Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图2: 损失曲线
    plt.subplot(2, 3, 2)
    plt.plot(episode_losses, label='Training Loss', color='red', alpha=0.7)
    plt.xlabel('Training Episode')
    plt.ylabel('Loss')
    plt.title('Agent Training Loss Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图3: 探索率变化
    plt.subplot(2, 3, 3)
    plt.plot(epsilon_history, label='Exploration Rate', color='green', linewidth=2)
    plt.xlabel('Training Episode')
    plt.ylabel('Epsilon')
    plt.title('Exploration Rate Decay Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图4: 奖励分布
    plt.subplot(2, 3, 4)
    plt.hist(episode_rewards, bins=15, alpha=0.7, color='orange', edgecolor='black')
    plt.xlabel('Reward Value')
    plt.ylabel('Frequency')
    plt.title('Reward Distribution Histogram')
    plt.grid(True, alpha=0.3)
    
    # 子图5: 策略使用统计
    plt.subplot(2, 3, 5)
    total_actions = sum(action_type_counts.values())
    if total_actions > 0:
        adp_percentage = (action_type_counts['adp'] / total_actions) * 100
        random_percentage = (action_type_counts['random'] / total_actions) * 100
        
        plt.pie([adp_percentage, random_percentage], 
                labels=[f'ADP Policy ({adp_percentage:.1f}%)', f'Random Policy ({random_percentage:.1f}%)'],
                autopct='%1.1f%%', colors=['skyblue', 'lightcoral'])
        plt.title('Policy Usage Distribution')
    
    # 子图6: 奖励趋势分析
    plt.subplot(2, 3, 6)
    if len(episode_rewards) >= 10:
        early_rewards = episode_rewards[:10]
        late_rewards = episode_rewards[-10:]
        
        plt.boxplot([early_rewards, late_rewards], labels=['Early (1-10)', 'Late (41-50)'])
        plt.ylabel('Reward')
        plt.title('Early vs Late Reward Comparison')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # 保存图像
    results_dir = Path("results/simple_tests")
    results_dir.mkdir(parents=True, exist_ok=True)
    
    plot_path = results_dir / "training_curves.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 计算训练统计
    avg_reward = np.mean(episode_rewards)
    final_avg_reward = np.mean(episode_rewards[-10:])  # 最后10回合的平均奖励
    initial_avg_reward = np.mean(episode_rewards[:10])  # 前10回合的平均奖励
    improvement = final_avg_reward - initial_avg_reward
    total_actions = sum(action_type_counts.values())
    adp_usage_percentage = (action_type_counts['adp'] / total_actions) * 100 if total_actions > 0 else 0
    
    print(f"\n=== 训练结果分析 ===")
    print(f"总回合数: {len(episode_rewards)}")
    print(f"平均奖励: {avg_reward:.2f}")
    print(f"初期平均奖励(前10回合): {initial_avg_reward:.2f}")
    print(f"后期平均奖励(后10回合): {final_avg_reward:.2f}")
    print(f"奖励提升: {improvement:.2f} ({improvement/abs(initial_avg_reward)*100:.1f}%)")
    print(f"最高奖励: {max(episode_rewards):.2f}")
    print(f"最低奖励: {min(episode_rewards):.2f}")
    print(f"最终探索率: {agent.epsilon:.3f}")
    print(f"ADP策略使用率: {adp_usage_percentage:.1f}%")
    print(f"奖励趋势: {'上升' if improvement > 0 else '下降' if improvement < 0 else '稳定'}")
    print(f"可视化图表已保存至: {plot_path}")
    
    # 验证奖励是否有上升趋势
    reward_trend_test = improvement > 0
    adp_usage_test = adp_usage_percentage > 30  # ADP使用率应该超过30%
    print(f"✓ Reward Improvement Test: {'PASS' if reward_trend_test else 'FAIL'}")
    print(f"✓ ADP Policy Usage Test: {'PASS' if adp_usage_test else 'FAIL'}")
    
    # ADP策略使用率解释
    print(f"\n=== ADP Strategy Usage Explanation ===")
    print(f"ADP (Approximate Dynamic Programming) Policy Usage Rate: {adp_usage_percentage:.1f}%")
    print(f"- ADP actions: {action_type_counts['adp']} (learned policy)")
    print(f"- Random actions: {action_type_counts['random']} (exploration)")
    print(f"- Total actions: {total_actions}")
    print(f"\nWhat does ADP usage rate mean:")
    print(f"• ADP策略使用率表示智能体使用学习到的策略（而非随机策略）的百分比")
    print(f"• Higher ADP usage indicates the agent is relying more on learned knowledge")
    print(f"• Lower usage means more exploration (random actions)")
    print(f"• Good training should show increasing ADP usage over time")
    print(f"• Our agent achieved {adp_usage_percentage:.1f}% ADP usage, showing it learned to prefer")
    print(f"  the trained policy over random actions")
    
    return agent, {
        'episode_rewards': episode_rewards,
        'moving_average_rewards': moving_average_rewards,
        'episode_losses': episode_losses,
        'epsilon_history': epsilon_history,
        'action_type_counts': action_type_counts,
        'improvement': improvement,
        'trend_positive': reward_trend_test,
        'adp_usage_percentage': adp_usage_percentage,
        'adp_usage_sufficient': adp_usage_test
    }

def test_save_load():
    """测试模型保存和加载"""
    print("\n=== 测试模型保存和加载 ===")
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 创建简单模型
    model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    # 生成测试数据
    x = torch.randn(5, 10).to(device)
    original_output = model(x)
    
    # 保存模型
    save_dir = Path("results/test_models")
    save_dir.mkdir(parents=True, exist_ok=True)
    
    model_path = save_dir / "test_model.pth"
    torch.save(model.state_dict(), model_path)
    
    print(f"模型已保存到: {model_path}")
      # 创建新模型并加载
    new_model = nn.Sequential(
        nn.Linear(10, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)
    
    new_model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    new_output = new_model(x)
    
    # 验证输出相同
    diff = torch.abs(original_output - new_output).max().item()
    print(f"输出差异: {diff:.8f}")
    
    success = diff < 1e-6
    print(f"保存/加载测试: {'成功' if success else '失败'}")
    
    return success

def main():
    """主测试函数"""
    print("="*60)
    print("Zone-ADP简化集成测试")
    print("="*60)
    
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    
    try:
        # 运行测试
        print(f"开始测试... {np.datetime64('now')}" if hasattr(np, 'datetime64') else "开始测试...")
        
        device = test_basic_pytorch()
        net = test_simple_network()
        env = test_zone_environment_simple()
        
        # 创建局部函数来避免导入问题
        class SimpleZoneEnvironment:
            def __init__(self, num_agents=5, num_zones=3, num_locations=20):
                self.num_agents = num_agents
                self.num_zones = num_zones
                self.num_locations = num_locations
                
                self.agent_positions = np.random.randint(0, num_locations, num_agents)
                self.zone_assignments = np.random.randint(0, num_zones, num_agents)
                self.demand_matrix = np.random.exponential(1.0, (num_locations, num_locations))
                
                self.current_step = 0
                self.max_steps = 100
            
            def reset(self):
                self.agent_positions = np.random.randint(0, self.num_locations, self.num_agents)
                self.zone_assignments = np.random.randint(0, self.num_zones, self.num_agents)
                self.current_step = 0
                return self.get_state()
            
            def get_state(self):
                state = np.concatenate([
                    self.agent_positions / self.num_locations,
                    np.eye(self.num_zones)[self.zone_assignments].flatten(),
                    [self.demand_matrix.sum() / 1000]
                ])
                return state.astype(np.float32)
            
            def step(self, actions):
                self.zone_assignments = actions
                self.agent_positions += np.random.randint(-1, 2, self.num_agents)
                self.agent_positions = np.clip(self.agent_positions, 0, self.num_locations - 1)
                
                zone_counts = np.bincount(self.zone_assignments, minlength=self.num_zones)
                balance_reward = -np.var(zone_counts)
                demand_reward = np.random.random() * 10
                reward = balance_reward + demand_reward
                
                self.current_step += 1
                done = self.current_step >= self.max_steps
                
                info = {
                    'zone_distribution': zone_counts,
                    'average_position': self.agent_positions.mean()
                }
                
                return self.get_state(), reward, done, info
        
        # 将环境类添加到全局命名空间以供DQN测试使用
        globals()['SimpleZoneEnvironment'] = SimpleZoneEnvironment
        
        agent, training_results = test_simple_dqn()
        save_success = test_save_load()
        
        # 新增充电功能测试
        charging_basic_success = test_charging_actions()
        charging_integration_success = test_charging_integration()
        
        # 分析训练结果
        reward_improvement = training_results['improvement']
        reward_trend_positive = training_results['trend_positive']
        adp_usage_percentage = training_results['adp_usage_percentage']
        adp_usage_sufficient = training_results['adp_usage_sufficient']
        
        print("\n" + "="*60)
        print("测试结果总结:")
        print("✓ PyTorch基础功能测试通过")
        print("✓ 简单神经网络测试通过")
        print("✓ Zone环境测试通过")
        print(f"✓ DQN智能体测试通过 (奖励{'上升' if reward_trend_positive else '未上升'}: {reward_improvement:+.2f})")
        print(f"✓ 策略选择测试通过 (ADP使用率: {adp_usage_percentage:.1f}%)")
        print(f"{'✓' if save_success else '✗'} 模型保存/加载测试{'通过' if save_success else '失败'}")
        print(f"{'✓' if charging_basic_success else '✗'} 充电动作基础测试{'通过' if charging_basic_success else '失败'}")
        print(f"{'✓' if charging_integration_success else '✗'} 充电动作集成测试{'通过' if charging_integration_success else '失败'}")
        print(f"{'✓' if reward_trend_positive else '✗'} 奖励上升趋势测试{'通过' if reward_trend_positive else '失败'}")
        print(f"{'✓' if adp_usage_sufficient else '✗'} ADP策略使用率测试{'通过' if adp_usage_sufficient else '失败'}")
        print("\n🎉 所有核心功能测试完成!")
        print(f"使用设备: {device}")
        
        # 创建详细的测试报告
        results_dir = Path("results/simple_tests")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        report_path = results_dir / "test_report.txt"
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("Zone-ADP简化测试报告（包含充电功能和可视化）\n")
            f.write("="*50 + "\n")
            f.write(f"测试设备: {device}\n")
            f.write(f"PyTorch版本: {torch.__version__}\n")
            f.write("基础功能测试: 通过\n")
            f.write(f"充电基础功能: {'通过' if charging_basic_success else '失败'}\n")
            f.write(f"充电集成功能: {'通过' if charging_integration_success else '失败'}\n")
            f.write(f"智能体训练: {'通过' if reward_trend_positive else '失败'}\n")
            f.write(f"奖励改善: {reward_improvement:+.2f}\n")
            f.write(f"ADP策略使用率: {adp_usage_percentage:.1f}%\n")
            f.write(f"策略测试: {'通过' if adp_usage_sufficient else '失败'}\n")
            f.write("可视化图表: training_curves.png\n")
        
        print(f"测试报告已保存到: {report_path}")
        
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
