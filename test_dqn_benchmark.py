"""
DQN Benchmark Test: Compare a simple DQN policy against the integrated environment
Mimics the structure and logging style of test_integrated_charging.py
"""
import os
import random
import numpy as np
import torch
from pathlib import Path
import pandas as pd
# Pretty prints
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Environment + DQN utilities
from src.Environment import ChargingIntegratedEnvironment
from src.ValueFunction_pytorch import DQNAgent, create_dqn_state_features

# Optional visualization helpers (only used for font settings)
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


def set_random_seeds(seed: int = 42):
    """Set all RNG seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"✓ Random seeds set to {seed} (Python/NumPy/PyTorch)")


from typing import Optional

def test_time_performance_dqn(carnumlist,iftense = True):
    time_list = []
    for carnum in carnumlist:
        import time
        start_time = time.time()
        run_dqn_benchmark(num_episodes=100, episode_length=200, num_vehicles=carnum, num_stations=4, use_intense_requests=iftense, save_results=False)
        end_time = time.time()
        print(f"DQN Benchmark with {carnum} vehicles took {end_time - start_time:.2f} seconds.")
        time_list.append(end_time - start_time)
    np.save(f'dqn_result/dqn_time_performance_{"intense" if iftense else "normal"}.npy', time_list, allow_pickle=True)
    return time_list


def run_dqn_benchmark(num_episodes: int = 3, episode_length: Optional[int] = None,
                      num_vehicles: int = 10, num_stations: int = 4,
                      use_intense_requests: bool = True,
                      save_results: bool = True,
                      # DQN training config
                      train_steps_per_tick: int = 1,
                      train_steps_per_episode: int = 2,
                      batch_size: int = 256,
                      warmup_steps: int = 200,
                      save_model_every: int = 0,
                      model_dir: Optional[str] = None):
    """
    Run a compact DQN benchmark over the ChargingIntegratedEnvironment.
    Follows the test_integrated_charging structure: setup -> loop episodes -> print stats -> save.
    """
    print("=== Starting DQN Benchmark Test ===")
    set_random_seeds(42)

    # Build environment
    env = ChargingIntegratedEnvironment(
        num_vehicles=num_vehicles,
        num_stations=num_stations,
        random_seed=42
    )
    if episode_length is not None:
        env.episode_length = int(episode_length)
    env.use_intense_requests = use_intense_requests

    # Create a vanilla DQN Agent
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dqn_agent = DQNAgent(state_dim=64, action_dim=32, device=device)
    print(f"✓ DQN agent initialized on {device}")

    # Results containers
    results = {
        'episode_rewards': [],
        'episode_completed_orders': [],
        'episode_stats': [],
        'per_step_metrics': []
    }

    total_steps = 0
    for ep in range(num_episodes):
        # Per-episode request sequence seed for controlled variability
        env.set_request_generation_seed(32 + ep)
        env.reset()
        print(f"Episode {ep+1}/{num_episodes} — request seed = {32 + ep}")

        episode_reward = 0.0
        episode_losses = []
        step_log_every = max(1, env.episode_length // 8)

        for step in range(env.episode_length):
            # Use current active requests snapshot when mapping assign actions
            current_requests = list(env.active_requests.values())

            # Run one DQN decision tick (includes internal environment update and one internal train step)
            sim = env.simulate_motion_dqn(dqn_agent=dqn_agent, current_requests=current_requests, training=True)
            if sim is None:
                raise RuntimeError("simulate_motion_dqn returned None — ensure DQN classes are available.")

            episode_reward += sim.get('total_reward', 0.0)
            total_steps += 1


            # Periodic logging
            if step % step_log_every == 0:
                util = sim.get('vehicle_utilization', 0.0)
                comp = sim.get('request_completion_rate', 0.0)
                avg_bat = sim.get('average_battery_level', 0.0)
                dist = sim.get('action_distribution', {})
                print(f"  Step {step:3d}: Util={util:.2f}, Complete={comp:.2f}, Bat={avg_bat:.2f}, Dist={dist}")


            if step % 2 == 0:
                if len(dqn_agent.memory) >= max(batch_size, 1) :
                    loss = dqn_agent.train_step(batch_size=batch_size)
                    if loss is not None:
                        episode_losses.append(loss)



            # Collect per-step metrics if needed
            results['per_step_metrics'].append({
                'episode': ep + 1,
                'step': step,
                'utilization': sim.get('vehicle_utilization', 0.0),
                'completion_rate': sim.get('request_completion_rate', 0.0),
                'avg_battery': sim.get('average_battery_level', 0.0),
            })

        # Episode-level stats from environment
        stats = env.get_episode_stats()
        results['episode_rewards'].append(episode_reward)
        results['episode_completed_orders'].append(stats.get('completed_orders', 0))
        # 采样 DQN 的 idle/assign/charge 三类 Q 值（取各自动作区间的最大值）
        try:
            sample_vid = 0 if env.vehicles else 0
            sample_state = create_dqn_state_features(env, sample_vid, current_time=getattr(env, 'current_time', 0))
            with torch.no_grad():
                q_all = dqn_agent.policy_net(
                    sample_state['vehicle'].unsqueeze(0),
                    sample_state['request'].unsqueeze(0),
                    sample_state['global'].unsqueeze(0)
                ).squeeze(0)
            ad = getattr(dqn_agent, 'action_dim', 32)
            # 区间安全裁剪
            def max_q(start, end):
                s = max(0, min(start, ad))
                e = max(0, min(end, ad))
                if e <= s:
                    return float('nan')
                return float(q_all[s:e].max().item())
            sample_assign_q = max_q(0, 10)      # 0-9: assign
            sample_charge_q = max_q(20, 25)     # 20-24: charge
            sample_idle_q = max_q(28, 32)       # 28-31: idle
            stats['sample_idle_q_value'] = sample_idle_q
            stats['sample_assign_q_value'] = sample_assign_q
            stats['sample_charge_q_value'] = sample_charge_q
        except Exception as e:
            # 不阻塞主流程，记录为 NaN 并打印一次性提示
            stats['sample_idle_q_value'] = float('nan')
            stats['sample_assign_q_value'] = float('nan')
            stats['sample_charge_q_value'] = float('nan')
            print(f"! Q-value sampling failed this episode: {e}")

        # 汇总并记录本轮平均训练损失
        avg_loss = float(np.mean(episode_losses)) if episode_losses else float('nan')
        stats['episode_reward'] = episode_reward
        stats['episode_number'] = ep + 1
        stats['episode_loss'] = avg_loss
        results['episode_stats'].append(stats)

        print(f"Episode {ep+1} done: Reward={episode_reward:.2f}, Completed={stats.get('completed_orders', 0)}, Battery={stats.get('avg_battery_level', 0.0):.2f}, AvgLoss={avg_loss:.6f}, Qs(idle/assign/charge)={stats['sample_idle_q_value']:.3f}/{stats['sample_assign_q_value']:.3f}/{stats['sample_charge_q_value']:.3f}")
        pd_result = pd.DataFrame(results['episode_stats'])
        pd_result.to_excel(f'dqn_result/dqn_benchmark_episode_stats_{use_intense_requests}_{num_vehicles}_{num_stations}.xlsx', index=False)
        # Optional checkpointing
        if save_model_every and (ep + 1) % int(save_model_every) == 0 and hasattr(dqn_agent, 'save_model'):
            ckpt_dir = Path(model_dir or "results/dqn_tests/models")
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = ckpt_dir / f"dqn_ep{ep+1}.pt"
            try:
                dqn_agent.save_model(str(ckpt_path))
                print(f"✓ Saved DQN checkpoint: {ckpt_path}")
            except Exception as e:
                print(f"! Failed to save DQN checkpoint: {e}")

    print("\n=== DQN Benchmark Complete ===")
    print(f"Episodes: {num_episodes}")
    print(f"Avg reward: {np.mean(results['episode_rewards']):.2f}")
    print(f"Avg completed: {np.mean(results['episode_completed_orders']):.2f}")

    if save_results:
        out_dir = Path("results/dqn_tests")
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "dqn_results.npy", results, allow_pickle=True)
        print(f"✓ Results saved to {out_dir / 'dqn_results.npy'}")

    return results, env


if __name__ == "__main__":
    # Quick local run
    # carlist = [(i+1)*5 for i in range(5)]
    # test_time_performance_dqn(carlist,iftense = False)
    run_dqn_benchmark(num_episodes=100, episode_length=200, num_vehicles=10, num_stations=4, use_intense_requests=True)
    run_dqn_benchmark(num_episodes=100, episode_length=200, num_vehicles=10, num_stations=4, use_intense_requests=False)