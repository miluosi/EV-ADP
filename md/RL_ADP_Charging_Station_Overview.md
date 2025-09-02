# RL-ADP Charging Station Management System

## Project Overview

This project implements an intelligent charging station management system based on Reinforcement Learning and Approximate Dynamic Programming (RL-ADP), designed to optimize charging scheduling and service allocation for Electric Vehicles (EV) and Autonomous Electric Vehicles (AEV).

### Core Features

- **Dynamic Charging Station Generation**: Automatically generates specified number of charging stations based on input parameters
- **Vehicle Type Classification**: Supports EV (70%) and AEV (30%) vehicle types
- **Intelligent Rejection Mechanism**: EV vehicles with distance-based exponential rejection probability
- **Gurobi Optimization Integration**: Linear programming optimization combined with Q-learning
- **Real-time Statistical Analysis**: Complete episode statistics and Excel export functionality

## RL-ADP Algorithm Principles

### Approximate Dynamic Programming (ADP)

Approximate Dynamic Programming is an effective method for solving large-scale dynamic programming problems, working through:

1. **State Representation**: Abstract system states into feature vectors
2. **Value Function Approximation**: Approximate value functions using neural networks or tabular methods
3. **Policy Improvement**: Improve decisions through learning optimal policies

### Q-Learning Integration

Q-learning is a model-free reinforcement learning algorithm：

```python
# Q-value update formula
Q(s, a) = Q(s, a) + α[r + γ * max(Q(s', a')) - Q(s, a)]
```

Where:
- `s`: Current state
- `a`: Action taken
- `r`: Immediate reward
- `α`: Learning rate (0.1)
- `γ`: Discount factor (0.9)

## System Architecture

### ChargingIntegratedEnvironment

Core environment class, inheriting from base Environment class：

```python
class ChargingIntegratedEnvironment(Environment):
    def __init__(self, num_vehicles=5, num_stations=3, grid_size=10):
        # Initialize parameters
        # Dynamically generate charging stations
        # Set up vehicle type distribution
```

#### Main Components

1. **Charging Station Manager (ChargingStationManager)**
   - Dynamic charging station generation
   - Unified capacity set to 10
   - Real-time occupancy monitoring

2. **Vehicle Management System**
   - EV/AEV type classification
   - Battery state tracking
   - Location and task management

3. **Request Processing System**
   - Passenger request generation
   - Charging demand identification
   - Intelligent allocation mechanism

### Vehicle Types and Rejection Mechanism

#### Vehicle Type Distribution

- **EV (Electric Vehicle)**: 70% - Traditional electric vehicles
- **AEV (Autonomous Electric Vehicle)**: 30% - Autonomous electric vehicles

#### Rejection Probability Calculation

EV vehicles with distance-based exponential rejection probability：

```python
def _calculate_rejection_probability(self, vehicle_id, request):
    if vehicle['type'] == 'AEV':
        return 0.0  # AEV never rejects

    distance = manhattan_distance(vehicle_pos, pickup_pos)
    base_rate = 0.1
    distance_factor = 0.3

    rejection_prob = base_rate * exp(distance * distance_factor)
    return min(0.9, rejection_prob)  # Maximum 90% rejection rate
```

## Gurobi Optimization Integration

### Optimization Objective

Combining distance cost and Q-learning value：

```python
# Objective function
distance_weight = 0.7
q_weight = 0.3

obj = distance_weight * distance_cost + q_weight * (-q_value_benefit)
model.setObjective(obj, GRB.MINIMIZE)
```

### Constraints

1. **Single Vehicle Single Target**: Each vehicle can only be assigned to one target
2. **Target Capacity Limit**: Limited number of vehicles each target can serve
3. **Continuous Variables**: Using 0-1 continuous variables for assignment

### Q-Value Retrieval Method

```python
def get_assignment_q_value(self, agent_id, target_id, agent_pos, target_pos):
    state = self.get_state_representation(agent_pos, target_pos, current_time)
    action = f"assign_{target_id}"
    return self.get_q_value(state, action)
```

## Q-Learning Implementation

### State Representation

State vector includes:
- Vehicle position (x, y coordinates)
- Target position (pickup, dropoff)
- Current time
- Battery state

### Action Space

- Accept request
- Reject request
- Charging action
- Movement action

### Reward Function

```python
def _execute_action(self, vehicle_id, action):
    if isinstance(action, ChargingAction):
        # Charging rewards
        if vehicle['battery'] < 0.2:
            reward = 5.0  # Emergency charging bonus
        elif vehicle['battery'] < 0.3:
            reward = 4.0  # Low battery bonus

    elif isinstance(action, ServiceAction):
        # Service rewards
        if pickup_successful:
            reward = 3.0  # Successful pickup
        if dropoff_successful:
            reward = earnings + 5.0  # Order completion + bonus
```

## Project Setup and Running

### Prerequisites

```bash
# Required Python packages
pip install gurobipy numpy matplotlib pandas torch scikit-learn

# Note: Gurobi requires a valid license
# Academic licenses are available for free at: https://www.gurobi.com/academia/academic-program-and-licenses/
```

### Basic Usage

#### 1. Import Required Modules

```python
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# Import project modules
from src.Environment import ChargingIntegratedEnvironment
from src.LearningAgent import LearningAgent
from src.Action import Action, ChargingAction, ServiceAction
from src.Request import Request
from src.charging_station import ChargingStationManager
```

#### 2. Initialize Environment

```python
# Create integrated charging environment
env = ChargingIntegratedEnvironment(
    num_vehicles=5,      # Number of vehicles
    num_stations=3,      # Number of charging stations
    grid_size=10         # Grid size for simulation
)

# Initialize environment
initial_states = env.reset()
print(f"Environment initialized with {len(initial_states)} vehicles")
```

#### 3. Run Simulation Episode

```python
def run_episode(env, max_steps=100):
    """Run a single episode simulation"""
    states = env.reset()
    total_rewards = {vid: 0 for vid in states.keys()}
    episode_data = []

    for step in range(max_steps):
        actions = {}

        # Generate actions for each vehicle
        for vehicle_id in states.keys():
            # Simple random action selection (can be replaced with learned policy)
            if np.random.random() < 0.3:
                # Charging action
                station_id = np.random.randint(1, env.num_stations + 1)
                actions[vehicle_id] = ChargingAction(station_id, duration=5)
            else:
                # Random movement or service action
                actions[vehicle_id] = Action()  # Default movement action

        # Execute actions
        next_states, rewards, done, info = env.step(actions)

        # Accumulate rewards
        for vid in rewards:
            total_rewards[vid] += rewards[vid]

        # Record step data
        step_stats = env.get_episode_stats()
        step_stats['step'] = step
        episode_data.append(step_stats)

        states = next_states

        if done:
            break

    return total_rewards, episode_data

# Run simulation
rewards, episode_data = run_episode(env, max_steps=100)
print(f"Episode completed. Total rewards: {rewards}")
```

#### 4. Advanced ADP Agent Usage

```python
from test_integrated_charging import ADPAgent

# Initialize ADP agent
state_dim = 5  # State vector dimension
action_dim = 10  # Number of possible actions
agent = ADPAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    learning_rate=0.001,
    epsilon=0.1,
    gamma=0.95
)

# Training loop
num_episodes = 1000
for episode in range(num_episodes):
    states = env.reset()
    episode_reward = 0

    for step in range(100):
        actions = {}

        # Agent selects actions
        for vehicle_id, state in states.items():
            action_idx = agent.select_action(state)
            # Convert action index to actual action
            actions[vehicle_id] = convert_action_index_to_action(action_idx)

        # Execute actions
        next_states, rewards, done, _ = env.step(actions)

        # Agent learns
        for vehicle_id in states:
            agent.update(states[vehicle_id], action_idx, rewards[vehicle_id],
                        next_states[vehicle_id], done)

        states = next_states
        episode_reward += sum(rewards.values())

        if done:
            break

    print(f"Episode {episode + 1}/{num_episodes}, Reward: {episode_reward:.2f}")
```

#### 5. Statistical Analysis and Export

```python
# Get comprehensive episode statistics
final_stats = env.get_episode_stats()

print("=== Episode Statistics ===")
print(f"Total Orders: {final_stats['total_orders']}")
print(f"Accepted Orders: {final_stats['accepted_orders']}")
print(f"Rejected Orders: {final_stats['rejected_orders']}")
print(f"Average Battery Level: {final_stats['avg_battery_level']:.3f}")
print(f"Average Vehicles per Station: {final_stats['avg_vehicles_per_station']:.2f}")
print(f"Station Utilization: {final_stats['station_utilization_rate']:.3f}")

# Export to Excel
def save_episode_statistics_to_excel(env, filename=None):
    """Save episode statistics to Excel file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"results/episode_statistics_{timestamp}.xlsx"

    stats = env.get_episode_stats()

    # Prepare data for Excel
    data = {
        'Metric': [
            'Total Orders', 'Accepted Orders', 'Rejected Orders',
            'Active Orders', 'Completed Orders', 'Average Battery Level',
            'Average Vehicles per Station', 'Station Utilization Rate',
            'Total Vehicles', 'EV Count', 'AEV Count',
            'EV Rejected', 'AEV Rejected', 'Total Stations',
            'Vehicles Charging', 'Total Earnings'
        ],
        'Value': [
            stats['total_orders'], stats['accepted_orders'], stats['rejected_orders'],
            stats['active_orders'], stats['completed_orders'], stats['avg_battery_level'],
            stats['avg_vehicles_per_station'], stats['station_utilization_rate'],
            stats['total_vehicles'], stats['ev_count'], stats['aev_count'],
            stats['ev_rejected'], stats['aev_rejected'], stats['total_stations'],
            stats['vehicles_charging'], stats['total_earnings']
        ]
    }

    df = pd.DataFrame(data)
    df.to_excel(filename, index=False)
    print(f"Statistics saved to {filename}")

# Save statistics
save_episode_statistics_to_excel(env)
```

#### 6. Visualization

```python
def plot_episode_results(episode_data):
    """Plot episode results"""
    steps = [d['step'] for d in episode_data]
    battery_levels = [d['avg_battery_level'] for d in episode_data]
    active_orders = [d['active_orders'] for d in episode_data]
    completed_orders = [d['completed_orders'] for d in episode_data]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 8))

    # Battery levels over time
    ax1.plot(steps, battery_levels)
    ax1.set_title('Average Battery Level Over Time')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Battery Level')
    ax1.grid(True)

    # Order statistics
    ax2.plot(steps, active_orders, label='Active Orders')
    ax2.plot(steps, completed_orders, label='Completed Orders')
    ax2.set_title('Order Statistics')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Number of Orders')
    ax2.legend()
    ax2.grid(True)

    # Station utilization
    station_util = [d['station_utilization_rate'] for d in episode_data]
    ax3.plot(steps, station_util)
    ax3.set_title('Station Utilization Rate')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Utilization Rate')
    ax3.grid(True)

    # Vehicles per station
    vehicles_per_station = [d['avg_vehicles_per_station'] for d in episode_data]
    ax4.plot(steps, vehicles_per_station)
    ax4.set_title('Average Vehicles per Station')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Vehicles per Station')
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig('results/episode_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

# Plot results
plot_episode_results(episode_data)
```

## Experimental Design

### Test Scenarios

1. **Integrated Test**: Comprehensive charging and service scheduling
2. **Zoning Test**: Optimization strategies for different regions
3. **Simple Test**: Basic functionality verification

### Performance Metrics

- **Order Completion Rate**: Proportion of successfully completed passenger requests
- **Rejection Rate**: Request rejection rates for different vehicle types
- **Battery Utilization**: Average battery levels and charging efficiency
- **Charging Station Utilization**: Station occupancy rates and average serviced vehicles

### Data Export

Automatically generates Excel reports containing:
- Episode statistical data
- Vehicle performance metrics
- Charging station utilization
- Time series analysis

## Experimental Results Analysis

### Typical Results

Based on current configuration experimental results：

```
Episode Statistics:
- Total Orders: Variable (depends on episode length)
- Accepted Orders: ~70-80%
- Rejected Orders: ~20-30%
- Average Battery Level: 0.4-0.6
- Average Vehicles per Station: 2-4 vehicles
```

### Vehicle Type Comparison

| Metric | EV | AEV |
|--------|----|-----|
| Rejection Rate | High (distance-related) | 0% |
| Service Efficiency | Medium | High |
| Charging Frequency | High | Medium |

## Future Improvement Directions

### Algorithm Optimization

1. **Deep Q-Network (DQN)**
   - Use neural networks to approximate Q-value functions
   - Handle continuous state spaces
   - Improve learning efficiency

2. **Multi-Agent Reinforcement Learning**
   - Consider collaboration between vehicles
   - Joint optimization strategies
   - Distributed decision making

### System Extensions

1. **Real-time Optimization**
   - Dynamic charging station capacity adjustment
   - Real-time traffic condition consideration
   - Energy price optimization

2. **Advanced Features**
   - Predictive maintenance
   - Energy storage system integration
   - Multi-modal transportation systems

### Technical Improvements

1. **Computational Efficiency**
   - Parallel optimization algorithms
   - Incremental learning methods
   - Cloud computing support

2. **Robustness Enhancement**
   - Uncertainty modeling
   - Fault recovery mechanisms
   - Adaptive parameter adjustment

## Conclusion

This RL-ADP charging station management system successfully integrates:
- Reinforcement learning's value learning capabilities
- Linear programming's precise optimization
- Practical charging station management constraints

Through Q-learning's learning mechanisms and Gurobi's optimization solving, the system can learn better allocation strategies while considering distance costs, providing effective solutions for intelligent transportation and new energy management.

---

*Documentation Generated: September 2, 2025*
*Project Version: RL-ADP Charging Station v1.0*
