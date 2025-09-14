# PyTorch Q-Network Parameter Documentation

## Network Architecture Overview

The EV-ADP PyTorch Q-Network (`PyTorchPathBasedNetwork`) is a sophisticated neural network designed for reinforcement learning in electric vehicle fleet management. It processes multiple input modalities to produce Q-values for decision making.

## Architecture Components

### 1. Input Features

| Feature | Dimension | Description | Range |
|---------|-----------|-------------|-------|
| `path_locations` | [batch_size, seq_len] | Sequence of location IDs in vehicle path | 0 to num_locations |
| `path_delays` | [batch_size, seq_len, 1] | Delay information for each location | 0 to max_delay |
| `current_time` | [batch_size, 1] | Normalized current time step | 0.0 to 1.0 |
| `other_agents` | [batch_size, 1] | Number of nearby agents | 0 to num_vehicles |
| `num_requests` | [batch_size, 1] | Current active requests | 0 to max_requests |
| `battery_level` | [batch_size, 1] | Vehicle battery level | 0.0 to 1.0 |
| `request_value` | [batch_size, 1] | Target request value | 0.0 to 1.0 |
| `action_type` | [batch_size, 1] | Action category (1=idle, 2=assign, 3=charge) | 1 to 3 |
| `vehicle_id` | [batch_size, 1] | Vehicle identifier | 1 to num_vehicles |
| `vehicle_type` | [batch_size, 1] | Vehicle type (1=EV, 2=AEV) | 1 to 2 |

### 2. Embedding Layers

#### Location Embedding
- **Input**: Location IDs (0 to num_locations)
- **Output**: 100-dimensional vectors
- **Purpose**: Convert discrete locations to continuous representations
- **Parameters**: (num_locations + 1) × 100

#### Vehicle ID Embedding  
- **Input**: Vehicle IDs (0 to num_vehicles)
- **Output**: 25-dimensional vectors
- **Purpose**: Capture vehicle-specific characteristics
- **Parameters**: (num_vehicles + 1) × 25

#### Vehicle Type Embedding
- **Input**: Vehicle type (0=unknown, 1=EV, 2=AEV)
- **Output**: 25-dimensional vectors
- **Purpose**: Distinguish between electric vehicle types
- **Parameters**: 3 × 25

#### Action Type Embedding
- **Input**: Action type (0=padding, 1=idle, 2=assign, 3=charge)
- **Output**: 50-dimensional vectors
- **Purpose**: Encode action semantics
- **Parameters**: 4 × 50

#### Time Embedding
- **Architecture**: Linear(1 → 100) + ELU
- **Purpose**: Transform scalar time to rich representation
- **Parameters**: 100 + 100 (weights + biases)

#### Context Embedding
- **Architecture**: Linear(2 → 50) + ELU + Dropout(0.1)
- **Input**: [battery_level, request_value]
- **Purpose**: Process contextual state information
- **Parameters**: 2 × 50 + 50

#### Vehicle Feature Embedding
- **Architecture**: Linear(50 → 50) + ELU + Dropout(0.1)
- **Input**: Concatenated vehicle_id and vehicle_type embeddings
- **Purpose**: Combine vehicle-specific features
- **Parameters**: 50 × 50 + 50

### 3. LSTM Processing

#### Path LSTM
- **Architecture**: LSTM(input_size=101, hidden_size=200, batch_first=True)
- **Input**: Concatenated location embeddings + delays [100 + 1 = 101]
- **Output**: Hidden state representation (200D)
- **Purpose**: Process sequential path information
- **Parameters**: ~325,600 (4 × (101 × 200 + 200 × 200 + 200))

### 4. Dense Network

#### State Embedding Network
- **Architecture**: 
  ```
  Linear(452 → 300) + ELU + Dropout(0.1)
  Linear(300 → 300) + ELU + Dropout(0.1)  
  Linear(300 → 1)
  ```
- **Input Dimension**: 452 (path:200 + time:100 + other_agents:1 + num_requests:1 + action:50 + context:50 + vehicle:50)
- **Output**: Single Q-value
- **Parameters**: 
  - Layer 1: 452 × 300 + 300 = 135,900
  - Layer 2: 300 × 300 + 300 = 90,300  
  - Layer 3: 300 × 1 + 1 = 301

## Hyperparameters

### Default Architecture Parameters
```python
num_locations: int = 100        # Grid locations
num_vehicles: int = 50          # Fleet size
max_capacity: int = 4           # Vehicle capacity
embedding_dim: int = 100        # Location embedding size
lstm_hidden: int = 200          # LSTM hidden units
dense_hidden: int = 300         # Dense layer size
```

### Training Parameters
```python
learning_rate: float = 1e-4     # Adam optimizer learning rate
batch_size: int = 256           # Training batch size
dropout_rate: float = 0.1       # Dropout probability
gradient_clip: float = 1.0      # Gradient clipping threshold
```

## Parameter Count Estimation

| Component | Parameters |
|-----------|------------|
| Location Embedding | ~10,100 |
| Vehicle ID Embedding | ~1,275 |
| Vehicle Type Embedding | 75 |
| Action Type Embedding | 200 |
| Time Embedding | 200 |
| Context Embedding | 150 |
| Vehicle Feature Embedding | 2,550 |
| LSTM Layer | ~325,600 |
| Dense Layers | ~226,501 |
| **Total** | **~566,651** |

## Input/Output Specifications

### Forward Pass Input
```python
def forward(self, 
            path_locations: torch.Tensor,      # [batch, seq_len]
            path_delays: torch.Tensor,         # [batch, seq_len, 1]
            current_time: torch.Tensor,        # [batch, 1]
            other_agents: torch.Tensor,        # [batch, 1] 
            num_requests: torch.Tensor,        # [batch, 1]
            battery_level: torch.Tensor = None, # [batch, 1]
            request_value: torch.Tensor = None, # [batch, 1]
            action_type: torch.Tensor = None,   # [batch, 1]
            vehicle_id: torch.Tensor = None,    # [batch, 1]
            vehicle_type: torch.Tensor = None) -> torch.Tensor: # [batch, 1]
```

### Output
- **Shape**: [batch_size, 1]
- **Type**: torch.Tensor
- **Range**: Real numbers (Q-values)
- **Interpretation**: Expected discounted future reward for state-action pair

## Usage Example

```python
# Initialize network
network = PyTorchPathBasedNetwork(
    num_locations=100,
    num_vehicles=50, 
    max_capacity=4
)

# Prepare inputs
batch_size = 32
seq_len = 10

path_locations = torch.randint(0, 100, (batch_size, seq_len))
path_delays = torch.rand(batch_size, seq_len, 1)
current_time = torch.rand(batch_size, 1)
other_agents = torch.randint(0, 10, (batch_size, 1)).float()
num_requests = torch.randint(0, 20, (batch_size, 1)).float()

# Forward pass
q_values = network(path_locations, path_delays, current_time, 
                  other_agents, num_requests)

print(f"Q-values shape: {q_values.shape}")  # [32, 1]
```

## Performance Considerations

### Memory Usage
- **Model Size**: ~2.3 MB (float32)
- **Peak Memory**: ~50 MB for batch_size=256
- **GPU Memory**: Recommended 2GB+ VRAM

### Computational Complexity
- **Forward Pass**: O(batch_size × seq_len × embedding_dim)
- **LSTM Complexity**: O(seq_len × hidden_dim²)
- **Training Step**: ~10-50ms on modern GPU

### Optimization Tips
1. Use mixed precision training (float16) to reduce memory
2. Gradient accumulation for larger effective batch sizes
3. Learning rate scheduling for stable convergence
4. Early stopping based on validation Q-value stability

---

*Generated automatically by EV-ADP Neural Network Visualizer*
*Date: 2025-09-14*
