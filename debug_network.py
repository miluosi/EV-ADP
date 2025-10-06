"""
Debug script to test PyTorchPathBasedNetwork initialization
"""

import torch
import torch.nn as nn
import sys
sys.path.append('c:/Users/miaoz/Downloads/EV-ADP-1')

try:
    from src.ValueFunction_pytorch import PyTorchPathBasedNetwork
    print("✓ Successfully imported PyTorchPathBasedNetwork")
    
    # Test basic initialization
    print("\n=== Testing Network Initialization ===")
    network = PyTorchPathBasedNetwork(
        num_locations=100,  # 10x10 grid
        num_vehicles=10,
        max_capacity=6,
        embedding_dim=128,
        lstm_hidden=256,
        dense_hidden=512,
        pretrained_embeddings=None
    )
    print(f"✓ Network initialized successfully")
    
    # Check parameters
    total_params = 0
    trainable_params = 0
    for name, param in network.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        print(f"  {name}: {param.shape}, requires_grad={param.requires_grad}")
    
    print(f"\nTotal parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    
    # Test forward pass
    print("\n=== Testing Forward Pass ===")
    batch_size = 2
    seq_len = 4
    
    # Create dummy inputs
    path_locations = torch.randint(1, 101, (batch_size, seq_len))
    path_delays = torch.randn(batch_size, seq_len, 1)
    current_time = torch.randn(batch_size, 1)
    other_agents = torch.randint(0, 5, (batch_size, 1)).float()
    num_requests = torch.randint(0, 10, (batch_size, 1)).float()
    battery_level = torch.rand(batch_size, 1)
    request_value = torch.rand(batch_size, 1)
    action_type = torch.randint(1, 4, (batch_size, 1))
    vehicle_id = torch.randint(1, 11, (batch_size, 1))
    vehicle_type = torch.randint(1, 3, (batch_size, 1))
    
    print("Input shapes:")
    print(f"  path_locations: {path_locations.shape}")
    print(f"  path_delays: {path_delays.shape}")
    print(f"  current_time: {current_time.shape}")
    print(f"  other_agents: {other_agents.shape}")
    print(f"  num_requests: {num_requests.shape}")
    print(f"  battery_level: {battery_level.shape}")
    print(f"  request_value: {request_value.shape}")
    print(f"  action_type: {action_type.shape}")
    print(f"  vehicle_id: {vehicle_id.shape}")
    print(f"  vehicle_type: {vehicle_type.shape}")
    
    # Forward pass
    output = network(
        path_locations=path_locations,
        path_delays=path_delays,
        current_time=current_time,
        other_agents=other_agents,
        num_requests=num_requests,
        battery_level=battery_level,
        request_value=request_value,
        action_type=action_type,
        vehicle_id=vehicle_id,
        vehicle_type=vehicle_type
    )
    
    print(f"✓ Forward pass successful, output shape: {output.shape}")
    print(f"  Output values: {output.detach().numpy().flatten()}")
    
    # Test optimizer initialization
    print("\n=== Testing Optimizer ===")
    import torch.optim as optim
    optimizer = optim.Adam(network.parameters(), lr=2e-3, weight_decay=1e-5)
    print(f"✓ Optimizer initialized successfully")
    print(f"  Parameter groups: {len(optimizer.param_groups)}")
    
    # Test loss computation
    print("\n=== Testing Loss Computation ===")
    target = torch.randn(batch_size, 1)
    loss_fn = nn.MSELoss()
    loss = loss_fn(output, target)
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Test backpropagation
    print("\n=== Testing Backpropagation ===")
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print("✓ Backpropagation successful")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()