"""
Test script for reject behavior in Gurobi optimization
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.Environment import ChargingIntegratedEnvironment
from src.GurobiOptimizer import GurobiOptimizer
from src.Request import Request
from src.ValueFunction_pytorch import PyTorchChargingValueFunction

def test_reject_behavior():
    """Test EV reject behavior in Gurobi optimization"""
    print("=== Testing EV Reject Behavior in Gurobi Optimization ===")
    
    # Create small environment for testing
    env = ChargingIntegratedEnvironment(num_vehicles=4, num_stations=2, grid_size=10)
    
    # Set up neural network value function
    value_function = PyTorchChargingValueFunction(
        grid_size=env.grid_size, 
        num_vehicles=4,
        device='cpu'
    )
    env.set_value_function(value_function)
    
    # Create Gurobi optimizer
    gurobi_optimizer = GurobiOptimizer(env)
    
    # Create test requests at different distances
    test_requests = []
    
    # Close request (should be accepted by EV)
    close_request = Request(
        request_id=1,
        source=10,  # Close to vehicles at grid positions
        destination=15,
        current_time=0,
        travel_time=5,
        value=10.0
    )
    test_requests.append(close_request)
    
    # Far request (should be rejected by EV)
    far_request = Request(
        request_id=2,
        source=90,  # Far from vehicles
        destination=95,
        current_time=0,
        travel_time=5,
        value=15.0
    )
    test_requests.append(far_request)
    
    # Get vehicle IDs
    vehicle_ids = list(env.vehicles.keys())
    
    print(f"\n📍 Vehicle positions:")
    for vid in vehicle_ids:
        vehicle = env.vehicles[vid]
        print(f"  Vehicle {vid} ({vehicle['type']}): position {vehicle['coordinates']}, battery {vehicle['battery']:.2f}")
    
    print(f"\n📨 Test requests:")
    for request in test_requests:
        pickup_coords = (request.pickup // env.grid_size, request.pickup % env.grid_size)
        print(f"  Request {request.request_id}: pickup {pickup_coords}, value {request.value}")
        
        # Check rejection probability for each vehicle
        for vid in vehicle_ids:
            vehicle = env.vehicles[vid]
            rejection_prob = env._calculate_rejection_probability(vid, request)
            print(f"    → Vehicle {vid} ({vehicle['type']}): rejection prob {rejection_prob:.3f}")
    
    print(f"\n🔍 Testing Original Gurobi Method:")
    try:
        original_assignments = gurobi_optimizer._gurobi_vehicle_rebalancing(
            vehicle_ids, test_requests, env.charging_manager.get_available_stations()
        )
        print(f"  Original assignments: {original_assignments}")
    except Exception as e:
        print(f"  Original method failed: {e}")
    
    print(f"\n🔍 Testing New Reject-Aware Gurobi Method:")
    try:
        reject_aware_assignments = gurobi_optimizer._gurobi_vehicle_rebalancing_knownreject(
            vehicle_ids, test_requests, env.charging_manager.get_available_stations()
        )
        print(f"  Reject-aware assignments: {reject_aware_assignments}")
        
        # Analyze assignments
        for vehicle_id, assignment in reject_aware_assignments.items():
            vehicle = env.vehicles[vehicle_id]
            if hasattr(assignment, 'request_id'):  # It's a request
                rejection_prob = env._calculate_rejection_probability(vehicle_id, assignment)
                print(f"    → Vehicle {vehicle_id} ({vehicle['type']}) assigned to request {assignment.request_id}")
                print(f"      Rejection probability: {rejection_prob:.3f}")
                if vehicle['type'] == 'EV' and rejection_prob >= 0.5:
                    print(f"      ⚠️  WARNING: EV assigned to request it would likely reject!")
            else:
                print(f"    → Vehicle {vehicle_id} ({vehicle['type']}) assigned to {assignment}")
                
    except Exception as e:
        print(f"  Reject-aware method failed: {e}")
    
    print(f"\n✅ Test completed!")

if __name__ == "__main__":
    test_reject_behavior()
