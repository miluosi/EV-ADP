"""
Test charging state synchronization fix
"""

from src.Environment import ChargingIntegratedEnvironment
from src.charging_station import ChargingStationManager, ChargingStation

def test_charging_sync():
    """Test that charging states are properly synchronized"""
    print("=== Testing Charging State Synchronization ===")
    
    env = ChargingIntegratedEnvironment(num_vehicles=5, num_stations=2)
    
    # Get first station
    station_id = list(env.charging_manager.stations.keys())[0]
    station = env.charging_manager.stations[station_id]
    
    print(f"Initial state:")
    print(f"  Station {station_id} capacity: {station.max_capacity}")
    print(f"  Station {station_id} current vehicles: {station.current_vehicles}")
    print(f"  Station {station_id} queue: {station.charging_queue}")
    
    # Manually add vehicles to station to simulate the bug
    vehicle_ids = ['0', '1', '2', '3']
    
    print(f"\nManually adding vehicles to station (simulating auto-queue processing):")
    for vid in vehicle_ids:
        success = station.start_charging(vid)
        print(f"  Vehicle {vid} start_charging result: {success}")
        if not success:
            print(f"  Station full, adding {vid} to queue")
    
    print(f"\nAfter manual additions:")
    print(f"  Station {station_id} current vehicles: {station.current_vehicles}")
    print(f"  Station {station_id} queue: {station.charging_queue}")
    
    # Check vehicle states before sync
    print(f"\nVehicle charging states before sync:")
    vehicles_with_charging_station = 0
    for vid, vehicle in env.vehicles.items():
        if vehicle['charging_station'] is not None:
            vehicles_with_charging_station += 1
            print(f"  Vehicle {vid}: charging at station {vehicle['charging_station']}")
    print(f"  Total vehicles with charging_station set: {vehicles_with_charging_station}")
    
    # Call update environment to trigger sync
    print(f"\nCalling _update_environment() to trigger sync...")
    env._update_environment()
    
    # Check vehicle states after sync  
    print(f"\nVehicle charging states after sync:")
    vehicles_with_charging_station = 0
    for vid, vehicle in env.vehicles.items():
        if vehicle['charging_station'] is not None:
            vehicles_with_charging_station += 1
            print(f"  Vehicle {vid}: charging at station {vehicle['charging_station']}")
    print(f"  Total vehicles with charging_station set: {vehicles_with_charging_station}")
    
    # Get statistics
    stats = env.get_episode_stats()
    print(f"\nFinal statistics:")
    print(f"  Station Usage: {stats['avg_vehicles_per_station']:.1f} vehicles/station")
    print(f"  Vehicles charging (from stats): {stats['vehicles_charging']}")
    print(f"  Station vehicles count: {len(station.current_vehicles)}")
    
    print(f"\n=== Test Complete ===")

if __name__ == "__main__":
    test_charging_sync()
