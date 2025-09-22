from .Request import Request
from .Path import Path
from typing import Set, FrozenSet, List, Iterable, Optional


class Action(object):
    """
    An Action is the output of an Agent for a decision epoch.

    In our formulation corresponds to an Agent accepting a given set
    of Requests.
    """

    def __init__(self, requests: Iterable[Request]) -> None:
        self.requests = frozenset(requests)
        self.new_path: Optional[Path] = None

    def __eq__(self, other):
        return (self.requests == other.requests)

    def __hash__(self):
        return hash(self.requests)


class ChargingAction(Action):
    """
    A ChargingAction represents a vehicle going to a charging station.
    
    This action allows vehicles to charge their batteries at designated
    charging stations in the environment.
    """
    
    def __init__(self, requests: Iterable[Request], charging_station_id: int, charging_duration: float = 30.0, vehicle_loc: tuple = None, vehicle_battery: float = None, next_action = None,next_value = 0) -> None:
        super().__init__(requests)
        self.charging_station_id = charging_station_id
        self.charging_duration = charging_duration  # minutes
        self.action_type = "charging"
        self.dur_reward = 0
        self.vehicle_loc = vehicle_loc
        self.vehicle_battery = vehicle_battery
        self.next_action = next_action  # 用于存储充电后的后续动作
        self.next_value = next_value  # 用于存储充电后的后续动作的价值
        self.vehicle_loc_post = None  # 用于存储服务后的车辆位置
        self.vehicle_battery_post = None  # 用于存储服务后的车辆电量
    def __eq__(self, other):
        if isinstance(other, ChargingAction):
            return (self.requests == other.requests and 
                   self.charging_station_id == other.charging_station_id)
        return False
    
    def __hash__(self):
        return hash((self.requests, self.charging_station_id))
    
    def get_charging_info(self):
        """Return charging station information"""
        return {
            'station_id': self.charging_station_id,
            'duration': self.charging_duration,
            'action_type': self.action_type
        }


class ServiceAction(Action):
    """
    A ServiceAction represents accepting and serving a passenger request.
    """

    def __init__(self, requests: Iterable[Request], request_id: int, vehicle_loc: tuple, vehicle_battery: float, next_action = None, next_value = 0) -> None:
        super().__init__(requests)
        self.request_id = request_id
        self.action_type = "service"
        self.dur_reward = 0
        self.vehicle_loc = vehicle_loc
        self.vehicle_battery = vehicle_battery
        self.next_action = next_action  # 用于存储服务后的后续动作
        self.next_value = next_value  # 用于存储服务后的后续动作的价值
        self.vehicle_loc_post = None  # 用于存储服务后的车辆位置
        self.vehicle_battery_post = None  # 用于存储服务后的车辆电量
    def __eq__(self, other):
        if isinstance(other, ServiceAction):
            return (self.requests == other.requests and 
                   self.request_id == other.request_id)
        return False
    
    def __hash__(self):
        return hash((self.requests, self.request_id))
    
    def get_service_info(self):
        """Return service request information"""
        return {
            'request_id': self.request_id,
            'action_type': self.action_type
        }


class IdleAction(Action):
    """
    An IdleAction represents a vehicle moving randomly while idle.
    
    This action allows vehicles to move from current coordinates to
    randomly selected target coordinates when they have no assigned tasks.
    """
    
    def __init__(self, requests: Iterable[Request], current_coords: tuple, target_coords: tuple,vehicle_loc=None, vehicle_battery=None, next_action = None,next_value = 0) -> None:
        super().__init__(requests)
        self.current_coords = current_coords  # (x, y) current position
        self.target_coords = target_coords    # (x, y) target position
        self.action_type = "idle"
        self.dur_reward = 0
        self.vehicle_loc = vehicle_loc
        self.vehicle_battery = vehicle_battery
        self.next_action = next_action  # 用于存储闲置移动后的后续动作
        self.next_value = next_value  # 用于存储闲置移动后的后续动作的价值
        self.vehicle_loc_post = None  # 用于存储服务后的车辆位置
        self.vehicle_battery_post = None  # 用于存储服务后的车辆电量
    def __eq__(self, other):
        if isinstance(other, IdleAction):
            return (self.requests == other.requests and 
                   self.current_coords == other.current_coords and
                   self.target_coords == other.target_coords)
        return False
    
    def __hash__(self):
        return hash((self.requests, self.current_coords, self.target_coords))
    
    def get_idle_info(self):
        """Return idle movement information"""
        return {
            'current_coords': self.current_coords,
            'target_coords': self.target_coords,
            'action_type': self.action_type
        }
