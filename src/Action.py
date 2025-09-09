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
    
    def __init__(self, requests: Iterable[Request], charging_station_id: int, charging_duration: float = 30.0) -> None:
        super().__init__(requests)
        self.charging_station_id = charging_station_id
        self.charging_duration = charging_duration  # minutes
        self.action_type = "charging"
    
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
    
    def __init__(self, requests: Iterable[Request], request_id: int) -> None:
        super().__init__(requests)
        self.request_id = request_id
        self.action_type = "service"
    
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
    
    def __init__(self, requests: Iterable[Request], current_coords: tuple, target_coords: tuple) -> None:
        super().__init__(requests)
        self.current_coords = current_coords  # (x, y) current position
        self.target_coords = target_coords    # (x, y) target position
        self.action_type = "idle"
    
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
