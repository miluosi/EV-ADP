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
