from .LearningAgent import LearningAgent
from .Action import Action
from .Request import Request
from .Path import PathNode, RequestInfo

from typing import Type, List, Generator, Tuple, Deque, Dict
import math
from abc import ABCMeta, abstractmethod
from random import choice, randint
from pandas import read_csv
from collections import deque
import gurobipy as gp  # type: ignore
from gurobipy import GRB  # type: ignore
import re
import random
import numpy as np
from .charging_station import ChargingStationManager, ChargingStation
from .Action import Action, ChargingAction, ServiceAction, IdleAction
from src.GurobiOptimizer import GurobiOptimizer
import time
class Environment(metaclass=ABCMeta):
    """Defines a class for simulating the Environment for the RL agent"""

    REQUEST_HISTORY_SIZE: int = 1000

    def __init__(self, NUM_LOCATIONS: int, MAX_CAPACITY: int, EPOCH_LENGTH: float, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float, DATA_DIR: str):
        # Load environment
        self.NUM_LOCATIONS = NUM_LOCATIONS
        self.MAX_CAPACITY = MAX_CAPACITY
        self.EPOCH_LENGTH = EPOCH_LENGTH
        self.NUM_AGENTS = NUM_AGENTS
        self.START_EPOCH = START_EPOCH
        self.STOP_EPOCH = STOP_EPOCH
        self.DATA_DIR = DATA_DIR
        self.adp_value = 0.5
        self.num_days_trained = 0
        self.recent_request_history: Deque[Request] = deque(maxlen=self.REQUEST_HISTORY_SIZE)
        self.current_time: float = 0.0
        self.idle_vehicle_requirement = 1
        self.idle_penalty = 0.5
        self.charging_penalty = 0.2
        self.chargeincrease_per_epoch = 0.1  # Battery increase per epoch when charging
        self.min_battery_level = 0.1
        self.complete_ratio_reward = 0.5
        # Q-learning components for ADP integration
        self.q_table = {}  # Q-table for state-action values
        self.learning_rate = 0.1
        self.discount_factor = 0.9
        self.epsilon = 0.1  # Exploration rate
    @abstractmethod
    def initialise_environment(self):
        raise NotImplementedError

    @abstractmethod
    def get_request_batch(self):
        raise NotImplementedError

    @abstractmethod
    def get_travel_time(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_next_location(self, source, destination):
        raise NotImplementedError

    @abstractmethod
    def get_initial_states(self, num_agents, is_training):
        raise NotImplementedError

    def simulate_motion(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True) -> None:
        # Move all agents
        agents_to_rebalance: List[Tuple[LearningAgent, float]] = []
        for agent in agents:
            time_remaining: float = self.EPOCH_LENGTH
            time_remaining = self._move_agent(agent, time_remaining)
            # If it has visited all the locations it needs to and has time left, rebalance
            if (time_remaining > 0):
                agents_to_rebalance.append((agent, time_remaining))

        # Update recent_requests list
        self.update_recent_requests(current_requests)

        # Perform Rebalancing
        if (rebalance and agents_to_rebalance):
            rebalancing_targets = self._get_rebalance_targets([agent for agent, _ in agents_to_rebalance])

            # Move cars according to the rebalancing_targets
            for idx, target in enumerate(rebalancing_targets):
                agent, time_remaining = agents_to_rebalance[idx]

                # Insert dummy target
                agent.path.requests.append(RequestInfo(target, False, True))
                agent.path.request_order.append(PathNode(False, 0))  # adds pickup location to 'to-visit' list

                # Move according to dummy target
                self._move_agent(agent, time_remaining)

                # Undo impact of creating dummy target
                agent.path.request_order.clear()
                agent.path.requests.clear()
                agent.path.current_capacity = 0
                agent.path.total_delay = 0

    def _move_agent(self, agent: LearningAgent, time_remaining: float) -> float:
        while(time_remaining >= 0):
            time_remaining -= agent.position.time_to_next_location

            # If we reach an intersection, make a decision about where to go next
            if (time_remaining >= 0):
                # If the intersection is an existing pick-up or drop-off location, update the Agent's path
                if (agent.position.next_location == agent.path.get_next_location()):
                    agent.path.visit_next_location(self.current_time + self.EPOCH_LENGTH - time_remaining)

                # Go to the next location in the path, if it exists
                if (not agent.path.is_empty()):
                    next_location = self.get_next_location(agent.position.next_location, agent.path.get_next_location())
                    agent.position.time_to_next_location = self.get_travel_time(agent.position.next_location, next_location)
                    agent.position.next_location = next_location

                # If no additional locations need to be visited, stop
                else:
                    agent.position.time_to_next_location = 0
                    break
            # Else, continue down the road you're on
            else:
                agent.position.time_to_next_location -= (time_remaining + agent.position.time_to_next_location)

        return time_remaining

    def get_state_representation(self, agent_position: int, target_position: int, 
                                current_time: float) -> str:
        """Get state representation for Q-learning"""
        # Discretize time for state representation
        time_slot = int(current_time // 10)  # 10-minute time slots
        return f"{agent_position}_{target_position}_{time_slot}"
    
    def get_q_value(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair"""
        key = f"{state}_{action}"
        return self.q_table.get(key, 0.0)
    
    def update_q_value(self, state: str, action: str, reward: float, next_state: str):
        """Update Q-value using Q-learning algorithm"""
        key = f"{state}_{action}"
        
        # Get current Q-value
        current_q = self.get_q_value(state, action)
        
        # Get max Q-value for next state (assuming action is assignment)
        next_q_values = [self.get_q_value(next_state, f"assign_{i}") for i in range(10)]
        max_next_q = max(next_q_values) if next_q_values else 0.0
        
        # Q-learning update
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[key] = new_q
    
    def get_assignment_q_value(self, agent_id: int, target_id: int, 
                              agent_position: int, target_position: int) -> float:
        """Get Q-value for a specific assignment"""
        state = self.get_state_representation(agent_position, target_position, self.current_time)
        action = f"assign_{target_id}"
        return self.get_q_value(state, action)
    
    def _get_rebalance_targets(self, agents: List) -> List:
        """Get rebalancing targets using Gurobi optimization with Q-learning integration"""
        # Get a list of possible targets by sampling from recent_requests
        possible_targets: List[Request] = []
        num_targets = min(500, len(agents))
        for _ in range(num_targets):
            target = choice(self.recent_request_history)
            possible_targets.append(target)

        # Solve an LP to assign each agent to closest possible target
        model = gp.Model()
        model.setParam('OutputFlag', 0)  # Suppress output

        # Define variables, a matrix defining the assignment of agents to targets
        assignments = {}
        for agent_id in range(len(agents)):
            for target_id in range(len(possible_targets)):
                assignments[agent_id, target_id] = model.addVar(vtype=GRB.Binary, 
                                                              name=f'assignment_{agent_id}_{target_id}')

        # Make sure one agent can only be assigned to one target
        for agent_id in range(len(agents)):
            model.addConstr(gp.quicksum(assignments[agent_id, target_id] 
                                      for target_id in range(len(possible_targets))) == 1)

        # Make sure one target can only be assigned to *ratio* agents
        num_fractional_targets = len(agents) - (int(len(agents) / num_targets) * num_targets)
        for target_id in range(len(possible_targets)):
            num_agents_to_target = int(len(agents) / num_targets) + (1 if target_id < num_fractional_targets else 0)
            model.addConstr(gp.quicksum(assignments[agent_id, target_id] 
                                      for agent_id in range(len(agents))) == num_agents_to_target)

        # Define the objective: Combine distance cost and Q-value benefit
        distance_weight = 0.7
        q_weight = 0.3
        
        # Distance cost component
        distance_obj = gp.quicksum(assignments[agent_id, target_id] * 
                                 self.get_travel_time(agents[agent_id].position.next_location, 
                                                    possible_targets[target_id].pickup) 
                                 for target_id in range(len(possible_targets)) 
                                 for agent_id in range(len(agents)))
        
        # Q-value benefit component (negative because we want to maximize benefit)
        q_value_obj = gp.quicksum(assignments[agent_id, target_id] * 
                                (-self.get_assignment_q_value(agent_id, target_id,
                                                            agents[agent_id].position.next_location,
                                                            possible_targets[target_id].pickup))
                                for target_id in range(len(possible_targets)) 
                                for agent_id in range(len(agents)))
        
        # Combined objective function
        obj = distance_weight * distance_obj + q_weight * q_value_obj
        model.setObjective(obj, GRB.MINIMIZE)

        # Solve
        model.optimize()
        assert model.status == GRB.OPTIMAL  # making sure that the model doesn't fail

        # Get the assigned targets
        assigned_targets: List[Request] = []
        for agent_id in range(len(agents)):
            for target_id in range(len(possible_targets)):
                if (assignments[agent_id, target_id].x == 1):
                    assigned_targets.append(possible_targets[target_id])
                    break

        return assigned_targets

    def get_reward(self, action: Action) -> float:
        """
        Return the reward to an agent for a given (feasible) action.

        (Feasibility is not checked!)
        Defined in Environment class because of Reinforcement Learning
        convention in literature.
        """
        return sum([request.value for request in action.requests])

    def update_recent_requests(self, recent_requests: List[Request]):
        self.recent_request_history.extend(recent_requests)


class NYEnvironment(Environment):
    """Define an Environment using the cleaned NYC Yellow Cab dataset."""

    NUM_MAX_AGENTS: int = 3000
    NUM_LOCATIONS: int = 4461

    def __init__(self, NUM_AGENTS: int, START_EPOCH: float, STOP_EPOCH: float, MAX_CAPACITY, DATA_DIR: str='../data/ny/', EPOCH_LENGTH: float = 60.0):
        super().__init__(NUM_LOCATIONS=self.NUM_LOCATIONS, MAX_CAPACITY=MAX_CAPACITY, EPOCH_LENGTH=EPOCH_LENGTH, NUM_AGENTS=NUM_AGENTS, START_EPOCH=START_EPOCH, STOP_EPOCH=STOP_EPOCH, DATA_DIR=DATA_DIR)
        self.initialise_environment()

    def initialise_environment(self):
        print('Loading Environment...')

        TRAVELTIME_FILE: str = self.DATA_DIR + 'zone_traveltime.csv'
        self.travel_time = read_csv(TRAVELTIME_FILE, header=None).values

        SHORTESTPATH_FILE: str = self.DATA_DIR + 'zone_path.csv'
        self.shortest_path = read_csv(SHORTESTPATH_FILE, header=None).values

        IGNOREDZONES_FILE: str = self.DATA_DIR + 'ignorezonelist.txt'
        self.ignored_zones = read_csv(IGNOREDZONES_FILE, header=None).values.flatten()

        INITIALZONES_FILE: str = self.DATA_DIR + 'taxi_3000_final.txt'
        self.initial_zones = read_csv(INITIALZONES_FILE, header=None).values.flatten()

        assert (self.EPOCH_LENGTH == 60) or (self.EPOCH_LENGTH == 30) or (self.EPOCH_LENGTH == 10)
        self.DATA_FILE_PREFIX: str = "{}files_{}sec/test_flow_5000_".format(self.DATA_DIR, int(self.EPOCH_LENGTH))

    def get_request_batch(self,
                          day: int=2,
                          downsample: float=1) -> Generator[List[Request], None, None]:

        assert 0 < downsample <= 1
        request_id = 0

        def is_in_time_range(current_time):
            current_hour = int(current_time / 3600)
            return True if (current_hour >= self.START_EPOCH / 3600 and current_hour < self.STOP_EPOCH / 3600) else False

        # Open file to read
        with open(self.DATA_FILE_PREFIX + str(day) + '.txt', 'r') as data_file:
            num_batches: int = int(data_file.readline().strip())

            # Defines the 2 possible RE for lines in the data file
            new_epoch_re = re.compile(r'Flows:(\d+)-\d+')
            request_re = re.compile(r'(\d+),(\d+),(\d+)\.0')

            # Parsing rest of the file
            request_list: List[Request] = []
            is_first_epoch = True
            for line in data_file.readlines():
                line = line.strip()

                is_new_epoch = re.match(new_epoch_re, line)
                if (is_new_epoch is not None):
                    if not is_first_epoch:
                        if is_in_time_range(self.current_time):
                            yield request_list
                        request_list.clear()  # starting afresh for new batch
                    else:
                        is_first_epoch = False

                    current_epoch = int(is_new_epoch.group(1))
                    self.current_time = current_epoch * self.EPOCH_LENGTH
                else:
                    request_data = re.match(request_re, line)
                    assert request_data is not None  # Make sure there's nothing funky going on with the formatting

                    num_requests = int(request_data.group(3))
                    for _ in range(num_requests):
                        # Take request according to downsampled rate
                        rand_num = random()
                        if (rand_num <= downsample):
                            source = int(request_data.group(1))
                            destination = int(request_data.group(2))
                            if (source not in self.ignored_zones and destination not in self.ignored_zones and source != destination):
                                    travel_time = self.get_travel_time(source, destination)
                                    request_list.append(Request(request_id, source, destination, self.current_time, travel_time))
                                    request_id += 1

            if is_in_time_range(self.current_time):
                yield request_list

    def get_travel_time(self, source: int, destination: int) -> float:
        return self.travel_time[source, destination]

    def get_next_location(self, source: int, destination: int) -> int:
        return self.shortest_path[source, destination]

    def get_initial_states(self, num_agents: int, is_training: bool) -> List[int]:
        """Give initial states for num_agents agents"""
        if (num_agents > self.NUM_MAX_AGENTS):
            print('Too many agents. Starting with random states.')
            is_training = True

        # If it's training, get random states
        if is_training:
            initial_states = []

            for _ in range(num_agents):
                initial_state = randint(0, self.NUM_LOCATIONS - 1)
                # Make sure it's not an ignored zone
                while (initial_state in self.ignored_zones):
                    initial_state = randint(0, self.NUM_LOCATIONS - 1)

                initial_states.append(initial_state)
        # Else, pick deterministic initial states
        else:
            initial_states = self.initial_zones[:num_agents]

        return initial_states

    def has_valid_path(self, agent: LearningAgent) -> bool:
        """Attempt to check if the request order meets deadline and capacity constraints"""
        def invalid_path_trace(issue: str) -> bool:
            print(issue)
            print('Agent {}:'.format(agent.id))
            print('Requests -> {}'.format(agent.path.requests))
            print('Request Order -> {}'.format(agent.path.request_order))
            print()
            return False

        # Make sure that its current capacity is sensible
        if (agent.path.current_capacity < 0 or agent.path.current_capacity > self.MAX_CAPACITY):
            return invalid_path_trace('Invalid current capacity')

        # Make sure that it visits all the requests that it has accepted
        if (not agent.path.is_complete()):
            return invalid_path_trace('Incomplete path.')

        # Start at global_time and current_capacity
        current_time = self.current_time + agent.position.time_to_next_location
        current_location = agent.position.next_location
        current_capacity = agent.path.current_capacity

        # Iterate over path
        available_delay: float = 0
        for node_idx, node in enumerate(agent.path.request_order):
            next_location, deadline = agent.path.get_info(node)

            # Delay related checks
            travel_time = self.get_travel_time(current_location, next_location)
            if (current_time + travel_time > deadline):
                return invalid_path_trace('Does not meet deadline at node {}'.format(node_idx))

            current_time += travel_time
            current_location = next_location

            # Updating available delay
            if (node.expected_visit_time != current_time):
                invalid_path_trace("(Ignored) Visit time incorrect at node {}".format(node_idx))
                node.expected_visit_time = current_time

            if (node.is_dropoff):
                available_delay += deadline - node.expected_visit_time

            # Capacity related checks
            if (current_capacity > self.MAX_CAPACITY):
                return invalid_path_trace('Exceeds MAX_CAPACITY at node {}'.format(node_idx))

            if (node.is_dropoff):
                next_capacity = current_capacity - 1
            else:
                next_capacity = current_capacity + 1
            if (node.current_capacity != next_capacity):
                invalid_path_trace("(Ignored) Capacity incorrect at node {}".format(node_idx))
                node.current_capacity = next_capacity
            current_capacity = node.current_capacity

        # Check total_delay
        if (agent.path.total_delay != available_delay):
            invalid_path_trace("(Ignored) Total delay incorrect.")
        agent.path.total_delay = available_delay

        return True



class ChargingIntegratedEnvironment(Environment):
    """
    Integrated charging environment class, inheriting from src.Environment
    """

    def __init__(self, num_vehicles=5, num_stations=3, grid_size=12, use_intense_requests=True, assignmentgurobi=True):  # Increased grid size
        # Provide required parameters for base class
        NUM_LOCATIONS = grid_size * grid_size  # Total locations in grid
        MAX_CAPACITY = 4  # Maximum capacity per location
        EPOCH_LENGTH = 1.0  # Length of each epoch
        NUM_AGENTS = num_vehicles  # Number of vehicles/agents
        START_EPOCH = 0.0  # Start time
        STOP_EPOCH = 100.0  # Stop time
        DATA_DIR = "data"  # Data directory (not used in this implementation)
        
        super().__init__(NUM_LOCATIONS, MAX_CAPACITY, EPOCH_LENGTH, NUM_AGENTS, START_EPOCH, STOP_EPOCH, DATA_DIR)
        self.assignmentgurobi = assignmentgurobi  # Whether to use Gurobi for assignment
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.grid_size = grid_size
        self.minimum_charging_level = 0.2  # Minimum battery level before needing to charge
        # Parameters for reward alignment with Gurobi optimization
        self.charging_penalty = 0.5  # Penalty for charging action (reduced from 10.0)
        self.adp_value = 1.0  # Weight for Q-value contribution
        self.unserved_penalty = 1.5  # Penalty for unserved requests (reduced from 5.0)
        self.idle_vehicle_requirement = 1  # Minimum idle vehicles required
        self.charge_duration = 2
        self.chargeincrease_per_epoch = 0.5
        self.chargeincrease_whole = self.chargeincrease_per_epoch * self.charge_duration
        self.min_battery_level = 0.2
        self.charge_finished = 0.0
        self.charge_stats = {}
        # Initialize charging station manager
        self.charging_manager = ChargingStationManager()
        self._setup_charging_stations()
        self.unserve_penalty = -100  # Penalty for unserved requests
        self.movingpenalty = -1e-1
        # Vehicle states
        self.rebalance_battery_threshold = 0.5
        self.vehicles = {}
        
        
        self._setup_vehicles()
        
        # Environment state
        self.current_time = 0
        self.episode_length = 200
        
        # Request system
        self.active_requests = {}  # Active passenger requests
        self.completed_requests = []  # Completed requests for analysis
        self.rejected_requests = []  # Rejected requests for analysis
        self.request_counter = 0
        self.request_generation_rate = 0.8  # Increased to 60% for more active environment
        self.use_intense_requests = use_intense_requests  # Whether to use concentrated request generation
        self.battery_consum = 0.01  # Battery consumption per epoch when moving
        # Assignment tracking for rebalancing analysis
        self.rebalancing_assignments_per_step = []  # Store assignments count per step
        self.total_rebalancing_calls = 0
        self.penalty_for_passenger_stranding = -50  
        # Tracking for visualization
        self.request_generation_history = []  # Track where requests are generated
        self.vehicle_position_history = {}  # Track vehicle movement patterns
        
        # Charging station usage tracking for episode-wide statistics
        self.charging_usage_history = []  # Track charging station usage over time
        
        # Initialize ValueFunction for Q-value calculation (will be set externally)
        self.value_function = None
        
        print(f"✓ Initialized integrated environment: {num_vehicles} vehicles, {num_stations} charging stations")
    
    def set_value_function(self, value_function):
        """Set the value function for Q-value calculation"""
        self.value_function = value_function
        print(f"✓ Value function set: {type(value_function).__name__}")
    
    def get_assignment_q_value(self, vehicle_id: int, target_id: int, 
                              vehicle_location: int, target_location: int) -> float:
        """Get Q-value for vehicle assignment using ValueFunction if available"""
        if self.value_function and hasattr(self.value_function, 'get_assignment_q_value'):
            # Provide additional context for neural network including battery level and request value
            vehicle = self.vehicles.get(vehicle_id, {})
            battery_level = vehicle.get('battery', 1.0)  # 获取车辆电池电量
            other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
            num_requests = len(self.active_requests)
            
            # 获取请求的价值信息 - 使用final_value确保与奖励一致
            request_value = 0.0
            if target_id in self.active_requests:
                # 使用value而不是final_value，确保与实际奖励计算一致
                request_value = self.active_requests[target_id].final_value
            
            return self.value_function.get_assignment_q_value(
                vehicle_id, target_id, vehicle_location, target_location, 
                self.current_time, other_vehicles, num_requests, battery_level, request_value)  # 添加request_value参数
        else:
            # Fallback to parent class method
            return super().get_assignment_q_value(vehicle_id, target_id, vehicle_location, target_location)
    
    def get_charging_q_value(self, vehicle_id: int, station_id: int,
                           vehicle_location: int, station_location: int) -> float:
        """Get Q-value for vehicle charging decision"""
        if self.value_function and hasattr(self.value_function, 'get_charging_q_value'):
            # Provide additional context for neural network including battery level
            vehicle = self.vehicles.get(vehicle_id, {})
            battery_level = vehicle.get('battery', 1.0)  # 获取车辆电池电量
            # 对齐训练时的计数口径：使用空闲车辆数量（不含当前车）而非“正在充电的车辆数”
            idle_count = len([v for vid, v in self.vehicles.items()
                              if v.get('assigned_request') is None and
                                 v.get('passenger_onboard') is None and
                                 v.get('charging_station') is None])
            other_vehicles = max(0, idle_count - 1)
            num_requests = len(self.active_requests)
            
            return self.value_function.get_charging_q_value(
                vehicle_id, station_id, vehicle_location, station_location, 
                self.current_time, other_vehicles, num_requests, battery_level)  # 添加battery_level参数
        else:
            # Fallback calculation
            distance = abs(vehicle_location - station_location)
            return 5.0 - distance * 0.1  # Simple heuristic

    # --- Option-value evaluators for optimizer alignment ---
    def _loc_to_xy(self, loc: int) -> tuple:
        return (loc % self.grid_size, loc // self.grid_size)

    def _manhattan_distance_loc(self, a_loc: int, b_loc: int) -> int:
        ax, ay = self._loc_to_xy(a_loc)
        bx, by = self._loc_to_xy(b_loc)
        return abs(ax - bx) + abs(ay - by)
    def _manhattan_distance_loc_time(self, a_loc: int, b_loc: int) -> int:
        ax, ay = self._loc_to_xy(a_loc)
        bx, by = self._loc_to_xy(b_loc)
        return max(abs(ax - bx), abs(ay - by))
    def _estimate_future_state_value(self, vehicle_id: int, future_loc: int, future_battery: float, future_time: float,actiontype) -> float:
        """Minimally estimate V(s_after) using current NN: take max of idle/waiting value at future state.
        Falls back to 0 if no value_function.
        """
        if not self.value_function:
            return 0.0
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        num_reqs = len(self.active_requests)
        if actiontype == 'charge':
            assign_v = self.value_function.get_charging_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=future_loc,
                current_time=future_time,
                other_vehicles=max(0, other_idle),
                num_requests=num_reqs,
                battery_level=future_battery,
            )
            return assign_v
        elif actiontype == 'serve':
            assign_v = self.value_function.get_assignment_q_value(
                vehicle_id=vehicle_id,
                request_id=-1,  # Request ID is not used in this context
                vehicle_location=future_loc,
                current_time=future_time,
                other_vehicles=max(0, other_idle),
                num_requests=num_reqs,
                battery_level=future_battery,
            )
            return assign_v
        elif actiontype == 'idle':
            idle_v = self.value_function.get_idle_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=future_loc,
                battery_level=future_battery,
                current_time=future_time,
                other_vehicles=max(0, other_idle),
                num_requests=num_reqs
            )
            return idle_v

    def evaluate_service_option(self, vehicle_id: int, request) -> float:
        """Estimate completion Q for serving a request (option value):
        Q_opt ≈ R_exec + gamma^tau * V(s_after).
        Minimal R_exec: movement penalties + final_value; battery feasibility check.
        """
        # Resolve request object from id
        if isinstance(request, (int, str)) and request in self.active_requests:
            request = self.active_requests[request]
        if request is None:
            return 0.0

        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']

        # distances (in steps)
        d1 = self._manhattan_distance_loc(cur_loc, request.pickup)
        d2 = self._manhattan_distance_loc(request.pickup, request.dropoff)
        tau = d1 + d2
        tau_time = self._manhattan_distance_loc_time(cur_loc, request.pickup) + self._manhattan_distance_loc_time(request.pickup, request.dropoff)
        # execution reward: movement penalty + completion earnings
        moving_cost = self.movingpenalty * (d1 + d2)
        earnings = getattr(request, 'final_value', getattr(request, 'value', 0.0))

        # battery feasibility
        travel_battery = (d1 + d2) * self.battery_consum
        if cur_bat - travel_battery < 0:
            # Not enough battery to complete without charging; penalize
            return -1e6

        r_exec = moving_cost + earnings

        # future state
        future_loc = request.dropoff
        future_battery = max(0.0, cur_bat - travel_battery)
        future_time = min(self.episode_length, self.current_time + tau_time)

        # discount
        gamma = getattr(self.value_function, 'gamma', 0.99) if self.value_function else 0.99
        v_after = self._estimate_future_state_value(vehicle_id, future_loc, future_battery, future_time, actiontype='serve')
        return  gamma * v_after

    def evaluate_charging_option(self, vehicle_id: int, station) -> float:
        """Estimate completion Q for going to charge at a station (option value)."""
        # Resolve station object/id
        station_obj = None
        station_id = None
        if hasattr(station, 'id'):
            station_id = station.id
            station_obj = station
        else:
            station_id = station
            station_obj = self.charging_manager.stations.get(station_id) if hasattr(self, 'charging_manager') else None
        if station_obj is None:
            return 0.0

        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']
        station_loc = station_obj.location

        # travel to station
        d_travel = self._manhattan_distance_loc(cur_loc, station_loc)
        d_travel_time = self._manhattan_distance_loc_time(cur_loc, station_loc)
        travel_battery = d_travel * self.battery_consum
        if cur_bat - travel_battery < 0:
            return -1e6

        # charging duration and battery gain
        charge_steps = getattr(self, 'charge_duration', 2)
        charge_gain = getattr(self, 'chargeincrease_whole', 0.5)
        tau = d_travel_time + charge_steps

        # execution reward: movement penalty + charging penalty/time cost
        moving_cost = self.movingpenalty * d_travel
        charging_penalty = -getattr(self, 'charging_penalty', 0.5) * charge_steps
        r_exec = moving_cost + charging_penalty

        # future state after charge
        future_loc = station_loc
        future_battery = max(0.0, min(1.0, cur_bat - travel_battery + charge_gain))
        future_time = min(self.episode_length, self.current_time + tau)

        gamma = getattr(self.value_function, 'gamma', 0.99) if self.value_function else 0.99
        v_after = self._estimate_future_state_value(vehicle_id, future_loc, future_battery, future_time,actiontype='charge')
        return  gamma * v_after


    def evaluate_idle_option(self, vehicle_id: int,target_loc) -> float:
        """Estimate completion Q for idling/waiting (option value)."""
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']

        # minimal execution reward: small movement penalty for staying put
        r_exec = self.movingpenalty * 1.0  # small penalty for idling
        d_travel = self._manhattan_distance_loc(cur_loc, target_loc)
        d_travel_time = self._manhattan_distance_loc_time(cur_loc, target_loc)
        # future state after idling one epoch
        future_loc = target_loc
        future_battery = max(0.0, cur_bat - self.battery_consum*d_travel)  # battery drains slightly even when idle
        future_time = min(self.episode_length, self.current_time + d_travel_time)
        tau = d_travel_time
        gamma = getattr(self.value_function, 'gamma', 0.99) if self.value_function else 0.99
        v_after = self._estimate_future_state_value(vehicle_id, future_loc, future_battery, future_time, actiontype='idle')
        return  gamma * v_after

    def evaluate_waiting_option(self, vehicle_id: int) -> float:
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']

        # minimal execution reward: small movement penalty for staying put
        r_exec = self.movingpenalty * 1.0  # small penalty for idling

        # future state after idling one epoch
        future_loc = cur_loc
        future_battery = veh['battery']  # battery drains slightly even when idle
        future_time = min(self.episode_length, self.current_time + 1.0)

        gamma = getattr(self.value_function, 'gamma', 0.99) if self.value_function else 0.99
        v_after = self._estimate_future_state_value(vehicle_id, future_loc, future_battery, future_time, actiontype='idle')
        return  gamma * v_after


    def get_idle_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = None, 
                        other_vehicles: int = None, num_requests: int = None) -> float:
        """
        Get Q-value for vehicle idle action
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 电池电量 (0-1)
            current_time: 当前时间 (可选，如果不提供则使用self.current_time)
            other_vehicles: 其他车辆数量 (可选，如果不提供则自动计算)
            num_requests: 当前请求数量 (可选，如果不提供则自动计算)
            
        Returns:
            float: idle动作的Q值
        """
        if self.value_function and hasattr(self.value_function, 'get_idle_q_value'):
            # 使用神经网络计算idle Q值，提供所有必要的上下文信息
            other_vehicles = len([v for v in self.vehicles.values() 
                                if v['assigned_request'] is None and 
                                   v['passenger_onboard'] is None and 
                                   v['charging_station'] is None]) - 1  # 减去当前车辆
            num_requests = len(self.active_requests)
            
            return self.value_function.get_idle_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                battery_level=battery_level,
                current_time=self.current_time,
                other_vehicles=max(0, other_vehicles),  # 确保非负
                num_requests=num_requests
            )
        else:
            # 后备计算：简单的基于电池电量和时间的启发式
            base_idle_value = -0.1  # idle的基础成本
            battery_bonus = battery_level * 0.5  # 高电量时idle的价值更高
            time_penalty = self.current_time / self.episode_length * 0.2  # 后期idle惩罚更大
            
            return base_idle_value + battery_bonus - time_penalty


    def get_waiting_q_value(self, vehicle_id: int, vehicle_location: int, 
                        battery_level: float, current_time: float = None, 
                        other_vehicles: int = None, num_requests: int = None) -> float:
        """
        Get Q-value for vehicle idle action
        
        Args:
            vehicle_id: 车辆ID
            vehicle_location: 车辆当前位置
            battery_level: 电池电量 (0-1)
            current_time: 当前时间 (可选，如果不提供则使用self.current_time)
            other_vehicles: 其他车辆数量 (可选，如果不提供则自动计算)
            num_requests: 当前请求数量 (可选，如果不提供则自动计算)
            
        Returns:
            float: idle动作的Q值
        """
        if self.value_function and hasattr(self.value_function, 'get_idle_q_value'):
            # 使用神经网络计算idle Q值，提供所有必要的上下文信息
            other_vehicles = len([v for v in self.vehicles.values() 
                                if v['assigned_request'] is None and 
                                   v['passenger_onboard'] is None and 
                                   v['charging_station'] is None]) - 1  # 减去当前车辆
            num_requests = len(self.active_requests)
            
            return self.value_function.get_waiting_q_value(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                battery_level=battery_level,
                current_time=self.current_time,
                other_vehicles=max(0, other_vehicles),  # 确保非负
                num_requests=num_requests
            )
        else:
            # 后备计算：简单的基于电池电量和时间的启发式
            base_idle_value = -0.1  # idle的基础成本
            battery_bonus = battery_level * 0.5  # 高电量时idle的价值更高
            time_penalty = self.current_time / self.episode_length * 0.2  # 后期idle惩罚更大
            
            return base_idle_value + battery_bonus - time_penalty
    
    # Note: store_q_learning_experience is now integrated into _update_q_learning
    # for better consistency between Q-table and neural network training
    
    def _setup_charging_stations(self):
        """Setup charging stations dynamically based on num_stations"""
        # Generate charging stations evenly distributed across the grid
        stations = []
        for i in range(self.num_stations):
            # Distribute stations evenly across the grid
            if self.num_stations == 1:
                x, y = self.grid_size // 2, self.grid_size // 2
            elif self.num_stations == 2:
                positions = [(2, 2), (7, 7)]
                x, y = positions[i]
            elif self.num_stations == 3:
                positions = [(2, 2), (5, 7), (8, 3)]
                x, y = positions[i]
            else:
                # For more stations, distribute more evenly
                cols = int(np.sqrt(self.num_stations))
                rows = (self.num_stations + cols - 1) // cols
                row = i // cols
                col = i % cols
                x = (col + 1) * self.grid_size // (cols + 1)
                y = (row + 1) * self.grid_size // (rows + 1)
            
            location_index = y * self.grid_size + x
            stations.append({
                'id': i + 1,
                'location': location_index,
                'capacity': 10  # Unified capacity of 10
            })
        
        for station_info in stations:
            self.charging_manager.add_station(
                station_info['id'],
                station_info['location'],
                station_info['capacity']
            )
    
    def _setup_vehicles(self):
        """Setup initial vehicle states with EV and AEV types"""
        for i in range(self.num_vehicles):
            # Convert grid coordinates to location index
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            location_index = y * self.grid_size + x
            
            if i % 2 == 0:
                vehicle_type = 1  # EV (Electric Vehicle)
                vehicle_type_name = 'EV'
            else:
                vehicle_type = 2  # AEV (Autonomous Electric Vehicle)
                vehicle_type_name = 'AEV'
            
            self.vehicles[i] = {
                'type': vehicle_type,  # Vehicle type: 1=EV, 2=AEV (数值编码)
                'type_name': vehicle_type_name,  # Vehicle type name: 'EV' or 'AEV' (字符串名称)
                'location': location_index,  # Use location index instead of coordinates
                'coordinates': (x, y),  # Keep coordinates for visualization
                'battery': random.uniform(0.3, 0.9),  # 30%-90% battery
                'charging_station': None,
                'charging_time_left': 0,
                'total_distance': 0,
                'charging_count': 0,
                'assigned_request': None,  # Currently assigned passenger request
                'passenger_onboard': None,  # Passenger being transported
                'service_earnings': 0,  # Total earnings from completed requests
                'rejected_requests': 0,  # Track rejected requests for analysis
                'unserved_penalty': 0,  # Accumulated penalty for unserved requests
                'is_stationary': False,  # Whether the vehicle is in waiting/stationary state
                'stationary_duration': 0  # Duration to remain stationary
            }
    
    def _calculate_rejection_probability(self, vehicle_id, request):
        """Calculate the probability that an EV rejects a request based on distance"""
        vehicle = self.vehicles[vehicle_id]
        
        # AEV never rejects
        if vehicle['type'] == 2:  # AEV
            return 0.0
        
        # Calculate distance to pickup location
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
        
        # Exponential distribution for rejection probability
        # Base rejection rate increases exponentially with distance
        base_rate = 0.01  # 5% base rejection rate for distance 0
        distance_factor = 0.5  # Exponential growth factor
        
        rejection_prob = base_rate * np.exp(distance * distance_factor)
        # Cap at 90% maximum rejection probability
        return min(0.9, rejection_prob)
    
    def _should_reject_request(self, vehicle_id, request):
        """Determine if a vehicle should reject a request"""
        rejection_prob = self._calculate_rejection_probability(vehicle_id, request)
        return random.random() < rejection_prob
    

    def _generate_random_requests(self):
        """Generate new passenger requests in batches"""
        generated_requests = []
        
        # Determine how many requests to generate this step
        if random.random() < self.request_generation_rate:
            # Generate between 1 and 10 requests with higher probability for fewer requests
            # 50% chance for 1-3 requests, 30% for 4-6, 20% for 7-10
            rand_val = random.random()
            if rand_val < 0.5:
                num_requests = random.randint(4, 8)
            elif rand_val < 0.8:
                num_requests = random.randint(8, 10)
            else:
                num_requests = random.randint(10,12)
            
            for _ in range(num_requests):
                self.request_counter += 1
                
                # Random pickup and dropoff locations
                pickup_x = random.randint(0, self.grid_size - 1)
                pickup_y = random.randint(0, self.grid_size - 1)
                pickup_location = pickup_y * self.grid_size + pickup_x
                
                dropoff_x = random.randint(0, self.grid_size - 1)
                dropoff_y = random.randint(0, self.grid_size - 1)
                dropoff_location = dropoff_y * self.grid_size + dropoff_x
                
                # Ensure pickup and dropoff are different
                attempts = 0
                while dropoff_location == pickup_location and attempts < 5:
                    dropoff_x = random.randint(0, self.grid_size - 1)
                    dropoff_y = random.randint(0, self.grid_size - 1)
                    dropoff_location = dropoff_y * self.grid_size + dropoff_x
                    attempts += 1
                
                # Calculate travel time (Manhattan distance)
                travel_time = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                
                # Create request with dynamic pricing based on demand
                base_value = 10
                distance_value = travel_time * (3 + np.random.rand()*0.1)
                surge_factor = 1.0 + (num_requests - 1) * 0.1  # More requests = higher prices
                final_value = base_value * surge_factor + distance_value
                
                request = Request(
                    request_id=self.request_counter,
                    source=pickup_location,
                    destination=dropoff_location,
                    current_time=self.current_time,
                    travel_time=travel_time,
                    value=final_value
                )
                
                self.active_requests[self.request_counter] = request
                generated_requests.append(request)
                
                # Track request generation for visualization
                if not hasattr(self, 'request_generation_history'):
                    self.request_generation_history = []
                self.request_generation_history.append({
                    'pickup_coords': (pickup_x, pickup_y),
                    'dropoff_coords': (dropoff_x, dropoff_y),
                    'hotspot_idx': None,  # No hotspot for random requests
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            return generated_requests
        
        return []
    
    def _generate_intense_requests(self):
        """Generate multiple requests concentrated in 3 hotspots with probability weights"""
        generated_requests = []
        
        # Determine how many requests to generate this step
        if random.random() < self.request_generation_rate:
            # Generate between 1 and 10 requests with higher probability for fewer requests
            # 50% chance for 1-3 requests, 30% for 4-6, 20% for 7-10
            rand_val = random.random()
            if rand_val < 0.8:
                num_requests = random.randint(24, 30)
            elif rand_val < 0.95:
                num_requests = random.randint(30, 36)
            else:
                num_requests = random.randint(36, 42)

            # Define 3 hotspot centers in the grid
            hotspots = [
                (self.grid_size // 4, self.grid_size // 4),           # Bottom-left hotspot
                (3 * self.grid_size // 4, self.grid_size // 4),       # Bottom-right hotspot
                (self.grid_size // 4, 3 * self.grid_size // 4),       # Top-left hotspot
                (3 * self.grid_size // 4, 3 * self.grid_size // 4)    # Top-right hotspot
            ]
            
            # Probability weights for each hotspot (should sum to 1.0)
            probability_weights = [0.4, 0.1, 0.25, 0.25]  # 40% for bottom-left, 10% for bottom-right, 30% for top-left, 20% for top-right
            selected_hotspot_idx_reward = [35, 15, 35, 15]  # Reward weights for each hotspot
            for _ in range(num_requests):
                self.request_counter += 1
                
                # Select hotspot based on probability weights
                rand_val = random.random()
                cumulative_prob = 0
                selected_hotspot_idx = 0
                for i, weight in enumerate(probability_weights):
                    cumulative_prob += weight
                    if rand_val <= cumulative_prob:
                        selected_hotspot_idx = i
                        break
                
                hotspot_center = hotspots[selected_hotspot_idx]
                
                # Generate pickup location near selected hotspot (with some randomness)
                hotspot_radius = max(2, self.grid_size // 8)  # Radius around hotspot
                pickup_x = max(0, min(self.grid_size - 1, 
                                    hotspot_center[0] + random.randint(-hotspot_radius, hotspot_radius)))
                pickup_y = max(0, min(self.grid_size - 1, 
                                    hotspot_center[1] + random.randint(-hotspot_radius, hotspot_radius)))
                pickup_location = pickup_y * self.grid_size + pickup_x

                available_hotspot_indices = [i for i in range(len(hotspots)) if i != selected_hotspot_idx]
                # if available_hotspot_indices:
                #     # Get weights for available hotspots and normalize them
                #     available_weights = [probability_weights[i] for i in available_hotspot_indices]
                #     total_weight = sum(available_weights)
                #     normalized_weights = [w / total_weight for w in available_weights]
                    
                #     # Select dropoff hotspot based on normalized weights
                #     rand_val = random.random()
                #     cumulative_prob = 0
                #     selected_dropoff_idx = 0
                #     for i, weight in enumerate(normalized_weights):
                #         cumulative_prob += weight
                #         if rand_val <= cumulative_prob:
                #             selected_dropoff_idx = available_hotspot_indices[i]
                #             break
                selected_hotspot_idx_1 = random.choice(available_hotspot_indices)
                dropoff_hotspot = hotspots[selected_hotspot_idx_1]
                dropoff_x = max(0, min(self.grid_size - 1, 
                                    dropoff_hotspot[0] + random.randint(-hotspot_radius, hotspot_radius)))
                dropoff_y = max(0, min(self.grid_size - 1, 
                                    dropoff_hotspot[1] + random.randint(-hotspot_radius, hotspot_radius)))
                
                dropoff_location = dropoff_y * self.grid_size + dropoff_x
                
                # Ensure pickup and dropoff are different
                attempts = 0
                while dropoff_location == pickup_location and attempts < 5:
                    dropoff_x = dropoff_x + random.randint(-2, 2)
                    dropoff_x = max(0, min(self.grid_size - 1, dropoff_x))
                    dropoff_y = dropoff_y + random.randint(-2, 2)
                    dropoff_y = max(0, min(self.grid_size - 1, dropoff_y))
                    dropoff_location = dropoff_y * self.grid_size + dropoff_x
                    attempts += 1
                
                # Calculate travel time (Manhattan distance)
                travel_time = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                
                #
                base_value = 20
                distance_value = travel_time * (2 + np.random.rand()*0.1)
                surge_factor = 1.0 + (num_requests - 1) * 0.1  # More requests = higher prices
                final_value = base_value * surge_factor + distance_value + selected_hotspot_idx_reward[selected_hotspot_idx]
                
                request = Request(
                    request_id=self.request_counter,
                    source=pickup_location,
                    destination=dropoff_location,
                    current_time=self.current_time,
                    travel_time=travel_time,
                    value=base_value,
                    final_value=final_value
                )
                
                self.active_requests[self.request_counter] = request
                generated_requests.append(request)
                
                # Track request generation for visualization
                if not hasattr(self, 'request_generation_history'):
                    self.request_generation_history = []
                self.request_generation_history.append({
                    'pickup_coords': (pickup_x, pickup_y),
                    'dropoff_coords': (dropoff_x, dropoff_y),
                    'hotspot_idx': selected_hotspot_idx,
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            return generated_requests
        
        return []
    
    def _assign_request_to_vehicle(self, vehicle_id, request_id):
        """Assign a request to a vehicle with rejection logic"""
        if request_id in self.active_requests and vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            request = self.active_requests[request_id]
            
            # Check if this request is already assigned to another vehicle
            for other_vehicle_id, other_vehicle in self.vehicles.items():
                if (other_vehicle_id != vehicle_id and 
                    (other_vehicle['assigned_request'] == request_id or 
                     other_vehicle['passenger_onboard'] == request_id)):
                    return False  # Request already assigned to another vehicle
            
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                # Check if the vehicle rejects the request
                if self._should_reject_request(vehicle_id, request):
                    vehicle['rejected_requests'] += 1
                    # Record rejected request if not already recorded
                    if request not in self.rejected_requests:
                        self.rejected_requests.append(request)
                    #print(f"DEBUG: Vehicle {vehicle_id} REJECTED request {request_id} at step {self.current_time}")
                    return False  # Request rejected
                
                # Request accepted
                vehicle['assigned_request'] = request_id
                #print(f"DEBUG: Vehicle {vehicle_id} ASSIGNED to request {request_id} at step {self.current_time}")
                return True
        return False
    
    def _move_vehicle_to_charging_station(self, vehicle_id, station_id):
        """Move a vehicle to a charging station for rebalancing"""
        if vehicle_id in self.vehicles and hasattr(self, 'charging_manager'):
            vehicle = self.vehicles[vehicle_id]
            
            # Check if vehicle is available for charging assignment
            if (vehicle['assigned_request'] is None and 
                vehicle['passenger_onboard'] is None and
                vehicle['charging_station'] is None and
                vehicle['charging_time_left'] == 0):
                
                # Check if station exists and has available slots
                if (station_id in self.charging_manager.stations and 
                    self.charging_manager.stations[station_id].available_slots > 0):
                    
                    station = self.charging_manager.stations[station_id]
                    # Convert station location to coordinates
                    station_x = station.location // self.grid_size
                    station_y = station.location % self.grid_size
                    station_coords = (station_x, station_y)
                    
                    # Set vehicle destination to charging station
                    vehicle['target_location'] = station_coords
                    vehicle['charging_target'] = station_id
                    
                    return True
        return False
    
    def _pickup_passenger(self, vehicle_id):
        """Vehicle picks up passenger at request pickup location"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成pickup
        if vehicle['battery'] <= 0.0:
            print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成pickup - 订单未完成")
            # 将未完成的订单重新放回active_requests等待其他车辆
            if vehicle['assigned_request'] is not None:
                request_id = vehicle['assigned_request']
                vehicle['assigned_request'] = None
                #print(f"   订单 {request_id} 因车辆电池耗尽被重新分配")
            return False
            
        if vehicle['assigned_request'] is not None:
            # Check if the assigned request still exists
            if vehicle['assigned_request'] not in self.active_requests:
                # Request has expired or been removed, clear the assignment
                vehicle['assigned_request'] = None
                return False
                
            request = self.active_requests[vehicle['assigned_request']]
            vehicle_coords = vehicle['coordinates']
            pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
            
            # Check if vehicle is at pickup location
            if vehicle_coords == pickup_coords:
                vehicle['passenger_onboard'] = vehicle['assigned_request']
                vehicle['assigned_request'] = None
                return True
        return False
    
    def _dropoff_passenger(self, vehicle_id):
        """Vehicle drops off passenger at destination"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成dropoff
        if vehicle['battery'] <= 0.0:
            print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成dropoff - 乘客滞留")
            # 乘客滞留在车上，订单未完成
            if vehicle['passenger_onboard'] is not None:
                request_id = vehicle['passenger_onboard']
                print(f"   乘客 {request_id} 因车辆电池耗尽而滞留在车上")
                # 保持passenger_onboard状态，等待车辆充电后继续
            return self.unserve_penalty
            
        if vehicle['passenger_onboard'] is not None:
            # Check if the passenger request still exists
            if vehicle['passenger_onboard'] not in self.active_requests:
                # Request has expired or been removed, clear the passenger
                vehicle['passenger_onboard'] = None
                vehicle['assigned_request'] = None
                return 0
                
            request = self.active_requests[vehicle['passenger_onboard']]
            vehicle_coords = vehicle['coordinates']
            dropoff_coords = (request.dropoff % self.grid_size, request.dropoff // self.grid_size)
            
            # Check if vehicle is at dropoff location
            if vehicle_coords == dropoff_coords:
                # Complete the request
                completed_request = self.active_requests.pop(vehicle['passenger_onboard'])
                self.completed_requests.append(completed_request)
                
                # Calculate earnings
                earnings = completed_request.final_value
                vehicle['passenger_onboard'] = None
                
                return earnings
        return 0
    
    def get_initial_states(self, num_agents=None, is_training=True):
        """Get initial states - implementing abstract method"""
        if num_agents is None:
            num_agents = self.num_vehicles
        
        states = {}
        for vehicle_id in range(num_agents):
            if vehicle_id in self.vehicles:
                states[vehicle_id] = self._get_vehicle_state(vehicle_id)
            else:
                # Create default state for additional agents
                states[vehicle_id] = np.array([0.5, 0.5, 0.5, 0, 0])
        return states
    
    def initialise_environment(self):
        """Initialize environment - implementing abstract method"""
        self.current_time = 0
        self._setup_vehicles()
        return self.get_initial_states()
    
    def get_request_batch(self):
        """Get request batch - implementing abstract method"""
        # Return both passenger requests and charging needs
        requests = []
        
        # Add passenger requests
        for request_id, request in self.active_requests.items():
            requests.append(request)
        
        # Add charging requests for low battery vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery'] < 0.3:  # Low battery vehicles need charging
                charging_request = Request(
                    request_id=f"charge_{vehicle_id}",
                    source=vehicle['location'],
                    destination=vehicle['location'],  # Stay at same location for charging
                    current_time=self.current_time,
                    travel_time=0,
                    value=0.5  # Small value for charging necessity
                )
                requests.append(charging_request)
        
        return requests
    
    def get_travel_time(self, source, destination):
        """Get travel time between locations - implementing abstract method"""
        if isinstance(source, tuple) and isinstance(destination, tuple):
            # Manhattan distance as travel time
            return abs(source[0] - destination[0]) + abs(source[1] - destination[1])
        else:
            # Default travel time
            return 1.0
    
    def get_next_location(self, source, destination):
        """Get next location towards destination - implementing abstract method"""
        if isinstance(source, tuple) and isinstance(destination, tuple):
            x_diff = destination[0] - source[0]
            y_diff = destination[1] - source[1]
            
            # Move one step towards destination
            next_x = source[0]
            next_y = source[1]
            
            if x_diff > 0:
                next_x += 1
            elif x_diff < 0:
                next_x -= 1
            elif y_diff > 0:
                next_y += 1
            elif y_diff < 0:
                next_y -= 1
                
            return (next_x, next_y)
        else:
            return source
    
    def _get_vehicle_state(self, vehicle_id):
        """Get vehicle state vector"""
        vehicle = self.vehicles[vehicle_id]
        coords = vehicle['coordinates']
        state = [
            coords[0] / self.grid_size,  # Normalized x coordinate
            coords[1] / self.grid_size,  # Normalized y coordinate
            vehicle['battery'],                        # Battery level
            float(vehicle['charging_station'] is not None),  # Whether charging
            self.current_time / self.episode_length    # Time progress
        ]
        return np.array(state)
    
    def step(self, actions):
        """执行一步环境交互"""
        rewards = {}
        dur_rewards = {}
        next_states = {}
        charging_events = []
        
        # Initialize step counters
        self.step_assignments = 0
        self.step_rejections = 0

        # Store pre-action vehicle states for Q-learning
        pre_action_states = {}
        for vehicle_id, act in actions.items():
            if vehicle_id not in self.vehicles:
                continue
            # New-style Action objects carry vehicle_loc & vehicle_battery as attributes
            vehicle_loc = None
            vehicle_battery = None
            try:
                # Attribute path (ChargingAction/ServiceAction/IdleAction)
                vehicle_loc = getattr(act, 'vehicle_loc', None)
                vehicle_battery = getattr(act, 'vehicle_battery', None)
            except Exception:
                vehicle_loc = None
                vehicle_battery = None
            # Backward compatibility: dict-like action payload
            if vehicle_loc is None or vehicle_battery is None:
                if isinstance(act, dict):
                    vehicle_loc = act.get('vehicle_loc', None)
                    vehicle_battery = act.get('vehicle_battery', None)
            # Fallback to current environment state if still missing
            if vehicle_loc is None:
                vehicle_loc = self.vehicles[vehicle_id]['location']
            if vehicle_battery is None:
                vehicle_battery = self.vehicles[vehicle_id]['battery']

            pre_action_states[vehicle_id] = {
                'location': vehicle_loc,
                'battery': vehicle_battery,
            }

        # 处理每个车辆的动作
        for vehicle_id, action in actions.items():
            reward = self._execute_action(vehicle_id, action)
            rewards[vehicle_id] = reward
            next_states[vehicle_id] = self._get_vehicle_state(vehicle_id)

            # Record charging events
            if isinstance(action, ChargingAction):
                charging_events.append({
                    'vehicle_id': vehicle_id,
                    'station_id': action.charging_station_id,
                    'duration': action.charging_duration,
                    'time': self.current_time
                })

        # 更新环境状态
        self._update_environment()
        batterypenaltyv = self._check_dead_battery_vehicles()

        # # 将电池耗尽惩罚合并到主奖励中
        # for vehicle_id in batterypenaltyv:
        #     if vehicle_id in rewards:
        #         rewards[vehicle_id] -= 50.0


        # 集成Gurobi优化和Q-learning更新
        # current_requests = list(self.active_requests.values())
        # self.simulate_motion(agents=[], current_requests=current_requests, rebalance=True)

        # 执行Q-learning更新
        self._update_q_learning(actions, rewards, pre_action_states)
        
        # Record charging station usage for this time step
        self._record_charging_usage()

        # 检查是否结束
        done = self.current_time >= self.episode_length

        return next_states, rewards, done, {'charging_events': charging_events}
    
    def _record_charging_usage(self):
        """Record charging station usage for current time step"""
        if hasattr(self, 'charging_manager') and self.charging_manager.stations:
            total_occupied = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            total_stations = len(self.charging_manager.stations)
            
            usage_stats = {
                'time': self.current_time,
                'total_occupied': total_occupied,
                'total_stations': total_stations,
                'vehicles_per_station': total_occupied / max(1, total_stations),
                'station_details': {
                    station_id: len(station.current_vehicles) 
                    for station_id, station in self.charging_manager.stations.items()
                }
            }
            self.charging_usage_history.append(usage_stats)

    def _check_dead_battery_vehicles(self):
        """检查电池耗尽的车辆"""
        dead_battery_vehicles = []
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery'] <= 0.0:
                # 只处理不在充电站的电池耗尽车辆
                if vehicle['charging_station'] is None:
                    dead_battery_vehicles.append(vehicle_id)
        return dead_battery_vehicles





    def simulate_motion(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []

        # Initialize actions dictionary for all vehicles
        actions = {}
        
        # For ChargingIntegratedEnvironment, we handle rebalancing differently
        # Convert our vehicle states to a format compatible with Gurobi optimization

        if rebalance and self.vehicles:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            for vehicle_id, vehicle in self.vehicles.items():
                if (vehicle['assigned_request'] is None and
                    vehicle['passenger_onboard'] is None and
                    vehicle['charging_station'] is None and
                    vehicle.get('idle_target') is None ) or (vehicle['battery'] <= self.rebalance_battery_threshold+1e+3) or vehicle['is_stationary']:
                    vehicles_to_rebalance.append(vehicle_id)

            if vehicles_to_rebalance:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Idle vehicles: {len(vehicles_to_rebalance)}, Available requests: {available_requests_count}")
                
                if self.assignmentgurobi:
                    rebalancing_assignments = self.gurobi_optimizer.optimize_vehicle_rebalancing_reject(vehicles_to_rebalance)
                else:
                    available_requests = []
                    if hasattr(self, 'active_requests') and self.active_requests:
                        available_requests = list(self.active_requests.values())
                    charging_stations = []
                    charging_stations = [station for station in self.charging_manager.stations.values() 
                               if station.available_slots > 0]
                    rebalancing_assignments = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance, available_requests, charging_stations)
                
                # Debug: Count assignments made
                new_assignments = 0
                charging_assignments = 0
                self.total_rebalancing_calls += 1
                
                # Apply the rebalancing assignments and generate corresponding actions
                for vehicle_id, target_request in rebalancing_assignments.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            # Handle charging assignment
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery)
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to charge
                        elif hasattr(target_request, 'request_id'):
                            # Handle regular request assignment  
                            self._assign_request_to_vehicle(vehicle_id, target_request.request_id)
                            new_assignments += 1
                            # Generate service action
                            from src.Action import ServiceAction
                            actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery)
                        elif isinstance(target_request, str) and target_request == "waiting":
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery)  # Stay in place
                        else:
                            self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                            print(f"Warning: Unknown assignment type for vehicle {vehicle_id}: {target_request}")
                    else:
                        # No assignment for this vehicle - generate idle action
                        from src.Action import IdleAction
                        vehicle = self.vehicles[vehicle_id]
                        current_coords = vehicle['coordinates']
                        # Generate random target for unassigned vehicles
                        vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                        target_x = max(0, min(self.grid_size-1, 
                                            current_coords[0] + random.randint(-3, 3)))
                        target_y = max(0, min(self.grid_size-1, 
                                            current_coords[1] + random.randint(-3, 3)))
                        target_coords = (target_x, target_y)
                        actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                
                # Store the count of request assignments for this rebalancing call
                self.rebalancing_assignments_per_step.append(new_assignments)
                
                #print(f"DEBUG Assignment Result: New request assignments: {new_assignments}, Charging assignments: {charging_assignments}, Idle: {len(vehicles_to_rebalance) - new_assignments - charging_assignments}")

        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords)
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'])
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'])
                else:
                    # Generate idle action with random target coordinates
                    current_coords = vehicle['coordinates']
                    # Generate random target coordinates within grid bounds
                    target_x = max(0, min(self.grid_size-1, 
                                        current_coords[0] + random.randint(-2, 2)))  # Random move within 2 steps
                    target_y = max(0, min(self.grid_size-1, 
                                        current_coords[1] + random.randint(-2, 2)))
                    target_coords = (target_x, target_y)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords)

        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
            
        return actions

    def _update_q_learning(self, actions, rewards, pre_action_states):
        """Update Q-learning based on actions taken and rewards received - unified with neural network training"""
        from .Action import ServiceAction, ChargingAction, IdleAction
        
        # Helper for Manhattan distance on location index
        def _manhattan_distance_loc(a_loc: int, b_loc: int) -> int:
            ax, ay = (a_loc % self.grid_size, a_loc // self.grid_size)
            bx, by = (b_loc % self.grid_size, b_loc // self.grid_size)
            return abs(ax - bx) + abs(ay - by)

        for vehicle_id in actions.keys():
            if vehicle_id in self.vehicles and vehicle_id in pre_action_states:
                action = actions[vehicle_id]
                batterynow = self.vehicles[vehicle_id]['battery']
                # Use pre-action as decision state
                current_location = pre_action_states[vehicle_id]['location']
                current_battery = pre_action_states[vehicle_id]['battery']
                veh_curloc = self.vehicles[vehicle_id]['location']
                # Only store at decision points; compute r_exec-style reward per option
                if (self.value_function and hasattr(self.value_function, 'store_experience')):
                    other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                    num_requests = len(self.active_requests)
                    store_threshold = 5
                    # Service option at assignment decision
                    if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and action.request_id in self.active_requests and rewards[vehicle_id] > store_threshold:
                        # Only at real assignment moment (vehicle not already busy)

                        req = self.active_requests[action.request_id]
                        veqid = vehicle_id
                        earnings = getattr(req, 'final_value', getattr(req, 'value', 0.0))
                        r_exec = rewards[vehicle_id]  # Use actual reward from step
                        next_battery = batterynow
                        target_location = req.pickup
                        next_location = req.dropoff

                        # request_value 用 r_exec 对齐优化器语义
                        self.value_function.store_experience(
                            vehicle_id=vehicle_id,
                            action_type=f"assign_{action.request_id}",
                            vehicle_location=current_location,
                            target_location=target_location,
                            current_time=self.current_time,
                            reward=r_exec,
                            next_vehicle_location=next_location,
                            battery_level=current_battery,
                            next_battery_level=next_battery,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=r_exec
                        )

                    # Charging option at decision
                    elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                        st_id = action.charging_station_id
                        if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch:
                            st = self.charging_manager.stations[st_id]
                            station_loc = st.location
                            r_exec = rewards[vehicle_id]  # Use actual reward from step
                            next_battery = batterynow
                            target_location = station_loc
                            next_location = station_loc

                            self.value_function.store_experience(
                                vehicle_id=vehicle_id,
                                action_type=f"charge_{st_id}",
                                vehicle_location=current_location,
                                target_location=target_location,
                                current_time=self.current_time,
                                reward=r_exec,
                                next_vehicle_location=next_location,
                                battery_level=current_battery,
                                next_battery_level=next_battery,
                                other_vehicles=other_vehicles,
                                num_requests=num_requests,
                                request_value=r_exec
                            )

                    # Idle/wait decision treated as single-step option
                    elif isinstance(action, IdleAction):
                        r_exec = rewards[vehicle_id]  # Use actual reward from step

                        self.value_function.store_experience(
                            vehicle_id=vehicle_id,
                            action_type="idle",
                            vehicle_location=current_location,
                            target_location=veh_curloc,
                            current_time=self.current_time,
                            reward=r_exec,
                            next_vehicle_location= veh_curloc,
                            battery_level=current_battery,
                            next_battery_level=batterynow,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=r_exec
                        )

    def _execute_action(self, vehicle_id, action):
        """Execute vehicle action with immediate reward aligned to Gurobi optimization objective"""
        from src.Action import ChargingAction, ServiceAction, IdleAction

        vehicle = self.vehicles[vehicle_id]
        
        # Check if vehicle is in stationary state
        if vehicle.get('is_stationary', False):
            # Reduce stationary duration
            vehicle['stationary_duration'] = 1
            
            # If stationary duration is finished, remove stationary status
            if vehicle['stationary_duration'] <= 0:
                vehicle['is_stationary'] = False
                vehicle['stationary_duration'] = 0
            

            active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
            active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
            avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
            # Return minimal reward for stationary period (no action taken)
            return -avg_request_value*(np.abs(np.random.normal(0, 0.05))+1e-3)
        
        reward = 0

        # Get parameters for consistency with Gurobi
        charging_penalty = getattr(self, 'charging_penalty', 2.0)

        if isinstance(action, ChargingAction):
            # Charging action - check if vehicle needs to move to station first
            if vehicle['charging_station'] is None:
                # Get the charging station and check location
                station_id = action.charging_station_id
                if station_id in self.charging_manager.stations:
                    station = self.charging_manager.stations[station_id]
                    current_location = vehicle['location']  # Use location index, not coordinates[0]
                    station_location = station.location
                    # Check if vehicle is already at the charging station
                    if current_location == station_location:
                        # Vehicle is at station - can start charging
                        success = station.start_charging(str(vehicle_id))
                        if success:
                            reward = -charging_penalty - np.random.random()*0.2
                        else:
                            reward = -charging_penalty - np.random.random()*0.2  # Station full penalty
                    else:
                        # Vehicle needs to move to charging station
                        vehicle['target_charging_station'] = station_id
                        #print(f"DEBUG: Vehicle {vehicle_id} moving towards charging station {station_id}")
                        reward = self._execute_movement_towards_charging_station(vehicle_id, station_id)
                else:
                    reward = -charging_penalty - np.random.random()*0.2  # Invalid station penalty
            else:
                reward = -charging_penalty - np.random.random()*0.2  # Invalid station penalty
            action.dur_reward += reward  # Store for reference
            reward = action.dur_reward  # Total reward over charging duration
        elif isinstance(action, ServiceAction):
            # Service action - immediate reward is request.value (same as Gurobi)
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                # Try to assign the request
                if self._assign_request_to_vehicle(vehicle_id, action.request_id):
                    if action.request_id in self.active_requests:
                        request = self.active_requests[action.request_id]
                        reward = np.random.normal(0, 0.1)  # Request not found
                    else:
                        reward = np.random.normal(0, 0.1)  # Request not found
                else:
                    reward = np.random.normal(0, 0.1)
                action.dur_reward += reward  # Store for reference
            elif vehicle['assigned_request'] is not None:
                # Progress towards pickup - check if我们能pickup
                if self._pickup_passenger(vehicle_id):
                    reward = 0.5 + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:
                        print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法继续前往pickup位置")
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                action.dur_reward += reward  # Store for reference
            elif vehicle['passenger_onboard'] is not None:
                earnings = self._dropoff_passenger(vehicle_id)
                if earnings > 0:
                    reward = earnings + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:

                        print(f"⚠️  车辆 {vehicle_id} 电池耗尽，乘客滞留无法到达dropoff位置")
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                action.dur_reward += reward  # Store for reference
            reward = action.dur_reward  # Total reward over charging duration
        elif isinstance(action, IdleAction):
            # Idle action - move towards specified target coordinates using unified method
            reward = self._execute_movement_towards_idle(vehicle_id, action.target_coords)
            action.dur_reward += reward  # Store for reference
            reward = action.dur_reward  # Total reward over charging duration
        return reward
    
    def _should_store_experience(self, action_type: str, reward: float, battery_level: float) -> bool:
        """
        决定是否存储experience，只存储关键决策点：
        1. 完成订单的experience（reward >= 15）
        2. 充电决策的experience（低电量情况）
        3. idle决策的experience（空闲移动决策）
        排除pickup/dropoff执行过程中的中间experience
        """
        # 1. 完成订单的experience（最高优先级）
        if reward >= 15:
            #print(f"✓ Storing COMPLETED ORDER experience: reward={reward}")
            return True
        
        # 2. 充电决策experience（电池管理的关键决策）
        if action_type.startswith('charge') and battery_level < 0.5:
            #print(f"✓ Storing CHARGING decision experience: battery={battery_level}")
            return True
        
        # 3. Idle决策experience（空闲状态的移动决策）
        if action_type == 'idle':
            return True  # 所有idle决策都存储
        
        # 4. 初始assignment决策（只存储决策时刻，不存储执行过程）
        if action_type.startswith('assign'):
            # 只存储真正的分配决策时刻或失败的分配
            if reward > 0 or reward <= -10:  # 成功分配或严重失败
                #print(f"✓ Storing assignment decision: reward={reward}")
                return True
            else:
                # 排除pickup/dropoff执行过程中的中间状态
                return False
        
        # 5. 其他情况不存储
        return False
    
    def _execute_movement_towards_target(self, vehicle_id):
        """Execute intelligent movement towards target (pickup/dropoff/charging)"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is not None:
            return -0.2  # Charging penalty for movement
            
        old_coords = vehicle['coordinates']
        target_coords = None
        movement_purpose = "idle"
        
        # 1. Priority 1: Move towards dropoff if passenger onboard
        if vehicle['passenger_onboard'] is not None:
            if vehicle['passenger_onboard'] in self.active_requests:
                request = self.active_requests[vehicle['passenger_onboard']]
                target_coords = (request.dropoff % self.grid_size, request.dropoff // self.grid_size)
                movement_purpose = "dropoff"
        
        # 2. Priority 2: Move towards pickup if request assigned
        elif vehicle['assigned_request'] is not None:
            if vehicle['assigned_request'] in self.active_requests:
                request = self.active_requests[vehicle['assigned_request']]
                target_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
                movement_purpose = "pickup"
        
        # 3. Priority 3: Move towards charging station if low battery
        elif vehicle['battery'] < self.min_battery_level and hasattr(vehicle, 'charging_target'):
            if vehicle['charging_target'] in self.charging_manager.stations:
                station = self.charging_manager.stations[vehicle['charging_target']]
                target_coords = (station.location % self.grid_size, station.location // self.grid_size)
                movement_purpose = "charging"
        
        # 4. Priority 4: Move towards charging station if target_location set
        elif 'target_location' in vehicle and vehicle['target_location'] is not None:
            target_coords = vehicle['target_location']
            movement_purpose = "rebalance_charging"
        
        # Calculate intelligent movement towards target
        if target_coords:
            current_x, current_y = old_coords
            target_x, target_y = target_coords
            
            # Move one step towards target (Manhattan distance)
            if current_x < target_x:
                new_x = current_x + 1
                new_y = current_y
            elif current_x > target_x:
                new_x = current_x - 1
                new_y = current_y
            elif current_y < target_y:
                new_x = current_x
                new_y = current_y + 1
            elif current_y > target_y:
                new_x = current_x
                new_y = current_y - 1
            else:
                # Already at target
                new_x, new_y = current_x, current_y
        else:
            # No specific target - random movement (exploration)
            new_x = max(0, min(self.grid_size-1, 
                             old_coords[0] + random.randint(-1, 1)))
            new_y = max(0, min(self.grid_size-1, 
                             old_coords[1] + random.randint(-1, 1)))
            movement_purpose = "exploration"
        
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        new_location_index = new_y * self.grid_size + new_x
        
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        vehicle['total_distance'] += distance
        
        # Track vehicle position for visualization
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []
        self.vehicle_position_history[vehicle_id].append({
            'coords': (new_x, new_y),
            'time': self.current_time,
            'action_type': movement_purpose
        })
        
        # Movement consumes battery
        vehicle['battery'] -= distance * (self.battery_consum + np.abs(np.random.random() * 0.0005))
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            print(f"⚠️  车辆 {vehicle_id} 在智能移动后电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return (self.movingpenalty  -  np.abs(np.random.normal(0, 0.05)))*distance if distance > 0 else -0.05 
    



    def _assign_idle_vehicle(self, vehicle_id):
        """Assign idle vehicle a target for random movement (without actually moving)"""
        if vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            
            # Only assign if vehicle is truly idle
            if (vehicle['assigned_request'] is None and 
                vehicle['passenger_onboard'] is None and
                vehicle['charging_station'] is None):
                
                # Set a random target for the vehicle (don't move yet, just set target)
                current_coords = vehicle['coordinates']
                target_x = max(0, min(self.grid_size-1, 
                                    current_coords[0] + random.randint(-1, 1)))
                target_y = max(0, min(self.grid_size-1, 
                                    current_coords[1] + random.randint(-1, 1)))
                
                # Store target for later use in action execution
                vehicle['idle_target'] = (target_x, target_y)
                
                return True
        return False
    def _execute_movement_towards_charging_station(self, vehicle_id, station_id):
        """Execute movement towards charging station"""
        vehicle = self.vehicles[vehicle_id]

            
        station = self.charging_manager.stations[station_id]
        current_location = vehicle['location']  # Use location index, not coordinates[0]
        station_location = station.location
        
        # Convert locations to coordinates
        current_x = current_location % self.grid_size
        current_y = current_location // self.grid_size
        target_x = station_location % self.grid_size
        target_y = station_location // self.grid_size
        
        # Move one step towards charging station (Manhattan distance)
        old_coords = vehicle['coordinates']
        if current_x < target_x:
            new_x = current_x + 1
            new_y = current_y
        elif current_x > target_x:
            new_x = current_x - 1
            new_y = current_y
        elif current_y < target_y:
            new_x = current_x
            new_y = current_y + 1
        elif current_y > target_y:
            new_x = current_x
            new_y = current_y - 1
            distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])

            vehicle['coordinates'] = (new_x, new_y)
            vehicle['location'] = new_y * self.grid_size + new_x
            vehicle['battery'] -= distance * (self.battery_consum + np.abs(np.random.random() * 0.0005))
            vehicle['battery'] = max(0, vehicle['battery'])
            return (self.movingpenalty  -  np.abs(np.random.normal(0, 0.05)))*distance 
        else:
            # Already at charging station - try to start charging
            success = station.start_charging(str(vehicle_id))
            #print(f"DEBUG: Vehicle {vehicle_id} at charging station {station_id}, trying to start: success={success}")
            if success:
                vehicle['charging_station'] = station_id
                vehicle['charging_time_left'] = getattr(self, 'charge_duration', 2)
                vehicle['charging_count'] += 1
                vehicle.pop('target_charging_station', None)  # Remove target
                #print(f"DEBUG: Vehicle {vehicle_id} started charging at station {station_id}")
                
                # Return charging penalty (same as Gurobi)
                charging_penalty = getattr(self, 'charging_penalty', 2.0)
                return -charging_penalty
            else:
                return 0
        
        # Update vehicle position
        new_location_index = new_y * self.grid_size + new_x
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        
        # Calculate distance moved
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        
        # Movement consumes battery
        vehicle['battery'] -= distance * (self.battery_consum + np.abs(np.random.random() * 0.0005))
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            print(f"⚠️  车辆 {vehicle_id} 前往充电站时电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return (self.movingpenalty  -  np.abs(np.random.normal(0, 0.05)))*distance 
    
    def _execute_movement_towards_idle(self, vehicle_id, target_coords):
        """Execute movement towards idle target coordinates"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is not None:
            return 0
        
        old_coords = vehicle['coordinates']
        current_x, current_y = old_coords
        target_x, target_y = target_coords
        
        # Move one step towards target coordinates (Manhattan distance)
        if current_x < target_x:
            new_x = current_x + 1
            new_y = current_y
        elif current_x > target_x:
            new_x = current_x - 1
            new_y = current_y
        elif current_y < target_y:
            new_x = current_x
            new_y = current_y + 1
        elif current_y > target_y:
            new_x = current_x
            new_y = current_y - 1
        else:
            # Already at target
            new_x, new_y = current_x, current_y
        if (new_x, new_y) == (current_x, current_y):
            # Reached target, clear idle target
            vehicle.pop('idle_target', None)
        # Update vehicle position
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        new_location_index = new_y * self.grid_size + new_x
        
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        vehicle['total_distance'] += distance
        
        # Track vehicle position for visualization
        if vehicle_id not in self.vehicle_position_history:
            self.vehicle_position_history[vehicle_id] = []
        self.vehicle_position_history[vehicle_id].append({
            'coords': (new_x, new_y),
            'time': self.current_time,
            'action_type': 'idle_movement'
        })
        
        # Movement consumes battery (same as other movement methods)
        vehicle['battery'] -= distance * (self.battery_consum + np.abs(np.random.random() * 0.0005))
        vehicle['battery'] = max(0, vehicle['battery'])
        
        # 检查电池是否耗尽，如果是则标记为需要紧急处理
        if vehicle['battery'] <= 0.0:
            vehicle['needs_emergency_charging'] = True
            #print(f"⚠️  车辆 {vehicle_id} 在闲置移动后电池耗尽 (位置: {new_x}, {new_y})")
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
        # Small time penalty for movement (consistent with other methods)
        return (self.movingpenalty  -  np.abs(np.random.normal(0, 0.05)))*distance   - avg_request_value*1e-3
    
    def _update_environment(self):
        """Update environment state"""
        self.current_time += 1
        
        # Generate new requests using selected method
        if self.use_intense_requests:
            new_requests = self._generate_intense_requests()  # Now returns a list

        else:
            new_requests = self._generate_random_requests()  # Now also returns a list
        
        # Update charging status
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['charging_station'] is not None:
                vehicle['charging_time_left'] -= 1
                #print(f"DEBUG: Vehicle {vehicle_id} charging at station {vehicle['charging_station']}, time left: {vehicle['charging_time_left']}")
                # Charging increases battery (reduced rate for more realistic charging)
                vehicle['battery'] += self.chargeincrease_per_epoch + np.random.random()*0.001
                vehicle['battery'] = min(1.0, vehicle['battery'])
                
                # Charging complete
                if vehicle['charging_time_left'] <= 0:
                    station_id = vehicle['charging_station']
                    if station_id in self.charging_manager.stations:
                        station = self.charging_manager.stations[station_id]
                        station.stop_charging(str(vehicle_id))
                    vehicle['charging_station'] = None
                    self.charge_finished += 1
                    self.charge_stats[station_id].append(self.current_time)
        # Synchronize charging states between vehicles and stations
        # This fixes the issue where stations auto-start vehicles from queue but don't update vehicle state
        for station_id, station in self.charging_manager.stations.items():
            for charging_vehicle_id in station.current_vehicles:
                vehicle_id = int(charging_vehicle_id)
                if vehicle_id in self.vehicles:
                    vehicle = self.vehicles[vehicle_id]
                    # If vehicle is in station but doesn't know it's charging, sync the state
                    if vehicle['charging_station'] is None:
                        vehicle['charging_station'] = station_id
                        vehicle['charging_time_left'] = getattr(self, 'default_charging_duration', 2)
                        # Don't increment charging_count as this is just a sync operation
        
        # Remove expired requests and apply unserved penalty
        current_time = self.current_time
        expired_requests = []
        unserved_penalty_total = 0
        
        for request_id, request in self.active_requests.items():
            if current_time > request.pickup_deadline:
                expired_requests.append(request_id)
                # Apply penalty for unserved request
                unserved_penalty_total += self.unserved_penalty
        
        # Move expired requests to rejected list
        for request_id in expired_requests:
            request = self.active_requests[request_id]
            self.rejected_requests.append(request)
            del self.active_requests[request_id]
        
        # Distribute unserved penalty among all vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            if self.vehicles[vehicle_id]['assigned_request'] and self.vehicles[vehicle_id]['assigned_request'] in expired_requests:
                vehicle['assigned_request'] = None  # Clear assigned request if it expired
            if self.vehicles[vehicle_id]['passenger_onboard'] and self.vehicles[vehicle_id]['battery'] <= self.min_battery_level:
                # Handle passenger stranding due to low battery
                request_id = vehicle['passenger_onboard']
                vehicle['passenger_onboard'] = None  
                # Add to unserved penalty instead of trying to modify non-existent reward field
                vehicle['unserved_penalty'] += self.penalty_for_passenger_stranding
                # Remove the stranded passenger's request from active requests
                if request_id in self.active_requests:
                    request = self.active_requests[request_id]
                    self.rejected_requests.append(request)
                    del self.active_requests[request_id]

            
    def reset(self):
        """重置环境"""
        self.current_time = 0
        # Reset request system
        self.active_requests = {}
        self.completed_requests = []
        self.rejected_requests = []
        self.request_counter = 0
        self.charge_finished = 0
        self.charge_stats = {station_id: [] for station_id in self.charging_manager.stations}
        # Reset rebalancing assignment tracking
        self.rebalancing_assignments_per_step = []
        self.total_rebalancing_calls = 0
        
        self._setup_vehicles()
        return self.get_initial_states()
    
    def get_episode_stats(self):
        """Get detailed statistics for current episode"""
        # Calculate average battery level
        total_battery = sum(v['battery'] for v in self.vehicles.values())
        avg_battery = total_battery / len(self.vehicles) if self.vehicles else 0
        
        # Calculate total rejected requests (unique orders that were rejected)
        total_rejected = len(self.rejected_requests)
        
        # Calculate average charging station utilization
        if not hasattr(self, 'charging_manager') or not self.charging_manager.stations:
            #print("DEBUG: No charging manager or stations found!")
            total_capacity = 0
            total_occupied = 0
            avg_station_utilization = 0
            avg_vehicles_per_station = 0
        else:
            total_capacity = sum(station.max_capacity for station in self.charging_manager.stations.values())
            total_occupied = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            
            # Debug: Check vehicle charging status consistency
            vehicles_with_charging_station = [v for v in self.vehicles.values() if v['charging_station'] is not None]
            station_vehicle_count = sum(len(station.current_vehicles) for station in self.charging_manager.stations.values())
            
            # Debug output for inconsistency detection
            # if len(vehicles_with_charging_station) != station_vehicle_count:
            #     print(f"DEBUG: Charging status inconsistency!")
            #     print(f"  Vehicles with charging_station set: {len(vehicles_with_charging_station)}")
            #     print(f"  Vehicles recorded in stations: {station_vehicle_count}")
                
            #     # Show individual station states
            #     for station_id, station in self.charging_manager.stations.items():
            #         if len(station.current_vehicles) > 0:
            #             print(f"  Station {station_id}: {len(station.current_vehicles)} vehicles: {station.current_vehicles}")
                        
            #     # Show vehicle charging states
            #     for vid, vehicle in self.vehicles.items():
            #         if vehicle['charging_station'] is not None:
            #             print(f"  Vehicle {vid}: charging at station {vehicle['charging_station']}")
            
            # Calculate charging station utilization using episode history
            if self.charging_usage_history:
                # Calculate average over entire episode
                avg_vehicles_per_station = sum(usage['vehicles_per_station'] for usage in self.charging_usage_history) / len(self.charging_usage_history)
                avg_total_occupied = sum(usage['total_occupied'] for usage in self.charging_usage_history) / len(self.charging_usage_history)
                avg_station_utilization = avg_total_occupied / max(1, total_capacity)
                
                #print(f"DEBUG: Episode charging stats - History points: {len(self.charging_usage_history)}, Avg occupied: {avg_total_occupied:.1f}, Avg per station: {avg_vehicles_per_station:.2f}")
            else:
                # Fallback to current state if no history
                avg_station_utilization = total_occupied / max(1, total_capacity)
                avg_vehicles_per_station = total_occupied / max(1, len(self.charging_manager.stations))
                #print(f"DEBUG: No charging history - using current state: {avg_vehicles_per_station:.2f}")
            
            # Additional debug info for station usage
            #print(f"DEBUG: Station stats - Total stations: {len(self.charging_manager.stations)}, Total occupied: {total_occupied}, Avg per station: {avg_vehicles_per_station}")
            
            # Debug: Show detailed station status
            # for station_id, station in self.charging_manager.stations.items():
            #     print(f"  Station {station_id}: current_vehicles={station.current_vehicles}, capacity={station.max_capacity}")
            
            # Debug: Show vehicles that think they're charging
            charging_vehicles = [vid for vid, v in self.vehicles.items() if v['charging_station'] is not None]
            #print(f"  Vehicles with charging_station set: {charging_vehicles}")
        
        # Count active and completed requests
        active_orders = len(self.active_requests)
        completed_orders = len(self.completed_requests)
        total_orders = active_orders + completed_orders + total_rejected
        accepted_orders = active_orders + completed_orders
        
        # Vehicle type breakdown
        ev_vehicles = [v for v in self.vehicles.values() if v['type'] == 1]  # EV
        aev_vehicles = [v for v in self.vehicles.values() if v['type'] == 2]  # AEV
        
        ev_rejected = sum(v['rejected_requests'] for v in ev_vehicles)
        aev_rejected = sum(v['rejected_requests'] for v in aev_vehicles)
        
        # Calculate rebalancing assignment statistics
        avg_rebalancing_assignments = 0
        total_rebalancing_assignments = 0
        if self.rebalancing_assignments_per_step:
            total_rebalancing_assignments = sum(self.rebalancing_assignments_per_step)
            avg_rebalancing_assignments = total_rebalancing_assignments / len(self.rebalancing_assignments_per_step)
        
        return {
            'episode_time': self.current_time,
            'total_orders': total_orders,
            'accepted_orders': accepted_orders,
            'rejected_orders': total_rejected,
            'active_orders': active_orders,
            'completed_orders': completed_orders,
            'avg_battery_level': avg_battery,
            'finished_charge': self.charge_finished,
            'charge_stats': self.charge_stats,
            'total_vehicles': len(self.vehicles),
            'ev_count': len(ev_vehicles),
            'aev_count': len(aev_vehicles),
            'ev_rejected': ev_rejected,
            'aev_rejected': aev_rejected,
            'total_stations': len(self.charging_manager.stations),
            'vehicles_charging': len([v for v in self.vehicles.values() if v['charging_station'] is not None]),
            # Rebalancing assignment statistics
            'total_rebalancing_calls': self.total_rebalancing_calls,
            'total_rebalancing_assignments': total_rebalancing_assignments,
            'avg_rebalancing_assignments_per_call': avg_rebalancing_assignments,
            'rebalancing_assignments_per_step': self.rebalancing_assignments_per_step.copy()
        }

    def get_stats(self):
        """Get environment statistics including request fulfillment and vehicle types"""
        total_battery = sum(v['battery'] for v in self.vehicles.values())
        avg_battery = total_battery / len(self.vehicles)
        
        total_charging = sum(v['charging_count'] for v in self.vehicles.values())
        total_rejected = sum(v['rejected_requests'] for v in self.vehicles.values())
        
        # Vehicle type statistics
        ev_vehicles = [v for v in self.vehicles.values() if v['type'] == 1]  # EV
        aev_vehicles = [v for v in self.vehicles.values() if v['type'] == 2]  # AEV
        
        ev_rejected = sum(v['rejected_requests'] for v in ev_vehicles)
        aev_rejected = sum(v['rejected_requests'] for v in aev_vehicles)
        
        vehicles_with_requests = len([v for v in self.vehicles.values() 
                                    if v['assigned_request'] is not None or v['passenger_onboard'] is not None])
        
        return {
            'average_battery': avg_battery,
            'total_charging_events': total_charging,
            'vehicles_charging': len([v for v in self.vehicles.values() 
                                    if v['charging_station'] is not None]),
            'active_requests': len(self.active_requests),
            'completed_requests': len(self.completed_requests),
            'vehicles_with_requests': vehicles_with_requests,
            'request_fulfillment_rate': len(self.completed_requests) / max(1, len(self.completed_requests) + len(self.active_requests)),
            'total_rejected_requests': total_rejected,
            'ev_count': len(ev_vehicles),
            'aev_count': len(aev_vehicles),
            'ev_rejected': ev_rejected,
            'aev_rejected': aev_rejected,
            'ev_rejection_rate': ev_rejected / max(1, ev_rejected + len(self.completed_requests)) if ev_vehicles else 0,
            'aev_rejection_rate': aev_rejected / max(1, aev_rejected + len(self.completed_requests)) if aev_vehicles else 0
        }
