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
import math
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
        self.charging_penalty = -1.0
        self.chargeincrease_per_epoch = 0.1  # Battery increase per epoch when charging
        self.min_battery_level = 0.2
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

    def __init__(self, num_vehicles=5, num_stations=3, ev_num_vehicles=None, grid_size=10, use_intense_requests=True, assignmentgurobi=True, random_seed=None,decision_mode="integrated"):  # Increased grid size
        # Provide required parameters for base class
        NUM_LOCATIONS = grid_size * grid_size  # Total locations in grid
        MAX_CAPACITY = 4  # Maximum capacity per location
        EPOCH_LENGTH = 1.0  # Length of each epoch
        NUM_AGENTS = num_vehicles  # Number of vehicles/agents
        START_EPOCH = 0.0  # Start time
        STOP_EPOCH = 100.0  # Stop time
        DATA_DIR = "data"  # Data directory (not used in this implementation)
        self.use_intense_requests = use_intense_requests
        # 设置随机数种子以确保可重复性
        self.initial_random_seed = random_seed  # 保存初始种子用于车辆初始化
        self.request_generation_seed = random_seed  # 请求生成的种子，可以单独设置
        if random_seed is not None:
            self.set_random_seed(random_seed)
        
        super().__init__(NUM_LOCATIONS, MAX_CAPACITY, EPOCH_LENGTH, NUM_AGENTS, START_EPOCH, STOP_EPOCH, DATA_DIR)
        self.assignmentgurobi = assignmentgurobi  # Whether to use Gurobi for assignment
        self.num_vehicles = num_vehicles
        self.ev_num_vehicles = ev_num_vehicles if (hasattr(self, 'ev_num_vehicles')) else int(num_vehicles // 2)
        self.num_stations = num_stations
        self.grid_size = grid_size
        self.num_zones = getattr(self, 'num_zones', 4)
        self.minimum_charging_level = 0.2  # Minimum battery level before needing to charge
        # Parameters for reward alignment with Gurobi optimization
        self.charging_penalty = 2 
        self.adp_value = 1.0  # Weight for Q-value contribution
        self.unserved_penalty = 50
        self.idle_vehicle_requirement = 1  # Minimum idle vehicles required
        self.charge_duration = 2
        self.chargeincrease_per_epoch = 0.5
        self.chargeincrease_whole = self.chargeincrease_per_epoch * self.charge_duration
        self.min_battery_level = 0.2
        self.charge_finished = 0.0
        self.penalty_reject_requestnum = 2
        self.penalty_time = 5
        self.charge_stats = {}
        self.decision_mode = "integrated"  # "integrated" or "sequential"
        self.decision_mode_set = {"integrated", "aev_first","ev_first"}
        # Initialize charging station manager
        self.charging_manager = ChargingStationManager()
        self._setup_charging_stations()
        self.unserve_penalty = -0.5  # Penalty for unserved requests
        self.movingpenalty = -5e-3
        # Vehicle states
        self.rebalance_battery_threshold = 0.3
        self.heuristic_battery_threshold = 0.5  # Battery threshold for heuristic rebalancing
        self.vehicles = {}
        self.storeactions = {}
        self.storeactions_ev = {}
        self.whole_req = 0
        # Total generated requests counter (used in _update_environment/get_episode_stats)
        self.whole_req_num = 0
        self.hotspot_locations = []
        self.initialise_environment()
        print(self.hotspot_locations)
        self.ev_requests = []
        # Environment state
        self.current_time = 0
        self.episode_length = 200


        self.idle_charging_num = {station_id: 0 for station_id in range(self.num_stations)}
        self.current_online = 0

        
        
        
        
        self.hotspot_locations_num = self.num_zones
        # Request system
        self.active_requests = {}  # Active passenger requests
        self.completed_requests = []  # Completed requests for analysis
        self.completed_requests_ev = []  # Completed EV requests for analysis
        self.rejected_requests = []  # Rejected requests for analysis
        self.request_counter = 0
        self.request_generation_rate = 0.8  # Increased to 60% for more active environment
        self.use_intense_requests = use_intense_requests  # Whether to use concentrated request generation
        self.battery_consum = 0.015  # Battery consumption per epoch when moving
        # Assignment tracking for rebalancing analysis
        self.rebalancing_assignments_per_step = []  # Store assignments count per step
        self.rebalancing_whole = []
        self.total_rebalancing_calls = 0
        self.penalty_for_passenger_stranding = -50  
        # Tracking for visualization
        self.request_generation_history = []  # Track where requests are generated
        self.vehicle_position_history = {}  # Track vehicle movement patterns
        
        # Charging station usage tracking for episode-wide statistics
        self.charging_usage_history = []  # Track charging station usage over time

        # =============================
        # Zone system (rsimulation_detail)
        # =============================
        # Node (location_id) -> zone_id mapping and zone definitions.
        # Default: partition grid into sqrt(Z) x sqrt(Z) blocks.
        self.loc_to_zone = {}
        self.zone_to_locs = {}
        self.zoneinfo = {"1": "Surge", "2": "HighDemand", "3": "CityCenter", "4": "Normal"}
        self.surge_zone_locs = set()
        self.high_demand_zone_locs = set()
        self.city_center_zone_locs = set()
        self._init_zones()

        # =============================
        # EV behavior (rsimulation_detail-inspired)
        # =============================
        # Idle time tracking: last completion -> next acceptance.
        self.ev_last_completed_time = {}
        self.ev_last_accepted_time = {}
        self.ev_current_idle_start_time = {}
        self.ev_idle_durations = []

        # Consecutive rejection penalty: 2 consecutive rejections triggers cooldown.
        self.ev_consecutive_rejections = {}
        self.ev_penalty_until_time = {}
        self.ev_penalty_duration = 5  # epochs; can be tuned externally

        # Probabilistic decision model parameters (kept simple/robust).
        self.ev_charge_soc_threshold = 0.25
        self.ev_charge_soc_slope = 12.0
        self.ev_station_choice_beta = 1.0
        self.relocation_beta = 1.0

        # =============================
        # rsimulation_detail nested-logit models
        # =============================
        from src.charging_models import ChargingProbabilityCalculator, RelocationManager
        self.charging_calculator = ChargingProbabilityCalculator(grid_size=self.grid_size)
        self.relocation_manager = RelocationManager(grid_size=self.grid_size)
        # Update relocation manager with initial zone sets
        self.relocation_manager.update_zone_info(
            surge_ids=list(self.surge_zone_locs),
            hd_ids=list(self.high_demand_zone_locs),
            city_center_ids=list(self.city_center_zone_locs)
        )
        
        # Initialize ValueFunction for Q-value calculation (will be set externally)
        self.value_function = None
        self.value_function_ev = None
        print(f"✓ Initialized integrated environment: {num_vehicles} vehicles, {num_stations} charging stations")

    # =============================
    # EV metrics & penalty helpers
    # =============================
    def _is_ev(self, vehicle_id: int) -> bool:
        v = self.vehicles.get(vehicle_id)
        return bool(v is not None and v.get('type', 0) == 1)

    def _in_ev_penalty(self, vehicle_id: int) -> bool:
        if not self._is_ev(vehicle_id):
            return False
        until_t = float(self.ev_penalty_until_time.get(vehicle_id, -1.0))
        return float(self.current_time) < until_t

    def _record_ev_acceptance(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_accepted_time[vehicle_id] = now
        idle_start = self.ev_current_idle_start_time.get(vehicle_id)
        if idle_start is not None:
            self.ev_idle_durations.append(max(0.0, now - float(idle_start)))
        self.ev_current_idle_start_time[vehicle_id] = None
        self.ev_consecutive_rejections[vehicle_id] = 0

    def _record_ev_completion(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        now = float(self.current_time)
        self.ev_last_completed_time[vehicle_id] = now
        self.ev_current_idle_start_time[vehicle_id] = now

    def _record_ev_rejection(self, vehicle_id: int):
        if not self._is_ev(vehicle_id):
            return
        cnt = int(self.ev_consecutive_rejections.get(vehicle_id, 0)) + 1
        self.ev_consecutive_rejections[vehicle_id] = cnt
        if cnt >= 2:
            self.ev_penalty_until_time[vehicle_id] = float(self.current_time) + float(self.ev_penalty_duration)
            self.ev_consecutive_rejections[vehicle_id] = 0

    # =============================
    # EV probabilistic decisions
    # =============================
    def _sigmoid(self, x: float) -> float:
        # numerically safe sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        z = math.exp(x)
        return z / (1.0 + z)

    def compute_ev_charge_probability(self, vehicle_id: int) -> Tuple[float, Dict[int, float]]:
        """Return (p_charge, station_probs) for an EV using nested-logit model.

        Calls ChargingProbabilityCalculator from rsimulation_detail.
        Returns: (p_charge, {station_id: probability})
        """
        if not self._is_ev(vehicle_id) or not hasattr(self, 'charging_manager'):
            return 0.0, {}

        vehicle = self.vehicles[vehicle_id]
        vehicle_loc = int(vehicle.get('location', 0))
        # SOC in [0,1] but calculator expects [0,100]
        soc_percent = float(vehicle.get('battery', 0.0)) * 100.0

        # Build charging station list for calculator
        stations_data = []
        for sid, station in self.charging_manager.stations.items():
            # Estimate time: queue_length * charge_duration + service_time
            queue_len = len(getattr(station, 'charging_queue', []))
            estimated_time = (queue_len * self.charge_duration) + self.charge_duration
            stations_data.append({
                'id': int(station.location),
                'estimated_time': float(estimated_time)
            })

        if not stations_data:
            return 0.0, {}

        # Call nested-logit calculator
        result = self.charging_calculator.calculate_probabilities(
            origin_id=vehicle_loc,
            dest_id=vehicle_loc,  # Idle EV: assume dest=origin for now
            current_soc=soc_percent,
            charging_stations=stations_data
        )

        p_charge = float(result['action_charge'])
        # Map station location_id back to station_id (sid)
        station_probs = {}
        for loc_id, prob in result['station_probs'].items():
            # Find station_id matching this location
            for sid, station in self.charging_manager.stations.items():
                if int(station.location) == int(loc_id):
                    station_probs[int(sid)] = float(prob)
                    break
        return p_charge, station_probs

    def compute_ev_relocation_probability(self, vehicle_id: int) -> Dict[str, float]:
        """Return probabilities over relocation actions after refusing to charge.

        Calls RelocationManager from rsimulation_detail (Ashkrof et al. 2024).
        Actions: Wait / Surge / HighDemand / Cruise.
        """
        if not self._is_ev(vehicle_id):
            return {'Wait': 1.0}

        vehicle = self.vehicles[vehicle_id]
        vehicle_loc = int(vehicle.get('location', 0))
        completed_orders = len([r for r in self.completed_requests if r.request_id in getattr(vehicle, 'completed_order_ids', [])])

        # Ensure idle_start_time is valid
        idle_start = self.ev_current_idle_start_time.get(vehicle_id)
        if idle_start is None:
            idle_start = self.current_time
        
        agent_state = {
            'location': vehicle_loc,
            'completed_trips': completed_orders,
            'current_wait_time': float(self.current_time - idle_start)
        }
        global_state = {
            'current_surge_price': 0.0,  # Can be dynamic if environment tracks surge pricing
            'is_weekend': 0  # Can be set from datetime if available
        }

        # Update relocation_manager zone sets (in case they changed)
        self.relocation_manager.update_zone_info(
            surge_ids=list(self.surge_zone_locs),
            hd_ids=list(self.high_demand_zone_locs),
            city_center_ids=list(self.city_center_zone_locs)
        )

        utils, targets = self.relocation_manager.calculate_relocation_utilities(agent_state, global_state)
        # Softmax
        u_vec = np.array(list(utils.values()))
        exp_u = np.exp(u_vec - np.max(u_vec))
        probs_array = exp_u / np.sum(exp_u)
        probs = dict(zip(utils.keys(), probs_array))
        return probs

    def sample_ev_relocation_target(self, vehicle_id: int, action_type: str):
        """Sample a target node (location_id) for a relocation action.
        
        action_type: 'Wait', 'Surge', 'HighDemand', 'Cruise'
        """
        vehicle = self.vehicles.get(vehicle_id, {})
        cur_loc = int(vehicle.get('location', 0))
        if action_type == 'Wait':
            return cur_loc

        # Use relocation_manager's nearest zone logic
        agent_state = {'location': cur_loc, 'completed_trips': 0, 'current_wait_time': 0.0}
        global_state = {'current_surge_price': 0.0, 'is_weekend': 0}
        _, targets = self.relocation_manager.calculate_relocation_utilities(agent_state, global_state)

        if action_type == 'Surge':
            # 在Surge区域内随机选择一个位置
            if self.surge_zone_locs:
                return int(random.choice(list(self.surge_zone_locs)))
            # 如果Surge区域为空，使用relocation_manager的目标
            elif targets.get('Surge') is not None:
                return int(targets['Surge'])
            return cur_loc
        elif action_type == 'HighDemand':
            # 在HighDemand区域内随机选择一个位置
            if self.high_demand_zone_locs:
                return int(random.choice(list(self.high_demand_zone_locs)))
            # 如果HighDemand区域为空，使用relocation_manager的目标
            elif targets.get('HighDemand') is not None:
                return int(targets['HighDemand'])
            return cur_loc
        elif action_type == 'Cruise':
            # Random neighbor within same zone
            zid = self.get_zone_id(cur_loc)
            candidate_locs = self.get_zone_locations(zid)
            if candidate_locs:
                return int(random.choice(candidate_locs))
        return cur_loc

    def _init_zones(self):
        """Initialize default zone partition and node<->zone membership."""
        self.loc_to_zone = {}
        self.zone_to_locs = {}

        grid_size = int(getattr(self, 'grid_size', 1))
        num_zones = int(getattr(self, 'num_zones', 4))
        if grid_size <= 0:
            return

        # Use near-square zone layout.
        zones_per_side = max(1, int(round(math.sqrt(num_zones))))
        block_w = max(1, int(math.ceil(grid_size / zones_per_side)))
        block_h = block_w

        def _zone_id_for_xy(x: int, y: int) -> int:
            zx = min(zones_per_side - 1, x // block_w)
            zy = min(zones_per_side - 1, y // block_h)
            return zy * zones_per_side + zx

        for loc in range(grid_size * grid_size):
            x = loc % grid_size
            y = loc // grid_size
            zid = _zone_id_for_xy(x, y)
            self.loc_to_zone[loc] = zid
            self.zone_to_locs.setdefault(zid, []).append(loc)
        
        # 根据 zoneinfo 定义填充各区域的位置集合
        # zoneinfo: {"1": "Surge", "2": "HighDemand", "3": "CityCenter", "4": "Normal"}
        for zid, zone_type in self.zoneinfo.items():
            zone_id_int = int(zid) - 1  # zoneinfo keys are 1-indexed, zone_to_locs keys are 0-indexed
            if zone_id_int in self.zone_to_locs:
                if zone_type == "Surge":
                    self.surge_zone_locs.update(self.zone_to_locs[zone_id_int])
                elif zone_type == "HighDemand":
                    self.high_demand_zone_locs.update(self.zone_to_locs[zone_id_int])
                elif zone_type == "CityCenter":
                    self.city_center_zone_locs.update(self.zone_to_locs[zone_id_int])


    
    
    def get_zone_id(self, location_id: int) -> int:
        """Return zone id for a node/location."""
        if not self.loc_to_zone:
            self._init_zones()
        return int(self.loc_to_zone.get(int(location_id), 0))

    def get_zone_locations(self, zone_id: int):
        """Return all node ids belonging to a zone."""
        if not self.zone_to_locs:
            self._init_zones()
        return list(self.zone_to_locs.get(int(zone_id), []))
    
    def set_random_seed(self, seed):
        """
        设置环境内部的随机数种子，确保可重复的实验结果
        
        Args:
            seed (int): 随机数种子
        """
        random.seed(seed)
        np.random.seed(seed)
        print(f"✓ Environment random seed set to {seed}")
    
    def set_request_generation_seed(self, seed):
        """
        专门设置请求生成的随机数种子，用于控制每个episode的请求序列
        
        Args:
            seed (int): 请求生成专用的随机数种子
        """
        self.request_generation_seed = seed
        # 临时保存当前随机状态
        current_random_state = random.getstate()
        current_numpy_state = np.random.get_state()
        
        # 设置请求生成专用种子
        random.seed(seed)
        np.random.seed(seed)
        
        # 重新保存请求生成的随机状态
        self._request_random_state = random.getstate()
        self._request_numpy_state = np.random.get_state()
        
        # 恢复原来的随机状态
        random.setstate(current_random_state)
        np.random.set_state(current_numpy_state)
        
        print(f"✓ Request generation seed set to {seed}")
    
    def _set_request_generation_random_state(self):
        """在生成请求前设置请求生成专用的随机状态"""
        if hasattr(self, '_request_random_state') and hasattr(self, '_request_numpy_state'):
            random.setstate(self._request_random_state)
            np.random.set_state(self._request_numpy_state)
    
    def _save_request_generation_random_state(self):
        """在生成请求后保存请求生成的随机状态"""
        if hasattr(self, '_request_random_state') and hasattr(self, '_request_numpy_state'):
            self._request_random_state = random.getstate()
            self._request_numpy_state = np.random.get_state()
    
    def set_value_function(self, value_function):
        """Set the value function for Q-value calculation"""
        self.value_function = value_function
        print(f"✓ Value function set: {type(value_function).__name__}")
    def set_value_function_ev(self, value_function):
        """Set the value function for Q-value calculation"""
        self.value_function_ev = value_function
        print(f"✓ Value function ev  set: {type(value_function).__name__}")
        
        
        
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
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        v_after = self.value_function.get_assignment_q_value(
                vehicle_id=vehicle_id,
                target_id=request.request_id,
                vehicle_location=cur_loc,
                target_reject=request.pickup,
                target_location=request.dropoff,
                current_time=self.current_time,
                other_vehicles=max(0, other_idle),
                num_requests=num_reqs,
                battery_level=cur_bat,
            )
        
        return  v_after

    def batch_evaluate_service_options(self, vehicle_request_pairs, ifEVQvalue = False):
        """
        批量计算多个vehicle-request对的拒绝感知调整Q值，提高计算效率
        现在集成神经网络预测的拒绝率: Q_value - immediate_reward * rejection_probability
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            
        Returns:
            List of rejection-aware adjusted Q-values corresponding to each vehicle-request pair
        """
        if not vehicle_request_pairs:
            return []
        
        # 准备批量输入数据
        batch_inputs = []
        valid_pairs = []
        
        for vehicle_id, request in vehicle_request_pairs:
            # 检查vehicle和request的有效性
            veh = self.vehicles.get(vehicle_id)
            if veh is None or request is None:
                continue
                
            # 如果request是ID，解析为对象
            if isinstance(request, (int, str)) and request in self.active_requests:
                request = self.active_requests[request]
            if request is None:
                continue
                
            cur_loc = veh['location']
            cur_bat = veh['battery']
            num_reqs = len(self.active_requests)
            other_idle = len([v for vid, v in self.vehicles.items() 
                             if vid != vehicle_id and v['assigned_request'] is None 
                             and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            # 准备神经网络输入
            input_data = {
                'vehicle_id': vehicle_id,
                'target_id': request.request_id,
                'vehicle_location': cur_loc,
                'target_reject': request.pickup,
                'target_location': request.dropoff,
                'current_time': self.current_time,
                'other_vehicles': max(0, other_idle),
                'num_requests': num_reqs,
                'battery_level': cur_bat,
            }
            
            batch_inputs.append(input_data)
            valid_pairs.append((vehicle_id, request))
        
        if not batch_inputs:
            return []
        
        # 批量计算基础Q值
        try:
            base_q_values = []
            if hasattr(self.value_function, 'batch_get_assignment_q_value'):
                # 如果value function支持批量计算
                if ifEVQvalue:
                    base_q_values = self.value_function_ev.batch_get_assignment_q_value(batch_inputs)
                else:
                    base_q_values = self.value_function.batch_get_assignment_q_value(batch_inputs)
            else:
                # 否则使用优化的单独计算
                for input_data in batch_inputs:
                    q_value = self.value_function.get_assignment_q_value(**input_data)
                    base_q_values.append(q_value)
            
            # 批量计算拒绝感知调整值
            adjusted_q_values = []
            for i, (vehicle_id, request) in enumerate(valid_pairs):
                base_q = base_q_values[i] if i < len(base_q_values) else 0.0
                
                # 计算拒绝感知调整: Q_value - immediate_reward * rejection_probability
                adjusted_q = self._calculate_rejection_aware_adjustment(
                    vehicle_id, request, base_q
                )
                adjusted_q_values.append(adjusted_q)
            
            return adjusted_q_values
            
        except Exception as e:
            print(f"Batch Q-value calculation failed: {e}")
            # 回退到逐个计算
            return [self.evaluate_service_option(vehicle_id, request) 
                    for vehicle_id, request in valid_pairs]
    
    


    def batch_evaluate_service_options_meanfield(self, vehicle_request_pairs, ifEVQvalue = False):
        """
        批量计算多个vehicle-request对的Mean Field Q值
        使用周围智能体的历史决策分布作为条件变量
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            ifEVQvalue: 是否使用EV的value function
            
        Returns:
            List of Mean Field Q-values: Q(s, a, μ) where μ is mean action distribution
        """
        if not vehicle_request_pairs:
            return []
        
        # 选择合适的 value function
        value_func = self.value_function_ev if ifEVQvalue else self.value_function
        
        # 检查 value function 是否支持 mean field
        if not hasattr(value_func, 'compute_mean_field') or not hasattr(value_func, 'batch_get_q_value_with_mean_field'):
            # 如果不支持 mean field，回退到普通方法
            return self.batch_evaluate_service_options(vehicle_request_pairs, ifEVQvalue)
        
        # 准备批量输入数据
        batch_inputs = []
        valid_pairs = []
        mean_fields = []
        
        # 收集所有车辆位置信息用于计算邻居
        agent_locations = {}
        for vid, vehicle in self.vehicles.items():
            loc = vehicle.get('location', 0)
            x = loc % self.grid_size
            y = loc // self.grid_size
            agent_locations[vid] = (x, y)
        
        for vehicle_id, request in vehicle_request_pairs:
            # 检查 vehicle 和 request 的有效性
            veh = self.vehicles.get(vehicle_id)
            if veh is None or request is None:
                continue
                
            # 如果 request 是 ID，解析为对象
            if isinstance(request, (int, str)) and request in self.active_requests:
                request = self.active_requests[request]
            if request is None:
                continue
            
            # 计算该车辆的邻居智能体的平均动作分布（mean field）
            mean_field = value_func.compute_mean_field(
                environment=self,
                agent_id=vehicle_id,
                agent_locations=agent_locations
            )
            mean_fields.append(mean_field)
            
            # 准备状态特征
            cur_loc = veh['location']
            cur_bat = veh['battery']
            num_reqs = len(self.active_requests)
            other_idle = len([v for vid, v in self.vehicles.items() 
                             if vid != vehicle_id and v['assigned_request'] is None 
                             and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            # 准备输入数据
            input_data = {
                'vehicle_id': vehicle_id,
                'target_id': request.request_id,
                'vehicle_location': cur_loc,
                'target_reject': request.pickup,
                'target_location': request.dropoff,
                'current_time': self.current_time,
                'other_vehicles': max(0, other_idle),
                'num_requests': num_reqs,
                'battery_level': cur_bat,
                'request_value': getattr(request, 'final_value', getattr(request, 'value', 0.0))
            }
            
            batch_inputs.append(input_data)
            valid_pairs.append((vehicle_id, request))
        
        if not batch_inputs:
            return []
        
        # 批量计算 Mean Field Q值
        try:
            # 使用 value function 的批量 mean field Q值计算方法
            mean_field_q_values = value_func.batch_get_q_value_with_mean_field(
                batch_inputs, 
                mean_fields
            )
            
            # 应用拒绝感知调整（如果需要）
            adjusted_q_values = []
            for i, (vehicle_id, request) in enumerate(valid_pairs):
                base_q = mean_field_q_values[i] if i < len(mean_field_q_values) else 0.0
                
                # 计算拒绝感知调整
                adjusted_q = self._calculate_rejection_aware_adjustment(
                    vehicle_id, request, base_q
                )
                adjusted_q_values.append(adjusted_q)
            
            return adjusted_q_values
            
        except Exception as e:
            print(f"Mean Field batch Q-value calculation failed: {e}")
            import traceback
            traceback.print_exc()
            # 回退到普通批量方法
            return self.batch_evaluate_service_options(vehicle_request_pairs, ifEVQvalue)


    
  
    
    
    def heuristic_find_nearest_v(self,reassignvehicles):
        """启发式寻找最近的车辆"""
        hotspot_locations = self.hotspot_locations[:self.hotspot_locations_num]
        nearest_vehicle = None
        for loc in hotspot_locations:
            available_vehicles = {}
            for vehicle_id, vehicle in reassignvehicles.items():
                loc = vehicle['location']
                battery_level = vehicle['battery']
                batt_loss = self._manhattan_distance_loc_time(loc, loc) * self.battery_consum
                if battery_level - batt_loss >= self.rebalance_battery_threshold:
                    available_vehicles[vehicle_id] = vehicle
            min_distance = float('inf')
            for vehicle_id, vehicle in available_vehicles.items():
                vehicle_loc = vehicle['location']
                distance = self._manhattan_distance_loc(vehicle_loc, loc)
                if distance < min_distance and distance !=0:
                    min_distance = distance
                    nearest_vehicle = vehicle_id
        if nearest_vehicle is not None:
            return nearest_vehicle
        else:
            loc = hotspot_locations[randint(0, len(hotspot_locations)-1)]
            available_vehicles = {}
            for vehicle_id, vehicle in reassignvehicles.items():
                loc = vehicle['location']
                battery_level = vehicle['battery']
                batt_loss = self._manhattan_distance_loc_time(loc, loc) * self.battery_consum
                if battery_level - batt_loss >= self.rebalance_battery_threshold:
                    available_vehicles[vehicle_id] = vehicle
            if available_vehicles:
                return list(available_vehicles.keys())[0]

    def _calculate_rejection_aware_adjustment(self, vehicle_id, request, base_q_value):
        """
        计算拒绝感知的Q值调整: Q_value - immediate_reward * rejection_probability
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            base_q_value: 基础Q值
            
        Returns:
            float: 调整后的Q值
        """
        vehicle = self.vehicles.get(vehicle_id)
        if vehicle is None:
            return base_q_value
        
        # 计算立即收益
        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
        
        # 计算移动成本
        cur_loc = vehicle['location']
        pickup_x = request.pickup % self.grid_size
        pickup_y = request.pickup // self.grid_size
        dropoff_x = request.dropoff % self.grid_size
        dropoff_y = request.dropoff // self.grid_size
        vehicle_x = cur_loc % self.grid_size
        vehicle_y = cur_loc // self.grid_size
        
        d1 = abs(vehicle_x - pickup_x) + abs(vehicle_y - pickup_y)
        d2 = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
        moving_cost = getattr(self, 'movingpenalty', -0.1) * (d1 + d2)
        
        # 净立即收益
        net_immediate_reward = immediate_reward + moving_cost
        
        # 获取拒绝概率
        rejection_prob = self._calculate_rejection_probability(vehicle_id, request)
        
        # 计算调整后的Q值: Q值 - 立即收益 * 拒绝概率
        # 如果拒绝概率高，从Q值中减去更多的立即收益
        adjusted_q = base_q_value - (net_immediate_reward * rejection_prob)
        
        return adjusted_q

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


        current_time = self.current_time
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        v_after = self.value_function.get_charging_q_value(vehicle_id=vehicle_id, station_id=station_id,
                                             vehicle_location=cur_loc, station_location=station_loc,
                                             current_time=current_time, other_vehicles=max(0, other_idle),
                                             num_requests=num_reqs, battery_level=cur_bat)
        return  v_after


    def evaluate_idle_option(self, vehicle_id: int,target_loc) -> float:
        """Estimate completion Q for idling/waiting (option value)."""
        veh = self.vehicles.get(vehicle_id)
        
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']

        current_time = self.current_time
        num_reqs = len(self.active_requests)
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        
        # 检查value_function是否可用
        if self.value_function is not None:
            v_after = self.value_function.get_idle_q_value(vehicle_id=vehicle_id, vehicle_location=cur_loc, 
                            battery_level=cur_bat, current_time=current_time, 
                            other_vehicles=max(0, other_idle), num_requests=num_reqs)
        else:
            # 如果没有value_function，使用简单的启发式计算
            v_after = getattr(self, 'idle_vehicle_reward', -0.1)
        return v_after

    def evaluate_waiting_option(self, vehicle_id: int) -> float:
        veh = self.vehicles.get(vehicle_id)
        if veh is None:
            return 0.0

        cur_loc = veh['location']
        cur_bat = veh['battery']
        num_reqs = len(self.active_requests)
        # minimal execution reward: small movement penalty for staying put
        r_exec = self.movingpenalty * 1.0  # small penalty for idling
        other_idle = len([v for vid, v in self.vehicles.items() if vid != vehicle_id and v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None])
        
        # 检查value_function是否可用
        if self.value_function is not None:
            v_after = self.value_function.get_waiting_q_value(vehicle_id=vehicle_id, vehicle_location=cur_loc, 
                            battery_level=cur_bat, current_time=self.current_time, 
                            other_vehicles=max(0, other_idle), num_requests=num_reqs)
        else:
            # 如果没有value_function，使用简单的启发式计算
            v_after = getattr(self, 'waiting_vehicle_reward', -0.1)
        return v_after


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
        """Setup initial vehicle states with EV and AEV types using fixed seed for consistency"""
        # 临时保存当前随机状态
        current_random_state = random.getstate()
        current_numpy_state = np.random.get_state()
        
        # 使用固定的初始种子确保每个episode车辆初始状态一致
        if hasattr(self, 'initial_random_seed') and self.initial_random_seed is not None:
            random.seed(self.initial_random_seed)
            np.random.seed(self.initial_random_seed)
        
        for i in range(self.num_vehicles):
            # Convert grid coordinates to location index
            x = random.randint(0, self.grid_size-1)
            y = random.randint(0, self.grid_size-1)
            location_index = y * self.grid_size + x
            
            if i < self.ev_num_vehicles:
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
                'battery': random.uniform(0.5, 0.95),  # 50%-90% battery
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
                'stationary_duration': 0 , # Duration to remain stationary
                'target_location': None,
                'charging_target': None,  # Target location for idling
                'idle_target': None,  # Target location for idling
                'idle_timer': 0,  # Timer for idle duration
                'continual_reject': 0,
                'penalty_timer': 0, 
                'needs_emergency_charging': False,  # Whether the vehicle needs emergency charging
            }
            # Initialize storeactions for each vehicle
            self.storeactions[i] = None
            self.storeactions_ev[i] = None
        
        # 恢复原来的随机状态
        random.setstate(current_random_state)
        np.random.set_state(current_numpy_state)
        
        print(f"✓ Vehicles initialized with fixed seed {getattr(self, 'initial_random_seed', 'None')} - consistent initial states")
    


    def _calculate_rejection_probability_disttest(self, vehicle_id, distance):
        """Calculate the probability that an EV rejects a request based on distance"""
        vehicle = self.vehicles[vehicle_id]
        
        # AEV never rejects
        if vehicle['type'] == 2:  # AEV
            return 0.0
        asc = 0.5
        eplison_t = np.random.normal(-0.5, 1, 1)
        beta_id = 0.5  # Significantly increased for visible idle time effect
        beta_distance = -0.30  # Increased for clearer distance pattern
        idle_time = 2
        idle_time = vehicle.get('idle_timer', 0)  # FIXED: was 'idle_timer'
        acc = asc + beta_id * idle_time + beta_distance * distance 
        rejection_prob  = np.exp(acc) / (1 + np.exp(acc))
        return min(0.999, rejection_prob)




    def _calculate_rejection_probability(self, vehicle_id, request):
        """Calculate the probability that an EV rejects a request based on distance"""
        vehicle = self.vehicles[vehicle_id]
        
        # AEV never rejects
        if vehicle['type'] == 2:  # AEV
            return 0.0
        asc = 0.5
        eplison_t = np.random.normal(-0.5, 1, 1)
        beta_id = 0.5  # Significantly increased for visible idle time effect
        beta_distance = -0.30  # Increased for clearer distance pattern
        idle_time = 2
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
        idle_time = vehicle.get('idle_timer', 0)  # FIXED: was 'idle_timer'
        acc = asc + beta_id * idle_time + beta_distance * distance 
        rejection_prob  = np.exp(acc) / (1 + np.exp(acc))
        return min(0.999, 1 -rejection_prob)
    

    def _calculate_rejection_probabilityreal(self, vehicle_id, request):
        """Calculate the probability that an EV rejects a request based on distance"""
        vehicle = self.vehicles[vehicle_id]
        
        # AEV never rejects
        if vehicle['type'] == 2:  # AEV
            return 0.0
        asc = 0.5
        eplison_t = np.random.normal(-0.5, 0.25, 1)
        beta_id = 0.5  # Increased for very significant idle time effect
        beta_distance = -0.30  # Increased for clearer distance pattern
        idle_time = 2
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
        idle_time = vehicle.get('idle_timer', 0)  # Fixed: was idle_timer
        acc = asc + beta_id * idle_time + beta_distance * distance + eplison_t
        rejection_prob  = np.exp(acc) / (1 + np.exp(acc))
        #print(f"Vehicle {vehicle_id} at {vehicle_coords} with idle_time {idle_time} and distance {distance} has rejection_prob {1 - rejection_prob}")
        if distance <= 0:
            return 0.0
        else:
            return min(0.999, 1 - rejection_prob)


    
    def _should_reject_request(self, vehicle_id, request):
        """Determine if a vehicle should reject a request"""
        rejection_prob = self._calculate_rejection_probabilityreal(vehicle_id, request)
        return random.random() < rejection_prob
    

    def _generate_random_requests(self):
        """Generate new passenger requests in batches"""
        generated_requests = []
        
        # 设置请求生成专用的随机状态
        self._set_request_generation_random_state()
        
        # Determine how many requests to generate this step
        if random.random() < self.request_generation_rate:

            rand_val = random.random()
            if self.current_time < self.episode_length * 0.5:
                rand_val = random.random()
                if rand_val < 0.3:
                    num_requests = random.randint(int(self.num_vehicles*0.25), int(self.num_vehicles*0.5))
                elif rand_val < 0.8:
                    num_requests = random.randint(int(self.num_vehicles*0.5), int(self.num_vehicles))
                else:
                    num_requests = random.randint(int(self.num_vehicles*1.5), int(self.num_vehicles*2))
            else:
                rand_val = random.random()
                if rand_val < 0.3:
                    num_requests = random.randint(int(self.num_vehicles*0.5), int(self.num_vehicles))
                elif rand_val < 0.8:
                    num_requests = random.randint(int(self.num_vehicles), int(self.num_vehicles*1.5))
                else:
                    num_requests = random.randint(int(self.num_vehicles*2), int(self.num_vehicles*3))

            
            for request_idx in range(num_requests):
                self.request_counter += 1
                
                # 为每个请求设置不同的随机种子，确保位置分布的随机性
                # 使用当前时间、请求计数器和循环索引组合作为种子
                request_seed = (self.request_generation_seed if hasattr(self, 'request_generation_seed') and self.request_generation_seed is not None else 12345) + \
                              self.current_time * 1000 + self.request_counter * 10 + request_idx
                random.seed(request_seed)
                np.random.seed(request_seed)
                
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
                    random.seed(request_seed+attempts)
                    np.random.seed(request_seed+attempts)
                    dropoff_x = random.randint(0, self.grid_size - 1)
                    dropoff_y = random.randint(0, self.grid_size - 1)
                    dropoff_location = dropoff_y * self.grid_size + dropoff_x
                    attempts += 1
                
                # Calculate travel time (Manhattan distance)
                travel_time = max(abs(pickup_x - dropoff_x), abs(pickup_y - dropoff_y))
                
                # Create request with dynamic pricing based on demand
                base_value = 25
                distance_value = travel_time * (2 + np.random.rand()*0.1)
                surge_factor = 1.0 + (num_requests - 1) * 0.1  # More requests = higher prices
                point_loc = pickup_y * self.grid_size + pickup_x
                zone_loc = self.loc_to_zone.get(point_loc, None)
                if zone_loc==1:
                    distance_value += 5  # Downtown has higher surge
                elif zone_loc==2:
                    distance_value -= 2  # Suburban has moderate surge
                elif zone_loc==3:
                    distance_value -= 2  # Outskirts have lower surge
                point_loc = pickup_y * self.grid_size + pickup_x
                zone_loc = self.loc_to_zone.get(point_loc, None)
                if zone_loc==1:
                    distance_value += 5  # Downtown has higher surge
                elif zone_loc==2:
                    distance_value -= 2  # Suburban has moderate surge
                elif zone_loc==3:
                    distance_value -= 2  # Outskirts have lower surge
                final_value = base_value * surge_factor + distance_value
                
                request = Request(
                    request_id=self.request_counter,
                    source=pickup_location,
                    destination=dropoff_location,
                    current_time=self.current_time,
                    travel_time=travel_time,
                    value=final_value,
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
                    'hotspot_idx': None,  # No hotspot for random requests
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            # 保存请求生成的随机状态
            self._save_request_generation_random_state()
            
            return generated_requests
        
        # 保存请求生成的随机状态
        self._save_request_generation_random_state()
        
        return []
    
    def _generate_intense_requests(self):
        """Generate multiple requests concentrated in 3 hotspots with probability weights"""
        generated_requests = []
        
        # 设置请求生成专用的随机状态
        self._set_request_generation_random_state()
        
        # Determine how many requests to generate this step
        if random.random() < self.request_generation_rate:
            # Generate between 1 and 10 requests with higher probability for fewer requests
            if self.current_time < self.episode_length * 0.5:
                rand_val = random.random()
                if rand_val < 0.3:
                    num_requests = random.randint(int(self.num_vehicles*0.25), int(self.num_vehicles*0.5))
                elif rand_val < 0.8:
                    num_requests = random.randint(int(self.num_vehicles*0.5), int(self.num_vehicles))
                else:
                    num_requests = random.randint(int(self.num_vehicles*1.5), int(self.num_vehicles*2))
            else:
                rand_val = random.random()
                if rand_val < 0.3:
                    num_requests = random.randint(int(self.num_vehicles*0.5), int(self.num_vehicles))
                elif rand_val < 0.8:
                    num_requests = random.randint(int(self.num_vehicles), int(self.num_vehicles*1.5))
                else:
                    num_requests = random.randint(int(self.num_vehicles*2), int(self.num_vehicles*3))

            # Define 3 hotspot centers in the grid
            hotspots = self.hotspot_locations

            # Probability weights for each hotspot (should sum to 1.0)
            probability_weights = [0.3, 0.1, 0.2, 0.4]  # 20% for bottom-left, 10% for bottom-right, 20% for center, 25% for top-left, 25% for top-right
            selected_hotspot_idx_reward = [35, 15, 20, 15]  # Reward weights for each hotspot
            for request_idx in range(num_requests):
                self.request_counter += 1
                
                # 为每个请求设置不同的随机种子，确保位置分布的随机性
                # 使用当前时间、请求计数器和循环索引组合作为种子
                request_seed = (self.request_generation_seed if hasattr(self, 'request_generation_seed') and self.request_generation_seed is not None else 12345) + \
                              self.current_time * 1000 + self.request_counter * 10 + request_idx
                random.seed(request_seed)
                np.random.seed(request_seed)
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

                random_dropoffx = random.randint(0, self.grid_size - 1)
                random_dropoffy = random.randint(0, self.grid_size - 1)

                dropoff_location = random_dropoffy * self.grid_size + random_dropoffx
                
                # Ensure pickup and dropoff are different
                attempts = 0
                while dropoff_location == pickup_location and attempts < 5:
                    random.seed(request_seed + attempts)
                    np.random.seed(request_seed + attempts)
                    random_dropoffx = random.randint(0, self.grid_size - 1)
                    random_dropoffy = random.randint(0, self.grid_size - 1)
                    dropoff_location = random_dropoffy * self.grid_size + random_dropoffx
                    attempts += 1
                dropoff_x = random_dropoffx
                dropoff_y = random_dropoffy
                # Calculate travel time (Manhattan distance)
                travel_time = max(abs(pickup_x - dropoff_x), abs(pickup_y - dropoff_y))
                
                # Create request with dynamic pricing based on demand
                base_value = 10
                distance_value = travel_time * (1 + np.random.rand()*0.1)
                surge_factor = 1.0 + (num_requests - 1) * 0.01  # More requests = higher prices
                point_loc = pickup_y * self.grid_size + pickup_x
                zone_loc = self.loc_to_zone.get(point_loc, None)
                if zone_loc==1:
                    distance_value += 5  # Downtown has higher surge
                elif zone_loc==2:
                    distance_value -= 2  # Suburban has moderate surge
                elif zone_loc==3:
                    distance_value -= 2  # Outskirts have lower surge
                final_value = base_value * surge_factor + distance_value+ selected_hotspot_idx_reward[selected_hotspot_idx]
                

                
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
                    'dropoff_coords': (random_dropoffx, random_dropoffy),
                    'hotspot_idx': selected_hotspot_idx,
                    'time': self.current_time,
                    'batch_size': num_requests
                })
            
            # 保存请求生成的随机状态
            self._save_request_generation_random_state()
            
            return generated_requests
        
        # 保存请求生成的随机状态
        self._save_request_generation_random_state()
        
        return []
    
    def _assign_request_to_vehicle(self, vehicle_id, request_id):
        """Assign a request to a vehicle with rejection logic"""
        #print("vehicle_id:", vehicle_id, "request_id:", request_id,"check if active_requests:", request_id in self.active_requests)
        if request_id in self.active_requests and vehicle_id in self.vehicles:
            vehicle = self.vehicles[vehicle_id]
            request = self.active_requests[request_id]

            # Check if this request is already assigned to another vehicle or onboard
            for other_vid, other_veh in self.vehicles.items():
                if other_vid != vehicle_id:
                    if other_veh['assigned_request'] == request_id:
                        print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: already assigned to vehicle {other_vid}")
                        return False
                    if other_veh['passenger_onboard'] == request_id:
                        print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: already onboard vehicle {other_vid}")
                        return False

            # EV penalty period: do not allow receiving orders.
            if self._in_ev_penalty(vehicle_id):
                return False
            #print("assign_request_to_vehicle: Vehicle {} request {} at step {}".format(vehicle_id, request_id, self.current_time))
            # Vehicle must be completely free (both assigned_request AND passenger_onboard must be None)
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                # Check if the vehicle rejects the request
                if self._should_reject_request(vehicle_id, request):
                    vehicle['rejected_requests'] += 1
                    self._record_ev_rejection(vehicle_id)
                    vehicle['assigned_request'] = request_id
                    self.rejected_requests.append(request)
                    
                    
                    if vehicle['type'] == 1 and hasattr(self, 'value_function') and self.value_function is not None:
                        # Calculate distance for rejection experience
                        vehicle_coords = vehicle['coordinates']
                        pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
                        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                        
                        # Store rejection experience in value function
                        self.value_function.store_rejection_experience(
                            vehicle_id=vehicle_id,
                            request_id=request_id,
                            vehicle_location=vehicle['location'],
                            pickup_location=request.pickup,
                            current_time=self.current_time,
                            distance=distance,
                            rejection_reason="distance"
                        )
                    
                    #print("assign_request_to_vehicle: Vehicle {} request {} at step {}".format(vehicle_id, request_id, self.current_time))
                    return False  # Request rejected
                # Request accepted
                vehicle['assigned_request'] = request_id
                self._record_ev_acceptance(vehicle_id)
                if vehicle['type']==1:
                    self.ev_requests.append(request)
                #print("✅  Vehicle {} accepted request {} at step {}".format(vehicle_id, request_id, self.current_time))
                return True
            else:
                # Vehicle is already busy
                print(f"⚠️  Cannot assign request {request_id} to vehicle {vehicle_id}: vehicle already has assigned_request={vehicle['assigned_request']} or passenger_onboard={vehicle['passenger_onboard']}")
                return False
        else:
            print("wrong assign_request_to_vehicle: Vehicle {} or request {} not found at step {}".format(vehicle_id, request_id, self.current_time))
            return False

    def _move_vehicle_to_charging_station(self, vehicle_id, station_id):
        """Move a vehicle to a charging station for rebalancing"""
        if vehicle_id in self.vehicles and hasattr(self, 'charging_manager'):
            vehicle = self.vehicles[vehicle_id]
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            vehicle['idle_target'] = None
            vehicle['charging_station'] = station_id
            station = self.charging_manager.stations[station_id]
            # Convert station location to coordinates
            station_x = station.location // self.grid_size
            station_y = station.location % self.grid_size
            station_coords = (station_x, station_y)
            
            # Set vehicle destination to charging station
            vehicle['target_location'] = station_coords
            vehicle['charging_target'] = station_id
            
            return True

    
    def _pickup_passenger(self, vehicle_id):
        """Vehicle picks up passenger at request pickup location"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成pickup
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成pickup - 订单未完成")
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
                if self.current_time % 50 == 0:
                    print(f"🚫 Vehicle {vehicle_id} assigned_request {vehicle['assigned_request']} expired/removed (not in active_requests)")
                vehicle['assigned_request'] = None
                return False
                
            request = self.active_requests[vehicle['assigned_request']]
            vehicle_coords = vehicle['coordinates']
            pickup_coords = (request.pickup % self.grid_size, request.pickup // self.grid_size)
            
            # Debug info every 50 steps
            if self.current_time % 50 == 0:
                distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                request_age = self.current_time - (request.pickup_deadline - request.MAX_PICKUP_DELAY)
                is_expired = self.current_time > request.pickup_deadline
                print(f"🚗 Vehicle {vehicle_id} moving to pickup: at {vehicle_coords}, target {pickup_coords}, distance={distance}, request_age={request_age:.0f}, expired={is_expired}")
            
            # Check if vehicle is at pickup location - allow pickup even if request expired (vehicle already committed)
            if vehicle_coords == pickup_coords:
                # Double-check: make sure no other vehicle has already picked up this passenger
                request_id = vehicle['assigned_request']
                already_picked_up = False
                for other_vid, other_veh in self.vehicles.items():
                    if other_vid != vehicle_id and other_veh['passenger_onboard'] == request_id:
                        already_picked_up = True
                        print(f"⚠️  Vehicle {vehicle_id} arrived at pickup but request {request_id} already picked up by vehicle {other_vid}")
                        break
                
                if not already_picked_up:
                    vehicle['passenger_onboard'] = vehicle['assigned_request']
                    vehicle['assigned_request'] = None
                    if self.current_time % 25 == 0 or self.current_time > request.pickup_deadline:
                        expired_status = "EXPIRED" if self.current_time > request.pickup_deadline else ""
                        print(f"✅ Vehicle {vehicle_id} picked up passenger (request {vehicle['passenger_onboard']}) at {vehicle_coords} {expired_status}")
                    return True
                else:
                    # Clear the assignment since another vehicle got there first
                    vehicle['assigned_request'] = None
                    return False
        return False
    


    def findchargerange_v(self):
        return_index = {}
        for j in self.charging_manager.stations.values():
            return_index[j.id] = []
            station_x = j.location % self.grid_size
            station_y = j.location // self.grid_size
            for vehicle_id, v in self.vehicles.items():
                vehicle_x = v['coordinates'][0]
                vehicle_y = v['coordinates'][1]
                distance = abs(vehicle_x - station_x) + abs(vehicle_y - station_y)
                if distance <= 5:
                    return_index[j.id].append(vehicle_id)
        return return_index

    def findchargerange_c(self):
        return_index = {}
        for vehicle_id, v in self.vehicles.items():
            return_index[vehicle_id] = 0
            vehicle_x = v['coordinates'][0]
            vehicle_y = v['coordinates'][1]
            v['in_chargerange'] = []
            sumcapa = 0
            for j in self.charging_manager.stations.values():
                station_x = j.location % self.grid_size
                station_y = j.location // self.grid_size
                distance = abs(vehicle_x - station_x) + abs(vehicle_y - station_y)
                if distance <= 5:
                    sumcapa += j.max_capacity - len(j.current_vehicles)
            return_index[vehicle_id] = sumcapa
        return return_index

    def _dropoff_passenger(self, vehicle_id):
        """Vehicle drops off passenger at destination"""
        vehicle = self.vehicles[vehicle_id]
        
        # 检查车辆电池：电池为0时无法完成dropoff
        if vehicle['battery'] <= 0.0:
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法完成dropoff - 乘客滞留")
            # 乘客滞留在车上，订单未完成
            if vehicle['passenger_onboard'] is not None:
                request_id = vehicle['passenger_onboard']
                vehicle['passenger_onboard'] = None
                vehicle['assigned_request'] = None
                vehicle['target_location'] = None
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
                if self.vehicles[vehicle_id]['type']==1:
                    self.completed_requests_ev.append(completed_request)
                self.request_value_sum += completed_request.final_value
                # Calculate earnings
                earnings = completed_request.final_value
                vehicle['passenger_onboard'] = None

                # EV idle time starts after completion
                self._record_ev_completion(vehicle_id)
                
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
        

        for i in range(self.num_zones):
            zone_centerx = (self.grid_size // (int(np.sqrt(self.num_zones))))//2 + (i % int(np.sqrt(self.num_zones))) * (self.grid_size // (int(np.sqrt(self.num_zones))))
            zone_centery = (self.grid_size // (int(np.sqrt(self.num_zones))))//2 + (i // int(np.sqrt(self.num_zones))) * (self.grid_size // (int(np.sqrt(self.num_zones))))
            self.hotspot_locations.append((zone_centerx, zone_centery))

        
        

    
    def get_request_batch(self):
        """Get request batch - implementing abstract method"""
        # Return both passenger requests and charging needs
        requests = []
        
        # Add passenger requests
        for request_id, request in self.active_requests.items():
            requests.append(request)
        
        # Add charging requests for low battery vehicles
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery'] < 0.005:  # Low battery vehicles need charging
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

    def generate_requests(self):
        """Public wrapper for request generation.

        Some workflows expect `env.generate_requests()`; the internal implementation
        uses `_generate_intense_requests()` / `_generate_random_requests()`.
        """
        if getattr(self, 'use_intense_requests', False):
            return self._generate_intense_requests()
        return self._generate_random_requests()
    
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
    
    def step(self, actions, storeactions,storeactions_ev = None):
        """执行一步环境交互"""
        rewards = {}
        dur_rewards = {}
        next_states = {}
        charging_events = []
        
        # Initialize step counters
        self.step_assignments = 0
        self.step_rejections = 0


        # 处理每个车辆的动作
        for vehicle_id, action in actions.items():
            reward,dur_reward = self._execute_action(vehicle_id, action)
            rewards[vehicle_id] = reward
            dur_rewards[vehicle_id] = dur_reward
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
        self._update_q_learning(storeactions, False)
        self._update_q_learning(storeactions_ev, True)
        # Record charging station usage for this time step
        self._record_charging_usage()

        # 检查是否结束
        done = self.current_time >= self.episode_length

        ev_idle_mean = float(np.mean(self.ev_idle_durations)) if getattr(self, 'ev_idle_durations', None) else 0.0
        ev_idle_count = int(len(self.ev_idle_durations)) if getattr(self, 'ev_idle_durations', None) else 0
        ev_in_penalty = int(sum(1 for vid in self.vehicles.keys() if self._in_ev_penalty(vid)))
        return next_states, rewards, dur_rewards, done, {
            'charging_events': charging_events,
            'ev_idle_mean': ev_idle_mean,
            'ev_idle_count': ev_idle_count,
            'ev_in_penalty': ev_in_penalty,
        }
    
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
                vehicle['target_location'] = None
                vehicle['idle_target'] = None
                vehicle['assigned_request'] = None
                vehicle['passenger_onboard'] = None
                vehicle['charging_target'] = None
                if vehicle['charging_station'] is None:
                    dead_battery_vehicles.append(vehicle_id)
        return dead_battery_vehicles





    def simulate_motion(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []

        # Initialize actions dictionary for all vehicles
        actions = {}
        
        # Initialize storeactions for all vehicles to prevent KeyError
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        # For ChargingIntegratedEnvironment, we handle rebalancing differently
        # Convert our vehicle states to a format compatible with Gurobi optimization
        from src.Action import ChargingAction, ServiceAction, IdleAction

        charging_ev = []
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None:
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and (random.random() < p_charge) or vehicle['battery']<=0.2:
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            # idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                if self.vehicles[vehicle_id]['assigned_request'] is not None  and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['passenger_onboard'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['charging_station'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['target_location'] is not None and vehicle_id in vehicles_to_rebalance and not self._is_ev(vehicle_id):
                    vehicles_to_rebalance.remove(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Total vehicles to rebalance: {len(vehicles_to_rebalance)}, Strict idle vehicles: {len(idle_vehicles_1)}, Available requests: {available_requests_count}")
                
                if self.assignmentgurobi: 
                    rebalancing_assignments = self.gurobi_optimizer.optimize_vehicle_rebalancing_integrated(vehicles_to_rebalance)
                else:
                    available_requests = []
                    if hasattr(self, 'active_requests') and self.active_requests:
                        available_requests = list(self.active_requests.values())
                    charging_stations = []
                    charging_stations = [station for station in self.charging_manager.stations.values() 
                               if station.available_slots > 0]
                    rebalancing_assignments = self.gurobi_optimizer._heuristic_assignment_with_reject(vehicles_to_rebalance, available_requests, charging_stations)
                    for vehicle_id, target_request in rebalancing_assignments.items():
                        if target_request is None:
                            rebalancing_assignments[vehicle_id] = None
                # Debug: Count assignments made
                new_assignments = 0
                re_assignments_len = len(rebalancing_assignments)
                charging_assignments = 0
                self.total_rebalancing_calls += 1
                if len(rebalancing_assignments) != len(vehicles_to_rebalance):
                    print(f"⚠️  Warning: Mismatch in assignments - vehicles: {len(vehicles_to_rebalance)}, assignments: {len(rebalancing_assignments)} at step {self.current_time}")
                # print("vehicle_length:", len(vehicles_to_rebalance))
                # print("rebalance_length:", len(rebalancing_assignments))
                quest_num_now = len(self.active_requests)
                for vehicle_id, target_request in rebalancing_assignments.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            
                            # Handle charging assignment
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']


                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to request {target_request.request_id} at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle regular request assignment  
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                                vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                vehicle['idle_target'] = None  # Clear idle target on new assignment
                                # Generate service action
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                if vehicle['type'] == 1:
                                    self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery,
                                                        target_coords=self.active_requests[target_request.request_id].dropoff,
                                                        next_value=self.active_requests[target_request.request_id].final_value)
                                else:
                                    if storeactions[vehicle_id] is None:
                                        storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].dropoff
                                    else:
                                        storeactions[vehicle_id].next_action = actions[vehicle_id]
                                        storeactions[vehicle_id].next_action.next_value = self.active_requests[target_request.request_id].final_value
                                        storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                        storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                        # Save the old current_time before replacing the action
                                        old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                        self.storeactions[vehicle_id] = None
                                        self.storeactions[vehicle_id] = actions[vehicle_id]
                                        self.storeactions[vehicle_id].dur_reward = 0
                                        self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                        self.storeactions[vehicle_id].current_time = self.current_time
                                        self.storeactions[vehicle_id].target_location = self.active_requests[target_request.request_id].dropoff
                            else:
                                vehicle['continual_reject'] += 1
                                vehicle['assigned_request'] = None  # Clear the rejected request assignment
                                #print(f"❌ Vehicle {vehicle_id} rejected request {target_request.request_id} at step {self.current_time} (continual_reject={vehicle['continual_reject']})")
                                if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                                    vehicle['penalty_timer'] = self.ev_penalty_duration
                                
                                # EV拒单后的relocation决策
                                if self._is_ev(vehicle_id):
                                    target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                    
                                    from src.Action import IdleAction
                                    vehicle['idle_target'] = target_coords
                                    current_coords = vehicle['coordinates']
                                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                    
                                    self._store_rejected_ev_action(vehicle_id, actions[vehicle_id], target_request.request_id, storeactions_ev, vehicle_location, vehicle_battery, target_coords)

                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].target_location = current_coords
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = current_coords
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = None
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            vehicle['target_location'] = idle_target
                            if storeactions[vehicle_id] is None:
                                storeactions[vehicle_id] = actions[vehicle_id]
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                            else:
                                storeactions[vehicle_id].next_action = actions[vehicle_id]
                                storeactions[vehicle_id].next_action.next_value = 0
                                storeactions[vehicle_id].vehicle_loc_post = vehicle_location
                                storeactions[vehicle_id].vehicle_battery_post = vehicle_battery
                                storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                                # Save the old current_time before replacing the action
                                old_current_time = getattr(storeactions[vehicle_id], 'current_time', self.current_time)
                                self.storeactions[vehicle_id] = None
                                self.storeactions[vehicle_id] = actions[vehicle_id]
                                self.storeactions[vehicle_id].dur_reward = 0
                                self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
                                self.storeactions[vehicle_id].current_time = self.current_time
                                self.storeactions[vehicle_id].target_location = self.vehicles[vehicle_id]['target_location']
                        elif isinstance(target_request, str) and target_request.startswith("reloc"):
                            if self._is_ev(vehicle_id):
                                target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                
                                from src.Action import IdleAction
                                vehicle['idle_target'] = target_coords
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                            
                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                           
                    else:
                        # No assignment for this vehicle - generate idle action
                        from src.Action import IdleAction
                        
                        self._assign_idle_vehicle(vehicle_id)
                        idle_target = vehicle.get('idle_target', None)
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery)

                for vehicle_id in vehicles_to_rebalance:
                    vehicle = self.vehicles[vehicle_id]
                    #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
                # Store the count of request assignments for this rebalancing call
                self.rebalancing_assignments_per_step.append(new_assignments)
                self.rebalancing_whole.append(re_assignments_len)
                #print(f"DEBUG Assignment Result: New request assignments: {new_assignments}, Charging assignments: {charging_assignments}, Idle assignments: {len(vehicles_to_rebalance) - new_assignments - charging_assignments}")
        
        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        return actions, storeactions, storeactions_ev

    def simulate_motion_evfirst(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []
        actions = {}
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        from src.Action import ChargingAction, ServiceAction, IdleAction

        charging_ev = []
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None:
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and (random.random() < p_charge) or vehicle['battery']<=0.2:
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_ev
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                if self.vehicles[vehicle_id]['assigned_request'] is not None  and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['passenger_onboard'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['charging_station'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['target_location'] is not None and vehicle_id in vehicles_to_rebalance and not self._is_ev(vehicle_id):
                    vehicles_to_rebalance.remove(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Total vehicles to rebalance: {len(vehicles_to_rebalance)}, Strict idle vehicles: {len(idle_vehicles_1)}, Available requests: {available_requests_count}")
                
                # Initialize counters for tracking assignments
                new_assignments = 0
                charging_assignments = 0
                quest_num_now = len(self.active_requests)
                re_assignments_len = len(vehicles_to_rebalance)
                self.total_rebalancing_calls += 1
                #print("videcles to rebalance:", vehicles_to_rebalance)
                vehicles_to_rebalance_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]
                requests_for_rebalance = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                rebalancing_assignments_ev,remaingrequests = self.gurobi_optimizer._gurobi_vehicle_rebalancing_ev(vehicles_to_rebalance_ev,available_requests)
                rejected_ev_requests = []
                for vehicle_id, target_request in rebalancing_assignments_ev.items():
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    if isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                        if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                            new_assignments += 1
                            vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                            vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                            vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                            vehicle['idle_target'] = None  # Clear idle target on new assignment
                            from src.Action import ServiceAction
                            actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if vehicle['type'] == 1:
                                self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery, 
                                                    target_coords=self.active_requests[target_request.request_id].dropoff,
                                                    next_value=self.active_requests[target_request.request_id].final_value)
                        else:
                            vehicle['continual_reject'] += 1
                            vehicle['assigned_request'] = None
                            if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                                vehicle['penalty_timer'] = self.ev_penalty_duration
                            
                            # EV拒单后的relocation决策
                            if self._is_ev(vehicle_id):
                                target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                
                                from src.Action import IdleAction
                                vehicle['idle_target'] = target_coords
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                
                                self._store_rejected_ev_action(vehicle_id, actions[vehicle_id], target_request.request_id, storeactions_ev, vehicle_location, vehicle_battery, target_coords)
                    else:
                        target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                        from src.Action import IdleAction
                        vehicle['idle_target'] = target_coords
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                        self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                vehicles_to_rebalance_aev = [vid for vid in vehicles_to_rebalance if vid not in vehicles_to_rebalance_ev]
                
                # 重新获取当前的active_requests（避免使用过期的requests_for_rebalance快照）
                current_active_requests = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                
                # print("Assigned requests:", assigned_request)
                # Filter out both assigned AND expired requests
                available_requests = [req for req in current_active_requests 
                                    if req.request_id not in assigned_request 
                                    and self.current_time <= req.pickup_deadline]
                

                charging_stations = [station for station in self.charging_manager.stations.values() 
                            if station.available_slots > 0]
                rebalancing_assignments_aev= self.gurobi_optimizer._gurobi_vehicle_rebalancing_aev(vehicles_to_rebalance_aev,available_requests,charging_stations)
                # for vehicle_id in vehicles_to_rebalance_aev:
                #     print("Vehicle to rebalance AEV:", vehicle_id, "Battery:", self.vehicles[vehicle_id]['battery'])
                quest_num_now = len(self.active_requests)
                for vehicle_id, target_request in rebalancing_assignments_aev.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            
                            # Handle charging assignment
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print("veh_id:", vehicle_id, "veh_loc:", vehicle_location, "veh_battery:", vehicle_battery)
                            #print("target_request:", target_request.request_id, "pickup:", target_request.pickup, "dropoff:", target_request.dropoff)
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                vehicle = self.vehicles[vehicle_id]
                                vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                                vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                vehicle['idle_target'] = None  # Clear idle target on new assignment
                                # Generate service action
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                if vehicle['type'] == 1:
                                    self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery,
                                                        target_coords=self.active_requests[target_request.request_id].dropoff,
                                                        next_value=self.active_requests[target_request.request_id].final_value)
                                else:
                                    self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                                     target_coords=self.active_requests[target_request.request_id].dropoff,
                                                     next_value=self.active_requests[target_request.request_id].final_value)
                        elif isinstance(target_request, Request) and target_request.request_id not in self.active_requests:
                            # Request no longer in active_requests (已被分配或过期)
                            print(f"⚠️ Vehicle {vehicle_id} 分配的请求 {target_request.request_id} 不在 active_requests 中 (step {self.current_time})")
                            # 给车辆分配idle action
                            self._assign_idle_vehicle(vehicle_id)
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                             target_coords=self.vehicles[vehicle_id]['location'])
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = None
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)

                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                           
                    else:
                        # No assignment for this vehicle - generate idle action
                        from src.Action import IdleAction
                        
                        self._assign_idle_vehicle(vehicle_id)
                        idle_target = vehicle.get('idle_target', None)
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery)

                for vehicle_id in vehicles_to_rebalance:
                    vehicle = self.vehicles[vehicle_id]
                    #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
                # Store the count of request assignments for this rebalancing call
                self.rebalancing_assignments_per_step.append(new_assignments)
                self.rebalancing_whole.append(re_assignments_len)
                #print(f"DEBUG Assignment Result: New request assignments: {new_assignments}, Charging assignments: {charging_assignments}, Idle assignments: {len(vehicles_to_rebalance) - new_assignments - charging_assignments}")
        
        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        return actions, storeactions, storeactions_ev








    def simulate_motion_aevfirst(self, agents: List[LearningAgent] = None, current_requests: List[Request] = None, rebalance: bool = True):
        """Override simulate_motion to integrate Gurobi optimization with Q-learning for charging environment"""
        if agents is None:
            agents = []
        actions = {}
        storeactions = {vid: self.storeactions.get(vid) for vid in self.vehicles.keys()}
        storeactions_ev = {vid: self.storeactions_ev.get(vid) for vid in self.vehicles.keys()}
        from src.Action import ChargingAction, ServiceAction, IdleAction

        charging_ev = []
        for vehicle_id, vehicle in self.vehicles.items():
            if self._is_ev(vehicle_id) and vehicle.get('charging_station') is None and vehicle.get('assigned_request') is None and vehicle.get('passenger_onboard') is None and vehicle.get('idle_target') is None and vehicle.get('target_location') is None:
                p_charge, station_probs = self.compute_ev_charge_probability(vehicle_id)
                if station_probs and (random.random() < p_charge) or vehicle['battery']<=0.2:
                    # Choose charging station by probability
                    r = random.random()
                    acc = 0.0
                    chosen_station = next(iter(station_probs.keys()))
                    for sid, prob in station_probs.items():
                        acc += float(prob)
                        if r <= acc:
                            chosen_station = int(sid)
                            break
                    # Extract vehicle state for action creation
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    self._move_vehicle_to_charging_station(vehicle_id, chosen_station)
                    actions[vehicle_id] = ChargingAction([], chosen_station, self.charge_duration, vehicle_location, vehicle_battery)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
        leftover_vehicleslist = [vid for vid in self.vehicles.keys() if vid not in actions]
        
        if rebalance and leftover_vehicleslist:
            # Get vehicles that need rebalancing (not currently assigned to tasks or charging)
            vehicles_to_rebalance = []
            
            # First priority: True idle vehicles (strict condition)
            idle_vehicles_1 = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['assigned_request'] is None and v['passenger_onboard'] is None and v['charging_station'] is None and v['target_location'] is None and  v['penalty_timer']==0]
            idle_vehicles_2  = [vehicle_id for vehicle_id, v in self.vehicles.items() 
                              if v['needs_emergency_charging']]
            idle_vehicles_ev = [vid for vid in idle_vehicles_1 if self._is_ev(vid) and self.vehicles[vid]['target_location'] is not None]
            idle_vehicles_1 = idle_vehicles_1 + idle_vehicles_2+idle_vehicles_ev
            for vehicle_id, vehicle in self.vehicles.items():
                # Include strict idle vehicles first
                if vehicle_id in leftover_vehicleslist:
                    if vehicle_id in idle_vehicles_1:
                        vehicles_to_rebalance.append(vehicle_id)
                    # Also include vehicles that need emergency rebalancing
                    elif (vehicle['battery'] <= self.rebalance_battery_threshold and vehicle['passenger_onboard'] == None and vehicle['assigned_request'] == None) :
                        vehicles_to_rebalance.append(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                if self.vehicles[vehicle_id]['assigned_request'] is not None  and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['passenger_onboard'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['charging_station'] is not None and vehicle_id in vehicles_to_rebalance:
                    vehicles_to_rebalance.remove(vehicle_id)
                if self.vehicles[vehicle_id]['target_location'] is not None and vehicle_id in vehicles_to_rebalance and not self._is_ev(vehicle_id):
                    vehicles_to_rebalance.remove(vehicle_id)
            for vehicle_id in vehicles_to_rebalance:
                vehicle = self.vehicles[vehicle_id]
                # print(f" {vehicle_id}  Status - Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
            if self.current_time % 50 == 0:
                print(f"🔄 Rebalancing Step {self.current_time}: Total vehicles to rebalance: {len(vehicles_to_rebalance)}")
            if len(vehicles_to_rebalance) > 0:
                # Use GurobiOptimizer for rebalancing
                if not hasattr(self, 'gurobi_optimizer'):
                    from src.GurobiOptimizer import GurobiOptimizer
                    self.gurobi_optimizer = GurobiOptimizer(self)
                
                # Debug: Count available requests before assignment
                available_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                #print(f"DEBUG Assignment: Step {self.current_time}, Total vehicles to rebalance: {len(vehicles_to_rebalance)}, Strict idle vehicles: {len(idle_vehicles_1)}, Available requests: {available_requests_count}")
                
                # Initialize counters for tracking assignments
                new_assignments = 0
                charging_assignments = 0
                quest_num_now = len(self.active_requests)
                re_assignments_len = len(vehicles_to_rebalance)
                self.total_rebalancing_calls += 1
                #print("videcles to rebalance:", vehicles_to_rebalance)
                vehicles_to_rebalance_aev = [vid for vid in vehicles_to_rebalance if not self._is_ev(vid)]
                requests_for_rebalance = list(self.active_requests.values()) if hasattr(self, 'active_requests') else []
                # 获取所有已分配的request_id（包括assigned_request和passenger_onboard）
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                charging_stations = [station for station in self.charging_manager.stations.values() 
                            if station.available_slots > 0]
                rebalancing_assignments_aev = self.gurobi_optimizer._gurobi_vehicle_rebalancing_aev(vehicles_to_rebalance_aev,available_requests,charging_stations)
                quest_num_now = len(self.active_requests)
                for vehicle_id, target_request in rebalancing_assignments_aev.items():
                    vehicle_location = self.vehicles[vehicle_id]['location']
                    vehicle_battery = self.vehicles[vehicle_id]['battery']
                    self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
                    self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
                    if target_request:
                        # Check if it's a charging assignment (string) or request assignment (object)
                        if isinstance(target_request, str) and target_request.startswith("charge_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            
                            # Handle charging assignment
                            station_id = int(target_request.replace("charge_", ""))
                            #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                            self._move_vehicle_to_charging_station(vehicle_id, station_id)
                            charging_assignments += 1
                            # Generate charging action
                            from src.Action import ChargingAction
                            
                            actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                            #print("veh_id:", vehicle_id, "veh_loc:", vehicle_location, "veh_battery:", vehicle_battery)
                            #print("target_request:", target_request.request_id, "pickup:", target_request.pickup, "dropoff:", target_request.dropoff)
                            if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                                new_assignments += 1
                                vehicle = self.vehicles[vehicle_id]
                                vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                                vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                                vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                                vehicle['idle_target'] = None  # Clear idle target on new assignment
                                # Generate service action
                                from src.Action import ServiceAction
                                actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                                if vehicle['type'] == 1:
                                    self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery,
                                                        target_coords=self.active_requests[target_request.request_id].dropoff,
                                                        next_value=self.active_requests[target_request.request_id].final_value)
                                else:
                                    self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                                     target_coords=self.active_requests[target_request.request_id].dropoff,
                                                     next_value=self.active_requests[target_request.request_id].final_value)
                        elif isinstance(target_request, Request) and target_request.request_id not in self.active_requests:
                            # Request no longer in active_requests (已被分配或过期)
                            print(f"⚠️ Vehicle {vehicle_id} 分配的请求 {target_request.request_id} 不在 active_requests 中 (step {self.current_time})")
                            # 给车辆分配idle action
                            self._assign_idle_vehicle(vehicle_id)
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                        elif isinstance(target_request, str) and target_request == "waiting":
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            # Handle waiting state - mark vehicle as stationary for next simulation
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = True
                            vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                            # Generate idle action to keep vehicle stationary
                            from src.Action import IdleAction
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                            self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                             target_coords=self.vehicles[vehicle_id]['location'])
                        elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                            #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            zone_id_str = target_request.replace("idle_at_", "")
                            zone_id = int(zone_id_str)
                            hotspot_coords = self.hotspot_locations[zone_id]
                            hot_x = hotspot_coords[0]
                            hot_y = hotspot_coords[1]

                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            idle_target = (hot_x, hot_y)
                            vehicle['assigned_request'] = None
                            vehicle['passenger_onboard'] = None
                            vehicle['charging_station'] = None
                            vehicle['target_location'] = None
                            vehicle['idle_target'] = idle_target
                            current_coords = vehicle['coordinates']
                            actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)

                            
                        else:
                            #print(f"DEBUG: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                            self._assign_idle_vehicle(vehicle_id)
                            # Generate idle action using the target set by _assign_idle_vehicle
                            from src.Action import IdleAction
                            vehicle = self.vehicles[vehicle_id]
                            vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                            current_coords = vehicle['coordinates']
                            target_coords = vehicle.get('idle_target', current_coords)  # Use assigned target
                            actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                            self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                assigned_request = []
                for vehicle_id in self.vehicles.keys():
                    if self.vehicles[vehicle_id]['assigned_request'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['assigned_request'])
                    if self.vehicles[vehicle_id]['passenger_onboard'] is not None:
                        assigned_request.append(self.vehicles[vehicle_id]['passenger_onboard'])
                available_requests = [req for req in requests_for_rebalance if req.request_id not in assigned_request]
                vehicles_to_rebalance_ev = [vid for vid in vehicles_to_rebalance if self._is_ev(vid)]
                rebalancing_assignments_ev,_= self.gurobi_optimizer._gurobi_vehicle_rebalancing_ev(vehicles_to_rebalance_ev,available_requests)
                for vehicle_id, target_request in rebalancing_assignments_ev.items():
                    vehicle = self.vehicles[vehicle_id]
                    vehicle_location = vehicle['location']
                    vehicle_battery = vehicle['battery']
                    if isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                        if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                            new_assignments += 1
                            vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                            vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                            vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                            vehicle['idle_target'] = None  # Clear idle target on new assignment
                            from src.Action import ServiceAction
                            actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                            if vehicle['type'] == 1:
                                self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery, 
                                                    target_coords=self.active_requests[target_request.request_id].dropoff,
                                                    next_value=self.active_requests[target_request.request_id].final_value)
                        else:
                            vehicle['continual_reject'] += 1
                            vehicle['assigned_request'] = None
                            if vehicle['continual_reject'] >= self.penalty_reject_requestnum:
                                vehicle['penalty_timer'] = self.ev_penalty_duration
                            
                            # EV拒单后的relocation决策
                            if self._is_ev(vehicle_id):
                                target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                                
                                from src.Action import IdleAction
                                vehicle['idle_target'] = target_coords
                                current_coords = vehicle['coordinates']
                                actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                                
                                self._store_rejected_ev_action(vehicle_id, actions[vehicle_id], target_request.request_id, storeactions_ev, vehicle_location, vehicle_battery, target_coords)
                    else:
                        target_coords, rel_action = self._handle_ev_rejection_relocation(vehicle_id)
                        from src.Action import IdleAction
                        vehicle['idle_target'] = target_coords
                        current_coords = vehicle['coordinates']
                        actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery, req_num=quest_num_now)
                        self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions_ev, is_ev=True)
                

        # Generate actions for vehicles not involved in rebalancing
        from src.Action import Action, ChargingAction, ServiceAction, IdleAction
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle_id not in actions:
                veh = self.vehicles[vehicle_id]
                vehicle_location = veh['location']
                vehicle_battery = veh['battery']
                # Check if vehicle is in stationary state
                if vehicle.get('is_stationary', False):
                    # Generate idle action to keep vehicle stationary
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)  # Stay in place
                # Generate action based on current vehicle state
                elif vehicle['charging_station'] is not None:
                    # Vehicle is charging - continue charging action
                    station_id = vehicle['charging_station']
                    charge_duration = vehicle.get('charging_time_left', 2)  # Use remaining time or default
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle['assigned_request'] is not None:
                    # Vehicle has assigned request - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['assigned_request'], vehicle_location, vehicle_battery)
                elif vehicle['passenger_onboard'] is not None:
                    # Vehicle has passenger - continue service
                    actions[vehicle_id] = ServiceAction([], vehicle['passenger_onboard'], vehicle_location, vehicle_battery)
                elif vehicle['charging_target'] is not None:
                    actions[vehicle_id] = ChargingAction([], vehicle['charging_target'], self.charge_duration, vehicle_location, vehicle_battery)
                elif vehicle.get('target_location') is not None:
                    # Vehicle has a target location - generate idle action to move there
                    current_coords = vehicle['coordinates']
                    target_coords = vehicle['target_location']
                    # Convert target_location to coordinates if it's a location index
                    if isinstance(target_coords, int):
                        target_coords = (target_coords % self.grid_size, target_coords // self.grid_size)
                    actions[vehicle_id] = IdleAction([], current_coords, target_coords, vehicle_location, vehicle_battery)
                else:
                    # No specific state - generate idle action at current location
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location, vehicle_battery)
        if len(actions) != len(self.vehicles):
            print(f"❌ CRITICAL ERROR: Action count mismatch at step {self.current_time}!")
            print(f"   Total vehicles: {len(self.vehicles)}, Actions generated: {len(actions)}")
            print(f"   Vehicles: {list(self.vehicles.keys())}")
            print(f"   Actions: {list(actions.keys())}")
            
            # Find missing vehicles
            missing_vehicles = [vid for vid in self.vehicles.keys() if vid not in actions]
            print("   Vehicles missing actions:")
            for vehicle_id in missing_vehicles:
                print(f"     - Vehicle ID: {vehicle_id}")
            print("   Detailed vehicle statuses for missing actions:")
            for action in actions.items():
                print(f"     - Vehicle ID with action: {action[0]}")
            print(f"   Missing actions for vehicles: {missing_vehicles}")
            
            # Show status of missing vehicles
            for vehicle_id in missing_vehicles:
                vehicle = self.vehicles[vehicle_id]
                print(f"   Vehicle {vehicle_id} status:")
                print(f"     - Assigned request: {vehicle['assigned_request']}")
                print(f"     - Passenger onboard: {vehicle['passenger_onboard']}")
                print(f"     - Charging station: {vehicle['charging_station']}")
                print(f"     - Target location: {vehicle['target_location']}")
                print(f"     - Is stationary: {vehicle.get('is_stationary', False)}")
                print(f"     - Battery: {vehicle['battery']:.3f}")
                print(f"     - Charging target: {vehicle.get('charging_target', None)}")
                print(f"     - Idle target: {vehicle.get('idle_target', None)}")
            
            # Force program termination with detailed context
            raise RuntimeError(f"Action generation failed - {len(missing_vehicles)} vehicles without actions")
            
        # Update recent requests list if provided
        if current_requests:
            self.update_recent_requests(current_requests)
        for vehicle_id in actions.keys():
            #print(f"Vehicle {vehicle_id} action: {actions[vehicle_id]}")
            vehicle = self.vehicles[vehicle_id]
            #print(f" {vehicle_id}  Status - finished Assigned: {vehicle['assigned_request']}, Onboard: {vehicle['passenger_onboard']}, Charging: {vehicle['charging_station']}, Target: {vehicle['target_location']}, Stationary: {vehicle['is_stationary']}")
        
        return actions, storeactions, storeactions_ev







    def _update_q_learning(self, actions, ifev = False):
        num_havenextaction = 0

        valuefunction = self.value_function
        valuefunction_ev = self.value_function_ev
        if valuefunction is None or not hasattr(valuefunction, 'experience_buffer'):
            return
        
        
        if ifev:
            offlinsedatalen = valuefunction_ev.experience_buffer.__len__()
            if self.current_time % 100 == 0:
                print(f"🔄 Updating EV Q-learning - current offline data size: {offlinsedatalen}")
                exp_analysis = self.value_function_ev.analyze_experience_data()
                if exp_analysis:
                    reward_stats = exp_analysis['reward_stats']
                    action_stats = exp_analysis['action_stats']
                    print(f" Assign: {action_stats['assign_count']}, chargelength: {action_stats['charge_count']}, idlelength: {action_stats['idle_count']}")
                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']} | 0{reward_stats['neutral_ratio']} | -{reward_stats['negative_ratio']}")
                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']}, Assign={action_stats['assign_mean_reward']}, Charge={action_stats['charge_mean_reward']}, Idle={action_stats['idle_mean_reward']}")
                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                else:
                    print("    ⚠️ No experience data available for analysis yet")
        else:
            offlinsedatalen = valuefunction.experience_buffer.__len__()
            if self.current_time % 100 == 0:
                print(f"🔄 Updating Q-learning - current offline data size: {offlinsedatalen}")
                exp_analysis = self.value_function.analyze_experience_data()
                if exp_analysis:
                    reward_stats = exp_analysis['reward_stats']
                    action_stats = exp_analysis['action_stats']
                    print(f" Assign: {action_stats['assign_count']}, chargelength: {action_stats['charge_count']}, idlelength: {action_stats['idle_count']}")
                    print(f"    📊 Experience Data Analysis (last 100 steps):")
                    print(f"      Reward Distribution: +{reward_stats['positive_ratio']} | 0{reward_stats['neutral_ratio']} | -{reward_stats['negative_ratio']}")
                    print(f"      Mean Rewards: Overall={reward_stats['mean_reward']}, Assign={action_stats['assign_mean_reward']}, Charge={action_stats['charge_mean_reward']}, Idle={action_stats['idle_mean_reward']}")
                    print(f"      Action Success Rates: Assign={action_stats['assign_positive_ratio']:.1%}, Charge={action_stats['charge_positive_ratio']:.1%}, Idle={action_stats['idle_positive_ratio']:.1%}")
                else:
                    print("    ⚠️ No experience data available for analysis yet")
        
        # Save training dataset at line 2085
        if ifev:
            if self.current_time % 200 == 0 and offlinsedatalen > 100:  # Save every 200 time steps with enough data
                #self._save_training_dataset(valuefunction_ev)
                self._analyze_q_value_issues(valuefunction_ev)
            
            # 额外分析：每50步检查Q-value趋势
            if self.current_time % 50 == 0 and offlinsedatalen > 50:
                self._quick_q_value_analysis(valuefunction_ev)

        else:
            if self.current_time % 200 == 0 and offlinsedatalen > 100:  # Save every 200 time steps with enough data
                #self._save_training_dataset(valuefunction)
                self._analyze_q_value_issues(valuefunction)
            
            # 额外分析：每50步检查Q-value趋势
            if self.current_time % 50 == 0 and offlinsedatalen > 50:
                self._quick_q_value_analysis(valuefunction)


        from .Action import ServiceAction, ChargingAction, IdleAction

        def _manhattan_distance_loc(a_loc: int, b_loc: int) -> int:
            ax, ay = (a_loc % self.grid_size, a_loc // self.grid_size)
            bx, by = (b_loc % self.grid_size, b_loc // self.grid_size)
            return abs(ax - bx) + abs(ay - by)


        whether_finish = self.current_time >= self.episode_length


        for vehicle_id in actions.keys():
            action = actions[vehicle_id]
            # Skip None actions to prevent AttributeError
            if action is None:
                continue
            batterynow = self.vehicles[vehicle_id]['battery']
            # Use pre-action as decision state
            current_location = action.vehicle_loc
            current_battery = action.vehicle_battery
            current_request_num = action.req_num
            veh_curloc = self.vehicles[vehicle_id]['location']
            veh_type = self.vehicles[vehicle_id]['type']
            if veh_type == 2:
                
                if (self.value_function and hasattr(self.value_function, 'store_experience')):
                    other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                    num_requests = len(self.active_requests)
                    store_threshold = 5
                    # Service option at assignment decision
                    next_action = getattr(action, 'next_action', None)
                    if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and next_action is not None and action.dur_reward >store_threshold:
                        #print(f"🚖 Storing service experience for vehicle {vehicle_id} at step {self.current_time}")
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                        next_value = getattr(next_action, 'next_value', r_exec)
                        req = action.target_location
                        next_battery = batterynow
                        target_location = req
                        next_location = req
                        if isinstance(next_action, ServiceAction) :
                            next_action_type = "assign"
                        elif isinstance(next_action, ChargingAction) :
                            next_action_type = "charge"
                        else:
                            next_action_type = "idle"
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
                            request_value=r_exec,
                            next_action_type = next_action_type,
                            next_request_value = next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False)
                        )

                    elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                        # print(f"🔋 Storing charging experience for vehicle {vehicle_id} at step {self.current_time}")
                        st_id = action.charging_station_id
                        next_action = getattr(action, 'next_action', None)
                        if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch and next_action is not None:
                            #print(f"🔋 Storing charging experience for vehicle {vehicle_id} at step {self.current_time}")
                            next_value = getattr(next_action, 'next_value', 0)
                            st = self.charging_manager.stations[st_id]
                            station_loc = st.location
                            r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                            next_battery = batterynow
                            target_location = station_loc
                            next_location = station_loc
                            if isinstance(next_action, ServiceAction) :
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction) :
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
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
                                request_value=r_exec,
                                next_action_type=next_action_type,
                                next_request_value=next_value,
                                dur_time=getattr(action, 'dur_time', 1.0),
                                is_system_done=getattr(self, 'done', False)
                            )

                    # Idle/wait decision treated as single-step option
                    elif isinstance(action, IdleAction):

                        # print(f"⏳ Storing idle experience for vehicle {vehicle_id} at step {self.current_time}")
                        next_action = getattr(action, 'next_action', None)
                        # 确定next_action类型和价值，即使next_action为None也要存储
                        if next_action is not None:
                            next_value = getattr(next_action, 'next_value', actions[vehicle_id].dur_reward)
                            if isinstance(next_action, ServiceAction):
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction):
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                        else:
                            # 没有next_action时，使用当前的dur_reward作为价值估计
                            next_value = actions[vehicle_id].dur_reward
                            next_action_type = "idle"
                        
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action

                        self.value_function.store_experience(
                            vehicle_id=vehicle_id,
                            action_type="idle",
                            vehicle_location=current_location,
                            target_location=veh_curloc,
                            current_time=self.current_time,
                            reward=-100,
                            next_vehicle_location=veh_curloc,
                            battery_level=current_battery,
                            next_battery_level=batterynow,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=r_exec,
                            next_action_type=next_action_type,
                            next_request_value=next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False)
                        )
            else:
                if (self.value_function_ev and hasattr(self.value_function_ev, 'store_experience')):
                    other_vehicles = len([v for v in self.vehicles.values() if v['assigned_request'] is not None])
                    num_requests = len(self.active_requests)
                    store_threshold = 5
                    # Service option at assignment decision
                    next_action = getattr(action, 'next_action', None)
                    if isinstance(action, ServiceAction) and hasattr(action, 'request_id') and next_action is not None and action.dur_reward >store_threshold:
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                        next_value = getattr(next_action, 'next_value', r_exec)
                        req = action.target_location
                        next_battery = batterynow
                        target_location = req
                        next_location = req
                        if isinstance(next_action, ServiceAction) :
                            next_action_type = "assign"
                        elif isinstance(next_action, ChargingAction) :
                            next_action_type = "charge"
                        else:
                            next_action_type = "idle"
                        # request_value 用 r_exec 对齐优化器语义
                        self.value_function_ev.store_experience(
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
                            request_value=r_exec,
                            next_action_type = next_action_type,
                            next_request_value = next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False)
                        )

                    # Service action when vehicle is stationary (rejection case)
                    elif isinstance(action, ServiceAction) and self.vehicles[vehicle_id].get('idle_target') is not None and hasattr(action, 'request_id'):
                        next_action = getattr(action, 'next_action', None)
                        if next_action is not None:
                            r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                            next_value = getattr(next_action, 'next_value', r_exec)
                            next_battery = batterynow
                            target_location = action.target_location
                            next_location = veh_curloc  # Vehicle doesn't move when stationary
                            if isinstance(next_action, ServiceAction):
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction):
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                            # Store experience for stationary service action (rejection case)
                            self.value_function_ev.store_experience(
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
                                request_value=r_exec,
                                next_action_type=next_action_type,
                                next_request_value=next_value,
                                dur_time=getattr(action, 'dur_time', 1.0),
                                is_system_done=getattr(self, 'done', False)
                            )

                    # Charging option at decision
                    elif isinstance(action, ChargingAction) and hasattr(action, 'charging_station_id'):
                        st_id = action.charging_station_id
                        next_action = getattr(action, 'next_action', None)
                        if hasattr(self, 'charging_manager') and st_id in self.charging_manager.stations and batterynow > self.chargeincrease_per_epoch and next_action is not None:
                            next_value = getattr(next_action, 'next_value', 0)
                            st = self.charging_manager.stations[st_id]
                            station_loc = st.location
                            r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action
                            next_battery = batterynow
                            target_location = station_loc
                            next_location = station_loc
                            if isinstance(next_action, ServiceAction) :
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction) :
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                            self.value_function_ev.store_experience(
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
                                request_value=r_exec,
                                next_action_type=next_action_type,
                                next_request_value=next_value,
                                dur_time=getattr(action, 'dur_time', 1.0),
                                is_system_done=getattr(self, 'done', False)
                            )

                    # Idle/wait decision treated as single-step option
                    elif isinstance(action, IdleAction):
                        next_action = getattr(action, 'next_action', None)
                        # 确定next_action类型和价值，即使next_action为None也要存储
                        if next_action is not None:
                            next_value = getattr(next_action, 'next_value', actions[vehicle_id].dur_reward)
                            if isinstance(next_action, ServiceAction):
                                next_action_type = "assign"
                            elif isinstance(next_action, ChargingAction):
                                next_action_type = "charge"
                            else:
                                next_action_type = "idle"
                        else:
                            # 没有next_action时，使用当前的dur_reward作为价值估计
                            next_value = actions[vehicle_id].dur_reward
                            next_action_type = "idle"
                        
                        r_exec = actions[vehicle_id].dur_reward  # Use accumulated reward from action

                        self.value_function_ev.store_experience(
                            vehicle_id=vehicle_id,
                            action_type="idle",
                            vehicle_location=current_location,
                            target_location=veh_curloc,
                            current_time=self.current_time,
                            reward=r_exec,
                            next_vehicle_location=veh_curloc,
                            battery_level=current_battery,
                            next_battery_level=batterynow,
                            other_vehicles=other_vehicles,
                            num_requests=num_requests,
                            request_value=r_exec,
                            next_action_type=next_action_type,
                            next_request_value=next_value,
                            dur_time=getattr(action, 'dur_time', 1.0),
                            is_system_done=getattr(self, 'done', False)
                        )

    def _execute_action(self, vehicle_id, action):
        """Execute vehicle action with immediate reward aligned to Gurobi optimization objective"""
        from src.Action import ChargingAction, ServiceAction, IdleAction

        vehicle = self.vehicles[vehicle_id]
        
        # Ensure storeactions[vehicle_id] exists
        if vehicle_id not in self.storeactions or self.storeactions[vehicle_id] is None:
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].current_time = self.current_time
        
        # Check if vehicle is in stationary state
        
        
        reward = 0
        dur_reward = 0.0
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
                        vehicle['charging_target'] = None
                        
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
            if vehicle['type'] == 1:
                reward += -0.1  # Extra penalty for EVs to encourage efficiency
                if self.storeactions_ev[vehicle_id] is not None:
                    self.storeactions_ev[vehicle_id].dur_reward += reward  # Store for reference
            else:
                reward += -0.01  # Smaller penalty for ICE vehicles
                if self.storeactions[vehicle_id] is not None:
                    self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
            dur_reward = action.dur_reward  # Total reward over charging duration
        elif isinstance(action, ServiceAction):
            # Service action - immediate reward is request.value (same as Gurobi)
            # Check if vehicle is in stationary state (rejection case)
            if vehicle['idle_target'] is not None:
                # Vehicle rejected the request - apply penalty and don't move
                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                
                penalty_reward = - avg_request_value * 0.1
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += penalty_reward
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += penalty_reward
                            
                return 0, penalty_reward  # Penalty for rejection
            elif vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:
                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += -avg_request_value*0.1
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += -avg_request_value*0.1
                return 0, -avg_request_value*0.1  # Penalty for invalid service action
            elif vehicle['assigned_request'] is not None:
                # Progress towards pickup - check if我们能pickup
                if self._pickup_passenger(vehicle_id):
                    reward = 0.5 + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:
                        vehicle['target_location'] = None
                        vehicle['idle_target'] = None
                        vehicle['assigned_request'] = None
                        vehicle['passenger_onboard'] = None
                        vehicle['charging_target'] = None
                        #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，无法继续前往pickup位置")
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += reward  # Store for reference
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
            elif vehicle['passenger_onboard'] is not None:
                earnings = self._dropoff_passenger(vehicle_id)
                if earnings > 0:
                    reward = earnings + np.random.normal(0, 0.2)
                else:
                    # 检查电池是否耗尽
                    if vehicle['battery'] <= 0.0:
                        #print(f"⚠️  车辆 {vehicle_id} 电池耗尽，乘客滞留无法到达dropoff位置")
                        vehicle['target_location'] = None
                        vehicle['idle_target'] = None
                        vehicle['assigned_request'] = None
                        vehicle['passenger_onboard'] = None
                        vehicle['charging_target'] = None
                    else:
                        reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += reward  # Store for reference
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += reward  # Store for reference
        
        elif isinstance(action, IdleAction):
            if vehicle.get('is_stationary', False):
                vehicle['stationary_duration'] = 1
        
                # If stationary duration is finished, remove stationary status
                if vehicle['stationary_duration'] <= 0:
                    vehicle['is_stationary'] = False
                    vehicle['stationary_duration'] = 0
                

                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += - avg_request_value*0.1
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += - avg_request_value*0.1
                return - avg_request_value*0.1, - avg_request_value*0.1
            else:
                active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
                reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target', None))
                if vehicle['type'] == 1:
                    if self.storeactions_ev[vehicle_id] is not None:
                        self.storeactions_ev[vehicle_id].dur_reward += reward  - avg_request_value*0.1
                        dur_reward = self.storeactions_ev[vehicle_id].dur_reward  # Total reward over charging duration
                else:
                    if self.storeactions[vehicle_id] is not None:
                        self.storeactions[vehicle_id].dur_reward += reward  - avg_request_value*0.1
                        dur_reward = self.storeactions[vehicle_id].dur_reward  # Total reward over charging duration
        # if vehicle.get('is_stationary', False):
        #     # Reduce stationary duration
        #     vehicle['stationary_duration'] = 1
            
        #     # If stationary duration is finished, remove stationary status
        #     if vehicle['stationary_duration'] <= 0:
        #         vehicle['is_stationary'] = False
        #         vehicle['stationary_duration'] = 0
            

        #     active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        #     active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        #     avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
        #     self.storeactions[vehicle_id].dur_reward += - avg_request_value*0.01
        #     return - avg_request_value*0.01, - avg_request_value*0.01
        return reward, dur_reward
    
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
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            vehicle['needs_emergency_charging'] = True
            #print(f"⚠️  车辆 {vehicle_id} 在智能移动后电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return (self.movingpenalty  +  np.abs(np.random.normal(0, 0.05)))*distance if distance > 0 else -0.05 
    

    def return_nearest_idle_target(self, vehicle_id):
        if self.use_intense_requests:
            return self.hotspot_locations[0]
        else:
            vehicle_loc = self.vehicles[vehicle_id]['location']
            x = vehicle_loc % self.grid_size
            y = vehicle_loc // self.grid_size
            distance_list = []
            for loc in self.hotspot_locations:
                hx, hy = loc
                dist = abs(hx - x) + abs(hy - y)
                distance_list.append((dist, loc))
            if distance_list:
                # Return the closest hotspot location
                return min(distance_list, key=lambda item: item[0])[1]
            return None

    
    
    
    
    
    def return_nearest_hotspot_index(self, vehicle_id):
        vehicle_loc = self.vehicles[vehicle_id]['location']
        x = vehicle_loc % self.grid_size
        y = vehicle_loc // self.grid_size
        distance_list = []
        for idx, loc in enumerate(self.hotspot_locations):
            hx, hy = loc
            dist = abs(hx - x) + abs(hy - y)
            distance_list.append((dist, idx))
        if distance_list:
            # Return the index of the closest hotspot location
            return min(distance_list, key=lambda item: item[0])[1]
        return None


    def _store_aev_action(self, rebalancing_assignments):

        for vehicle_id, target_request in rebalancing_assignments.items():
            vehicle_location = self.vehicles[vehicle_id]['location']
            vehicle_battery = self.vehicles[vehicle_id]['battery']
            self.vehicles[vehicle_id]['needs_emergency_charging'] = False  # Reset emergency flag after assignment
            self.vehicles[vehicle_id]['is_stationary'] = False  # Reset stationary state if moving to charge
            if target_request:
                # Check if it's a charging assignment (string) or request assignment (object)
                if isinstance(target_request, str) and target_request.startswith("charge_"):
                    #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to charging at step {self.current_time}, battery: {vehicle_battery:.2f}")
                    
                    # Handle charging assignment
                    station_id = int(target_request.replace("charge_", ""))
                    #print(f"ASSIGN: Vehicle {vehicle_id} assigned to charging station {station_id} at step {self.current_time}")
                    self._move_vehicle_to_charging_station(vehicle_id, station_id)
                    charging_assignments += 1
                    # Generate charging action
                    from src.Action import ChargingAction
                    
                    actions[vehicle_id] = ChargingAction([], station_id, self.charge_duration, vehicle_location,vehicle_battery,req_num = quest_num_now)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)
                elif isinstance(target_request, Request) and target_request.request_id in self.active_requests:
                    #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to request {target_request.request_id} at step {self.current_time}, battery: {vehicle_battery:.2f}")
                    # Handle regular request assignment  
                    if self._assign_request_to_vehicle(vehicle_id, target_request.request_id):
                        new_assignments += 1
                        vehicle['idle_timer'] = 0  # Reset idle timer on new assignment
                        vehicle['continual_reject'] = 0  # Reset continual reject counter on new assignment
                        vehicle['penalty_timer'] = 0  # Clear any penalty timer on new assignment
                        vehicle['idle_target'] = None  # Clear idle target on new assignment
                        # Generate service action
                        from src.Action import ServiceAction
                        actions[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location,vehicle_battery,req_num = quest_num_now)
                        if vehicle['type'] == 1:
                            self._store_action_ev(vehicle_id, actions[vehicle_id], storeactions_ev, vehicle_location, vehicle_battery,
                                                target_coords=self.active_requests[target_request.request_id].dropoff,
                                                next_value=self.active_requests[target_request.request_id].final_value)
                        else:
                            self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                                target_coords=self.active_requests[target_request.request_id].dropoff,
                                                next_value=self.active_requests[target_request.request_id].final_value)

                elif isinstance(target_request, str) and target_request == "waiting":
                    #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to waiting at step {self.current_time}, battery: {vehicle_battery:.2f}")
                    # Handle waiting state - mark vehicle as stationary for next simulation
                    vehicle = self.vehicles[vehicle_id]
                    vehicle['is_stationary'] = True
                    vehicle['stationary_duration'] = getattr(target_request, 'duration', 1)  # Default 2 steps
                    # Generate idle action to keep vehicle stationary
                    from src.Action import IdleAction
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, current_coords, vehicle_location,vehicle_battery,req_num = quest_num_now)  # Stay in place
                    self._store_action(vehicle_id, actions[vehicle_id], storeactions, vehicle_location, vehicle_battery,
                                        target_coords=self.vehicles[vehicle_id]['location'])
                elif isinstance(target_request, str) and target_request.startswith("idle_at_"):
                    #print(f"DEBUG Assignment: Vehicle {vehicle_id} assigned to idle at step {self.current_time}, battery: {vehicle_battery:.2f}")
                    zone_id_str = target_request.replace("idle_at_", "")
                    zone_id = int(zone_id_str)
                    hotspot_coords = self.hotspot_locations[zone_id]
                    hot_x = hotspot_coords[0]
                    hot_y = hotspot_coords[1]

                    from src.Action import IdleAction
                    vehicle = self.vehicles[vehicle_id]
                    # 不需要调用_assign_idle_vehicle，因为我们手动设置idle_target
                    vehicle['is_stationary'] = False  # Reset stationary state if moving to idle target
                    idle_target = (hot_x, hot_y)
                    vehicle['assigned_request'] = None
                    vehicle['passenger_onboard'] = None
                    vehicle['charging_station'] = None
                    vehicle['target_location'] = None
                    vehicle['idle_target'] = idle_target
                    current_coords = vehicle['coordinates']
                    actions[vehicle_id] = IdleAction([], current_coords, idle_target, vehicle_location, vehicle_battery,req_num = quest_num_now)
                    self._update_storeaction(vehicle_id, actions[vehicle_id], storeactions, is_ev=False)





    def _store_action(self, vehicle_id, action, storeactions_dict, vehicle_location, vehicle_battery, target_coords=None, next_value=0):
        """封装storeactions的赋值逻辑
        
        Args:
            vehicle_id: 车辆ID
            action: 要存储的action对象
            storeactions_dict: 本地storeactions字典（用于immediate更新）
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 可选的目标坐标，如果不提供则使用vehicle['target_location']
            next_value: next_action的value值，默认为0（用于ServiceAction时可传入final_value）
        """
        if target_coords is None:
            target_coords = self.vehicles[vehicle_id]['target_location']
        
        if self.storeactions[vehicle_id] is None:
            # 新action - 直接存储
            storeactions_dict[vehicle_id] = action
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].current_time = self.current_time
            self.storeactions[vehicle_id].target_location = target_coords
        else:
            # 替换action - 保存旧信息
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].next_action.next_value = next_value
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            
            self.storeactions[vehicle_id] = None
            self.storeactions[vehicle_id] = action
            self.storeactions[vehicle_id].dur_reward = 0
            self.storeactions[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions[vehicle_id].current_time = self.current_time
            self.storeactions[vehicle_id].target_location = target_coords
    
    def _store_action_ev(self, vehicle_id, action, storeactions_ev_dict, vehicle_location, vehicle_battery, target_coords=None, next_value=0):
        """封装storeactions_ev的赋值逻辑
        
        Args:
            vehicle_id: 车辆ID
            action: 要存储的action对象
            storeactions_ev_dict: 本地storeactions_ev字典（用于immediate更新）
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 可选的目标坐标，如果不提供则使用vehicle['target_location']
            next_value: next_action的value值，默认为0（用于ServiceAction时可传入final_value）
        """
        if target_coords is None:
            target_coords = self.vehicles[vehicle_id]['target_location']
        
        if self.storeactions_ev[vehicle_id] is None:
            # 新action - 直接存储
            storeactions_ev_dict[vehicle_id] = action
            self.storeactions_ev[vehicle_id] = action
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
        else:
            # 替换action - 保存旧信息
            storeactions_ev_dict[vehicle_id].next_action = action
            storeactions_ev_dict[vehicle_id].next_action.next_value = next_value
            storeactions_ev_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_ev_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_ev_dict[vehicle_id], 'current_time', self.current_time)
            
            self.storeactions_ev[vehicle_id] = None
            self.storeactions_ev[vehicle_id] = action
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
    
    def _store_rejected_ev_action(self, vehicle_id, idle_action, rejected_request_id, storeactions_ev_dict, vehicle_location, vehicle_battery, target_coords):
        """封装EV拒单后的storeactions_ev逻辑，存储ServiceAction用于拒单penalty计算
        
        Args:
            vehicle_id: 车辆ID
            idle_action: 实际执行的IdleAction
            rejected_request_id: 被拒绝的请求ID
            storeactions_ev_dict: 本地storeactions_ev字典
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 目标坐标
        """
        quest_num_now = len(self.active_requests)
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
        penalty_reward = - avg_request_value * 0.1
        
        from src.Action import ServiceAction
        
        if self.storeactions_ev[vehicle_id] is None:
            # 新action - 存储IdleAction但记录为拒单的ServiceAction用于计算
            storeactions_ev_dict[vehicle_id] = idle_action
            self.storeactions_ev[vehicle_id] = ServiceAction([], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            self.storeactions_ev[vehicle_id].dur_reward = penalty_reward
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
        else:
            # 替换action - 保存旧信息
            storeactions_ev_dict[vehicle_id].next_action = ServiceAction([], rejected_request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            storeactions_ev_dict[vehicle_id].next_action.next_value = 0
            storeactions_ev_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_ev_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_ev_dict[vehicle_id], 'current_time', self.current_time)
            
            self.storeactions_ev[vehicle_id] = None
            self.storeactions_ev[vehicle_id] = idle_action
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
    
    def _handle_ev_rejection_relocation(self, vehicle_id):
        """封装EV拒单后的relocation决策逻辑
        
        Args:
            vehicle_id: 车辆ID
            
        Returns:
            tuple: (target_coords, rel_action) 目标坐标和relocation动作类型
        """
        rel_probs = self.compute_ev_relocation_probability(vehicle_id)
        r = random.random()
        acc = 0.0
        rel_action = 'Wait'  # Default action is to wait
        for action_name, prob in rel_probs.items():
            acc += float(prob)
            if r <= acc:
                rel_action = action_name
                break
        
        vehicle = self.vehicles[vehicle_id]
        current_coords = vehicle['coordinates']
        
        if rel_action == 'Wait':
            # 停在原地
            target_coords = current_coords
        else:
            # 选择目标位置
            target_loc = self.sample_ev_relocation_target(vehicle_id, rel_action)
            target_coords = (target_loc % self.grid_size, target_loc // self.grid_size)
        
        return target_coords, rel_action

    def _assign_idle_vehicle(self, vehicle_id):

        vehicle = self.vehicles[vehicle_id]
        vehicle['assigned_request'] = None
        vehicle['passenger_onboard'] = None
        vehicle['charging_station'] = None
        vehicle['idle_target'] = None
        vehicle['target_location'] = None
        veh_location = vehicle['location']
        current_coords = (veh_location % self.grid_size, veh_location // self.grid_size)
        # Choose a neighboring cell to ensure at least one-step movement (avoid staying in place)
        cx, cy = current_coords
        candidates = []
        if cx + 1 < self.grid_size:
            candidates.append((cx + 1, cy))
        if cx - 1 >= 0:
            candidates.append((cx - 1, cy))
        if cy + 1 < self.grid_size:
            candidates.append((cx, cy + 1))
        if cy - 1 >= 0:
            candidates.append((cx, cy - 1))
        if candidates:
            target_x, target_y = random.choice(candidates)
        else:
            target_x, target_y = cx, cy
        
        # # Use hotspot location if available, otherwise use the calculated random neighbor

        # target_x, target_y = self.return_nearest_idle_target(vehicle_id)

        # vehicle['idle_target'] = (target_x, target_y)

        return True

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
        if (current_x, current_y) == (target_x, target_y):
            vehicle['charging_target'] = None
            success = station.start_charging(str(vehicle_id))
            #print(f"DEBUG: Vehicle {vehicle_id} at charging station {station_id}, trying to start: success={success}")
            if success:
                vehicle['charging_station'] = station_id
                vehicle['charging_time_left'] = getattr(self, 'charge_duration', 2)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None  # Clear any rebalance target
                vehicle.pop('target_charging_station', None)  # Remove target
                #print(f"DEBUG: Vehicle {vehicle_id} started charging at station {station_id}")
                
                # Return charging penalty (same as Gurobi)
                charging_penalty = getattr(self, 'charging_penalty', 2.0)
                return -charging_penalty
            else:
                return 0
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
        else:
            # Already at target
            new_x, new_y = current_x, current_y
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])

        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_y * self.grid_size + new_x
        vehicle['battery'] -= distance * (self.battery_consum + np.abs(np.random.random() * 0.0005))
        vehicle['battery'] = max(0, vehicle['battery'])
        
        if (new_x, new_y) == (target_x, target_y):
            vehicle['charging_target'] = None
            success = station.start_charging(str(vehicle_id))
            #print(f"DEBUG: Vehicle {vehicle_id} at charging station {station_id}, trying to start: success={success}")
            if success:
                vehicle['charging_station'] = station_id
                vehicle['charging_time_left'] = getattr(self, 'charge_duration', 2)
                vehicle['charging_count'] += 1
                vehicle['target_location'] = None  # Clear any rebalance target
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
            vehicle['battery'] = 1
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            #print(f"⚠️  车辆 {vehicle_id} 前往充电站时电池耗尽 (位置: {new_x}, {new_y})")
        
        # Small time penalty for movement (consistent with other movement methods)
        return (self.movingpenalty  +  np.abs(np.random.normal(0, 0.05)))*distance 
    
    def _execute_movement_towards_idle(self, vehicle_id, target_coords):
        """Execute movement towards idle target coordinates"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is not None:
            return 0
        
        # No target to move towards
        if not target_coords:
            return 0

        old_coords = vehicle['coordinates']
        current_x, current_y = old_coords
        target_x, target_y = target_coords
        if current_x == target_x and current_y == target_y:
            vehicle['idle_target'] = None
            # 已经在目标点：清空idle目标，打印一次调试信息
            #print(f"DEBUG: Idle movement - already at target {target_coords}")
            return 0
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
            vehicle['idle_target'] = None
            #print("DEBUG: Idle movement - already at target (redundant branch)")
        # Update vehicle position
        distance = abs(new_x - old_coords[0]) + abs(new_y - old_coords[1])
        new_location_index = new_y * self.grid_size + new_x
        
        vehicle['coordinates'] = (new_x, new_y)
        vehicle['location'] = new_location_index
        vehicle['total_distance'] += distance
        # If reached target after this move, clear and log
        if (new_x, new_y) == (target_x, target_y):
            vehicle['idle_target'] = None
            #print(f"DEBUG: Idle movement - reached target {target_coords}")
        
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
            vehicle['target_location'] = None
            vehicle['idle_target'] = None
            vehicle['assigned_request'] = None
            vehicle['passenger_onboard'] = None
            vehicle['charging_target'] = None
            vehicle['needs_emergency_charging'] = True
            #print(f"⚠️  车辆 {vehicle_id} 在闲置移动后电池耗尽 (位置: {new_x}, {new_y})")
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
        # Small time penalty for movement (consistent with other methods)
        return (self.movingpenalty  +  np.abs(np.random.normal(0, 0.05)))*distance   
    
    def _update_environment(self):
        """Update environment state"""
        self.current_time += 1
        
        # Generate new requests using selected method
        if self.use_intense_requests:
            new_requests = self._generate_intense_requests()  # Now returns a list

        else:
            new_requests = self._generate_random_requests()  # Now also returns a list
        request_num = len(new_requests)
        self.whole_req_num += request_num
        # Update charging status
        
        
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['assigned_request'] is None and vehicle['passenger_onboard'] is None:  
                vehicle['idle_timer'] += 1
            if vehicle['penalty_timer'] > 0:
                vehicle['penalty_timer'] -= 1
            
                    
        
        
        
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
                    

        # This fixes the issue where stations auto-start vehicles from queue but don't update vehicle state
        for station_id, station in self.charging_manager.stations.items():
            for charging_vehicle_id in station.current_vehicles:
                vehicle_id = int(charging_vehicle_id)
                if vehicle_id in self.vehicles:
                    vehicle = self.vehicles[vehicle_id]
                    vehicle['charging_target'] = None
                    
                    # If vehicle is in station but doesn't know it's charging, sync the state
                    if vehicle['charging_station'] is None:
                        vehicle['charging_station'] = station_id
                        vehicle['charging_time_left'] = getattr(self, 'default_charging_duration', 2)
                        # Don't increment charging_count as this is just a sync operation
            self.idle_charging_num[station_id] = station.max_capacity - len(station.current_vehicles)
        self.current_online = sum(1 for vehicle in self.vehicles.values() if vehicle['idle_target'] is not None)

        # Remove expired requests and apply unserved penalty
        current_time = self.current_time
        expired_requests = []
        unserved_penalty_total = 0
        
        for request_id, request in self.active_requests.items():
            if current_time > request.pickup_deadline:
                expired_requests.append(request_id)
                # Apply penalty for unserved request
                unserved_penalty_total += self.unserved_penalty

        # Debug info
        if len(expired_requests) > 0 and self.current_time % 50 == 0:
            print(f"⏰ Time {current_time}: {len(expired_requests)} requests expired (pickup_deadline passed)")

        # Distribute unserved penalty among all vehicles
        expired_being_served = 0
        for request_id in expired_requests:
            # 检查订单是否正在被车辆服务（assigned或onboard）
            request_being_served = False
            for vehicle_id, vehicle in self.vehicles.items():
                if (vehicle['assigned_request'] == request_id or 
                    vehicle['passenger_onboard'] == request_id):
                    request_being_served = True
                    expired_being_served += 1
                    if self.current_time % 50 == 0:
                        print(f"   ⚠️  Request {request_id} expired but still being served by vehicle {vehicle_id}")
                    break
            
            # 只删除不在服务中的过期订单
            if not request_being_served:
                request = self.active_requests[request_id]
                #self.rejected_requests.append(request)
                del self.active_requests[request_id]
        
        if expired_being_served > 0 and self.current_time % 50 == 0:
            print(f"   📊 {expired_being_served} expired requests still being served (kept in active_requests)")
        
        # Distribute unserved penalty among all vehicles
        # for vehicle_id, vehicle in self.vehicles.items():
        #     if self.vehicles[vehicle_id]['assigned_request'] and self.vehicles[vehicle_id]['assigned_request'] in expired_requests:
        #         # 只有当订单真的被删除时才清空车辆状态
        #         if self.vehicles[vehicle_id]['assigned_request'] not in self.active_requests:
        #             vehicle['assigned_request'] = None 
        #     if self.vehicles[vehicle_id]['passenger_onboard'] and self.vehicles[vehicle_id]['passenger_onboard'] in expired_requests:
        #         # 只有当订单真的被删除时才清空车辆状态
        #         if self.vehicles[vehicle_id]['passenger_onboard'] not in self.active_requests:
        #             vehicle['passenger_onboard'] = None
        # NOTE: EV penalty filtering is handled in simulate_motion().
            # if self.vehicles[vehicle_id]['passenger_onboard'] and self.vehicles[vehicle_id]['battery'] <= self.min_battery_level:
            #     # Handle passenger stranding due to low battery
            #     request_id = vehicle['passenger_onboard']
            #     vehicle['passenger_onboard'] = None  
            #     # Add to unserved penalty instead of trying to modify non-existent reward field
            #     vehicle['unserved_penalty'] += self.penalty_for_passenger_stranding
            #     # Remove the stranded passenger's request from active requests
            #     if request_id in self.active_requests:
            #         request = self.active_requests[request_id]
            #         self.rejected_requests.append(request)
            #         del self.active_requests[request_id]

            
    def reset(self):
        """重置环境"""
        self.current_time = 0
        # Reset request system
        self.request_value_sum = 0
        self.whole_req = 0
        self.ev_requests = []
        self.active_requests = {}
        self.whole_req_num = 0
        self.completed_requests = []
        self.completed_requests_ev = []
        self.rejected_requests = []
        self.request_counter = 0
        self.charge_finished = 0
        self.charge_stats = {station_id: [] for station_id in self.charging_manager.stations}
        # Reset rebalancing assignment tracking
        self.rebalancing_assignments_per_step = []
        self.rebalancing_whole = []
        self.total_rebalancing_calls = 0
        self.storeactions = {}
        self.storeactions_ev = {}
        self.storeactions_next = {}
        self._setup_vehicles()
        return self.get_initial_states()
    
    def get_episode_stats(self):
        """Get detailed statistics for current episode"""
        # Calculate average battery level
        total_battery = sum(v['battery'] for v in self.vehicles.values())
        avg_battery = total_battery / len(self.vehicles) if self.vehicles else 0
        
        # Calculate total rejected requests (unique orders that were rejected)
        total_rejected = len(self.rejected_requests)
        total_ev_request = len(self.ev_requests)
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
        completed_ev_orders = len(self.completed_requests_ev)
        service_ratio = completed_orders/self.whole_req_num if self.whole_req_num > 0 else 0
        avg_request_value1 = self.request_value_sum/completed_orders if completed_orders > 0 else 0
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
        avg_rebalance_whole = 0
        if self.rebalancing_assignments_per_step:
            total_rebalancing_assignments = sum(self.rebalancing_assignments_per_step)
            avg_rebalancing_assignments = total_rebalancing_assignments / len(self.rebalancing_assignments_per_step)
            avg_rebalance_whole = sum(self.rebalancing_whole) / len(self.rebalancing_whole) if self.rebalancing_whole else 0
        return {
            'episode_time': self.current_time,
            'total_orders': total_orders,
            'accepted_orders': accepted_orders,
            'active_orders': active_orders,
            'rejected_orders': total_rejected,
            'ev_accept': total_ev_request,
            'completed_orders': completed_orders,
            'completed_ev_orders': completed_ev_orders,
            'service_ratio': service_ratio,
            'avg_request_value': avg_request_value1,
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
            'avg_rebalancing_assignments_per_whole': avg_rebalance_whole,
            'rebalancing_assignments_per_step': self.rebalancing_assignments_per_step.copy(),
            'rebalance_whole': self.rebalancing_whole.copy()
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
            'completed_orders_req': self.request_value_sum/len(self.completed_requests) if len(self.completed_requests) > 0 else 0,
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
        
        
        
    def simulate_motion_dqn(self, dqn_agent=None, current_requests: List[Request] = None, training=True):
        """
        DQN-based simulation for vehicle dispatch as benchmark comparison to ILP-ADP.
        存储的 transition 以“动作完成”为 done（接单完成/失败、充电完成、wait/idle 单步）。
        """
        # Import lightweight DQN utilities without changing other components
        try:
            from .ValueFunction_pytorch import DQNAgent, create_dqn_state_features
        except Exception:
            print("Warning: DQN components not available. Please ensure ValueFunction_pytorch.py defines DQNAgent and create_dqn_state_features.")
            return None

        # If caller did not provide a request list, we'll derive a per-vehicle Top-K (by distance) later
        provided_requests = None if current_requests is None else list(current_requests)

        # Instantiate a default agent if needed (benchmark use only)
        if dqn_agent is None:
            device = 'cuda' if hasattr(self, 'device') else 'cpu'
            dqn_agent = DQNAgent(state_dim=64, action_dim=32, device=device)

        results = {
            'total_reward': 0.0,
            'actions_taken': [],
            'vehicle_utilization': 0.0,
            'request_completion_rate': 0.0,
            'average_battery_level': 0.0,
            'dqn_decisions': [],
            'transitions_added': 0
        }

        # Snapshot counts for metrics
        completed_before = len(self.completed_requests)

        # Prepare per-action buffer (persist across ticks)
        if not hasattr(self, '_dqn_action_buffers'):
            self._dqn_action_buffers = {}

        # 1) Progress ongoing actions for vehicles already in buffer
        for vehicle_id, buf in list(self._dqn_action_buffers.items()):
            v = self.vehicles.get(vehicle_id)
            if not v:
                del self._dqn_action_buffers[vehicle_id]
                continue

            a_type = buf.get('action_type', 'idle')
            step_reward = 0.0
            action_done = False

            if a_type == 'assign':
                # Continue towards pickup/dropoff
                if self._pickup_passenger(vehicle_id):
                    step_reward += 0.5 + np.random.normal(0, 0.2)
                else:
                    step_reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)
                # Try dropoff if onboard
                if v.get('passenger_onboard') is not None:
                    drop_r = self._dropoff_passenger(vehicle_id)
                    if drop_r > 0:
                        step_reward += drop_r + np.random.normal(0, 0.2)
                        # Mark successful completion for this buffered assign
                        buf['dropoff_done'] = True
                    else:
                        step_reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)
                # Done when vehicle becomes free (no assigned and no onboard)
                if v.get('assigned_request') is None and v.get('passenger_onboard') is None:
                    # If we became free without a recorded dropoff, treat as assignment failure
                    if not buf.get('dropoff_done', False):
                        active_requests_count = len(self.active_requests)
                        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
                        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
                        step_reward += -avg_request_value*0.01
                    action_done = True

            elif a_type == 'charge':
                station_id = buf.get('station_id')
                if v.get('charging_station') is None:
                    # Move towards station / try start charging
                    vloc = v.get('location', 0)
                    # Ensure station_id is valid; remap to nearest if missing
                    if not hasattr(self, 'charging_manager') or not getattr(self.charging_manager, 'stations', None):
                        action_done = True  # no stations; consider action finished
                    else:
                        stations = self.charging_manager.stations
                        if station_id not in stations:
                            best_sid, best_d = None, 1e9
                            for sid, st in stations.items():
                                d = self._manhattan_distance_loc(vloc, st.location)
                                if d < best_d:
                                    best_sid, best_d = sid, d
                            station_id = best_sid if best_sid is not None else list(stations.keys())[0]
                            buf['station_id'] = station_id
                        self._move_vehicle_to_charging_station(vehicle_id, station_id)
                        step_reward += self._execute_movement_towards_charging_station(vehicle_id, station_id)
                # Detect completion: charging_duration == 0 if present; else charging_time_left <= 0; or finished charging (was charging -> now not charging)
                cd = v.get('charging_duration', None)
                ctl = v.get('charging_time_left', None)
                finished_by_time = (cd == 0) if cd is not None else (ctl is not None and ctl <= 0)
                if finished_by_time:
                    action_done = True
                else:
                    if buf.get('was_charging', False) and v.get('charging_station') is None:
                        action_done = True
                    if v.get('charging_station') is not None:
                        buf['was_charging'] = True

            # wait/idle are stored immediately at creation; buffers shouldn't exist for them

            # Accumulate and finalize if needed
            buf['acc_reward'] = buf.get('acc_reward', 0.0) + float(step_reward)
            results['total_reward'] += float(step_reward)

            if action_done and training and hasattr(dqn_agent, 'store_transition'):
                next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                dqn_agent.store_transition(buf['state'], int(buf['action_idx']), float(buf['acc_reward']), next_state, done=True)
                results['transitions_added'] += 1
                del self._dqn_action_buffers[vehicle_id]

        # 2) Iterate idle-capable vehicles only (not assigned/onboard/charging) to start new actions
        for vehicle_id, v in self.vehicles.items():
            if vehicle_id in self._dqn_action_buffers:
                continue
            # Skip busy vehicles
            is_free = (v.get('assigned_request') is None and
                       v.get('passenger_onboard') is None and
                       v.get('charging_station') is None or v.get('is_stationary') is True)
            if not is_free and v.get('battery_level', 1.0) > self.min_battery_level:
                continue

            vehicle = self.vehicles[vehicle_id]
            # 根据车辆类型构建候选请求：
            # - EV(type==1): 最近且电量可完成
            # - AEV(type==2): 价值最高且电量可完成
            if provided_requests is not None:
                veh_requests = provided_requests
            else:
                all_reqs = list(self.active_requests.values())
                if not all_reqs:
                    veh_requests = []
                else:
                    vloc = v.get('location', 0)
                    battery = v.get('battery', v.get('battery_level', 1.0))

                    def feasible(req):
                        pick = getattr(req, 'pickup', 0)
                        drop = getattr(req, 'dropoff', 0)
                        d1 = self._manhattan_distance_loc(vloc, pick)
                        d2 = self._manhattan_distance_loc(pick, drop)
                        total_d = d1 + d2
                        est_use = total_d * (self.battery_consum)
                        return est_use <= battery

                    feasible_reqs = [req for req in all_reqs if feasible(req)]

                    if vehicle['type'] == 1:  # EV: 最近优先
                        feasible_reqs.sort(key=lambda r: (
                            self._manhattan_distance_loc(vloc, getattr(r, 'pickup', 0)),
                            getattr(r, 'request_id', 0)
                        ))
                    else:  # AEV: 价值最高优先
                        feasible_reqs.sort(key=lambda r: (
                            -float(getattr(r, 'final_value', getattr(r, 'value', 0.0))),
                            getattr(r, 'request_id', 0)
                        ))

                    veh_requests = feasible_reqs[:10]
                
            # Build DQN state features
            state = create_dqn_state_features(self, vehicle_id, self.current_time)

            # 检查当前idle车辆数量，确定是否需要考虑idle约束
            current_idle_count = self._count_idle_vehicles()
            min_required_idle = getattr(self, 'idle_vehicle_requirement', 1)
            
            # 计算还需要多少车辆保持idle状态
            idle_deficit = max(0, min_required_idle - current_idle_count)
            need_idle_constraint = idle_deficit > 0
            
            if need_idle_constraint:
                print(f"  📊 Idle constraint active: current={current_idle_count}, required={min_required_idle}, deficit={idle_deficit}")

            # Select action (考虑idle约束)
            action_idx, q_values = dqn_agent.select_action(
                state['vehicle'], state['request'], state['global'], 
                training=training,
                force_idle_constraint=need_idle_constraint  # 传递约束信息
            )

            # Map to environment-level action spec
            env_action = self._map_dqn_action_to_env(action_idx, vehicle_id, veh_requests)

            # Execute action via DQN executor (uses existing helpers internally)
            reward = self._execute_dqn_action(vehicle_id, env_action, veh_requests)

            # Record
            results['total_reward'] += float(reward)
            results['actions_taken'].append({
                'vehicle_id': vehicle_id,
                'dqn_action': int(action_idx),
                'env_action': env_action,
                'reward': float(reward),
                'q_values': q_values.detach().cpu().numpy().tolist() if hasattr(q_values, 'detach') else None
            })

            # Store transition logic based on action completeness semantics
            if training and hasattr(dqn_agent, 'store_transition'):
                a_type = env_action.get('type', 'idle')
                if a_type == 'assign':
                    # If assignment failed this step (仍然空闲且没有乘客)，立即存储并结束
                    if v.get('assigned_request') is None and v.get('passenger_onboard') is None:
                        next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                        dqn_agent.store_transition(state, int(action_idx), float(reward), next_state, done=True)
                        results['transitions_added'] += 1
                    else:
                        # 成功接单：进入缓冲，累计到 dropoff 完成
                        self._dqn_action_buffers[vehicle_id] = {
                            'state': state,
                            'action_idx': int(action_idx),
                            'env_action': env_action,
                            'action_type': 'assign',
                            'acc_reward': float(reward)
                        }
                elif a_type == 'charge':
                    # 充电：进入缓冲，直到 charging_duration==0 或 charging_time_left<=0 完成
                    self._dqn_action_buffers[vehicle_id] = {
                        'state': state,
                        'action_idx': int(action_idx),
                        'env_action': env_action,
                        'action_type': 'charge',
                        'station_id': env_action.get('station_id'),
                        'acc_reward': float(reward),
                        'was_charging': v.get('charging_station') is not None
                    }
                else:
                    # wait/idle 单步即完成
                    next_state = create_dqn_state_features(self, vehicle_id, self.current_time)
                    dqn_agent.store_transition(state, int(action_idx), float(reward), next_state, done=True)
                    results['transitions_added'] += 1

        # Advance environment by one tick to keep parity with built-in step()
        self._update_environment()

        # Compute benchmark metrics from current env
        self._calculate_dqn_performance_metrics(results, current_requests)

        # Optional policy update
        if training and hasattr(dqn_agent, 'train_step'):
            loss = dqn_agent.train_step(batch_size=32)
            if loss is not None:
                results['training_loss'] = float(loss)

        # Snapshot deltas
        results['orders_completed_delta'] = max(0, len(self.completed_requests) - completed_before)

        return results
    
    def _map_dqn_action_to_env(self, dqn_action, vehicle_id, current_requests):
        """
        Map DQN action index to environment action
        
        Args:
            dqn_action: Action index from DQN
            vehicle_id: ID of the vehicle
            current_requests: Available requests
        
        Returns:
            dict: Environment action
        """
        # Define comprehensive action mapping for vehicle dispatch:
        # Actions 0-9: Assign to request (if available)
        # Actions 10-19: Move to location for rebalancing/repositioning
        # Actions 20-24: Charge at station (EVs only)
        # Actions 25-27: Wait for better requests
        # Actions 28-31: Idle/do nothing
        
        if current_requests and dqn_action < min(10, len(current_requests)):
            # Assign to request (接受订单)
            request = current_requests[dqn_action]
            return {
                'type': 'assign',
                'request_id': getattr(request, 'request_id', dqn_action),
                'pickup_location': getattr(request, 'pickup', 0),
                'dropoff_location': getattr(request, 'dropoff', 0)
            }
        elif 10 <= dqn_action < 20:
            # Rebalance to strategic location (重新平衡)
            target_location = (dqn_action - 10) * (self.NUM_LOCATIONS // 10)
            return {
                'type': 'rebalance',
                'target_location': min(target_location, self.NUM_LOCATIONS - 1)
            }
        elif 20 <= dqn_action < 25:
            # Charge at station (充电) - map to an existing station key
            station_id = dqn_action - 20
            if hasattr(self, 'charging_manager') and getattr(self.charging_manager, 'stations', None):
                station_keys = list(self.charging_manager.stations.keys())
                if station_keys:
                    station_id = station_keys[(dqn_action - 20) % len(station_keys)]
            return {
                'type': 'charge',
                'station_id': station_id
            }
        elif 25 <= dqn_action < 28:
            # Wait for better requests (等待更好的订单)
            wait_duration = (dqn_action - 25 + 1) * 5  # 5, 10, or 15 minutes
            return {
                'type': 'wait',
                'duration': wait_duration,
                'reason': 'better_requests'
            }
        else:
            # Idle (空闲)
            return {
                'type': 'idle'
            }
    
    def _execute_dqn_action(self, vehicle_id, action, current_requests):
        """
        Execute DQN action in environment and return reward
        
        Args:
            vehicle_id: ID of the vehicle
            action: Action to execute
            current_requests: Available requests
        
        Returns:
            float: Reward for the action
        """
        vehicle = self.vehicles.get(vehicle_id)
        if not vehicle:
            return -1.0

        a_type = action.get('type', 'idle')

        # Helper to safely extract request from provided mapping or index
        def _resolve_request(req_id_or_index):
            # Prefer direct id lookup
            if isinstance(req_id_or_index, (int, str)) and req_id_or_index in self.active_requests:
                return self.active_requests[req_id_or_index]
            # Fallback: index into current_requests if valid
            if isinstance(req_id_or_index, int) and 0 <= req_id_or_index < len(current_requests or []):
                candidate = current_requests[req_id_or_index]
                rid = getattr(candidate, 'request_id', None)
                if rid in self.active_requests:
                    return self.active_requests[rid]
                return candidate
            return None

        # Assign/serve: accept one existing order
        if a_type == 'assign':
            req_id = action.get('request_id')
            # Also accept alternate keys from mapper
            if req_id is None:
                req_id = action.get('req_id')
            req = _resolve_request(req_id if req_id is not None else 0)
            if req is None:
                return -0.5  # No valid request to accept

            # Ensure the id we use exists in active_requests
            rid = getattr(req, 'request_id', None)
            if rid is None or rid not in self.active_requests:
                return -0.5

            # Attempt to assign (may reject based on EV behaviour)
            accepted = self._assign_request_to_vehicle(vehicle_id, rid)
            if not accepted:
                # Vehicle chose to reject; mark stationary penalty via stationary branch
                vehicle['is_stationary'] = True
                vehicle['stationary_duration'] = 1
                active_requests_count = len(self.active_requests)
                active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
                avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 100.0
                vehicle['waiting_for_requests'] = True
                return -avg_request_value * 0.01

            # Move/pickup/dropoff using existing helpers
            reward = 0.0
            if self._pickup_passenger(vehicle_id):
                reward += 0.5 + np.random.normal(0, 0.2)
            else:
                # Move one step toward target (pickup or dropoff)
                reward += self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.05)

            # If passenger onboard after movement, attempt dropoff
            if vehicle.get('passenger_onboard') is not None:
                drop_reward = self._dropoff_passenger(vehicle_id)
                if drop_reward > 0:
                    reward += drop_reward + np.random.normal(0, 0.2)
                else:
                    reward = self._execute_movement_towards_target(vehicle_id) + np.random.normal(0, 0.1)
            return float(reward)

        # Charge: send to a station and let charging start when reached
        if a_type == 'charge' and vehicle.get('type') == 1:
            # Choose station: prefer provided id else nearest available
            station_id = action.get('station_id', None)
            if not hasattr(self, 'charging_manager') or not self.charging_manager.stations:
                return -1.0
            stations = self.charging_manager.stations
            if station_id not in stations:
                # Pick nearest station by manhattan distance
                vloc = vehicle.get('location', 0)
                best_sid, best_d = None, 1e9
                for sid, st in stations.items():
                    d = self._manhattan_distance_loc(vloc, st.location)
                    if d < best_d:
                        best_sid, best_d = sid, d
                station_id = best_sid if best_sid is not None else list(stations.keys())[0]

            # Set charging goal and move one step
            self._move_vehicle_to_charging_station(vehicle_id, station_id)
            reward = self._execute_movement_towards_charging_station(vehicle_id, station_id)
            vehicle['waiting_for_requests'] = False
            return float(reward)

        # Wait: stay in place for one step with small opportunity penalty
        if a_type == 'wait':
            vehicle['is_stationary'] = True
            vehicle['stationary_duration'] = int(action.get('duration', 1))
            active_requests_count = len(self.active_requests)
            active_requests_value = sum(req.final_value for req in self.active_requests.values()) if self.active_requests else 0.0
            avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 50.0
            vehicle['waiting_for_requests'] = True
            return float(-avg_request_value * 0.01)

        # Idle: move toward a simple idle target for exploration
        # If rebalance target provided, treat as idle to that target
        if a_type in ('idle', 'rebalance'):
            if a_type == 'rebalance' and 'target_location' in action:
                # Convert target index to coordinates
                tloc = int(action['target_location'])
                tx, ty = (tloc % self.grid_size, tloc // self.grid_size)
                vehicle['idle_target'] = (tx, ty)
            else:
                self._assign_idle_vehicle(vehicle_id)
            reward = self._execute_movement_towards_idle(vehicle_id, vehicle.get('idle_target'))
            vehicle['waiting_for_requests'] = False
            return float(reward)

        # Default fallback
        vehicle['waiting_for_requests'] = False
        return -0.1
    
    def _update_storeaction(self, vehicle_id, action, storeactions_dict, is_ev=False):
        """
        封装storeaction更新逻辑，避免代码重复
        
        Args:
            vehicle_id: 车辆ID
            action: 新的action对象
            storeactions_dict: storeactions或storeactions_ev字典
            is_ev: 是否为EV车辆
        """
        vehicle = self.vehicles[vehicle_id]
        vehicle_location = vehicle['location']
        vehicle_battery = vehicle['battery']
        target_coords = vehicle.get('target_location')
        
        # 获取对应的全局存储
        global_store = self.storeactions_ev if is_ev else self.storeactions
        
        if storeactions_dict[vehicle_id] is None:
            # 首次创建action
            storeactions_dict[vehicle_id] = action
            global_store[vehicle_id] = action
            global_store[vehicle_id].dur_reward = 0
            global_store[vehicle_id].current_time = self.current_time
            global_store[vehicle_id].target_location = target_coords
        else:
            # 更新已有action
            storeactions_dict[vehicle_id].next_action = action
            storeactions_dict[vehicle_id].next_action.next_value = 0
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            
            # 替换为新action
            global_store[vehicle_id] = action
            global_store[vehicle_id].dur_reward = 0
            global_store[vehicle_id].dur_time = self.current_time - old_current_time
            global_store[vehicle_id].current_time = self.current_time
            global_store[vehicle_id].target_location = target_coords
    
    def _update_storeaction_ev_rejection(self, vehicle_id, action, target_request, storeactions_dict, vehicle_location, vehicle_battery, target_coords):
        """
        EV拒单后的特殊storeaction更新逻辑，包含penalty_reward计算
        
        Args:
            vehicle_id: 车辆ID
            action: 新的IdleAction对象
            target_request: 被拒绝的请求对象
            storeactions_dict: storeactions_ev字典
            vehicle_location: 车辆当前位置
            vehicle_battery: 车辆当前电量
            target_coords: 目标坐标
        """
        from src.Action import ServiceAction
        
        # 计算penalty reward
        active_requests_count = len(self.active_requests) if hasattr(self, 'active_requests') else 0
        active_requests_value = sum(req.final_value for req in self.active_requests.values()) if hasattr(self, 'active_requests') else 0.0
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 500.0
        penalty_reward = - avg_request_value * 0.1
        quest_num_now = len(self.active_requests)
        
        if self.storeactions_ev[vehicle_id] is None:
            storeactions_dict[vehicle_id] = action
            self.storeactions_ev[vehicle_id] = ServiceAction([], target_request.request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            self.storeactions_ev[vehicle_id].dur_reward = penalty_reward
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
        else:
            storeactions_dict[vehicle_id].next_action = ServiceAction([], target_request.request_id, vehicle_location, vehicle_battery, req_num=quest_num_now)
            storeactions_dict[vehicle_id].next_action.next_value = 0
            storeactions_dict[vehicle_id].vehicle_loc_post = vehicle_location
            storeactions_dict[vehicle_id].vehicle_battery_post = vehicle_battery
            old_current_time = getattr(storeactions_dict[vehicle_id], 'current_time', self.current_time)
            self.storeactions_ev[vehicle_id] = action
            self.storeactions_ev[vehicle_id].dur_reward = 0
            self.storeactions_ev[vehicle_id].dur_time = self.current_time - old_current_time
            self.storeactions_ev[vehicle_id].current_time = self.current_time
            self.storeactions_ev[vehicle_id].target_location = target_coords
    
    def _calculate_dqn_performance_metrics(self, results, current_requests):
        """
        Calculate performance metrics for DQN simulation
        
        Args:
            results: Simulation results dictionary to update
            current_requests: List of current requests
        """
        # Vehicle utilization: vehicles engaged in any activity (assigned, onboard, charging)
        total_vehicles = len(self.vehicles)
        engaged = sum(1 for v in self.vehicles.values() if (
            v.get('assigned_request') is not None or
            v.get('passenger_onboard') is not None or
            v.get('charging_station') is not None
        ))
        results['vehicle_utilization'] = engaged / max(1, total_vehicles)

        # Request completion rate based on environment accounting
        completed = len(self.completed_requests)
        total_requests = completed + len(self.active_requests) + len(self.rejected_requests)
        results['request_completion_rate'] = completed / max(1, total_requests)

        # Average EV battery level
        evs = [v for v in self.vehicles.values() if v.get('type') == 1]
        results['average_battery_level'] = (sum(v.get('battery', 1.0) for v in evs) / len(evs)) if evs else 1.0

        # Action distribution inferred from chosen DQN indices if available
        act_indices = [a.get('dqn_action') for a in results.get('actions_taken', []) if 'dqn_action' in a]
        if act_indices:
            n = len(act_indices)
            results['action_distribution'] = {
                'assign': sum(1 for a in act_indices if a is not None and a < 10) / n,
                'rebalance': sum(1 for a in act_indices if a is not None and 10 <= a < 20) / n,
                'charge': sum(1 for a in act_indices if a is not None and 20 <= a < 25) / n,
                'wait': sum(1 for a in act_indices if a is not None and 25 <= a < 28) / n,
                'idle': sum(1 for a in act_indices if a is not None and a >= 28) / n,
            }
        else:
            results['action_distribution'] = {'assign': 0, 'rebalance': 0, 'charge': 0, 'wait': 0, 'idle': 1}

        # Waiting stats
        waiting = sum(1 for v in self.vehicles.values() if v.get('is_stationary', False) or v.get('waiting_for_requests', False))
        results['vehicles_waiting'] = waiting
        results['wait_utilization'] = waiting / max(1, total_vehicles)

        return results

    def _save_training_dataset(self, value_function):
        """
        保存Q-network训练的experience数据集到本地
        """
        import json
        import pickle
        import os
        from datetime import datetime
        
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            print("⚠️ No experience buffer found or empty buffer")
            return
        
        # 创建保存目录
        save_dir = "results/training_datasets"
        os.makedirs(save_dir, exist_ok=True)
        
        # 生成时间戳文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 转换经验数据为可保存格式
        experiences = list(value_function.experience_buffer)
        dataset = {
            'timestamp': timestamp,
            'current_time': self.current_time,
            'dataset_size': len(experiences),
            'experiences': experiences,
            'environment_info': {
                'grid_size': self.grid_size,
                'num_vehicles': self.NUM_AGENTS,
                'num_charging_stations': len(self.charging_stations) if hasattr(self, 'charging_stations') else 0
            }
        }
        
        # 保存为pickle文件（用于后续训练）
        pickle_file = f"{save_dir}/training_dataset_{timestamp}.pkl"
        with open(pickle_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        # 保存为JSON文件（便于查看和分析）
        json_file = f"{save_dir}/training_dataset_{timestamp}.json"
        # 将numpy类型转换为Python原生类型以便JSON序列化
        def convert_for_json(obj):
            if hasattr(obj, 'item'):  # numpy types
                return obj.item()
            elif isinstance(obj, (list, tuple)):
                return [convert_for_json(x) for x in obj]
            elif isinstance(obj, dict):
                return {k: convert_for_json(v) for k, v in obj.items()}
            else:
                return obj
        
        json_dataset = convert_for_json(dataset)
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(json_dataset, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Training dataset saved:")
        print(f"   📁 Pickle file: {pickle_file}")
        print(f"   📄 JSON file: {json_file}")
        print(f"   📊 Dataset size: {len(experiences)} experiences")
        
    def _analyze_q_value_issues(self, value_function):
        """
        分析为什么接受订单的Q-value比idle还要小的问题
        """
        print("\n🔍 Analyzing Q-value issues (accept vs idle)...")
        
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            print("⚠️ No experience buffer found")
            return
        
        experiences = list(value_function.experience_buffer)
        
        # 分离不同类型的动作
        assign_experiences = [exp for exp in experiences if exp['action_type'].startswith('assign')]
        idle_experiences = [exp for exp in experiences if exp['action_type'] == 'idle']
        charge_experiences = [exp for exp in experiences if exp['action_type'].startswith('charge')]
        
        print(f"📈 Action type distribution:")
        print(f"   Assign actions: {len(assign_experiences)}")
        print(f"   Idle actions: {len(idle_experiences)}")
        print(f"   Charge actions: {len(charge_experiences)}")
        
        if len(assign_experiences) > 0 and len(idle_experiences) > 0:
            # 计算奖励统计
            assign_rewards = [exp['reward'] for exp in assign_experiences]
            idle_rewards = [exp['reward'] for exp in idle_experiences]
            
            assign_mean = sum(assign_rewards) / len(assign_rewards)
            idle_mean = sum(idle_rewards) / len(idle_rewards)
            
            assign_positive = len([r for r in assign_rewards if r > 0])
            idle_positive = len([r for r in idle_rewards if r > 0])
            
            print(f"\n🎯 Reward Analysis:")
            print(f"   Assign - Mean: {assign_mean:.3f}, Positive: {assign_positive}/{len(assign_rewards)} ({assign_positive/len(assign_rewards)*100:.1f}%)")
            print(f"   Idle   - Mean: {idle_mean:.3f}, Positive: {idle_positive}/{len(idle_rewards)} ({idle_positive/len(idle_rewards)*100:.1f}%)")
            
            # 计算当前Q值
            sample_vehicle_id = 0
            sample_location = 50  # 网格中心位置
            sample_time = self.current_time
            
            try:
                # 获取sample state下的Q值
                assign_q = value_function.get_q_value(
                    vehicle_id=sample_vehicle_id,
                    action_type="assign_0",
                    vehicle_location=sample_location,
                    target_location=sample_location + 10,
                    current_time=sample_time,
                    battery_level=0.8,
                    request_value=10.0
                )
                
                idle_q = value_function.get_q_value(
                    vehicle_id=sample_vehicle_id,
                    action_type="idle",
                    vehicle_location=sample_location,
                    target_location=sample_location,
                    current_time=sample_time,
                    battery_level=0.8,
                    request_value=0.0
                )
                
                print(f"\n🧠 Current Q-values (sample state):")
                print(f"   Assign Q-value: {assign_q:.3f}")
                print(f"   Idle Q-value:   {idle_q:.3f}")
                print(f"   Difference:     {assign_q - idle_q:.3f}")
                
                if assign_q < idle_q:
                    print(f"⚠️  ISSUE DETECTED: Assign Q-value is lower than idle!")
                    
                    # 分析可能的原因
                    print(f"\n🔍 Possible causes:")
                    print(f"   1. Assign actions getting more negative rewards: {assign_mean < idle_mean}")
                    print(f"   2. Idle actions more consistently positive: {idle_positive/len(idle_rewards) > assign_positive/len(assign_rewards) if len(assign_rewards) > 0 else False}")
                    print(f"   3. Training imbalance - more negative assign examples")
                    
                    # 分析距离对奖励的影响
                    assign_with_distance = [(exp['reward'], abs(exp['vehicle_location'] - exp['target_location'])) 
                                          for exp in assign_experiences 
                                          if 'vehicle_location' in exp and 'target_location' in exp]
                    
                    if assign_with_distance:
                        avg_distance = sum(d[1] for d in assign_with_distance) / len(assign_with_distance)
                        high_distance_rewards = [r for r, d in assign_with_distance if d > avg_distance]
                        low_distance_rewards = [r for r, d in assign_with_distance if d <= avg_distance]
                        
                        print(f"\n📏 Distance analysis for assign actions:")
                        print(f"   Average distance: {avg_distance:.1f}")
                        if high_distance_rewards:
                            print(f"   High distance rewards: {sum(high_distance_rewards)/len(high_distance_rewards):.3f}")
                        if low_distance_rewards:
                            print(f"   Low distance rewards:  {sum(low_distance_rewards)/len(low_distance_rewards):.3f}")
                
            except Exception as e:
                print(f"❌ Error calculating Q-values: {e}")
        
        else:
            print("⚠️ Not enough data for both assign and idle actions")

    def _quick_q_value_analysis(self, value_function):
        """
        快速Q-value分析 - 每50步运行一次，检查Q-value趋势和矛盾
        """
        if not hasattr(value_function, 'experience_buffer') or len(value_function.experience_buffer) == 0:
            return
        
        experiences = list(value_function.experience_buffer)
        
        # 只分析最近100个experience
        recent_experiences = experiences[-100:] if len(experiences) > 100 else experiences
        
        # 快速统计
        assign_rewards = [exp['reward'] for exp in recent_experiences if exp['action_type'].startswith('assign')]
        idle_rewards = [exp['reward'] for exp in recent_experiences if exp['action_type'] == 'idle']
        
        if len(assign_rewards) > 0 and len(idle_rewards) > 0:
            assign_mean = sum(assign_rewards) / len(assign_rewards)
            idle_mean = sum(idle_rewards) / len(idle_rewards)
            
            print(f"🔍 Quick Q-Value Check (last {len(recent_experiences)} experiences):")
            print(f"   Assign avg reward: {assign_mean:.3f} (n={len(assign_rewards)})")
            print(f"   Idle avg reward:   {idle_mean:.3f} (n={len(idle_rewards)})")
            print(f"   Difference: {assign_mean - idle_mean:.3f}")
            
            # 检查Q值与奖励的矛盾
            if hasattr(value_function, 'get_q_value'):
                try:
                    # 快速获取当前Q值估计 (使用平均状态)
                    avg_location = 160  # 网格中心位置
                    avg_time = self.current_time
                    avg_battery = 0.7
                    other_vehicles = max(0, len([v for v in self.vehicles.values() 
                                               if v.get('assigned_request') is None and 
                                                  v.get('passenger_onboard') is None]) - 1)
                    num_requests = len(self.active_requests)
                    
                    # 获取当前Q值预测
                    assign_q = value_function.get_q_value(
                        vehicle_id=1, action_type='assign_1', 
                        vehicle_location=avg_location, target_location=avg_location,
                        current_time=avg_time, other_vehicles=other_vehicles,
                        num_requests=num_requests, battery_level=avg_battery,
                        request_value=10.0
                    )
                    
                    idle_q = value_function.get_idle_q_value(
                        vehicle_id=1, vehicle_location=avg_location,
                        battery_level=avg_battery, current_time=avg_time,
                        other_vehicles=other_vehicles, num_requests=num_requests
                    )
                    
                    charge_q = value_function.get_charging_q_value(
                        vehicle_id=1, station_id=1,
                        vehicle_location=avg_location, station_location=avg_location,
                        current_time=avg_time, other_vehicles=other_vehicles,
                        num_requests=num_requests, battery_level=avg_battery
                    )
                    
                    print(f"   Current Q-predictions: Assign={assign_q:.3f}, Idle={idle_q:.3f}, Charge={charge_q:.3f}")
                    
                    # 检测矛盾
                    if assign_mean > idle_mean + 5.0 and idle_q > assign_q + 0.5:
                        print(f"🚨 CONTRADICTION DETECTED!")
                        print(f"   Assign rewards ({assign_mean:.1f}) > Idle rewards ({idle_mean:.1f})")
                        print(f"   But Idle Q-value ({idle_q:.3f}) > Assign Q-value ({assign_q:.3f})")
                        print(f"   💡 Possible causes:")
                        print(f"      1. Training hasn't converged yet (need more steps)")
                        print(f"      2. Sample imbalance in training batch")
                        print(f"      3. Network capacity insufficient")
                        print(f"      4. Learning rate too high/low")
                        
                        # 调用详细的矛盾分析
                        self._analyze_q_reward_contradiction(value_function, recent_experiences)
                        
                except Exception as e:
                    print(f"   Q-value prediction error: {e}")
            
            if assign_mean < idle_mean - 0.1:  # 阈值0.1
                print(f"⚠️  WARNING: Assign rewards significantly lower than idle!")
                
                # 导出最近的experience数据为CSV
                self._export_recent_experiences_csv(recent_experiences)
    
    def _export_recent_experiences_csv(self, experiences):
        """
        导出最近的experience数据为CSV文件
        """
        import pandas as pd
        import os
        from datetime import datetime
        
        # 创建导出目录
        export_dir = "results/q_value_analysis"
        os.makedirs(export_dir, exist_ok=True)
        
        # 准备数据
        rows = []
        for i, exp in enumerate(experiences):
            # 计算距离 - 安全处理位置坐标
            v_loc = exp.get('vehicle_location', 0)
            t_loc = exp.get('target_location', 0)
            grid_size = getattr(self, 'grid_size', 10)
            
            # 安全转换位置为整数索引
            def _safe_location_to_int(loc):
                if isinstance(loc, tuple) and len(loc) == 2:
                    # 如果是坐标元组，转换为索引
                    x, y = loc
                    return y * grid_size + x
                elif isinstance(loc, (int, float)):
                    return int(loc)
                else:
                    return 0
            
            v_loc_int = _safe_location_to_int(v_loc)
            t_loc_int = _safe_location_to_int(t_loc)
            
            # 计算坐标
            vx, vy = v_loc_int % grid_size, v_loc_int // grid_size
            tx, ty = t_loc_int % grid_size, t_loc_int // grid_size
            distance = abs(vx - tx) + abs(vy - ty)
            
            # 简化动作类型
            action_type = exp.get('action_type', '')
            if action_type == 'idle':
                action_category = 'idle'
            elif action_type.startswith('assign'):
                action_category = 'assign'
            elif action_type.startswith('charge'):
                action_category = 'charge'
            else:
                action_category = 'other'
            
            row = {
                'exp_id': i,
                'vehicle_id': exp.get('vehicle_id', 0),
                'action_type': action_type,
                'action_category': action_category,
                'vehicle_location': v_loc,
                'target_location': t_loc,
                'distance': distance,
                'battery_level': exp.get('battery_level', 1.0),
                'current_time': exp.get('current_time', 0.0),
                'reward': exp.get('reward', 0.0),
                'next_battery_level': exp.get('next_battery_level', 1.0),
                'num_requests': exp.get('num_requests', 0),
                'request_value': exp.get('request_value', 0.0),
                'is_rejection': exp.get('is_rejection', False)
            }
            rows.append(row)
        
        # 创建DataFrame并保存
        df = pd.DataFrame(rows)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_file = os.path.join(export_dir, f"recent_experiences_{timestamp}.csv")
        df.to_csv(csv_file, index=False, encoding='utf-8')
        
        print(f"📊 Recent experiences exported to: {csv_file}")
        
        # 显示基础统计
        print(f"📈 Quick Statistics:")
        reward_by_action = df.groupby('action_category')['reward'].agg(['count', 'mean', 'std']).round(3)
        print(reward_by_action)
    
    def _analyze_q_reward_contradiction(self, value_function, experiences):
        """
        深入分析Q值与奖励矛盾的详细原因
        """
        print(f"\n🔬 详细矛盾分析:")
        print("=" * 50)
        
        try:
            # 1. 分析training step和buffer状态
            training_step = getattr(value_function, 'training_step', 0)
            buffer_size = len(value_function.experience_buffer) if hasattr(value_function, 'experience_buffer') else 0
            
            print(f"📊 训练状态:")
            print(f"   训练步数: {training_step}")
            print(f"   缓冲区大小: {buffer_size}")
            print(f"   分析样本数: {len(experiences)}")
            
            # 2. 分析最近的训练批次构成
            if hasattr(value_function, '_action_balanced_sample'):
                try:
                    sample_batch = value_function._action_balanced_sample(64)
                    batch_assign = len([exp for exp in sample_batch if exp['action_type'].startswith('assign')])
                    batch_idle = len([exp for exp in sample_batch if exp['action_type'] == 'idle'])
                    batch_charge = len([exp for exp in sample_batch if exp['action_type'].startswith('charge')])
                    
                    print(f"🎲 最近训练批次构成:")
                    print(f"   Assign: {batch_assign}/64 ({batch_assign/64:.1%})")
                    print(f"   Idle: {batch_idle}/64 ({batch_idle/64:.1%})")
                    print(f"   Charge: {batch_charge}/64 ({batch_charge/64:.1%})")
                    
                    # 检查是否过度倾向于idle
                    if batch_idle > batch_assign * 2:
                        print(f"   ⚠️  Idle样本过多，可能影响学习")
                        
                except Exception as e:
                    print(f"   采样分析失败: {e}")
            
            # 3. 分析奖励分布的细节
            assign_rewards = [exp['reward'] for exp in experiences if exp['action_type'].startswith('assign')]
            idle_rewards = [exp['reward'] for exp in experiences if exp['action_type'] == 'idle']
            
            if assign_rewards and idle_rewards:
                import numpy as np
                
                # 正负奖励分布
                assign_pos = len([r for r in assign_rewards if r > 0])
                assign_neg = len([r for r in assign_rewards if r <= 0])
                idle_pos = len([r for r in idle_rewards if r > 0])
                idle_neg = len([r for r in idle_rewards if r <= 0])
                
                print(f"\n🎯 奖励分布分析:")
                print(f"   Assign: {assign_pos} 正奖励, {assign_neg} 负/零奖励")
                print(f"   Idle: {idle_pos} 正奖励, {idle_neg} 负/零奖励")
                
                # 奖励量级分析
                if assign_rewards:
                    print(f"   Assign奖励范围: [{np.min(assign_rewards):.1f}, {np.max(assign_rewards):.1f}]")
                if idle_rewards:
                    print(f"   Idle奖励范围: [{np.min(idle_rewards):.1f}, {np.max(idle_rewards):.1f}]")
                
                # 检查奖励scale问题
                assign_scale = np.std(assign_rewards) if len(assign_rewards) > 1 else 0
                idle_scale = np.std(idle_rewards) if len(idle_rewards) > 1 else 0
                print(f"   奖励变异性: Assign std={assign_scale:.2f}, Idle std={idle_scale:.2f}")
                
                if assign_scale > idle_scale * 3:
                    print(f"   ⚠️  Assign奖励变异性过大，可能影响学习稳定性")
            
            # 4. 提供具体的改进建议
            print(f"\n💡 改进建议:")
            
            if training_step < 1000:
                print(f"   🔄 训练步数较少({training_step})，建议继续训练至少2000步")
                
            if buffer_size < 5000:
                print(f"   📊 缓冲区数据较少({buffer_size})，建议积累更多经验")
                
            # 检查学习率
            if hasattr(value_function, 'optimizer'):
                current_lr = value_function.optimizer.param_groups[0]['lr']
                print(f"   📈 当前学习率: {current_lr:.6f}")
                if current_lr > 0.01:
                    print(f"      建议降低学习率到 0.001-0.005 范围")
                elif current_lr < 0.0001:
                    print(f"      学习率可能过低，建议提高到 0.0005-0.001")
            
            print(f"   🎯 建议启用更强的assign奖励bonus")
            print(f"   🔧 考虑调整action-balanced采样比例 (增加assign权重)")
            print(f"   📚 使用prioritized experience replay优先训练高价值样本")
            
        except Exception as e:
            print(f"❌ 矛盾分析失败: {e}")
        
        return csv_file

    def _count_idle_vehicles(self):
        """
        统计当前处于idle状态的车辆数量
        Idle车辆定义：没有分配请求、没有乘客在车、没有在充电站充电
        """
        idle_count = 0
        for vehicle_id, vehicle in self.vehicles.items():
            is_idle = (
                vehicle.get('assigned_request') is None and
                vehicle.get('passenger_onboard') is None and
                vehicle.get('charging_station') is None and
                vehicle.get('battery_level', 1.0) > self.min_battery_level
            )
            if is_idle:
                idle_count += 1
        
        return idle_count
    
    def _get_idle_action_index(self):
        """
        获取idle动作对应的DQN动作索引
        这需要根据DQN动作空间的定义来确定
        假设idle动作是最后一个动作（索引为动作空间大小-1）
        """
        # 根据DQN动作空间定义，通常idle/wait动作在末尾
        # 这里假设动作空间大小为32，idle动作索引为31
        # 实际使用时需要根据具体的DQN实现调整
        return 31  # 或者根据实际的DQN动作定义返回正确的索引

    def _count_idle_vehicles(self):
        """
        统计当前处于idle状态的车辆数量
        Idle车辆定义：没有分配请求、没有乘客在车、没有在充电站充电
        """
        idle_count = 0
        for vehicle_id, vehicle in self.vehicles.items():
            is_idle = (
                vehicle.get('assigned_request') is None and
                vehicle.get('passenger_onboard') is None and
                vehicle.get('charging_station') is None and
                vehicle.get('battery_level', 1.0) > self.min_battery_level
            )
            if is_idle:
                idle_count += 1
        
        return idle_count



