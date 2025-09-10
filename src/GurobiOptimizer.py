from typing import List, Dict
from .Request import Request
import random
import gurobipy as gp
class GurobiOptimizer:
    """Gurobi-based optimization for vehicle assignment and rebalancing"""
    
    def __init__(self, env):
        self.env = env
        # Only import Gurobi if it's available
        try:
            import gurobipy as gp
            from gurobipy import GRB
            self.gp = gp
            self.GRB = GRB
            self.available = True
            print("✓ Gurobi optimizer available")
        except ImportError:
            print("⚠ Gurobi not available, using heuristic methods")
            self.available = False
    
    
    
    
    
    
    def optimize_vehicle_rebalancing(self, vehicle_ids):
        """Optimize vehicle rebalancing using Gurobi or heuristic methods"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        try:
            return self._gurobi_vehicle_rebalancing(vehicle_ids)
        except Exception as e:
            print(f"Gurobi rebalancing failed: {e}, using heuristic")
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        
        
    def optimize_vehicle_rebalancing_reject(self, vehicle_ids):
        """Optimize vehicle rebalancing using Gurobi or heuristic methods with reject consideration"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids)
        
        # Get available requests from environment
        available_requests = []
        if hasattr(self.env, 'active_requests') and self.env.active_requests:
            available_requests = list(self.env.active_requests.values())
        
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        try:
            return self._gurobi_vehicle_rebalancing_knownreject(vehicle_ids, available_requests, charging_stations)
        except Exception as e:
            print(f"Gurobi rebalancing with reject failed: {e}, using heuristic")
            return self._heuristic_rebalancing_assignment(vehicle_ids)

    def _gurobi_vehicle_rebalancing(self, vehicle_ids):
        """Use Gurobi optimization to assign vehicles to available requests"""
        assignments = {}

        if not hasattr(self.env, 'active_requests') or not self.env.active_requests:
            return assignments

        # Convert active requests to list
        available_requests = list(self.env.active_requests.values())
        request_count = len(available_requests)
        # Get available charging stations
        charging_stations = []
        if hasattr(self.env, 'charging_manager') and self.env.charging_manager.stations:
            charging_stations = [station for station in self.env.charging_manager.stations.values() 
                               if station.available_slots > 0]
        
        # Return empty if no requests and no need for charging rebalancing
        if not available_requests and not charging_stations:
            return assignments

        # Create Gurobi model for vehicle-to-request assignment
        model = self.gp.Model("vehicle_rebalancing")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', 30)  # Set time limit

        # Decision variables: x[i,j] = 1 if vehicle i is assigned to request j
        request_decision = {}

        charge_decision = {}
        idle_vehicle = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, request in enumerate(available_requests):
                request_decision[i, j] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_id}_request_{request.request_id}')

        for i, vehicle_id in enumerate(vehicle_ids):
            for j, station in enumerate(charging_stations):
                charge_decision[i, j] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_id}_charge_{station.id}')
        
        # Constraint 1: Each vehicle can be assigned to at most one request

        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_ids[i]}_idle')
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            for j in range(len(available_requests)):
                actionv += request_decision[i, j]
            for j in range(len(charging_stations)):
                actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv == 1) 
        idlevehicle = self.gp.LinExpr()
        for i in range(len(vehicle_ids)):
            idlevehicle += idle_vehicle[i]
        model.addConstr(idlevehicle >= getattr(self.env, 'idle_vehicle_requirement', 0)) 
        servedrequest = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                servedrequest += request_decision[i, j]     
        # Constraint 2: Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            model.addConstr(self.gp.quicksum(request_decision[i, j] for i in range(len(vehicle_ids))) <= 1)

        # Objective: Maximize total value (considering distance and Q-value)
        objective_terms  = self.gp.LinExpr()
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            vehicle_pos = vehicle['coordinates']

            # Process request assignments
            for j, request in enumerate(available_requests):
                # Get Q-value benefit for request assignment
                q_value = 0
                if hasattr(self.env, 'get_assignment_q_value'):
                    q_value = self.env.get_assignment_q_value(vehicle_id, request.request_id,
                                                        vehicle['location'], request.pickup)

                # Combined objective: request value + Q-value benefit
                total_value = request.value + q_value * self.env.adp_value
                objective_terms += total_value * request_decision[i, j]
            
            # Process charging assignments  
            for j, station in enumerate(charging_stations):
                # Get Q-value benefit for charging decision
                charging_q_value = 0
                if hasattr(self.env, 'get_charging_q_value'):
                    charging_q_value = self.env.get_charging_q_value(vehicle_id, station.id,
                                                                   vehicle['location'], 
                                                                   station.location)
                elif hasattr(self.env, 'get_assignment_q_value'):
                    # Use assignment Q-value as fallback for charging
                    charging_q_value = self.env.get_assignment_q_value(vehicle_id, f"charge_{station.id}",
                                                                     vehicle['location'], 
                                                                     station.location)
                
                # Charging penalty offset by Q-value benefit
                charging_penalty = getattr(self.env, 'charging_penalty', 2.0)
                charging_value = -charging_penalty + charging_q_value * self.env.adp_value
                objective_terms += charging_value * charge_decision[i, j]
        objective_terms -= getattr(self.env, 'unserved_penalty', 1.5) * (request_count - servedrequest)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)

        # Solve the optimization problem
        model.optimize()

        # Extract assignments
        if model.status == self.GRB.OPTIMAL:
            for i, vehicle_id in enumerate(vehicle_ids):
                # Check request assignments
                for j, request in enumerate(available_requests):
                    if request_decision[i, j].x > 0.5:  # Binary variable threshold
                        assignments[vehicle_id] = request
                        break
                
                # Check charging assignments if no request assigned
                if vehicle_id not in assignments:
                    for j, station in enumerate(charging_stations):
                        if charge_decision[i, j].x > 0.5:  # Binary variable threshold
                            assignments[vehicle_id] = f"charge_{station.id}"
                            break

        return assignments
    
    
    
    
    def optimize_vehicle_assignment(self, requests, vehicles):
        """Optimize assignment of vehicles to requests using Gurobi"""
        if not self.available or not requests:
            return self._heuristic_order_assignment(requests, vehicles)
        
        try:
            # Create optimization model
            model = self.gp.Model("vehicle_assignment")
            model.setParam('OutputFlag', 0)  # Suppress output
            
            # Decision variables: x[i,j] = 1 if vehicle i is assigned to request j
            x = {}
            for i, vehicle_id in enumerate(vehicles):
                for j, request in enumerate(requests):
                    x[i, j] = model.addVar(vtype=self.GRB.BINARY, 
                                         name=f'assign_{vehicle_id}_{request.request_id}')
            
            # Constraints: Each request can be assigned to at most one vehicle
            for j in range(len(requests)):
                model.addConstr(self.gp.quicksum(x[i, j] for i in range(len(vehicles))) <= 1)
            
            # Constraints: Each vehicle can be assigned to at most one request
            for i in range(len(vehicles)):
                model.addConstr(self.gp.quicksum(x[i, j] for j in range(len(requests))) <= 1)
            
            # Objective: Minimize total distance + maximize total value
            obj = self.gp.quicksum(
                x[i, j] * (requests[j].value - self._calculate_distance(vehicles[i], requests[j]))
                for i in range(len(vehicles))
                for j in range(len(requests))
            )
            model.setObjective(obj, self.GRB.MAXIMIZE)
            
            # Solve
            model.optimize()
            
            # Extract solution
            assignments = {}
            if model.status == self.GRB.OPTIMAL:
                for i, vehicle_id in enumerate(vehicles):
                    for j, request in enumerate(requests):
                        if x[i, j].x > 0.5:  # Binary variable is 1
                            assignments[vehicle_id] = request.request_id
            
            return assignments
            
        except Exception as e:
            print(f"Gurobi optimization failed: {e}, using heuristic")
            return self._heuristic_order_assignment(requests, vehicles)
    
    
    
    
    def _calculate_distance(self, vehicle_id, request):
        """Calculate distance from vehicle to request pickup"""
        vehicle = self.env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
        return abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
    
    
    
    
    def _heuristic_rebalancing_assignment(self, vehicle_ids):
        """Advanced heuristic assignment for vehicle rebalancing when Gurobi is not available"""
        assignments = {}
        
        if not hasattr(self.env, 'active_requests') or not self.env.active_requests:
            return assignments

        available_requests = list(self.env.active_requests.values())
        if not available_requests:
            return assignments

        available_vehicles = set(vehicle_ids)

        # Strategy 1: Prioritize high-value requests with deadline urgency
        # Calculate request priorities based on value, urgency, and vehicle compatibility
        request_priorities = []
        for request in available_requests:
            # Calculate urgency factor (requests closer to deadline are more urgent)
            time_remaining = max(1, request.pickup_deadline - self.env.current_time)
            urgency_factor = 1.0 / time_remaining
            
            # Combined priority: value + urgency
            priority = request.value * 0.7 + urgency_factor * 0.3
            request_priorities.append((priority, request))

        # Sort requests by priority (highest first)
        request_priorities.sort(key=lambda x: x[0], reverse=True)

        # Strategy 2: Match vehicles optimally considering multiple factors
        for priority, request in request_priorities:
            if not available_vehicles:
                break

            best_vehicle = None
            best_score = float('-inf')

            pickup_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)

            for vehicle_id in available_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_pos = vehicle['coordinates']
                
                # Calculate distance
                distance = abs(vehicle_pos[0] - pickup_pos[0]) + abs(vehicle_pos[1] - pickup_pos[1])
                
                # Distance score (closer is better)
                distance_score = 1.0 / (1.0 + distance)
                
                # Battery level score (higher battery is better for service)
                battery_score = vehicle['battery']
                
                # Vehicle type compatibility score
                type_score = 1.0
                if vehicle['type'] == 'AEV':
                    type_score = 1.2  # AEV vehicles are preferred for service
                
                # Combined score: distance + battery + type
                total_score = (distance_score * 0.5 + 
                             battery_score * 0.3 + 
                             type_score * 0.2)
                
                if total_score > best_score:
                    best_score = total_score
                    best_vehicle = vehicle_id

            if best_vehicle:
                assignments[best_vehicle] = request
                available_vehicles.remove(best_vehicle)

        return assignments

    def _heuristic_order_assignment(self, requests, vehicles):
        """Enhanced heuristic assignment for order processing"""
        assignments = {}
        available_vehicles = set(vehicles)
        
        # Strategy: Multi-criteria optimization for order assignment
        # Calculate vehicle capabilities and request requirements
        vehicle_scores = {}
        for vehicle_id in vehicles:
            vehicle = self.env.vehicles[vehicle_id]
            # Base score considers battery level and vehicle type
            base_score = vehicle['battery'] * 0.6
            if vehicle['type'] == 'AEV':
                base_score += 0.3  # AEV bonus for reliability
            vehicle_scores[vehicle_id] = base_score

        # Sort requests by combined value and urgency
        enhanced_requests = []
        for request in requests:
            # Calculate time pressure
            if hasattr(request, 'pickup_deadline'):
                time_remaining = max(1, request.pickup_deadline - self.env.current_time)
                urgency = 1.0 / time_remaining
            else:
                urgency = 0.5  # Default urgency
            
            # Combined priority
            priority = request.value * 0.8 + urgency * 0.2
            enhanced_requests.append((priority, request))

        enhanced_requests.sort(key=lambda x: x[0], reverse=True)

        # Assign vehicles to requests using enhanced scoring
        for priority, request in enhanced_requests:
            if not available_vehicles:
                break

            best_vehicle = None
            best_combined_score = float('-inf')

            pickup_coords = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)

            for vehicle_id in available_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                
                # Distance factor
                distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                distance_factor = 1.0 / (1.0 + distance * 0.1)
                
                # Combine vehicle capability with distance efficiency
                combined_score = vehicle_scores[vehicle_id] * 0.6 + distance_factor * 0.4
                
                if combined_score > best_combined_score:
                    best_combined_score = combined_score
                    best_vehicle = vehicle_id

            if best_vehicle:
                assignments[best_vehicle] = request.request_id
                available_vehicles.remove(best_vehicle)

        return assignments

    def _heuristic_charge_assignment(self, requests, vehicles):
        """Enhanced heuristic assignment for charging optimization"""
        assignments = {}
        
        # Get vehicle data and sort by charging priority
        vehicle_data = {}
        for vehicle_id in vehicles:
            vehicle_data[vehicle_id] = self.env.vehicles[vehicle_id]
        
        # Strategy: Prioritize vehicles by charging need and strategic positioning
        charging_priorities = []
        for vehicle_id, vehicle in vehicle_data.items():
            # Calculate charging urgency (lower battery = higher urgency)
            battery_urgency = 1.0 - vehicle['battery']
            
            # Calculate strategic value (position relative to high-demand areas)
            strategic_value = 0.5  # Base strategic value
            
            # If we have active requests, consider proximity to demand
            if hasattr(self.env, 'active_requests') and self.env.active_requests:
                total_distance_to_requests = 0
                num_requests = len(self.env.active_requests)
                
                for request in self.env.active_requests.values():
                    pickup_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
                    vehicle_pos = vehicle['coordinates']
                    distance = abs(vehicle_pos[0] - pickup_pos[0]) + abs(vehicle_pos[1] - pickup_pos[1])
                    total_distance_to_requests += distance
                
                # Lower average distance = higher strategic value
                avg_distance = total_distance_to_requests / num_requests if num_requests > 0 else 10
                strategic_value = 1.0 / (1.0 + avg_distance * 0.1)
            
            # Combined priority: urgency + strategic positioning
            priority = battery_urgency * 0.7 + strategic_value * 0.3
            charging_priorities.append((priority, vehicle_id))
        
        # Sort by priority (highest first)
        charging_priorities.sort(key=lambda x: x[0], reverse=True)
        
        available_vehicles = set(vehicles)
        
        # Sort requests by value and accessibility
        enhanced_requests = []
        for request in requests:
            # For charging requests, prioritize based on value and station availability
            station_accessibility = 1.0  # Default accessibility
            
            # If request is related to charging stations, consider station load
            if hasattr(self.env, 'charging_manager'):
                # Calculate average distance to available charging stations
                available_stations = [s for s in self.env.charging_manager.stations.values() 
                                    if s.current_capacity < s.max_capacity]
                if available_stations:
                    min_station_distance = float('inf')
                    request_pos = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
                    
                    for station in available_stations:
                        station_distance = abs(station.location[0] - request_pos[0]) + abs(station.location[1] - request_pos[1])
                        min_station_distance = min(min_station_distance, station_distance)
                    
                    station_accessibility = 1.0 / (1.0 + min_station_distance * 0.1)
            
            enhanced_value = request.value * station_accessibility
            enhanced_requests.append((enhanced_value, request))
        
        enhanced_requests.sort(key=lambda x: x[0], reverse=True)
        
        # Assign prioritized vehicles to enhanced requests
        for enhanced_value, request in enhanced_requests:
            if not available_vehicles:
                break
            
            # Find the best vehicle from our priority list
            best_vehicle = None
            
            for priority, vehicle_id in charging_priorities:
                if vehicle_id in available_vehicles:
                    # Check if this vehicle can handle the request efficiently
                    vehicle = vehicle_data[vehicle_id]
                    distance = self._calculate_distance(vehicle_id, request)
                    
                    # Only assign if vehicle has sufficient battery or is close
                    if vehicle['battery'] > 0.2 or distance <= 3:
                        best_vehicle = vehicle_id
                        break
            
            if best_vehicle:
                assignments[best_vehicle] = request.request_id
                available_vehicles.remove(best_vehicle)
                # Remove assigned vehicle from priority list
                charging_priorities = [(p, v) for p, v in charging_priorities if v != best_vehicle]
        
        return assignments

    def _gurobi_vehicle_rebalancing_knownreject(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        if not self.available:
            return {}
        
        assignments = {}
        
        if not available_requests and not charging_stations:
            return assignments
        
        try:
            # Create optimization model
            model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
            model.setParam('OutputFlag', 0)  # Suppress output
            
            # Parameters
            min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
            charging_rate = 0.15     # Battery increase per charging period
            travel_consumption = 0.02 # Battery consumption per unit distance
            service_consumption = 0.05 # Battery consumption per service
            
            # Filter out rejected requests for each EV
            valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid
            
            for i, vehicle_id in enumerate(vehicle_ids):
                vehicle = self.env.vehicles[vehicle_id]
                for j, request in enumerate(available_requests):
                    # Check if EV would reject this request
                    if vehicle['type'] == 'EV':
                        # Calculate rejection probability
                        rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                        # If rejection probability is high (>50%), don't allow assignment
                        valid_assignments[(i, j)] = rejection_prob < 0.5
                    else:
                        # AEV never rejects
                        valid_assignments[(i, j)] = True
            
            # Decision variables for request assignments
            request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]
            
            # Constraint invalid assignments to 0
            for i in range(len(vehicle_ids)):
                for j in range(len(available_requests)):
                    if not valid_assignments.get((i, j), False):
                        model.addConstr(request_decision[i][j] == 0)
            
            
            # Decision variables for charging assignments
            charge_decision = {}
            if charging_stations:
                for i, vehicle_id in enumerate(vehicle_ids):
                    for j, station in enumerate(charging_stations):
                        charge_decision[i, j] = model.addVar(
                            vtype=self.GRB.BINARY,
                            name=f'charge_{vehicle_id}_{station.id}'
                        )
            
            # Battery level variables (t-1 and t)
            battery_t_minus_1 = {}  # Battery level at t-1 (current)
            battery_t = {}          # Battery level at t (after actions)
            
            for i, vehicle_id in enumerate(vehicle_ids):
                vehicle = self.env.vehicles[vehicle_id]
                
                # t-1 battery level (current battery level)
                battery_t_minus_1[i] = vehicle['battery']
                
                # t battery level (decision variable)
                battery_t[i] = model.addVar(
                    vtype=self.GRB.CONTINUOUS,
                    lb=0,  # Minimum battery constraint
                    ub=1.0,                # Maximum battery is 100%
                    name=f'battery_t_{vehicle_id}'
                )
            
            
            
            # for i in range(len(vehicle_ids)):
            #     model.addConstr(battery_t[i] >= min_battery_level)
            # Idle vehicle variables
            idle_vehicle = {}
            for i in range(len(vehicle_ids)):
                idle_vehicle[i] = model.addVar(
                    vtype=self.GRB.BINARY,
                    name=f'vehicle_{vehicle_ids[i]}_idle'
                )
            
            # Battery level transition constraints (t-1 to t relationship)
            for i, vehicle_id in enumerate(vehicle_ids):
                vehicle = self.env.vehicles[vehicle_id]
                battery_change = self.gp.LinExpr()
                
                # Base battery consumption for staying idle
                base_consumption = 0.01  # 1% per time step
                battery_change += -base_consumption
                
                # Battery gain from charging
                if charging_stations:
                    for j, station in enumerate(charging_stations):
                        battery_change += self.env.chargeincrease_per_epoch * charge_decision[i, j]
                
                
                model.addConstr(battery_t[i] == battery_t_minus_1[i] + battery_change)
                # model.addConstr(battery_t[i] >= self.env.min_battery_level)

            
            
            # Constraint 1: Each vehicle can only take one action
            for i in range(len(vehicle_ids)):
                actionv = self.gp.LinExpr()
                # Add valid request assignments
                for j in range(len(available_requests)):
                    actionv += request_decision[i][j]
                # Add charging assignments
                if charging_stations:
                    for j in range(len(charging_stations)):
                        actionv += charge_decision[i, j]
                model.addConstr(actionv <= 1)
                model.addConstr(idle_vehicle[i] + actionv == 1)
            
            # Minimum idle vehicles constraint
            idle_vehicles = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                idle_vehicles += idle_vehicle[i]
            #model.addConstr(idle_vehicles >= self.env.idle_vehicle_requirement)
            
            # Constraint 2: Each request can be assigned to at most one vehicle
            for j in range(len(available_requests)):
                valid_vehicles = self.gp.LinExpr()
                for i in range(len(vehicle_ids)):
                    valid_vehicles += request_decision[i][j]
                model.addConstr(valid_vehicles <= 1)
            
            # Constraint 3: Each charging station capacity
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    model.addConstr(
                        self.gp.quicksum(charge_decision[i, j] for i in range(len(vehicle_ids))) 
                        <= station.max_capacity
                    )
            
            # Objective: Maximize total value considering Q-values
            objective_terms = self.gp.LinExpr()
            
            for i, vehicle_id in enumerate(vehicle_ids):
                vehicle = self.env.vehicles[vehicle_id]
                
                # Process valid request assignments
                for j, request in enumerate(available_requests):
                    if (i, j) in request_decision:
                        # Get Q-value benefit for request assignment
                        q_value = 0
                        if hasattr(self.env, 'get_assignment_q_value'):
                            q_value = self.env.get_assignment_q_value(
                                vehicle_id, request.request_id,
                                vehicle['location'], request.pickup
                            )
                        
                        # Combined objective: request value + Q-value benefit
                        total_value = request.value + q_value * getattr(self.env, 'adp_value', 1.0)
                        objective_terms += total_value * request_decision[i, j]
                
                # Process charging assignments
                if charging_stations:
                    for j, station in enumerate(charging_stations):
                        # Get Q-value benefit for charging decision
                        charging_q_value = 0
                        if hasattr(self.env, 'get_charging_q_value'):
                            charging_q_value = self.env.get_charging_q_value(
                                vehicle_id, station.id,
                                vehicle['location'], station.location
                            )
                        elif hasattr(self.env, 'get_assignment_q_value'):
                            charging_q_value = self.env.get_assignment_q_value(
                                vehicle_id, f"charge_{station.id}",
                                vehicle['location'], station.location
                            )
                        
                        # Charging penalty offset by Q-value benefit
                        charging_penalty = getattr(self.env, 'charging_penalty', 2.0)
                        charging_value = -charging_penalty + charging_q_value * getattr(self.env, 'adp_value', 1.0)
                        objective_terms += charging_value * charge_decision[i, j]
            

            
            # Add penalty for unserved requests (considering reject behavior)
            served_requests = self.gp.LinExpr()
            for j in range(len(available_requests)):
                for i in range(len(vehicle_ids)):
                    served_requests += request_decision[i][j]
            
            for i in range(len(vehicle_ids)):
                # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
                vehicle_id = vehicle_ids[i]
                vehicle = self.env.vehicles[vehicle_id]
                
                # 获取神经网络预测的idle Q值
                idle_q_value = 0
                if hasattr(self.env, 'get_idle_q_value'):
                    try:
                        idle_q_value = self.env.get_idle_q_value(
                            vehicle_id=vehicle_id,
                            vehicle_location=vehicle['location'],
                            battery_level=vehicle['battery'],
                            current_time=getattr(self.env, 'current_time', 0.0),
                            other_vehicles=len(vehicle_ids) - 1,  # 其他车辆数量
                            num_requests=len(available_requests)
                        )
                    except Exception as e:
                        print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_id}: {e}")
                        # 使用默认的idle奖励作为后备
                        idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
                else:
                    # 如果没有神经网络方法，使用默认奖励
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
                
                # 将神经网络预测的Q值加权到目标函数中
                adp_weight = getattr(self.env, 'adp_value', 1.0)
                objective_terms += idle_q_value * adp_weight * idle_vehicle[i]


            # Penalty for unserved requests
            unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
            objective_terms -= unserved_penalty * (len(available_requests) - served_requests)
            
            model.setObjective(objective_terms, self.GRB.MAXIMIZE)
            
            # Solve the optimization problem
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                # Print battery level optimization results for debugging

                
                for i, vehicle_id in enumerate(vehicle_ids):
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i][j].x > 0.5:
                            assignments[vehicle_id] = request
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                break
                                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                # Print constraint violations for debugging
                if model.status == self.GRB.INFEASIBLE:
                    print("Model is infeasible. Computing IIS...")
                    model.computeIIS()
                    print("Infeasible constraints:")
                    for c in model.getConstrs():
                        if c.IISConstr:
                            print(f"  {c.constrName}")
        
        except Exception as e:
            print(f"Gurobi optimization with reject and charging levels failed: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to heuristic with reject consideration
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return assignments
    
    def _heuristic_assignment_with_reject(self, vehicle_ids, available_requests, charging_stations=None):

        assignments = {}
        
        if not vehicle_ids:
            return assignments
        
        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['battery'] < 0.5:
                low_battery_vehicles.append(vehicle_id)
            else:
                high_battery_vehicles.append(vehicle_id)
        
        # 第二步：为低电量车辆分配充电站
        if charging_stations and low_battery_vehicles:
            for vehicle_id in low_battery_vehicles:
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                
                best_station = None
                best_distance = float('inf')
                
                # 找到最近的有容量的充电站
                for station in charging_stations:
                    if len(station.current_vehicles) < station.max_capacity:
                        station_coords = (
                            station.location // self.env.grid_size,
                            station.location % self.env.grid_size
                        )
                        distance = abs(vehicle_coords[0] - station_coords[0]) + \
                                  abs(vehicle_coords[1] - station_coords[1])
                        
                        if distance < best_distance:
                            best_distance = distance
                            best_station = station
                
                if best_station:
                    assignments[vehicle_id] = f"charge_{best_station.id}"
        
        # 第三步：按电池容量从高到低排序剩余车辆
        high_battery_vehicles.sort(
            key=lambda v_id: self.env.vehicles[v_id]['battery'], 
            reverse=True
        )
        
        # 第四步：为高电量车辆分配订单（考虑EV拒绝率）
        if available_requests and high_battery_vehicles:
            remaining_requests = list(available_requests)
            
            for vehicle_id in high_battery_vehicles:
                if vehicle_id in assignments:  # 已被分配充电
                    continue
                
                vehicle = self.env.vehicles[vehicle_id]
                vehicle_coords = vehicle['coordinates']
                battery_level = vehicle['battery']
                
                best_request = None
                best_score = -float('inf')
                
                # 为该车辆寻找最佳订单
                for request in remaining_requests[:]:  # 使用副本避免修改原列表
                    # 检查EV拒绝率
                    if vehicle['type'] == 'EV':
                        rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                        if rejection_prob >= 0.5:  # 高拒绝概率，跳过
                            continue
                    
                    # 计算距离
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )
                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                              abs(vehicle_coords[1] - pickup_coords[1])
                    
                    # 距离过远则跳过
                    if distance > 8:  # 最大服务距离
                        continue
                    
                    # 计算分配评分
                    if battery_level > 0.7:
                        # 高电量：优先选择远距离高价值订单
                        score = distance * 0.7 + request.value * 0.3
                    elif battery_level > 0.5:
                        # 中等电量：平衡距离和价值
                        score = request.value * 0.5 - distance * 0.5
                    else:
                        # 低电量：优先近距离订单
                        score = -distance
                    
                    if score > best_score:
                        best_score = score
                        best_request = request
                
                # 分配最佳订单
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)
        
        return assignments



