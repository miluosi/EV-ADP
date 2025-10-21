from typing import List, Dict
from .Request import Request
import random
import numpy as np
import gurobipy as gp

from src.ValueFunction_pytorch import PyTorchChargingValueFunction



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
        waiting_vehicle = {}
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
            waiting_vehicle[i] = model.addVar(vtype=self.GRB.BINARY,
                                     name=f'vehicle_{vehicle_ids[i]}_waiting')
        for i in range(len(vehicle_ids)):
            actionv = self.gp.LinExpr()
            for j in range(len(available_requests)):
                actionv += request_decision[i, j]
            for j in range(len(charging_stations)):
                actionv += charge_decision[i, j]
            model.addConstr(actionv <= 1)
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1) 
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

        # Objective: Maximize blended objective
        # - If adp_weight == 0: use immediate rewards (movement cost + request value / charging penalty)
        # - Else: use option-completion Q-values scaled by adp_weight (to avoid double-counting immediate rewards)
        objective_terms  = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]

            # Process request assignments using option value
            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # Immediate reward fallback
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._manhattan_loc(request.pickup, request.dropoff)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * (d1 + d2)
                    immediate = req_val 
                    objective_terms += immediate * request_decision[i, j]
                else:
                    option_q = 0.0
                    if hasattr(self.env, 'evaluate_service_option'):
                        try:
                            option_q = self.env.evaluate_service_option(vehicle_id, request)
                        except Exception:
                            option_q = 0.0
                    objective_terms += option_q * adp_weight * request_decision[i, j]
            
            # Process charging assignments using option value
            for j, station in enumerate(charging_stations):
                if adp_weight <= 0:
                    # Immediate charging cost fallback
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * d_travel
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    objective_terms += immediate * charge_decision[i, j]
                else:
                    charging_q = 0.0
                    if hasattr(self.env, 'evaluate_charging_option'):
                        try:
                            charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                        except Exception:
                            charging_q = 0.0
                    objective_terms += charging_q * adp_weight * charge_decision[i, j]
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
    
    def _manhattan_loc(self, a_loc: int, b_loc: int) -> int:
        """Manhattan distance between two location indices (grid flattened)."""
        gx = self.env.grid_size
        ax, ay = a_loc % gx, a_loc // gx
        bx, by = b_loc % gx, b_loc // gx
        return abs(ax - bx) + abs(ay - by)

    
    
    
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
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service
        request_decision =[[model.addVar(vtype=self.GRB.BINARY,
                     name=f'request_{vehicle_id}_{request.request_id}') for request in available_requests] for i, vehicle_id in enumerate(vehicle_ids)]
            

        # valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid
        # for i, vehicle_id in enumerate(vehicle_ids):
        #     vehicle = self.env.vehicles[vehicle_id]
        #     for j, request in enumerate(available_requests):
        #         # Check if EV would reject this request
        #         if vehicle['type'] == 1:
        #             # Calculate rejection probability
        #             rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
        #             valid_assignments[(i, j)] = rejection_prob < 0.5
        #         else:
        #             # AEV never rejects
        #             valid_assignments[(i, j)] = True
        # for i in range(len(vehicle_ids)):
        #     for j in range(len(available_requests)):
        #         if not valid_assignments.get((i, j), False):
        #             model.addConstr(request_decision[i][j] == 0)



    # Decision variables for request assignments
        
        # Constraint invalid assignments to 0

            
            
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

                name=f'battery_t_{vehicle_id}'
            )
        
        

        idle_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )
        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            
            # Battery consumption from charging (travel to station)
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    station_x = station.location % self.env.grid_size
                    station_y = station.location // self.env.grid_size
                    travel_distance = abs(vehicle['coordinates'][0] - station_x) + abs(vehicle['coordinates'][1] - station_y)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            
            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            battery_loss+=idle_vehicle[i]*2*battery_consum # idle consumption
            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level*(1 - waiting_vehicle[i]))  # If not idle, must meet min battery

            
            
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
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1)
        
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
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        
        # Objective: Maximize total value considering Q-values with rejection-aware ADP
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
        
        # 批量预计算所有vehicle-request对的Q值以提高性能
        option_q_cache = {}
        rejection_adjusted_values = {}  # 存储拒绝感知调整后的价值
        
        if adp_weight > 0:
            # 收集所有需要计算的vehicle-request对
            vehicle_request_pairs = []
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, request in enumerate(available_requests):
                    vehicle_request_pairs.append((vehicle_id, request))
            
            # 批量计算Q值和拒绝感知价值
            if hasattr(self.env, 'batch_evaluate_service_options'):
                try:
                    batch_q_values = self.env.batch_evaluate_service_options(vehicle_request_pairs)
                    
                    # 批量计算拒绝概率（只对EV）
                    batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                    
                    for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                        q_value = batch_q_values[i] if i < len(batch_q_values) else 0.0
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 计算拒绝感知调整价值
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                except Exception as e:
                    print(f"Batch evaluation failed: {e}, falling back to individual calculations")
            
            # 如果批量计算失败，使用单独计算
            if not option_q_cache:
                # 批量计算拒绝概率（只对EV）
                batch_rejection_probs = self._batch_calculate_reject_pro_network(vehicle_request_pairs)
                
                for i, (vehicle_id, request) in enumerate(vehicle_request_pairs):
                    try:
                        q_value = self.env.evaluate_service_option(vehicle_id, request)
                        option_q_cache[(vehicle_id, request.request_id)] = q_value
                        
                        # 使用批量计算的拒绝概率
                        rejection_prob = batch_rejection_probs[i] if i < len(batch_rejection_probs) else 0.0
                        
                        adjusted_value = self._calculate_rejection_aware_value(
                            vehicle_id, request, q_value, rejection_prob
                        )
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = adjusted_value
                    except Exception:
                        option_q_cache[(vehicle_id, request.request_id)] = 0.0
                        rejection_adjusted_values[(vehicle_id, request.request_id)] = 0.0
            
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]

            for j, request in enumerate(available_requests):
                if adp_weight <= 0:
                    # 回退到基础计算
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._manhattan_loc(request.pickup, request.dropoff)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * (d1 + d2)
                    immediate = req_val 
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    objective_terms += immediate* request_decision[i][j]
                else:
                    # 使用批量计算的Q值和拒绝感知的调整价值
                    base_q_value = option_q_cache.get((vehicle_id, request.request_id), 0.0)
                    #adjusted_value = rejection_adjusted_values.get((vehicle_id, request.request_id), base_q_value)
                    objective_terms += base_q_value * adp_weight * request_decision[i][j]
                
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * d_travel
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        objective_terms += charging_q * adp_weight * charge_decision[i, j]
            

            
            # Add penalty for unserved requests (considering reject behavior)
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idld_q_penalty = -5e+3
        for i in range(len(vehicle_ids)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids[i]
            vehicle = self.env.vehicles[vehicle_id]
            
            # 获取神经网络预测的idle Q值
            idle_q_value = 0
            wait_q_value = 0
            current_coords = vehicle['coordinates']
            target_x = max(0, min(self.env.grid_size-1, 
                                current_coords[0] + random.randint(-1, 1)))
            target_y = max(0, min(self.env.grid_size-1, 
                                current_coords[1] + random.randint(-1, 1)))
            target_loc = target_y * self.env.grid_size + target_x            
            if hasattr(self.env, 'evaluate_idle_option'):
                try:
                    idle_q_value = self.env.evaluate_idle_option(
                        vehicle_id=vehicle_id,
                        target_loc = target_loc,
                    )
                except Exception as e:
                    print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的idle奖励作为后备
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            else:
                # 如果没有神经网络方法，使用默认奖励
                idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            
            # 获取神经网络预测的waiting Q值
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )
                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += (-avg_request_value + wait_q_penalty) * idle_vehicle[i]
                objective_terms += (-avg_request_value + wait_q_penalty) * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += (idle_q_value+wait_q_penalty) * idle_vehicle[i]
                objective_terms += (wait_q_value+wait_q_penalty) * waiting_vehicle[i]  # Use neural network predicted waiting Q-value

            # Penalty for unserved requests
        unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
        # objective_terms -= avg_request_value * (len(available_requests) - served_requests)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
            # Solve the optimization problem
        try:
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
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    if idle_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"idle"
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
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












    def _gurobi_vehicle_rebalancing_knownreject_state(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Gurobi optimization with known reject behavior for EVs and charging level constraints
        EVs won't be assigned to requests they would reject
        Includes t-1 to t charging level progression with minimum battery requirements
        """
        if not self.available:
            return {}
        
        assignments = {}
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_with_reject_and_charging")
        model.setParam('OutputFlag', 0)  # Suppress output

        # Aggregate stats for opportunity costs (optional)
        active_requests_count = len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0
        active_requests_value = sum(getattr(req, 'final_value', getattr(req, 'value', 0.0)) for req in (self.env.active_requests.values() if hasattr(self.env, 'active_requests') else []))
        avg_request_value = (active_requests_value / active_requests_count) if active_requests_count > 0 else 0.0

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2

        battery_consum = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05 # Battery consumption per travel step
        service_consumption = 0.05 # Battery consumption per service
        
        # Filter out rejected requests for each EV
        valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid

        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            for j, request in enumerate(available_requests):
                # Check if EV would reject this request
                if vehicle['type'] == 1:
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

                name=f'battery_t_{vehicle_id}'
            )
        
        

        idle_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )
        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_waiting'
            )
            # Battery level transition constraints (t-1 to t relationship)
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Initialize battery expressions as Gurobi LinExpr
            battery_loss = self.gp.LinExpr()
            battery_increase = self.gp.LinExpr()
            
            # Battery consumption from charging (travel to station)
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    # Convert station location index to coordinates
                    station_x = station.location % self.env.grid_size
                    station_y = station.location // self.env.grid_size
                    travel_distance = abs(vehicle['coordinates'][0] - station_x) + abs(vehicle['coordinates'][1] - station_y)
                    battery_loss += travel_distance * battery_consum * charge_decision[i, j]
                    battery_increase +=  self.env.chargeincrease_whole*charge_decision[i, j]
            
            # Battery consumption from service requests (travel to pickup + pickup to dropoff)
            if available_requests:
                for j, request in enumerate(available_requests):
                    # Travel from vehicle current position to pickup
                    pickup_x = request.pickup % self.env.grid_size
                    pickup_y = request.pickup // self.env.grid_size
                    travel_distance_to_pickup = abs(vehicle['coordinates'][0] - pickup_x) + abs(vehicle['coordinates'][1] - pickup_y)
                    
                    # Travel from pickup to dropoff
                    dropoff_x = request.dropoff % self.env.grid_size
                    dropoff_y = request.dropoff // self.env.grid_size
                    travel_distance_pickup_to_dropoff = abs(pickup_x - dropoff_x) + abs(pickup_y - dropoff_y)
                    
                    # Total battery consumption for this request
                    total_travel_distance = travel_distance_to_pickup + travel_distance_pickup_to_dropoff
                    battery_loss += total_travel_distance * battery_consum * request_decision[i][j]
            battery_loss+=idle_vehicle[i]*2*battery_consum # idle consumption
            # Battery transition constraint (simplified to avoid infeasibility)
            model.addConstr(battery_t[i] == battery_t_minus_1[i] - battery_loss + battery_increase)
            # Ensure vehicle has enough battery for actions (but allow some flexibility)
            model.addConstr(battery_loss <= battery_t_minus_1[i] )  # Allow small battery deficit to avoid infeasibility
            # Ensure battery doesn't go below minimum (but allow some flexibility)
            model.addConstr(battery_t[i] >=min_battery_level*(1 - waiting_vehicle[i]))  # If not idle, must meet min battery

            
            
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
            model.addConstr(idle_vehicle[i] + actionv + waiting_vehicle[i] == 1)
        
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
                    <= max(
                        0,
                        station.max_capacity - len(station.current_vehicles)
                ))
        
        # Objective: Maximize total value considering Q-values
        objective_terms = self.gp.LinExpr()
        adp_weight = getattr(self.env, 'adp_value', 1.0)
            
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Process valid request assignments
            for j, request in enumerate(available_requests):
                if (i, j) in request_decision:
                    req_val = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                    cur_loc = vehicle['location']
                    d1 = self._manhattan_loc(cur_loc, request.pickup)
                    d2 = self._manhattan_loc(request.pickup, request.dropoff)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * (d1 + d2)
                    immediate = req_val 
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    if adp_weight <= 0:
                        objective_terms += immediate *(1 - rejection_prob)* request_decision[i, j]
                    else:
                        # Use option-completion Q-value for request assignment
                        print(f"Evaluating service option for vehicle {vehicle_id} and request {request}")
                        option_q = self.env.evaluate_service_option(vehicle_id, request)
                        objective_terms += option_q * adp_weight * request_decision[i, j]
                
                # Process charging assignments
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    cur_loc = vehicle['location']
                    d_travel = self._manhattan_loc(cur_loc, station.location)
                    moving_cost = getattr(self.env, 'movingpenalty', -0.1) * d_travel
                    charge_steps = getattr(self.env, 'charge_duration', 2)
                    charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * charge_steps
                    immediate = moving_cost + charging_penalty
                    if adp_weight <= 0:
                        # Immediate charging cost fallback
                        objective_terms += immediate * charge_decision[i, j]
                    else:
                        # Use option-completion Q-value for charging
                        charging_q = 0.0
                        if hasattr(self.env, 'evaluate_charging_option'):
                            try:
                                charging_q = self.env.evaluate_charging_option(vehicle_id, station)
                            except Exception:
                                charging_q = 0.0
                        objective_terms += charging_q * adp_weight * charge_decision[i, j]
            

            
            # Add penalty for unserved requests (considering reject behavior)
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                served_requests += request_decision[i][j]
        wait_q_penalty = -5e+3
        idle_q_penalty = -5e+3
        for i in range(len(vehicle_ids)):
            # 使用神经网络预测的idle Q值替代固定的idle_vehicle_reward
            vehicle_id = vehicle_ids[i]
            vehicle = self.env.vehicles[vehicle_id]
            
            # 获取神经网络预测的idle Q值
            idle_q_value = 0
            wait_q_value = 0
            current_coords = vehicle['coordinates']
            target_x = max(0, min(self.env.grid_size-1, 
                                current_coords[0] + random.randint(-1, 1)))
            target_y = max(0, min(self.env.grid_size-1, 
                                current_coords[1] + random.randint(-1, 1)))
            target_loc = target_y * self.env.grid_size + target_x            
            if hasattr(self.env, 'evaluate_idle_option'):
                try:
                    idle_q_value = self.env.evaluate_idle_option(
                        vehicle_id=vehicle_id,
                        target_loc = target_loc,
                    )
                except Exception as e:
                    print(f"Warning: Failed to get idle Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的idle奖励作为后备
                    idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            else:
                # 如果没有神经网络方法，使用默认奖励
                idle_q_value = getattr(self.env, 'idle_vehicle_reward', 0.0)
            
            # 获取神经网络预测的waiting Q值
            if hasattr(self.env, 'evaluate_waiting_option'):
                try:
                    wait_q_value = self.env.evaluate_waiting_option(
                        vehicle_id=vehicle_id,
                    )
                except Exception as e:
                    print(f"Warning: Failed to get waiting Q-value for vehicle {vehicle_id}: {e}")
                    # 使用默认的waiting奖励作为后备
                    wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            else:
                # 如果没有神经网络方法，使用默认奖励
                wait_q_value = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            
            if adp_weight <= 0:
                objective_terms += -avg_request_value * idle_vehicle[i]
                objective_terms += -avg_request_value * waiting_vehicle[i]  # Additional opportunity cost penalty
            else:
                objective_terms += idle_q_value * idle_vehicle[i]
                objective_terms += wait_q_value * waiting_vehicle[i]  # Use neural network predicted waiting Q-value

            # Penalty for unserved requests
        unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
        # objective_terms -= avg_request_value * (len(available_requests) - served_requests)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        objvalue = []
        try:
            model.optimize()
            
            # Extract assignments
            if model.status == self.GRB.OPTIMAL:
                
                for i in range(len(vehicle_ids)):
                    vehicle_id = vehicle_ids[i]  # Add this line to define vehicle_id
                    vehicle_obj = 0
                    for j in range(len(available_requests)):
                        if request_decision[i][j].x > 0.5:
                            request = available_requests[j]
                            option_q = self.env.evaluate_service_option_state(vehicle_id, request)
                            vehicle_obj += getattr(available_requests[j], 'final_value', getattr(available_requests[j], 'value', 0.0)) + option_q
                        if charging_stations:
                            for k, station in enumerate(charging_stations):
                                if charge_decision[i, k].x > 0.5:
                                    charging_q = self.env.evaluate_charging_option_state(vehicle_id, station)
                                    vehicle_obj += -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)+ charging_q
                        if idle_vehicle[i].x > 0.1:
                            idle_q = self.env.evaluate_idle_option_state(vehicle_id)
                            vehicle_obj += getattr(self.env, 'idle_vehicle_reward', -0.1)+  idle_q
                        if waiting_vehicle[i].x > 0.1:
                            wait_q = self.env.evaluate_waiting_option_state(vehicle_id)
                            vehicle_obj += getattr(self.env, 'waiting_vehicle_reward', -0.1)+ wait_q
                    objvalue.append(vehicle_obj)

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
                    if waiting_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"waiting"

                    if idle_vehicle[i].x > 0.1:
                        assignments[vehicle_id] = f"idle"
                
                # Update vehicle battery levels based on optimization results
                for i, vehicle_id in enumerate(vehicle_ids):
                    if hasattr(self.env.vehicles[vehicle_id], 'predicted_battery_t'):
                        self.env.vehicles[vehicle_id]['predicted_battery_t'] = battery_t[i].x
                        
            else:
                print(f"Optimization status: {model.status}")
                objvalue = [0 for _ in range(len(vehicle_ids))]
                for i, vehicle_id in enumerate(vehicle_ids):
                    assignments[vehicle_id] = f"waiting"
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
            objvalue = [0 for _ in range(len(vehicle_ids))]
            assignments = self._heuristic_assignment_with_reject(vehicle_ids, available_requests, charging_stations)
        
        return objvalue, assignments





    def _heuristic_assignment_with_reject(self, vehicle_ids, available_requests, charging_stations=None):

        assignments = {}
        battery_threshold = self.env.heuristic_battery_threshold if hasattr(self.env, 'heuristic_battery_threshold') else 0.5
        if not vehicle_ids:
            return assignments
        
        # 第一步：识别低电量车辆（电池 < 0.5）
        low_battery_vehicles = []
        high_battery_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = self.env.vehicles[vehicle_id]
            if vehicle['battery'] < battery_threshold:
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
                
                # 计算该车辆到所有订单的距离
                distance_list = {}
                for request in remaining_requests:
                    pickup_coords = (
                        request.pickup // self.env.grid_size,
                        request.pickup % self.env.grid_size
                    )
                    distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                              abs(vehicle_coords[1] - pickup_coords[1])
                    distance_list[request.request_id] = distance
                
                battery_consumption = self.env.battery_consum if hasattr(self.env, 'battery_consum') else 0.05
                min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
                
                # 为该车辆寻找最佳订单
                best_request = None
                
                if vehicle['type'] == 1:  # EV车辆：优先选择近距离订单
                    # 按距离从近到远排序
                    distance_sorted = sorted(distance_list.items(), key=lambda x: x[1])
                    
                    for req_id, distance in distance_sorted:
                        # 检查电池是否足够
                        estimated_consumption = distance * battery_consumption * 2  # 往返消耗
                        if battery_level - estimated_consumption >= min_battery_level:
                            # 检查拒绝概率
                            request = next(r for r in remaining_requests if r.request_id == req_id)
                            best_request = request
                
                else:  # AEV车辆：可以选择任何距离的订单，优先高价值
                    # 按订单价值排序
                    value_distance_list = []
                    for req_id, distance in distance_list.items():
                        request = next(r for r in remaining_requests if r.request_id == req_id)
                        value = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                        estimated_consumption = distance * battery_consumption * 2
                        
                        if battery_level - estimated_consumption >= min_battery_level:
                            value_distance_list.append((request, value, distance))
                    
                    # 按价值从高到低排序
                    if value_distance_list:
                        value_distance_list.sort(key=lambda x: x[1], reverse=True)
                        best_request = value_distance_list[0][0]
                
                # 分配最佳订单
                if best_request:
                    assignments[vehicle_id] = best_request
                    remaining_requests.remove(best_request)

        for vehicle_id in vehicle_ids:
            if vehicle_id not in assignments:
                assignments[vehicle_id] = "idle"
        return assignments

    def optimize_vehicle_rebalancing_state(self, vehicle_ids):
        """Optimize vehicle rebalancing using state-based value function (src2-style approach)"""
        if not self.available:
            return self._heuristic_rebalancing_assignment(vehicle_ids), []
        
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
            return self._gurobi_vehicle_rebalancing_knownreject_state_enhanced(vehicle_ids, available_requests, charging_stations)
        except Exception as e:
            print(f"Enhanced Gurobi rebalancing failed: {e}, using fallback")
            return self._heuristic_rebalancing_assignment(vehicle_ids), []

    def _gurobi_vehicle_rebalancing_knownreject_state_enhanced(self, vehicle_ids, available_requests, charging_stations=None):
        """
        Enhanced Gurobi optimization using state-based value function (src2-style approach)
        Returns both assignments and individual vehicle rewards (y_ei values)
        """
        if not self.available:
            return {}, []
        
        assignments = {}
        vehicle_rewards = []  # Store y_ei values for each vehicle
        
        # Create optimization model
        model = self.gp.Model("vehicle_assignment_state_based")
        model.setParam('OutputFlag', 0)  # Suppress output
        model.setParam('TimeLimit', 30)  # Set time limit

        # Parameters
        min_battery_level = self.env.min_battery_level if hasattr(self.env, 'min_battery_level') else 0.2
        
        # Filter out rejected requests for each EV
        valid_assignments = {}  # (vehicle_id, request_idx) -> is_valid
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            for j, request in enumerate(available_requests):
                # Check if EV would reject this request
                if vehicle['type'] == 1:
                    rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                    valid_assignments[(i, j)] = rejection_prob < 0.5
                else:
                    # AEV never rejects
                    valid_assignments[(i, j)] = True
        
        # Decision variables for request assignments
        request_decision = {}
        for i, vehicle_id in enumerate(vehicle_ids):
            for j, request in enumerate(available_requests):
                request_decision[i, j] = model.addVar(
                    vtype=self.GRB.BINARY,
                    name=f'request_{vehicle_id}_{request.request_id}'
                )
        
        # Decision variables for charging assignments
        charge_decision = {}
        if charging_stations:
            for i, vehicle_id in enumerate(vehicle_ids):
                for j, station in enumerate(charging_stations):
                    charge_decision[i, j] = model.addVar(
                        vtype=self.GRB.BINARY,
                        name=f'charge_{vehicle_id}_{station.id}'
                    )
        
        # Decision variables for idle/waiting
        idle_vehicle = {}
        waiting_vehicle = {}
        for i in range(len(vehicle_ids)):
            idle_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_idle'
            )
            waiting_vehicle[i] = model.addVar(
                vtype=self.GRB.BINARY,
                name=f'vehicle_{vehicle_ids[i]}_wait'
            )
        
        # Constraints
        # Each vehicle must choose exactly one action
        for i in range(len(vehicle_ids)):
            action_sum = idle_vehicle[i] + waiting_vehicle[i]
            for j in range(len(available_requests)):
                if valid_assignments.get((i, j), False):
                    action_sum += request_decision[i, j]
            if charging_stations:
                for j in range(len(charging_stations)):
                    action_sum += charge_decision[i, j]
            model.addConstr(action_sum == 1)
        
        # Each request can be assigned to at most one vehicle
        for j in range(len(available_requests)):
            request_sum = self.gp.LinExpr()
            for i in range(len(vehicle_ids)):
                if valid_assignments.get((i, j), False):
                    request_sum += request_decision[i, j]
            model.addConstr(request_sum <= 1)
        
        # Constraint invalid assignments to 0
        for i in range(len(vehicle_ids)):
            for j in range(len(available_requests)):
                if not valid_assignments.get((i, j), False):
                    model.addConstr(request_decision[i, j] == 0)
        
        # Objective function using state-based value function
        objective_terms = self.gp.LinExpr()
        
        for i, vehicle_id in enumerate(vehicle_ids):
            vehicle = self.env.vehicles[vehicle_id]
            
            # Process request assignments using state-based value function
            for j, request in enumerate(available_requests):
                if valid_assignments.get((i, j), False):
                    # Use state-based evaluation for service option
                    if hasattr(self.env, 'evaluate_service_option_state'):
                        try:
                            state_value = self.env.evaluate_service_option_state(vehicle_id, request)
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            objective_terms += (immediate_reward + state_value) * request_decision[i, j]
                        except Exception as e:
                            print(f"Warning: Failed to get state value for vehicle {vehicle_id}, request {getattr(request, 'id', 'unknown')}: {e}")
                            # Fallback to immediate reward only
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            objective_terms += immediate_reward * request_decision[i, j]
                    else:
                        # Fallback if state value function not available
                        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                        objective_terms += immediate_reward * request_decision[i, j]
            
            # Process charging assignments using state-based value function
            if charging_stations:
                for j, station in enumerate(charging_stations):
                    if hasattr(self.env, 'evaluate_charging_option_state'):
                        try:
                            state_value = self.env.evaluate_charging_option_state(vehicle_id, station)
                            charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                            objective_terms += (charging_penalty + state_value) * charge_decision[i, j]
                        except Exception as e:
                            print(f"Warning: Failed to get charging state value for vehicle {vehicle_id}: {e}")
                            charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                            objective_terms += charging_penalty * charge_decision[i, j]
                    else:
                        charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                        objective_terms += charging_penalty * charge_decision[i, j]
            
            # Process idle option using state-based value function
            if hasattr(self.env, 'evaluate_idle_option_state'):
                try:
                    idle_state_value = self.env.evaluate_idle_option_state(vehicle_id)
                    idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                    objective_terms += (idle_penalty + idle_state_value) * idle_vehicle[i]
                except Exception as e:
                    print(f"Warning: Failed to get idle state value for vehicle {vehicle_id}: {e}")
                    idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                    objective_terms += idle_penalty * idle_vehicle[i]
            else:
                idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                objective_terms += idle_penalty * idle_vehicle[i]
            
            # Process waiting option
            wait_penalty = getattr(self.env, 'waiting_vehicle_reward', -0.1)
            objective_terms += wait_penalty * waiting_vehicle[i]
        
        # Penalty for unserved requests
        served_requests = self.gp.LinExpr()
        for j in range(len(available_requests)):
            for i in range(len(vehicle_ids)):
                if valid_assignments.get((i, j), False):
                    served_requests += request_decision[i, j]
        
        unserved_penalty = getattr(self.env, 'unserved_penalty', 1.5)
        objective_terms -= unserved_penalty * (len(available_requests) - served_requests)
        
        model.setObjective(objective_terms, self.GRB.MAXIMIZE)
        
        # Solve the optimization problem
        try:
            model.optimize()
            
            # Extract assignments and calculate individual vehicle rewards (y_ei)
            if model.status == self.GRB.OPTIMAL:
                for i, vehicle_id in enumerate(vehicle_ids):
                    vehicle_obj = 0.0  # This will be the y_ei value for this vehicle
                    
                    # Check request assignments
                    for j, request in enumerate(available_requests):
                        if request_decision[i, j].x > 0.5:
                            assignments[vehicle_id] = request
                            # Calculate contribution to objective (immediate reward + state value)
                            immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
                            rejection_prob = self.env._calculate_rejection_probability(vehicle_id, request)
                            if hasattr(self.env, 'evaluate_service_option_state'):
                                try:
                                    state_value = self.env.evaluate_service_option_state(vehicle_id, request)
                                    vehicle_obj = immediate_reward + state_value
                                except Exception:
                                    vehicle_obj = immediate_reward
                            else:
                                vehicle_obj = immediate_reward
                            break
                    
                    # Check charging assignments if no request assigned
                    if vehicle_id not in assignments and charging_stations:
                        for j, station in enumerate(charging_stations):
                            if charge_decision[i, j].x > 0.5:
                                assignments[vehicle_id] = f"charge_{station.id}"
                                # Calculate charging contribution
                                charging_penalty = -getattr(self.env, 'charging_penalty', 0.5) * getattr(self.env, 'charge_duration', 2)
                                if hasattr(self.env, 'evaluate_charging_option_state'):
                                    try:
                                        state_value = self.env.evaluate_charging_option_state(vehicle_id, station)
                                        vehicle_obj = charging_penalty + state_value
                                    except Exception:
                                        vehicle_obj = charging_penalty
                                else:
                                    vehicle_obj = charging_penalty
                                break
                    
                    # Check idle/waiting assignments
                    if vehicle_id not in assignments:
                        if idle_vehicle[i].x > 0.5:
                            assignments[vehicle_id] = "idle"
                            idle_penalty = getattr(self.env, 'idle_vehicle_reward', -0.1)
                            if hasattr(self.env, 'evaluate_idle_option_state'):
                                try:
                                    state_value = self.env.evaluate_idle_option_state(vehicle_id)
                                    vehicle_obj = idle_penalty + state_value
                                except Exception:
                                    vehicle_obj = idle_penalty
                            else:
                                vehicle_obj = idle_penalty
                        elif waiting_vehicle[i].x > 0.5:
                            assignments[vehicle_id] = "waiting"
                            vehicle_obj = getattr(self.env, 'waiting_vehicle_reward', -0.1)
                    
                    vehicle_rewards.append(vehicle_obj)
                
                return assignments, vehicle_rewards
            else:
                print(f"Optimization failed with status: {model.status}")
                # Return fallback assignments and zero rewards
                fallback_assignments = self._heuristic_rebalancing_assignment(vehicle_ids)
                fallback_rewards = [0.0] * len(vehicle_ids)
                return fallback_assignments, fallback_rewards
                
        except Exception as e:
            print(f"Gurobi optimization failed: {e}")
            # Return fallback assignments and zero rewards
            fallback_assignments = self._heuristic_rebalancing_assignment(vehicle_ids)
            fallback_rewards = [0.0] * len(vehicle_ids)
            return fallback_assignments, fallback_rewards

    def store_and_train_state_experiences(self, vehicle_ids, vehicle_rewards, batch_size=32):
        """
        Store state experiences and perform training using src2-style approach
        
        Args:
            vehicle_ids: List of vehicle IDs that were optimized
            vehicle_rewards: List of y_ei values (target values from Gurobi optimization)
            batch_size: Batch size for training
        """
        if not hasattr(self.env, 'value_function_state') or self.env.value_function_state is None:
            print("Warning: State-based value function not available for training")
            return 0.0
        
        # Store experiences for each vehicle
        current_time = getattr(self.env, 'current_time', 0.0)
        num_requests = len(getattr(self.env, 'active_requests', {}))
        
        for i, (vehicle_id, y_ei) in enumerate(zip(vehicle_ids, vehicle_rewards)):
            vehicle = self.env.vehicles.get(vehicle_id)
            if vehicle is None:
                continue
            
            # Get current vehicle state
            vehicle_location = vehicle['location']
            battery_level = vehicle['battery']
            
            # Calculate other vehicles (excluding current vehicle)
            other_vehicles = len([v for vid, v in self.env.vehicles.items() 
                                 if vid != vehicle_id and v['assigned_request'] is None 
                                 and v['passenger_onboard'] is None and v['charging_station'] is None])
            
            # Get request value if vehicle is assigned to a request
            request_value = 0.0
            if vehicle.get('assigned_request') in getattr(self.env, 'active_requests', {}):
                assigned_request = self.env.active_requests[vehicle['assigned_request']]
                request_value = getattr(assigned_request, 'final_value', getattr(assigned_request, 'value', 0.0))
            
            # Store state experience
            self.env.value_function_state.store_experience_state(
                vehicle_id=vehicle_id,
                vehicle_location=vehicle_location,
                battery_level=battery_level,
                current_time=current_time,
                other_vehicles=max(0, other_vehicles),
                num_requests=num_requests,
                request_value=request_value,
                y_ei=y_ei  # Target value from Gurobi optimization
            )
        
        # Perform training step
        if hasattr(self.env.value_function_state, 'experience_buffer_state'):
            buffer_size = len(self.env.value_function_state.experience_buffer_state)
            if buffer_size >= batch_size:
                training_loss = self.env.value_function_state.train_step_state(batch_size=batch_size)
                if buffer_size % 100 == 0:  # Log every 100 experiences
                    print(f"State-based training: Buffer size={buffer_size}, Loss={training_loss:.4f}")
                return training_loss
        
        return 0.0

    def _batch_calculate_reject_pro_network(self, vehicle_request_pairs):
        """
        批量计算多个vehicle-request对的拒绝概率，提高计算效率
        只对EV车辆计算，AEV返回0
        
        Args:
            vehicle_request_pairs: List of (vehicle_id, request) tuples
            
        Returns:
            List of rejection probabilities corresponding to each vehicle-request pair
        """
        if not vehicle_request_pairs:
            return []
        
        # 检查ValueFunction是否有拒绝预测器
        value_function = getattr(self.env, 'value_function', None)
        if value_function is None or not hasattr(value_function, 'rejection_predictor'):
            # 回退到单独计算
            return [self._calculate_reject_pro_network(vehicle_id, request) 
                   for vehicle_id, request in vehicle_request_pairs]
        
        try:
            # 准备批量输入特征
            batch_features = []
            valid_pairs = []
            
            for vehicle_id, request in vehicle_request_pairs:
                vehicle = self.env.vehicles.get(vehicle_id)
                if vehicle is None:
                    continue
                
                # AEV永远不拒绝，EV才需要计算
                if vehicle.get('type') == 2:  # AEV
                    continue
                elif vehicle.get('type') != 1:  # 不是EV
                    continue
                
                # 计算到pickup的距离
                vehicle_coords = vehicle['coordinates']
                pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
                distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
                
                # 准备神经网络输入特征
                features = [
                    distance,                           # 距离
                    vehicle.get('battery', 1.0),       # 电池电量
                    self.env.current_time,             # 当前时间
                    len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0,  # 订单数量
                    vehicle.get('type', 1)             # 车辆类型
                ]
                
                batch_features.append(features)
                valid_pairs.append((vehicle_id, request))
            
            if not batch_features:
                # 没有需要计算的EV，返回全0
                return [0.0] * len(vehicle_request_pairs)
            
            # 批量神经网络推理
            import torch
            features_tensor = torch.tensor(batch_features, dtype=torch.float32).to(value_function.device)
            
            with torch.no_grad():
                batch_rejection_probs = value_function.rejection_predictor(features_tensor).squeeze()
                
                # 确保输出是一维的
                if batch_rejection_probs.dim() == 0:
                    batch_rejection_probs = batch_rejection_probs.unsqueeze(0)
                
                rejection_probs_list = batch_rejection_probs.cpu().numpy().tolist()
            
            # 将结果映射回原始的vehicle_request_pairs顺序
            result_probs = []
            valid_idx = 0
            
            for vehicle_id, request in vehicle_request_pairs:
                vehicle = self.env.vehicles.get(vehicle_id)
                if vehicle is None:
                    result_probs.append(0.0)
                elif vehicle.get('type') == 2:  # AEV
                    result_probs.append(0.0)
                elif vehicle.get('type') != 1:  # 不是EV
                    result_probs.append(0.0)
                else:  # EV车辆
                    if valid_idx < len(rejection_probs_list):
                        prob = rejection_probs_list[valid_idx]
                        # 确保概率在合理范围内
                        prob = max(0.0, min(0.95, prob))
                        result_probs.append(prob)
                        valid_idx += 1
                    else:
                        result_probs.append(0.0)
            
            return result_probs
            
        except Exception as e:
            print(f"Batch rejection probability calculation failed: {e}")
            # 回退到单独计算
            return [self._calculate_reject_pro_network(vehicle_id, request) 
                   for vehicle_id, request in vehicle_request_pairs]

    def _calculate_reject_pro_network(self, vehicle_id, request):
        """
        使用ValueFunction的神经网络预测器计算拒绝概率
        只对EV车辆计算，AEV返回0
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            
        Returns:
            float: 拒绝概率 (0-1之间)
        """
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return 0.0
        
        # AEV永远不拒绝
        if vehicle.get('type') == 2:  # AEV
            return 0.0
        
        # 只对EV计算拒绝概率
        if vehicle.get('type') != 1:  # 不是EV
            return 0.0
        
        # 检查ValueFunction是否有拒绝预测器
        value_function = getattr(self.env, 'value_function', None)
        if value_function is None or not hasattr(value_function, 'rejection_predictor'):
            # 回退到简单的距离基础计算
            return self._fallback_rejection_probability(vehicle_id, request)
        
        try:
            # 计算到pickup的距离
            vehicle_coords = vehicle['coordinates']
            pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
            distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
            
            # 准备神经网络输入特征
            import torch
            features = torch.tensor([
                distance,                           # 距离
                vehicle.get('battery', 1.0),       # 电池电量
                self.env.current_time,             # 当前时间
                len(self.env.active_requests) if hasattr(self.env, 'active_requests') else 0,  # 订单数量
                vehicle.get('type', 1)             # 车辆类型
            ], dtype=torch.float32).unsqueeze(0).to(value_function.device)
            
            # 使用神经网络预测拒绝概率
            with torch.no_grad():
                rejection_prob = value_function.rejection_predictor(features).item()
            
            # 确保概率在合理范围内
            rejection_prob = max(0.0, min(0.95, rejection_prob))
            
            return rejection_prob
            
        except Exception as e:
            print(f"Neural network rejection prediction failed for vehicle {vehicle_id}: {e}")
            # 回退到简单计算
            return self._fallback_rejection_probability(vehicle_id, request)
    
    def _fallback_rejection_probability(self, vehicle_id, request):
        """
        当神经网络不可用时的回退拒绝概率计算
        基于距离的简单模型
        """
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return 0.0
        
        # 计算距离
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup % self.env.grid_size, request.pickup // self.env.grid_size)
        distance = abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
        
        # 基于距离的简单拒绝概率模型
        distance_factor = 0.2
        rejection_prob = 1 - np.exp(-distance * distance_factor)
        
        # 限制最大拒绝概率为90%
        return min(0.9, rejection_prob)

    def _calculate_rejection_aware_value(self, vehicle_id, request, base_q_value, rejection_prob=None):
        """
        计算拒绝感知的调整价值: Q_value - immediate_reward * rejection_probability
        
        Args:
            vehicle_id: 车辆ID
            request: 请求对象
            base_q_value: 基础Q值（接受订单的正向价值）
            rejection_prob: 拒绝概率（如果为None则重新计算）
            
        Returns:
            float: 调整后的价值
        """
        # 计算立即收益
        vehicle = self.env.vehicles.get(vehicle_id)
        if vehicle is None:
            return base_q_value
            
        # 订单价值（立即收益）
        immediate_reward = getattr(request, 'final_value', getattr(request, 'value', 0.0))
        
        # 计算移动成本
        cur_loc = vehicle['location']
        d1 = self._manhattan_loc(cur_loc, request.pickup)
        d2 = self._manhattan_loc(request.pickup, request.dropoff)
        moving_cost = getattr(self.env, 'movingpenalty', -0.1) * (d1 + d2)
        
        # 净立即收益（考虑移动成本）
        net_immediate_reward = immediate_reward + moving_cost  # moving_cost通常是负数
        
        # 如果没有提供拒绝概率，则计算
        if rejection_prob is None:
            rejection_prob = self._calculate_reject_pro_network(vehicle_id, request)
        
        # 计算调整后的价值: Q值 - 立即收益 * 拒绝概率
        # 逻辑：如果拒绝概率高，则减去更多的立即收益价值
        adjusted_value = base_q_value - (net_immediate_reward * rejection_prob)
        
        return adjusted_value



