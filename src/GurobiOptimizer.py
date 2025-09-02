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
    
    def optimize_vehicle_assignment(self, requests, vehicles):
        """Optimize assignment of vehicles to requests using Gurobi"""
        if not self.available or not requests:
            return self._heuristic_assignment(requests, vehicles)
        
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
            return self._heuristic_assignment(requests, vehicles)
    
    def _calculate_distance(self, vehicle_id, request):
        """Calculate distance from vehicle to request pickup"""
        vehicle = self.env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        pickup_coords = (request.pickup // self.env.grid_size, request.pickup % self.env.grid_size)
        return abs(vehicle_coords[0] - pickup_coords[0]) + abs(vehicle_coords[1] - pickup_coords[1])
    
    def _heuristic_assignment(self, requests, vehicles):
        """Heuristic assignment when Gurobi is not available"""
        assignments = {}
        available_vehicles = set(vehicles)
        
        # Sort requests by value (highest first)
        sorted_requests = sorted(requests, key=lambda r: r.value, reverse=True)
        
        for request in sorted_requests:
            if not available_vehicles:
                break
            
            # Find closest available vehicle
            best_vehicle = None
            min_distance = float('inf')
            
            for vehicle_id in available_vehicles:
                distance = self._calculate_distance(vehicle_id, request)
                if distance < min_distance:
                    min_distance = distance
                    best_vehicle = vehicle_id
            
            if best_vehicle:
                assignments[best_vehicle] = request.request_id
                available_vehicles.remove(best_vehicle)
        
        return assignments

