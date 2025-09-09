"""
Heuristic Policy for Vehicle Assignment - Benchmark Implementation
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from src.Action import Action, ChargingAction, ServiceAction
from src.Request import Request


class HeuristicPolicy:
    
    def __init__(self, battery_threshold=0.5, max_service_distance=8):
        self.battery_threshold = battery_threshold
        self.max_service_distance = max_service_distance
        
    def get_actions(self, env) -> Dict[int, Action]:

        actions = {}
        
        # 获取所有可用车辆（不在充电、无分配任务、无乘客）
        available_vehicles = self._get_available_vehicles(env)
        
        # 第一步：处理低电量车辆的充电需求
        low_battery_vehicles = [
            vehicle_id for vehicle_id in available_vehicles 
            if env.vehicles[vehicle_id]['battery'] < self.battery_threshold
        ]
        
        charged_vehicles = set()
        for vehicle_id in low_battery_vehicles:
            charging_action = self._assign_nearest_charging_station(env, vehicle_id)
            if charging_action:
                actions[vehicle_id] = charging_action
                charged_vehicles.add(vehicle_id)
        
        # 第二步：剩余车辆按电池容量排序，处理订单分配
        remaining_vehicles = [
            vehicle_id for vehicle_id in available_vehicles 
            if vehicle_id not in charged_vehicles
        ]
        
        # 按电池容量从高到低排序
        remaining_vehicles.sort(
            key=lambda v_id: env.vehicles[v_id]['battery'], 
            reverse=True
        )
        
        # 获取未分配的订单
        available_requests = self._get_available_requests(env)
        assigned_requests = set()
        
        # 第三步：按电池容量分配订单（高电量车辆优先选择远距离订单）
        for vehicle_id in remaining_vehicles:
            if available_requests:
                service_action = self._assign_optimal_request(
                    env, vehicle_id, available_requests, assigned_requests
                )
                if service_action:
                    actions[vehicle_id] = service_action
                    assigned_requests.add(service_action.request_id)
                else:
                    # 没有合适订单，执行移动动作
                    actions[vehicle_id] = Action([])
            else:
                # 没有可用订单，执行移动动作
                actions[vehicle_id] = Action([])
        
        # 第四步：为其他车辆分配默认移动动作
        for vehicle_id in env.vehicles:
            if vehicle_id not in actions:
                actions[vehicle_id] = Action([])
        
        return actions
    
    def _get_available_vehicles(self, env) -> List[int]:
        """获取可用于分配任务的车辆列表"""
        available_vehicles = []
        for vehicle_id, vehicle in env.vehicles.items():
            if (vehicle['assigned_request'] is None and 
                vehicle['passenger_onboard'] is None and 
                vehicle['charging_station'] is None):
                available_vehicles.append(vehicle_id)
        return available_vehicles
    
    def _assign_nearest_charging_station(self, env, vehicle_id) -> Optional[ChargingAction]:
        """为车辆分配最近的有容量的充电站"""
        vehicle = env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        
        best_station = None
        min_distance = float('inf')
        
        for station_id, station in env.charging_manager.stations.items():
            if len(station.current_vehicles) < station.max_capacity:
                # 计算距离
                station_coords = (
                    station.location // env.grid_size,
                    station.location % env.grid_size
                )
                distance = abs(vehicle_coords[0] - station_coords[0]) + \
                          abs(vehicle_coords[1] - station_coords[1])
                
                if distance < min_distance:
                    min_distance = distance
                    best_station = station_id
        
        if best_station:
            # 根据电池水平确定充电时长
            battery_level = vehicle['battery']
            if battery_level < 0.2:
                charge_duration = 6  # 长时间充电
            elif battery_level < 0.3:
                charge_duration = 4  # 中等充电
            else:
                charge_duration = 3  # 快速补充
            
            return ChargingAction([], best_station, charge_duration)
        
        return None
    
    def _get_available_requests(self, env) -> List[Request]:
        """获取未分配的订单列表"""
        available_requests = []
        for req in env.active_requests.values():
            # 检查订单是否已被分配
            request_assigned = False
            for vehicle in env.vehicles.values():
                if (vehicle.get('assigned_request') == req.request_id or 
                    vehicle.get('passenger_onboard') == req.request_id):
                    request_assigned = True
                    break
            
            if not request_assigned:
                available_requests.append(req)
        
        return available_requests
    
    def _assign_optimal_request(self, env, vehicle_id, available_requests, assigned_requests) -> Optional[ServiceAction]:
        """
        为车辆分配最优订单
        
        策略：
        - 电池容量高的车辆优先选择距离较远的订单
        - 电池容量低的车辆选择距离较近的订单
        """
        vehicle = env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        battery_level = vehicle['battery']
        
        # 过滤掉已分配的订单
        candidate_requests = [
            req for req in available_requests 
            if req.request_id not in assigned_requests
        ]
        
        if not candidate_requests:
            return None
        
        # 计算所有订单的距离
        request_distances = []
        for request in candidate_requests:
            pickup_coords = (
                request.pickup // env.grid_size,
                request.pickup % env.grid_size
            )
            distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                      abs(vehicle_coords[1] - pickup_coords[1])
            
            if distance <= self.max_service_distance:
                request_distances.append((request, distance))
        
        if not request_distances:
            return None
        
        # 根据电池水平选择策略
        if battery_level > 0.7:
            # 高电量：优先选择距离较远的高价值订单
            # 按距离和价值的组合排序（远距离 + 高价值优先）
            request_distances.sort(
                key=lambda x: (-x[1] * 0.7 + x[0].value * 0.3, -x[0].value), 
                reverse=True
            )
        elif battery_level > 0.5:
            # 中等电量：平衡距离和价值
            request_distances.sort(
                key=lambda x: (x[1] * 0.5 - x[0].value * 0.5, x[1])
            )
        else:
            # 低电量：优先选择近距离订单
            request_distances.sort(key=lambda x: x[1])
        
        # 选择最优订单
        best_request = request_distances[0][0]
        return ServiceAction([], best_request.request_id)
    
    def get_assignments(self, env, vehicle_ids, available_requests, charging_stations=None) -> Dict[int, any]:
        """
        获取车辆分配结果，返回格式与GurobiOptimizer._gurobi_vehicle_rebalancing_knownreject相同
        
        Args:
            env: ChargingIntegratedEnvironment 环境实例
            vehicle_ids: 可用车辆ID列表
            available_requests: 可用请求列表
            charging_stations: 充电站列表
            
        Returns:
            Dict[int, any]: 车辆ID到分配的映射
                           值可以是Request对象（服务分配）或字符串"charge_{station_id}"（充电分配）
        """
        assignments = {}
        
        if not vehicle_ids:
            return assignments
        
        # 第一步：处理低电量车辆的充电需求
        low_battery_vehicles = []
        remaining_vehicles = []
        
        for vehicle_id in vehicle_ids:
            vehicle = env.vehicles[vehicle_id]
            if vehicle['battery'] < self.battery_threshold:
                low_battery_vehicles.append(vehicle_id)
            else:
                remaining_vehicles.append(vehicle_id)
        
        # 为低电量车辆分配充电站
        if charging_stations:
            for vehicle_id in low_battery_vehicles:
                charging_assignment = self._assign_nearest_charging_station_with_id(env, vehicle_id, charging_stations)
                if charging_assignment:
                    assignments[vehicle_id] = charging_assignment
        
        # 第二步：为剩余车辆分配订单（考虑EV拒绝率）
        if available_requests:
            # 按电池容量从高到低排序
            remaining_vehicles.sort(
                key=lambda v_id: env.vehicles[v_id]['battery'], 
                reverse=True
            )
            
            # 过滤有效的车辆-请求组合（考虑EV拒绝率）
            valid_assignments = self._get_valid_assignments(env, remaining_vehicles, available_requests)
            
            # 分配订单
            assigned_requests = set()
            for vehicle_id in remaining_vehicles:
                if vehicle_id not in assignments:  # 未被分配充电的车辆
                    request_assignment = self._assign_optimal_request_with_reject(
                        env, vehicle_id, available_requests, assigned_requests, valid_assignments
                    )
                    if request_assignment:
                        assignments[vehicle_id] = request_assignment
                        assigned_requests.add(request_assignment.request_id)
        
        return assignments
    
    def _assign_nearest_charging_station_with_id(self, env, vehicle_id, charging_stations) -> Optional[str]:
        """为车辆分配最近的有容量的充电站，返回格式为"charge_{station_id}" """
        vehicle = env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        
        best_station = None
        min_distance = float('inf')
        
        for station in charging_stations:
            if len(station.current_vehicles) < station.max_capacity:
                # 计算距离
                station_coords = (
                    station.location // env.grid_size,
                    station.location % env.grid_size
                )
                distance = abs(vehicle_coords[0] - station_coords[0]) + \
                          abs(vehicle_coords[1] - station_coords[1])
                
                if distance < min_distance:
                    min_distance = distance
                    best_station = station
        
        if best_station:
            return f"charge_{best_station.id}"
        
        return None
    
    def _get_valid_assignments(self, env, vehicle_ids, available_requests) -> Dict[tuple, bool]:
        """获取有效的车辆-请求分配组合，考虑EV拒绝率"""
        valid_assignments = {}
        
        for vehicle_id in vehicle_ids:
            vehicle = env.vehicles[vehicle_id]
            for request in available_requests:
                # 检查EV是否会拒绝此请求
                if vehicle['type'] == 'EV':
                    # 计算拒绝概率
                    rejection_prob = env._calculate_rejection_probability(vehicle_id, request)
                    # 如果拒绝概率高于50%，不允许分配
                    valid_assignments[(vehicle_id, request.request_id)] = rejection_prob < 0.5
                else:
                    # AEV从不拒绝
                    valid_assignments[(vehicle_id, request.request_id)] = True
        
        return valid_assignments
    
    def _assign_optimal_request_with_reject(self, env, vehicle_id, available_requests, assigned_requests, valid_assignments) -> Optional[Request]:
        """
        为车辆分配最优订单，考虑EV拒绝率
        
        策略：
        - 只考虑有效的分配（EV拒绝率<50%）
        - 电池容量高的车辆优先选择距离较远的订单
        - 电池容量低的车辆选择距离较近的订单
        """
        vehicle = env.vehicles[vehicle_id]
        vehicle_coords = vehicle['coordinates']
        battery_level = vehicle['battery']
        
        # 过滤掉已分配的订单和无效分配
        candidate_requests = []
        for request in available_requests:
            if (request.request_id not in assigned_requests and 
                valid_assignments.get((vehicle_id, request.request_id), False)):
                candidate_requests.append(request)
        
        if not candidate_requests:
            return None
        
        # 计算所有有效订单的距离
        request_distances = []
        for request in candidate_requests:
            pickup_coords = (
                request.pickup // env.grid_size,
                request.pickup % env.grid_size
            )
            distance = abs(vehicle_coords[0] - pickup_coords[0]) + \
                      abs(vehicle_coords[1] - pickup_coords[1])
            
            if distance <= self.max_service_distance:
                request_distances.append((request, distance))
        
        if not request_distances:
            return None
        
        # 根据电池水平选择策略
        if battery_level > 0.7:
            # 高电量：优先选择距离较远的高价值订单
            request_distances.sort(
                key=lambda x: (-x[1] * 0.7 + x[0].value * 0.3, -x[0].value), 
                reverse=True
            )
        elif battery_level > 0.5:
            # 中等电量：平衡距离和价值
            request_distances.sort(
                key=lambda x: (x[1] * 0.5 - x[0].value * 0.5, x[1])
            )
        else:
            # 低电量：优先选择近距离订单
            request_distances.sort(key=lambda x: x[1])
        
        # 选择最优订单
        return request_distances[0][0]
    
    def get_policy_name(self) -> str:
        """返回策略名称"""
        return f"Heuristic_BatteryThreshold{self.battery_threshold}_MaxDist{self.max_service_distance}"
    
    def get_policy_description(self) -> str:
        """返回策略描述"""
        return f"""
        启发式基准策略:
        - 电池阈值: {self.battery_threshold}
        - 最大服务距离: {self.max_service_distance}
        - 低电量车辆优先充电
        - 高电量车辆优先远距离订单
        - 低电量车辆优先近距离订单
        - 考虑EV拒绝率（>50%不分配）
        """