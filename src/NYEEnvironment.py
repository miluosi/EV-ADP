"""
New York Electric Taxi Environment
基于纽约市真实数据的电动出租车环境
包含真实充电站位置、出租车轨迹和需求数据
"""

import pandas as pd
import numpy as np
import random
import json
import requests
from datetime import datetime, timedelta
import pickle
import os
from typing import List, Dict, Tuple, Optional
from geopy.distance import geodesic
from .Environment import Environment
from .NYCRequest import NYCRequest as Request, NYCRequestGenerator
from .Action import ServiceAction, ChargingAction, IdleAction
from .charging_station import ChargingStation, ChargingStationManager


class NYEEnvironment(Environment):
    """
    New York Electric Taxi Environment
    基于纽约市真实地理坐标和出租车数据的电动出租车环境
    """
    
    def __init__(self, num_vehicles=20, num_stations=50, 
                 manhattan_bounds=None, use_real_data=True, 
                 data_date="2023-01-01", random_seed=None):
        """
        初始化纽约电动出租车环境
        
        Args:
            num_vehicles: 出租车数量
            num_stations: 充电站数量
            manhattan_bounds: 曼哈顿区域边界 [(min_lat, min_lon), (max_lat, max_lon)]
            use_real_data: 是否使用真实数据
            data_date: 数据日期
            random_seed: 随机种子
        """
        # 调用基类构造函数
        super().__init__(
            NUM_LOCATIONS=100,  # 简化为100个位置
            MAX_CAPACITY=4,     # 每车最大乘客数
            EPOCH_LENGTH=1.0,   # 每个epoch 1小时
            NUM_AGENTS=num_vehicles,  # 车辆数量
            START_EPOCH=0.0,    # 开始时间
            STOP_EPOCH=24.0,    # 结束时间(24小时)
            DATA_DIR="data/"    # 数据目录
        )
        
        # 曼哈顿地理边界 (纽约市核心区域)
        if manhattan_bounds is None:
            self.manhattan_bounds = [
                (40.7000, -74.0200),  # 西南角 (min_lat, min_lon)
                (40.8000, -73.9300)   # 东北角 (max_lat, max_lon)
            ]
        else:
            self.manhattan_bounds = manhattan_bounds
            
        self.num_vehicles = num_vehicles
        self.num_stations = num_stations
        self.use_real_data = use_real_data
        self.data_date = data_date
        
        # 环境参数
        self.current_time = datetime(2024, 1, 1, 6, 0)  # 早上6点开始统一使用datetime
        self.episode_length = 1440  # 24小时 (分钟)
        self.battery_consumption_per_km = 0.2  # 每公里消耗20%电池 (kWh/km)
        self.max_battery_capacity = 75  # kWh
        self.min_battery_level = 0.15  # 15%最低电量
        self.charge_rate = 50  # 50kW充电功率
        self.charge_duration = 30  # 充电时长(分钟)
        self.max_trip_distance = 20  # 最大行程距离(公里)
        
        # 奖励参数
        self.base_fare = 2.5  # 起步价
        self.per_km_rate = 1.75  # 每公里费率
        self.time_rate = 0.5  # 每分钟等待费率
        self.charging_penalty = -5.0  # 充电惩罚
        self.rejection_penalty = -2.0  # 拒绝请求惩罚
        self.battery_penalty = -10.0  # 电池耗尽惩罚
        
        # 数据存储
        self.vehicles = {}
        self.requests = []  # 当前可用请求列表
        self.active_requests = {}
        self.completed_requests = []
        self.charging_stations = {}
        self.charging_manager = None
        
        # 真实数据缓存
        self.taxi_data_cache = None
        self.charging_stations_data = None
        self.demand_patterns = None
        
        # 初始化环境
        if random_seed:
            random.seed(random_seed)
            np.random.seed(random_seed)
            
        self._load_real_data()
        self._setup_charging_stations()
        self._setup_vehicles()
        
    def _load_real_data(self):
        """加载纽约市真实数据"""
        print("🔄 Loading New York City real data...")
        
        # 1. 加载充电站数据
        self._load_charging_stations()
        
        # 2. 加载出租车历史数据 (用于需求模式)
        self._load_taxi_demand_patterns()
        
        # 3. 加载地理数据
        self._load_geographic_data()
        
    def _load_charging_stations(self):
        """加载纽约市电动车充电站数据"""
        print("📍 Loading NYC EV charging stations...")
        
        # 纽约市充电站数据 (部分真实坐标)
        # 数据来源: NYC Open Data + PlugShare + ChargePoint
        charging_stations_nyc = [
            # 曼哈顿中城
            {"id": 1, "name": "Times Square Charging", "lat": 40.7580, "lon": -73.9855, "power": 150, "ports": 8},
            {"id": 2, "name": "Central Park South", "lat": 40.7661, "lon": -73.9797, "power": 50, "ports": 4},
            {"id": 3, "name": "Penn Station Hub", "lat": 40.7505, "lon": -73.9934, "power": 100, "ports": 6},
            {"id": 4, "name": "Grand Central Area", "lat": 40.7527, "lon": -73.9772, "power": 75, "ports": 5},
            
            # 下曼哈顿
            {"id": 5, "name": "Wall Street Station", "lat": 40.7074, "lon": -74.0113, "power": 50, "ports": 4},
            {"id": 6, "name": "Brooklyn Bridge", "lat": 40.7061, "lon": -73.9969, "power": 100, "ports": 6},
            {"id": 7, "name": "SoHo Charging", "lat": 40.7230, "lon": -74.0030, "power": 75, "ports": 4},
            {"id": 8, "name": "Chinatown Hub", "lat": 40.7150, "lon": -73.9973, "power": 50, "ports": 3},
            
            # 上曼哈顿
            {"id": 9, "name": "Columbia University", "lat": 40.8075, "lon": -73.9626, "power": 100, "ports": 8},
            {"id": 10, "name": "Harlem Station", "lat": 40.8176, "lon": -73.9482, "power": 75, "ports": 5},
            {"id": 11, "name": "Upper West Side", "lat": 40.7831, "lon": -73.9712, "power": 50, "ports": 4},
            {"id": 12, "name": "Upper East Side", "lat": 40.7794, "lon": -73.9441, "power": 75, "ports": 6},
            
            # 中曼哈顿
            {"id": 13, "name": "Chelsea Market", "lat": 40.7420, "lon": -74.0063, "power": 100, "ports": 7},
            {"id": 14, "name": "Union Square", "lat": 40.7359, "lon": -73.9911, "power": 75, "ports": 5},
            {"id": 15, "name": "Madison Square", "lat": 40.7505, "lon": -73.9934, "power": 50, "ports": 4},
        ]
        
        # 扩展到指定数量的充电站
        if len(charging_stations_nyc) < self.num_stations:
            # 在曼哈顿范围内随机生成额外充电站
            for i in range(len(charging_stations_nyc), self.num_stations):
                lat = np.random.uniform(
                    self.manhattan_bounds[0][0], 
                    self.manhattan_bounds[1][0]
                )
                lon = np.random.uniform(
                    self.manhattan_bounds[0][1], 
                    self.manhattan_bounds[1][1]
                )
                charging_stations_nyc.append({
                    "id": i + 1,
                    "name": f"Generated Station {i+1}",
                    "lat": lat,
                    "lon": lon,
                    "power": np.random.choice([50, 75, 100, 150]),
                    "ports": np.random.randint(2, 9)
                })
        
        self.charging_stations_data = charging_stations_nyc[:self.num_stations]
        print(f"✓ Loaded {len(self.charging_stations_data)} charging stations")
        
    def _load_taxi_demand_patterns(self):
        """加载出租车需求模式数据"""
        print("🚕 Loading taxi demand patterns...")
        
        # 纽约市出租车需求热点 (基于历史数据分析)
        # 数据模式: 工作日vs周末, 不同时段的需求分布
        
        # 热点区域定义 (lat, lon, 需求强度权重)
        demand_hotspots = {
            "financial_district": {"center": (40.7074, -74.0113), "weight": 0.15, "peak_hours": [7, 8, 9, 17, 18, 19]},
            "midtown": {"center": (40.7580, -73.9855), "weight": 0.25, "peak_hours": [11, 12, 13, 14, 20, 21]},
            "upper_east_side": {"center": (40.7794, -73.9441), "weight": 0.12, "peak_hours": [8, 9, 18, 19, 22]},
            "upper_west_side": {"center": (40.7831, -73.9712), "weight": 0.10, "peak_hours": [8, 9, 17, 18, 19]},
            "soho": {"center": (40.7230, -74.0030), "weight": 0.08, "peak_hours": [12, 13, 14, 20, 21, 22]},
            "chelsea": {"center": (40.7420, -74.0063), "weight": 0.10, "peak_hours": [11, 12, 19, 20, 21]},
            "union_square": {"center": (40.7359, -73.9911), "weight": 0.08, "peak_hours": [12, 13, 17, 18]},
            "columbia": {"center": (40.8075, -73.9626), "weight": 0.05, "peak_hours": [8, 9, 16, 17, 18]},
            "jfk_route": {"center": (40.7505, -73.9934), "weight": 0.07, "peak_hours": [5, 6, 7, 22, 23, 0]}
        }
        
        # 时段需求系数 (24小时)
        hourly_demand_multiplier = [
            0.3, 0.2, 0.1, 0.1, 0.2, 0.4, 0.7, 1.0,  # 0-7点
            1.2, 1.1, 0.9, 1.0, 1.3, 1.2, 1.1, 1.0,  # 8-15点  
            1.1, 1.4, 1.5, 1.3, 1.1, 0.9, 0.7, 0.5   # 16-23点
        ]
        
        self.demand_patterns = {
            "hotspots": demand_hotspots,
            "hourly_multiplier": hourly_demand_multiplier,
            "base_requests_per_hour": 50,  # 基础每小时请求数
            "weekend_multiplier": 0.8      # 周末需求系数
        }
        
        print("✓ Loaded demand patterns for NYC")
        
    def _load_geographic_data(self):
        """加载地理数据和路网信息"""
        print("🗺️ Loading geographic data...")
        
        # 曼哈顿平均车速 (km/h) - 基于时段和区域
        self.average_speeds = {
            "peak_hours": 15,      # 高峰期 (7-9, 17-19)
            "normal_hours": 25,    # 正常时段
            "night_hours": 35,     # 夜间 (23-6)
            "weekend": 20          # 周末
        }
        
        # 曼哈顿街道网格系数 (用于距离计算修正)
        self.manhattan_coefficient = 1.3  # 实际行驶距离 = 直线距离 * 1.3
        
        print("✓ Geographic data loaded")
        
    def _setup_charging_stations(self):
        """设置充电站"""
        print("🔌 Setting up charging stations...")
        
        charging_stations = []
        for station_data in self.charging_stations_data:
            station = ChargingStation(
                id=station_data["id"],
                location=station_data["id"],  # 简化为整数位置
                max_capacity=station_data["ports"]
            )
            charging_stations.append(station)
            self.charging_stations[station_data["id"]] = station_data
        
        # 创建充电管理器
        # 充电站管理器
        self.charging_manager = ChargingStationManager()
        
        # 为充电站管理器添加充电站
        for station_data in self.charging_stations_data:
            self.charging_manager.add_station(
                station_id=station_data["id"],
                location=station_data["id"],  # 简化处理，使用ID作为位置
                capacity=station_data.get("ports", 8)
            )
        
        print(f"✓ Setup {len(charging_stations)} charging stations")
        
    def _setup_vehicles(self):
        """设置初始车辆状态"""
        print("🚗 Setting up vehicles...")
        
        for i in range(self.num_vehicles):
            # 随机分布在曼哈顿区域内
            lat = np.random.uniform(
                self.manhattan_bounds[0][0], 
                self.manhattan_bounds[1][0]
            )
            lon = np.random.uniform(
                self.manhattan_bounds[0][1], 
                self.manhattan_bounds[1][1]
            )
            
            self.vehicles[i] = {
                'id': i,
                'location': (lat, lon),  # (纬度, 经度)
                'battery_kwh': np.random.uniform(30, 70),  # kWh
                'battery_percentage': None,  # 会自动计算
                'charging_station': None,
                'charging_time_left': 0,
                'total_distance': 0.0,
                'charging_count': 0,
                'assigned_request': None,
                'passenger_onboard': None,
                'service_earnings': 0.0,
                'rejected_requests': 0,
                'status': 'idle',  # idle, driving_to_pickup, with_passenger, charging
                'last_update_time': self.current_time,
                'trip_history': [],
                'destination': None,
                'eta_minutes': 0
            }
            
            # 计算电池百分比
            self.vehicles[i]['battery_percentage'] = (
                self.vehicles[i]['battery_kwh'] / self.max_battery_capacity
            )
        
        print(f"✓ Setup {len(self.vehicles)} vehicles")
        
    def _calculate_distance_km(self, loc1: Tuple[float, float], 
                             loc2: Tuple[float, float]) -> float:
        """计算两点间的实际行驶距离(公里)"""
        # 使用geodesic计算地球表面距离
        straight_distance = geodesic(loc1, loc2).kilometers
        # 考虑曼哈顿街道网格，实际距离更长
        actual_distance = straight_distance * self.manhattan_coefficient
        return actual_distance
        
    def _calculate_travel_time(self, distance_km: float, current_hour: int = None) -> int:
        """计算行驶时间(分钟)"""
        if current_hour is None:
            current_hour = self.current_time.hour  # 使用datetime的hour属性
            
        # 根据时段确定平均速度
        if current_hour in [7, 8, 9, 17, 18, 19]:  # 高峰期
            speed = self.average_speeds["peak_hours"]
        elif current_hour in [23, 0, 1, 2, 3, 4, 5, 6]:  # 夜间
            speed = self.average_speeds["night_hours"]
        else:
            speed = self.average_speeds["normal_hours"]
            
        travel_time = (distance_km / speed) * 60  # 转换为分钟
        return max(1, int(travel_time))
        
    def _generate_realistic_requests(self) -> List[Request]:
        """基于真实需求模式生成请求"""
        current_hour = self.current_time.hour  # 使用datetime的hour属性
        current_day = self.current_time.weekday()  # 使用datetime的weekday() (0=Monday, 6=Sunday)
        is_weekend = current_day >= 5  # 周六日为周末
        
        # 基础请求生成率
        base_rate = self.demand_patterns["base_requests_per_hour"]
        hourly_multiplier = self.demand_patterns["hourly_multiplier"][current_hour]
        weekend_multiplier = self.demand_patterns["weekend_multiplier"] if is_weekend else 1.0
        
        # 计算该分钟应生成的请求数
        requests_per_minute = (base_rate * hourly_multiplier * weekend_multiplier) / 60
        num_requests = np.random.poisson(requests_per_minute)
        
        generated_requests = []
        
        for _ in range(num_requests):
            # 根据热点选择起点
            hotspot_name = self._select_demand_hotspot(current_hour)
            pickup_location = self._generate_location_near_hotspot(hotspot_name)
            
            # 生成目的地 (在曼哈顿范围内)
            dropoff_location = self._generate_random_location()
            
            # 计算距离和价值
            distance = self._calculate_distance_km(pickup_location, dropoff_location)
            
            # 过滤过远的请求
            if distance > self.max_trip_distance:
                continue
                
            # 计算请求价值
            trip_value = self._calculate_trip_value(distance)
            
            # 创建请求 - 使用基类Request的构造函数
            # 使用NYCRequest创建请求
            request = Request(
                request_id=f"nye_{self.current_time.strftime('%Y%m%d_%H%M%S')}_{len(generated_requests)}",
                pickup_location=pickup_location,
                dropoff_location=dropoff_location,
                request_time=self.current_time,
                trip_distance=distance,
                base_value=trip_value,
                passenger_count=np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02])
            )
            
            generated_requests.append(request)
            
        return generated_requests
    
    def _generate_requests(self) -> List[Request]:
        """生成请求 - 测试用别名方法"""
        return self._generate_realistic_requests()
        
    def _select_demand_hotspot(self, current_hour: int) -> str:
        """根据当前时间选择需求热点"""
        hotspots = self.demand_patterns["hotspots"]
        
        # 计算每个热点在当前时段的权重
        weighted_hotspots = []
        for name, data in hotspots.items():
            weight = data["weight"]
            # 如果是该热点的高峰时段，权重翻倍
            if current_hour in data["peak_hours"]:
                weight *= 2
            weighted_hotspots.append((name, weight))
        
        # 随机选择热点
        names, weights = zip(*weighted_hotspots)
        selected = np.random.choice(names, p=np.array(weights)/sum(weights))
        return selected
        
    def _generate_location_near_hotspot(self, hotspot_name: str) -> Tuple[float, float]:
        """在热点附近生成位置"""
        hotspot_center = self.demand_patterns["hotspots"][hotspot_name]["center"]
        
        # 在热点中心500米范围内随机生成
        lat_offset = np.random.normal(0, 0.0045)  # 约500米纬度偏移
        lon_offset = np.random.normal(0, 0.0055)  # 约500米经度偏移
        
        lat = np.clip(
            hotspot_center[0] + lat_offset,
            self.manhattan_bounds[0][0],
            self.manhattan_bounds[1][0]
        )
        lon = np.clip(
            hotspot_center[1] + lon_offset,
            self.manhattan_bounds[0][1],
            self.manhattan_bounds[1][1]
        )
        
        return (lat, lon)
        
    def _generate_random_location(self) -> Tuple[float, float]:
        """在曼哈顿范围内生成随机位置"""
        lat = np.random.uniform(
            self.manhattan_bounds[0][0],
            self.manhattan_bounds[1][0]
        )
        lon = np.random.uniform(
            self.manhattan_bounds[0][1],
            self.manhattan_bounds[1][1]
        )
        return (lat, lon)
        
    def _calculate_trip_value(self, distance_km: float) -> float:
        """计算行程价值"""
        # 纽约出租车费率 (简化版本)
        base_fare = self.base_fare
        distance_fare = distance_km * self.per_km_rate
        
        # 添加随机波动 (需求高峰期涨价等)
        surge_multiplier = np.random.uniform(1.0, 1.5)
        
        total_fare = (base_fare + distance_fare) * surge_multiplier
        return round(total_fare, 2)
        
    def _execute_action(self, vehicle_id: int, action) -> Tuple[float, float]:
        """执行车辆动作 - 基于真实地理坐标"""
        vehicle = self.vehicles[vehicle_id]
        reward = 0.0
        
        if isinstance(action, ServiceAction):
            reward = self._execute_service_action(vehicle_id, action)
            
        elif isinstance(action, ChargingAction):
            reward = self._execute_charging_action(vehicle_id, action)
            
        elif isinstance(action, IdleAction):
            reward = self._execute_idle_action(vehicle_id, action)
            
        # 更新车辆状态
        vehicle['last_update_time'] = self.current_time
        
        # 更新电池百分比
        vehicle['battery_percentage'] = vehicle['battery_kwh'] / self.max_battery_capacity
        
        return reward, reward
        
    def _execute_service_action(self, vehicle_id: int, action: ServiceAction) -> float:
        """执行服务动作"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['assigned_request'] is None:
            # 尝试分配请求
            if hasattr(action, 'request_id') and action.request_id in self.active_requests:
                request = self.active_requests[action.request_id]
                
                # 检查车辆是否能够完成这个请求
                pickup_distance = self._calculate_distance_km(
                    vehicle['location'], request.pickup_location
                )
                total_distance = pickup_distance + request.trip_distance
                required_battery = total_distance * self.battery_consumption_per_km
                
                if vehicle['battery_kwh'] < required_battery:
                    # 电量不足，拒绝请求
                    vehicle['rejected_requests'] += 1
                    return self.rejection_penalty
                
                # 分配请求
                vehicle['assigned_request'] = action.request_id
                vehicle['status'] = 'driving_to_pickup'
                vehicle['destination'] = request.pickup_location
                vehicle['eta_minutes'] = self._calculate_travel_time(pickup_distance)
                
                return 0.5  # 成功分配的小奖励
            else:
                return self.rejection_penalty
                
        else:
            # 执行已分配的请求
            return self._execute_assigned_service(vehicle_id)
            
    def _execute_assigned_service(self, vehicle_id: int) -> float:
        """执行已分配的服务"""
        vehicle = self.vehicles[vehicle_id]
        request_id = vehicle['assigned_request']
        
        if request_id not in self.active_requests:
            # 请求已过期
            vehicle['assigned_request'] = None
            vehicle['status'] = 'idle'
            vehicle['destination'] = None
            return 0
            
        request = self.active_requests[request_id]
        
        if vehicle['status'] == 'driving_to_pickup':
            # 前往接客
            return self._drive_to_pickup(vehicle_id, request)
            
        elif vehicle['status'] == 'with_passenger':
            # 载客前往目的地
            return self._drive_to_dropoff(vehicle_id, request)
            
        return 0
        
    def _drive_to_pickup(self, vehicle_id: int, request: Request) -> float:
        """驾驶到接客点"""
        vehicle = self.vehicles[vehicle_id]
        
        # 计算到达接客点需要的时间和距离
        distance_to_pickup = self._calculate_distance_km(
            vehicle['location'], request.pickup_location
        )
        
        # 移动车辆 (简化：假设1分钟能行驶的距离)
        max_distance_per_minute = self.average_speeds["normal_hours"] / 60  # km/min
        
        if distance_to_pickup <= max_distance_per_minute:
            # 到达接客点
            vehicle['location'] = request.pickup_location
            vehicle['status'] = 'with_passenger'
            vehicle['destination'] = request.dropoff_location
            
            # 消耗电池
            battery_consumed = distance_to_pickup * self.battery_consumption_per_km
            vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
            vehicle['total_distance'] += distance_to_pickup
            
            # 记录行程
            vehicle['trip_history'].append({
                'type': 'pickup',
                'time': self.current_time,
                'location': request.pickup_location,
                'distance': distance_to_pickup
            })
            
            return 1.0  # 成功接客奖励
        else:
            # 继续前往接客点
            # 沿直线方向移动
            new_location = self._move_towards_destination(
                vehicle['location'], request.pickup_location, max_distance_per_minute
            )
            
            vehicle['location'] = new_location
            
            # 消耗电池
            battery_consumed = max_distance_per_minute * self.battery_consumption_per_km
            vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
            vehicle['total_distance'] += max_distance_per_minute
            
            return -0.1  # 行驶成本
            
    def _drive_to_dropoff(self, vehicle_id: int, request: Request) -> float:
        """载客到目的地"""
        vehicle = self.vehicles[vehicle_id]
        
        # 计算到目的地的距离
        distance_to_dropoff = self._calculate_distance_km(
            vehicle['location'], request.dropoff_location
        )
        
        # 移动车辆
        max_distance_per_minute = self.average_speeds["normal_hours"] / 60  # km/min
        
        if distance_to_dropoff <= max_distance_per_minute:
            # 到达目的地，完成订单
            vehicle['location'] = request.dropoff_location
            vehicle['status'] = 'idle'
            vehicle['assigned_request'] = None
            vehicle['destination'] = None
            
            # 消耗电池
            battery_consumed = distance_to_dropoff * self.battery_consumption_per_km
            vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
            vehicle['total_distance'] += distance_to_dropoff
            
            # 计算收入
            earnings = request.base_value
            vehicle['service_earnings'] += earnings
            
            # 记录完成的订单
            self.completed_requests.append(request)
            if request.request_id in self.active_requests:
                del self.active_requests[request.request_id]
            
            # 记录行程
            vehicle['trip_history'].append({
                'type': 'dropoff',
                'time': self.current_time,
                'location': request.dropoff_location,
                'distance': distance_to_dropoff,
                'earnings': earnings
            })
            
            return earnings  # 获得订单收入
        else:
            # 继续前往目的地
            new_location = self._move_towards_destination(
                vehicle['location'], request.dropoff_location, max_distance_per_minute
            )
            
            vehicle['location'] = new_location
            
            # 消耗电池
            battery_consumed = max_distance_per_minute * self.battery_consumption_per_km
            vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
            vehicle['total_distance'] += max_distance_per_minute
            
            return -0.05  # 较小的行驶成本（载客中）
            
    def _move_towards_destination(self, current_loc: Tuple[float, float], 
                                 destination: Tuple[float, float], 
                                 max_distance: float) -> Tuple[float, float]:
        """向目的地移动指定距离"""
        # 计算方向向量
        lat_diff = destination[0] - current_loc[0]
        lon_diff = destination[1] - current_loc[1]
        
        # 计算当前距离
        current_distance = geodesic(current_loc, destination).kilometers
        
        if current_distance <= max_distance:
            return destination
        
        # 计算移动比例
        move_ratio = max_distance / current_distance
        
        # 新位置
        new_lat = current_loc[0] + lat_diff * move_ratio
        new_lon = current_loc[1] + lon_diff * move_ratio
        
        return (new_lat, new_lon)
        
    def _execute_charging_action(self, vehicle_id: int, action: ChargingAction) -> float:
        """执行充电动作"""
        vehicle = self.vehicles[vehicle_id]
        
        if vehicle['charging_station'] is None:
            # 寻找最近的充电站
            station = self._find_nearest_available_station(vehicle['location'])
            if station is None:
                return self.charging_penalty  # 没有可用充电站
            
            # 移动到充电站
            distance_to_station = self._calculate_distance_km(
                vehicle['location'], 
                (station["lat"], station["lon"])
            )
            
            max_distance_per_minute = self.average_speeds["normal_hours"] / 60
            
            if distance_to_station <= max_distance_per_minute:
                # 到达充电站
                vehicle['location'] = (station["lat"], station["lon"])
                vehicle['charging_station'] = station["id"]
                vehicle['charging_time_left'] = self.charge_duration
                vehicle['status'] = 'charging'
                
                # 开始充电
                station.start_charging(str(vehicle_id))
                vehicle['charging_count'] += 1
                
                return self.charging_penalty  # 充电启动成本
            else:
                # 继续前往充电站
                new_location = self._move_towards_destination(
                    vehicle['location'], 
                    (station["lat"], station["lon"]),
                    max_distance_per_minute
                )
                vehicle['location'] = new_location
                
                # 消耗电池
                battery_consumed = max_distance_per_minute * self.battery_consumption_per_km
                vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
                
                return -0.2  # 前往充电站的成本
        else:
            # 正在充电
            vehicle['charging_time_left'] = max(0, vehicle['charging_time_left'] - 1)
            
            if vehicle['charging_time_left'] <= 0:
                # 充电完成
                station = self.charging_manager.stations[vehicle['charging_station']]
                station.stop_charging(str(vehicle_id))
                
                vehicle['charging_station'] = None
                vehicle['status'] = 'idle'
                vehicle['battery_kwh'] = self.max_battery_capacity  # 充满电
                
                return 2.0  # 充电完成奖励
            else:
                # 继续充电
                charge_rate_per_minute = self.charge_rate / 60  # kW per minute
                battery_charged = min(
                    charge_rate_per_minute,
                    self.max_battery_capacity - vehicle['battery_kwh']
                )
                vehicle['battery_kwh'] += battery_charged
                
                return 0.1  # 充电进度奖励
                
    def _execute_idle_action(self, vehicle_id: int, action: IdleAction) -> float:
        """执行空闲动作（巡游或等待）"""
        vehicle = self.vehicles[vehicle_id]
        
        if hasattr(action, 'target_coords') and action.target_coords:
            # 移动到指定位置巡游
            target_location = action.target_coords
            distance_to_target = self._calculate_distance_km(
                vehicle['location'], target_location
            )
            
            max_distance_per_minute = self.average_speeds["normal_hours"] / 60
            
            if distance_to_target <= max_distance_per_minute:
                # 到达目标位置
                vehicle['location'] = target_location
                vehicle['status'] = 'idle'
                
                # 消耗电池
                battery_consumed = distance_to_target * self.battery_consumption_per_km
                vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
                
                return -0.1  # 巡游成本
            else:
                # 继续移动
                new_location = self._move_towards_destination(
                    vehicle['location'], target_location, max_distance_per_minute
                )
                vehicle['location'] = new_location
                
                # 消耗电池
                battery_consumed = max_distance_per_minute * self.battery_consumption_per_km
                vehicle['battery_kwh'] = max(0, vehicle['battery_kwh'] - battery_consumed)
                
                return -0.15  # 巡游移动成本
        else:
            # 原地等待
            vehicle['status'] = 'idle'
            return -0.05  # 等待的机会成本
            
    def _find_nearest_available_station(self, location: Tuple[float, float]):
        """寻找最近的可用充电站"""
        min_distance = float('inf')
        nearest_station = None
        
        # 使用charging_stations_data而不是charging_manager.stations
        for station_data in self.charging_stations_data:
            distance = self._calculate_distance_km(
                location, (station_data["lat"], station_data["lon"])
            )
            if distance < min_distance:
                min_distance = distance
                nearest_station = station_data
                
        return nearest_station
        
    def _update_environment(self):
        """更新环境状态"""
        self.current_time += timedelta(minutes=1)  # 每次更新增加1分钟
        
        # 1. 生成新请求
        new_requests = self._generate_realistic_requests()
        for request in new_requests:
            self.active_requests[request.request_id] = request
            
        # 2. 移除过期请求
        expired_requests = []
        for request_id, request in self.active_requests.items():
            if self.current_time - request.request_time > request.max_wait_time:
                expired_requests.append(request_id)
                
        for request_id in expired_requests:
            del self.active_requests[request_id]
            
        # 3. 更新充电站状态
        self.charging_manager.update_all_stations()
        
        # 4. 检查电池耗尽的车辆
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['battery_kwh'] <= 0:
                vehicle['status'] = 'stranded'
                # 紧急救援 (简化处理)
                vehicle['battery_kwh'] = 10  # 紧急充电
                
        # 5. 更新车辆ETA
        for vehicle_id, vehicle in self.vehicles.items():
            if vehicle['eta_minutes'] > 0:
                vehicle['eta_minutes'] -= 1
                
    def get_vehicle_state(self, vehicle_id: int) -> np.ndarray:
        """获取车辆状态向量"""
        vehicle = self.vehicles[vehicle_id]
        
        # 规范化地理坐标到0-1范围
        lat_norm = (vehicle['location'][0] - self.manhattan_bounds[0][0]) / \
                   (self.manhattan_bounds[1][0] - self.manhattan_bounds[0][0])
        lon_norm = (vehicle['location'][1] - self.manhattan_bounds[0][1]) / \
                   (self.manhattan_bounds[1][1] - self.manhattan_bounds[0][1])
                   
        state = [
            lat_norm,                                    # 标准化纬度
            lon_norm,                                    # 标准化经度
            vehicle['battery_percentage'],               # 电池百分比
            float(vehicle['status'] == 'charging'),      # 是否在充电
            float(vehicle['status'] == 'with_passenger'), # 是否载客
            len(self.active_requests) / 100,            # 标准化活跃请求数
            (self.current_time.hour * 60 + self.current_time.minute) / 1440,  # 标准化时间 (一天内)
            vehicle['service_earnings'] / 1000,         # 标准化收入
        ]
        
        return np.array(state, dtype=np.float32)
        
    def reset(self):
        """重置环境"""
        self.current_time = datetime(2024, 1, 1, 6, 0)  # 重置为早上6点
        self.active_requests.clear()
        self.completed_requests.clear()
        
        # 重新初始化车辆
        self._setup_vehicles()
        
        # 重置充电站
        for station in self.charging_manager.stations.values():
            station.reset()
            
        return {i: self.get_vehicle_state(i) for i in range(self.num_vehicles)}
        
    def get_episode_stats(self) -> Dict:
        """获取episode统计信息"""
        total_earnings = sum(v['service_earnings'] for v in self.vehicles.values())
        total_distance = sum(v['total_distance'] for v in self.vehicles.values())
        total_completed = len(self.completed_requests)
        total_rejected = sum(v['rejected_requests'] for v in self.vehicles.values())
        avg_battery = np.mean([v['battery_percentage'] for v in self.vehicles.values()])
        
        return {
            'total_earnings': total_earnings,
            'total_distance': total_distance,
            'completed_requests': total_completed,
            'rejected_requests': total_rejected,
            'average_battery': avg_battery,
            'active_requests': len(self.active_requests),
            'vehicles_charging': len([v for v in self.vehicles.values() 
                                    if v['status'] == 'charging']),
            'utilization_rate': total_completed / max(1, total_completed + total_rejected)
        }
        
    def save_episode_data(self, filepath: str):
        """保存episode数据"""
        episode_data = {
            'vehicles': self.vehicles,
            'completed_requests': [req.__dict__ for req in self.completed_requests],
            'stats': self.get_episode_stats(),
            'charging_stations': self.charging_stations_data
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(episode_data, f)
            
    def load_episode_data(self, filepath: str):
        """加载episode数据"""
        with open(filepath, 'rb') as f:
            episode_data = pickle.load(f)
            
        return episode_data
    
    # 实现基类的抽象方法
    def initialise_environment(self):
        """初始化环境 - 实现基类抽象方法"""
        # 重置环境状态
        self.current_time = datetime(2024, 1, 1, 6, 0)  # 早上6点开始
        self.current_step = 0
        self.total_requests = 0
        self.completed_requests = 0
        self.requests = []
        
        # 重置车辆状态
        self._setup_vehicles()
        
        print("🔄 NYC Environment initialized")
    
    def get_request_batch(self):
        """获取当前时间步的请求批次 - 实现基类抽象方法"""
        return self.requests
    
    def get_travel_time(self, source, destination):
        """
        计算两点间的旅行时间 - 实现基类抽象方法
        
        Args:
            source: 起点坐标 (lat, lon)
            destination: 终点坐标 (lat, lon)
            
        Returns:
            float: 旅行时间（分钟）
        """
        # 计算距离
        distance = geodesic(source, destination).kilometers
        
        # 根据距离和平均速度计算时间
        # 纽约市平均车速约15-20 km/h (考虑交通)
        avg_speed_kmh = np.random.normal(18.5, 3.0)  # 加入随机性
        avg_speed_kmh = max(10, min(30, avg_speed_kmh))  # 限制在合理范围
        
        travel_time_hours = distance / avg_speed_kmh
        travel_time_minutes = travel_time_hours * 60
        
        return max(1, travel_time_minutes)  # 最少1分钟
    
    def get_next_location(self, source, destination):
        """
        获取从源点到目标点的下一个位置 - 实现基类抽象方法
        
        Args:
            source: 起点坐标 (lat, lon) 
            destination: 目标点坐标 (lat, lon)
            
        Returns:
            tuple: 下一个位置的坐标 (lat, lon)
        """
        # 简化的直线移动（实际可以用路径规划算法）
        src_lat, src_lon = source
        dest_lat, dest_lon = destination
        
        # 计算总距离
        total_distance = geodesic(source, destination).kilometers
        
        # 如果距离很小，直接到达目标
        if total_distance < 0.5:  # 小于500米
            return destination
            
        # 每步移动距离（假设每分钟移动约300米）
        step_distance_km = 0.3  # 300米
        
        # 计算移动比例
        move_ratio = min(1.0, step_distance_km / total_distance)
        
        # 计算下一个位置
        next_lat = src_lat + (dest_lat - src_lat) * move_ratio
        next_lon = src_lon + (dest_lon - src_lon) * move_ratio
        
        return (next_lat, next_lon)
    
    def get_initial_states(self, num_agents, is_training):
        """
        获取初始状态 - 实现基类抽象方法
        
        Args:
            num_agents: 代理数量
            is_training: 是否为训练模式
            
        Returns:
            list: 初始状态列表
        """
        states = []
        
        for i in range(min(num_agents, self.num_vehicles)):
            vehicle = self.vehicles[i]
            
            # 创建状态字典
            state = {
                'vehicle_id': i,
                'location': vehicle['location'],
                'battery_level': vehicle['battery_kwh'] / 75.0,  # 标准化到0-1
                'status': vehicle['status'],
                'has_passenger': vehicle['passenger_onboard'] is not None,
                'assigned_request': vehicle['assigned_request'],
                'time_step': self.current_step,
                'available_requests': len(self.requests),
                'nearest_charging_station': self._find_nearest_charging_station(vehicle['location'])
            }
            
            states.append(state)
            
        return states
    
    def _find_nearest_charging_station(self, location):
        """
        找到最近的充电站
        
        Args:
            location: 车辆位置 (lat, lon)
            
        Returns:
            dict: 最近充电站信息
        """
        if not self.charging_stations:
            return None
            
        min_distance = float('inf')
        nearest_station = None
        
        for station_id, station_data in self.charging_stations.items():
            station_location = (station_data["lat"], station_data["lon"])
            distance = geodesic(location, station_location).kilometers
            
            if distance < min_distance:
                min_distance = distance
                nearest_station = {
                    'id': station_id,
                    'location': station_location,
                    'distance': distance,
                    'power': station_data.get("power", 50),
                    'name': station_data.get("name", f"Station {station_id}")
                }
                
        return nearest_station