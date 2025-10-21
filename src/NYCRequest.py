"""
NYC Request Class for NYC Electric Taxi Environment
纽约市电动出租车请求类
基于真实NYC数据结构设计，与data/download_data.py集成
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, Optional, Dict, Any, Union
from geopy.distance import geodesic
import os
import sys

# 添加根目录到path以便导入其他模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 尝试导入数据下载模块
try:
    from data.download_data import download_data, describe_data
    DATA_DOWNLOAD_AVAILABLE = True
except ImportError:
    DATA_DOWNLOAD_AVAILABLE = False
    print("Warning: data.download_data module not available")


class NYCRequest:
    """
    纽约市出租车请求类
    基于NYC Yellow Taxi数据格式设计，包含地理坐标和真实属性
    兼容基础Request类，同时支持NYC特有功能
    """
    
    # 类常量
    MAX_PICKUP_DELAY = timedelta(minutes=20)    # 最大接客延迟
    MAX_DROPOFF_DELAY = timedelta(minutes=50)   # 最大送达延迟
    MIN_TRIP_DISTANCE = 0.1   # 最小行程距离(km)
    MAX_TRIP_DISTANCE = 50.0  # 最大行程距离(km)
    
    # 费率参数 (基于NYC真实费率)
    BASE_FARE = 2.50         # 起步价
    PER_MILE_RATE = 2.50     # 每英里费率
    PER_MINUTE_RATE = 0.50   # 每分钟时间费率
    RUSH_HOUR_SURCHARGE = 1.0  # 高峰期附加费
    
    # 曼哈顿地理边界
    MANHATTAN_BOUNDS = {
        'min_lat': 40.7000,
        'max_lat': 40.8800,
        'min_lon': -74.0200,
        'max_lon': -73.9300
    }
    
    def __init__(self,
                 request_id: str,
                 pickup_location: Tuple[float, float],
                 dropoff_location: Tuple[float, float], 
                 request_time: Union[datetime, int],
                 passenger_count: int = 1,
                 trip_type: str = "standard",
                 payment_type: str = "credit_card",
                 rate_code: int = 1,
                 store_and_fwd_flag: str = "N",
                 max_wait_time: Optional[int] = None,
                 trip_distance: Optional[float] = None,
                 estimated_duration: Optional[int] = None,
                 base_value: Optional[float] = None,
                 **kwargs):
        """
        初始化NYC请求
        
        Args:
            request_id: 请求唯一标识符
            pickup_location: 上车地点 (纬度, 经度)
            dropoff_location: 下车地点 (纬度, 经度)
            request_time: 请求时间 (datetime对象或分钟数)
            passenger_count: 乘客数量
            trip_type: 行程类型 ("standard", "airport", "premium")
            payment_type: 支付方式
            rate_code: 费率代码
            store_and_fwd_flag: 存储转发标志
            max_wait_time: 最大等待时间(分钟，可选)
            trip_distance: 行程距离(可选，自动计算)
            estimated_duration: 预计时长(分钟，可选)
            base_value: 基础价值(可选，自动计算)
            **kwargs: 其他额外属性
        """
        # 基本信息
        self.request_id = request_id
        self.pickup_location = pickup_location  # (lat, lon)
        self.dropoff_location = dropoff_location  # (lat, lon)
        
        # 处理时间格式 (支持datetime和int)
        if isinstance(request_time, datetime):
            self.request_time = request_time
            self.request_time_minutes = int((request_time.hour * 60 + request_time.minute))
        else:
            self.request_time_minutes = request_time
            # 假设从某个基准时间开始的分钟数
            base_time = datetime(2024, 1, 1, 0, 0)
            self.request_time = base_time + timedelta(minutes=request_time)
        
        self.passenger_count = max(1, min(6, passenger_count))  # 限制1-6人
        
        # 行程信息
        self.trip_type = trip_type
        self.payment_type = payment_type
        self.rate_code = rate_code
        self.store_and_fwd_flag = store_and_fwd_flag
        
        # 计算或使用提供的距离和时间
        self.trip_distance = trip_distance if trip_distance is not None else self._calculate_trip_distance()
        self.estimated_duration = estimated_duration if estimated_duration is not None else self._calculate_estimated_duration()
        
        # NYC特有属性 (必须在计算费用之前设置)
        self.tip_amount = 0.0
        self.tolls_amount = 0.0
        self.improvement_surcharge = 0.30
        self.congestion_surcharge = 0.0
        
        # 计算费用和价值
        self.base_value = base_value if base_value is not None else self._calculate_fare()
        self.fare_amount = self.base_value
        self.total_amount = self._calculate_total_amount()
        
        # 时间限制
        self.max_wait_time = max_wait_time if max_wait_time is not None else self.MAX_PICKUP_DELAY.seconds // 60
        self.pickup_deadline = self.request_time + timedelta(minutes=self.max_wait_time)
        self.dropoff_deadline = (self.request_time + timedelta(minutes=self.estimated_duration) + 
                               self.MAX_DROPOFF_DELAY)
        
        # 请求状态
        self.status = "pending"  # pending, assigned, picked_up, completed, cancelled, expired
        self.assigned_vehicle_id = None
        self.assigned_vehicle = None  # 向后兼容
        self.pickup_time = None
        self.dropoff_time = None
        self.actual_fare = None
        
        # 兼容性属性 (与基类Request兼容)
        self.pickup = self._location_to_grid_id(pickup_location)
        self.dropoff = self._location_to_grid_id(dropoff_location)
        self.value = self.total_amount
        self.final_value = self.total_amount
        
        # 地理属性
        self.pickup_borough = self._get_borough(pickup_location)
        self.dropoff_borough = self._get_borough(dropoff_location)
        
        # 额外属性
        for key, value in kwargs.items():
            setattr(self, key, value)
        
    def _calculate_trip_distance(self) -> float:
        """计算行程距离 (公里)"""
        distance_km = geodesic(self.pickup_location, self.dropoff_location).kilometers
        # 考虑曼哈顿路网系数 (实际道路距离比直线距离长约30%)
        manhattan_factor = 1.3
        actual_distance = distance_km * manhattan_factor
        
        # 限制在合理范围内
        return max(self.MIN_TRIP_DISTANCE, min(self.MAX_TRIP_DISTANCE, actual_distance))
    
    def _calculate_estimated_duration(self) -> int:
        """计算预计行程时间 (分钟)"""
        # 基于时段确定平均速度
        hour = self.request_time.hour
        
        if hour in [7, 8, 9, 17, 18, 19]:  # 高峰期
            avg_speed_kmh = 12.0  # 12 km/h
        elif hour in [22, 23, 0, 1, 2, 3, 4, 5]:  # 深夜
            avg_speed_kmh = 25.0  # 25 km/h
        else:  # 正常时段
            avg_speed_kmh = 18.0  # 18 km/h
            
        # 添加随机波动 (±20%)
        speed_variation = np.random.uniform(0.8, 1.2)
        actual_speed = avg_speed_kmh * speed_variation
        
        # 计算时间 (分钟)
        duration_minutes = (self.trip_distance / actual_speed) * 60
        
        # 添加停车、等灯等额外时间
        extra_time = max(2, self.trip_distance * 0.5)  # 每公里额外0.5分钟
        
        total_minutes = duration_minutes + extra_time
        return max(1, int(total_minutes))
    
    def _calculate_fare(self) -> float:
        """计算基本费用"""
        # 基本费用结构
        fare = self.BASE_FARE
        
        # 距离费用 (转换为英里)
        distance_miles = self.trip_distance * 0.621371  # km to miles
        fare += distance_miles * self.PER_MILE_RATE
        
        # 时间费用
        duration_minutes = self.estimated_duration
        fare += duration_minutes * self.PER_MINUTE_RATE
        
        # 高峰期附加费
        hour = self.request_time.hour
        if hour in [16, 17, 18, 19, 20] and self.request_time.weekday() < 5:  # 工作日晚高峰
            fare += self.RUSH_HOUR_SURCHARGE
        elif hour in [7, 8, 9] and self.request_time.weekday() < 5:  # 工作日早高峰
            fare += self.RUSH_HOUR_SURCHARGE
            
        # 机场附加费
        if self.trip_type == "airport":
            fare += 5.00
        elif self.trip_type == "premium":
            fare *= 1.5  # 高端服务加价50%
            
        return round(fare, 2)
    
    def _calculate_total_amount(self) -> float:
        """计算总费用"""
        total = self.fare_amount
        total += self.improvement_surcharge  # 改善附加费
        
        # 拥堵费 (曼哈顿南部工作日收费)
        if (self.pickup_location[0] < 40.7500 and  # 南曼哈顿
            self.request_time.weekday() < 5 and     # 工作日
            6 <= self.request_time.hour <= 20):    # 6AM-8PM
            self.congestion_surcharge = 2.50
            total += self.congestion_surcharge
            
        return round(total, 2)
    
    def _location_to_grid_id(self, location: Tuple[float, float]) -> int:
        """将地理坐标转换为网格ID (兼容性)"""
        lat, lon = location
        # 简单的网格映射
        lat_grid = int((lat - self.MANHATTAN_BOUNDS['min_lat']) * 1000) % 100
        lon_grid = int((lon - self.MANHATTAN_BOUNDS['min_lon']) * 1000) % 100
        return lat_grid * 100 + lon_grid
    
    def _get_borough(self, location: Tuple[float, float]) -> str:
        """根据坐标判断所属区域"""
        lat, lon = location
        
        # 简化的纽约区域判断
        if (40.7000 <= lat <= 40.8800 and 
            -74.0200 <= lon <= -73.9300):
            return "Manhattan"
        elif (40.5700 <= lat <= 40.7000 and
              -74.0500 <= lon <= -73.7000):
            return "Brooklyn"
        elif (40.7500 <= lat <= 40.9500 and
              -73.9300 <= lon <= -73.7650):
            return "Queens"  
        elif (40.7800 <= lat <= 40.9200 and
              -73.9300 <= lon <= -73.8650):
            return "Bronx"
        else:
            return "Other"
    
    def update_status(self, new_status: str, vehicle_id: Optional[int] = None):
        """更新请求状态"""
        old_status = self.status
        self.status = new_status
        
        if new_status == "assigned" and vehicle_id is not None:
            self.assigned_vehicle_id = vehicle_id
            self.assigned_vehicle = vehicle_id  # 向后兼容
        elif new_status == "picked_up":
            self.pickup_time = datetime.now()
        elif new_status == "completed":
            self.dropoff_time = datetime.now()
            
    def assign_to_vehicle(self, vehicle_id: int):
        """分配给车辆 (兼容性方法)"""
        self.update_status("assigned", vehicle_id)
        
    def set_pickup(self, pickup_time: Union[int, datetime]):
        """乘客上车 (兼容性方法)"""
        if isinstance(pickup_time, int):
            # 假设是分钟数
            base_time = datetime(2024, 1, 1, 0, 0)
            self.pickup_time = base_time + timedelta(minutes=pickup_time)
        else:
            self.pickup_time = pickup_time
        self.status = "picked_up"
        
    def complete(self, dropoff_time: Union[int, datetime], actual_fare: float = None):
        """完成行程 (兼容性方法)"""
        if isinstance(dropoff_time, int):
            # 假设是分钟数
            base_time = datetime(2024, 1, 1, 0, 0)
            self.dropoff_time = base_time + timedelta(minutes=dropoff_time)
        else:
            self.dropoff_time = dropoff_time
            
        self.actual_fare = actual_fare if actual_fare is not None else self.total_amount
        self.status = "completed"
        
    def expire(self):
        """请求过期"""
        self.status = "expired"
        
    def is_expired(self, current_time: Union[int, datetime]) -> bool:
        """检查是否过期"""
        if isinstance(current_time, int):
            # 分钟格式
            return (current_time - self.request_time_minutes) > self.max_wait_time
        else:
            # datetime格式
            return current_time > self.pickup_deadline
        
    def calculate_wait_time(self, current_time: Union[int, datetime]) -> Union[int, timedelta]:
        """计算等待时间"""
        if isinstance(current_time, int):
            if self.pickup_time:
                pickup_minutes = int((self.pickup_time.hour * 60 + self.pickup_time.minute))
                return pickup_minutes - self.request_time_minutes
            else:
                return current_time - self.request_time_minutes
        else:
            return current_time - self.request_time
            
    def get_wait_time(self, current_time: int) -> int:
        """获取当前等待时间 (兼容性方法)"""
        wait_time = self.calculate_wait_time(current_time)
        return wait_time if isinstance(wait_time, int) else int(wait_time.total_seconds() / 60)
            
    def get_trip_duration(self) -> Optional[int]:
        """获取实际行程时间 (分钟)"""
        if self.pickup_time and self.dropoff_time:
            duration = self.dropoff_time - self.pickup_time
            return int(duration.total_seconds() / 60)
        return None
    
    def get_pickup_urgency(self, current_time: Union[int, datetime]) -> float:
        """获取接客紧急程度 (0-1, 1最紧急)"""
        if isinstance(current_time, int):
            current_dt = datetime(2024, 1, 1, 0, 0) + timedelta(minutes=current_time)
        else:
            current_dt = current_time
            
        time_left = self.pickup_deadline - current_dt
        total_wait_time = self.MAX_PICKUP_DELAY.total_seconds()
        
        if time_left.total_seconds() <= 0:
            return 1.0  # 已过期
        
        urgency = 1.0 - (time_left.total_seconds() / total_wait_time)
        return max(0.0, min(1.0, urgency))
    
    def calculate_reward(self, pickup_delay: Union[int, timedelta], trip_completed: bool = True) -> float:
        """计算完成请求的奖励"""
        base_reward = self.total_amount
        
        # 处理延迟格式
        if isinstance(pickup_delay, int):
            delay_minutes = pickup_delay
        else:
            delay_minutes = pickup_delay.total_seconds() / 60
        
        # 延迟惩罚
        delay_penalty = min(delay_minutes * 0.1, base_reward * 0.3)  # 最多扣30%
        
        # 完成奖励
        completion_bonus = base_reward * 0.1 if trip_completed else 0
        
        total_reward = base_reward - delay_penalty + completion_bonus
        return max(0, total_reward)
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'request_id': self.request_id,
            'pickup_latitude': self.pickup_location[0],
            'pickup_longitude': self.pickup_location[1],
            'dropoff_latitude': self.dropoff_location[0],
            'dropoff_longitude': self.dropoff_location[1],
            'request_time': self.request_time.isoformat(),
            'request_time_minutes': self.request_time_minutes,
            'passenger_count': self.passenger_count,
            'trip_distance': self.trip_distance,
            'estimated_duration': self.estimated_duration,
            'max_wait_time': self.max_wait_time,
            'fare_amount': self.fare_amount,
            'total_amount': self.total_amount,
            'base_value': self.base_value,
            'final_value': self.final_value,
            'trip_type': self.trip_type,
            'payment_type': self.payment_type,
            'status': self.status,
            'assigned_vehicle_id': self.assigned_vehicle_id,
            'pickup_borough': self.pickup_borough,
            'dropoff_borough': self.dropoff_borough,
            'pickup_time': self.pickup_time.isoformat() if self.pickup_time else None,
            'dropoff_time': self.dropoff_time.isoformat() if self.dropoff_time else None,
            'actual_fare': self.actual_fare
        }
    
    @classmethod
    def from_nyc_data(cls, row: pd.Series, request_id_prefix: str = "nyc") -> 'NYCRequest':
        """从NYC数据行创建请求对象"""
        # 处理NYC数据的列名变化
        pickup_lat = getattr(row, 'pickup_latitude', None)
        pickup_lon = getattr(row, 'pickup_longitude', None)
        dropoff_lat = getattr(row, 'dropoff_latitude', None)
        dropoff_lon = getattr(row, 'dropoff_longitude', None)
        
        # 如果没有坐标，尝试LocationID
        if pickup_lat is None or pickup_lon is None:
            pickup_location_id = getattr(row, 'PULocationID', 0)
            pickup_lat, pickup_lon = cls._location_id_to_coords(pickup_location_id)
            
        if dropoff_lat is None or dropoff_lon is None:
            dropoff_location_id = getattr(row, 'DOLocationID', 0)
            dropoff_lat, dropoff_lon = cls._location_id_to_coords(dropoff_location_id)
        
        # 处理时间
        request_time = getattr(row, 'tpep_pickup_datetime', datetime.now())
        if isinstance(request_time, str):
            request_time = pd.to_datetime(request_time)
        
        # 处理其他字段
        trip_distance = getattr(row, 'trip_distance', 0.0)
        if trip_distance == 0.0:
            # 如果没有距离数据，根据坐标计算
            pickup_loc = (pickup_lat, pickup_lon)
            dropoff_loc = (dropoff_lat, dropoff_lon)
            trip_distance = geodesic(pickup_loc, dropoff_loc).kilometers * 1.3
        
        return cls(
            request_id=f"{request_id_prefix}_{getattr(row, 'index', np.random.randint(100000))}",
            pickup_location=(pickup_lat, pickup_lon),
            dropoff_location=(dropoff_lat, dropoff_lon),
            request_time=request_time,
            passenger_count=getattr(row, 'passenger_count', 1),
            trip_distance=trip_distance,
            base_value=getattr(row, 'fare_amount', 0.0),
            trip_type="standard",
            payment_type=str(getattr(row, 'payment_type', 'credit_card'))
        )
    
    @classmethod
    def load_from_data(cls, start_date: str = "2024-01", 
                      num_requests: int = 100) -> list['NYCRequest']:
        """从下载的NYC数据加载请求"""
        if not DATA_DOWNLOAD_AVAILABLE:
            print("Warning: Data download module not available, generating synthetic data")
            return cls._generate_synthetic_requests(num_requests)
        
        try:
            # 下载NYC数据
            print(f"📥 Downloading NYC taxi data for {start_date}...")
            data_file = download_data(start_date)
            
            if data_file and os.path.exists(data_file):
                print(f"✓ Loading data from {data_file}")
                df = pd.read_parquet(data_file)
                
                # 过滤和采样数据
                df = df.head(num_requests)
                
                # 转换为NYCRequest对象
                requests = []
                for idx, row in df.iterrows():
                    try:
                        request = cls.from_nyc_data(row, f"data_{start_date}")
                        requests.append(request)
                    except Exception as e:
                        print(f"Warning: Error processing row {idx}: {e}")
                        continue
                
                print(f"✓ Successfully loaded {len(requests)} requests from NYC data")
                return requests
            else:
                print("Warning: Data download failed, generating synthetic data")
                return cls._generate_synthetic_requests(num_requests)
                
        except Exception as e:
            print(f"Error loading NYC data: {e}")
            print("Falling back to synthetic data generation")
            return cls._generate_synthetic_requests(num_requests)
    
    @staticmethod
    def _location_id_to_coords(location_id: int) -> Tuple[float, float]:
        """将LocationID转换为坐标 (简化映射)"""
        # NYC Taxi Zone简化映射
        zone_coords = {
            1: (40.7831, -73.9712),   # Newark Airport
            2: (40.6713, -73.8370),   # Jamaica Bay
            4: (40.7594, -73.9776),   # Algonquin
            7: (40.7505, -73.9934),   # Penn Station/Madison Sq West
            13: (40.7794, -73.9441),  # Battery Park City
            24: (40.7527, -73.9772),  # East Chelsea
            # 添加更多常用zone...
        }
        
        if location_id in zone_coords:
            return zone_coords[location_id]
        
        # 默认映射到曼哈顿中心附近
        base_lat = 40.7500 
        base_lon = -73.9800
        
        # 简单的网格映射
        lat_offset = (location_id % 20) * 0.005
        lon_offset = (location_id // 20) * 0.005
        
        return (base_lat + lat_offset, base_lon + lon_offset)
    
    @classmethod
    def _generate_synthetic_requests(cls, num_requests: int) -> list['NYCRequest']:
        """生成合成请求数据 (当真实数据不可用时)"""
        requests = []
        
        # 需求热点
        hotspots = [
            (40.7580, -73.9855),  # Times Square
            (40.7505, -73.9934),  # Penn Station
            (40.7527, -73.9772),  # Grand Central
            (40.7074, -74.0113),  # Financial District
            (40.7794, -73.9441),  # Upper East Side
        ]
        
        base_time = datetime(2024, 10, 20, 8, 0)
        
        for i in range(num_requests):
            # 随机选择起点
            pickup_idx = np.random.randint(len(hotspots))
            pickup_base = hotspots[pickup_idx]
            pickup_location = (
                pickup_base[0] + np.random.normal(0, 0.003),
                pickup_base[1] + np.random.normal(0, 0.003)
            )
            
            # 随机选择终点
            dropoff_idx = np.random.randint(len(hotspots))
            dropoff_base = hotspots[dropoff_idx]
            dropoff_location = (
                dropoff_base[0] + np.random.normal(0, 0.005),
                dropoff_base[1] + np.random.normal(0, 0.005)
            )
            
            # 随机时间
            request_time = base_time + timedelta(minutes=np.random.randint(0, 720))
            
            request = cls(
                request_id=f"synthetic_{i:06d}",
                pickup_location=pickup_location,
                dropoff_location=dropoff_location,
                request_time=request_time,
                passenger_count=np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02])
            )
            requests.append(request)
            
        return requests
        
    def __str__(self) -> str:
        """字符串表示"""
        return (f"NYCRequest({self.request_id}: "
                f"{self.pickup_location} -> {self.dropoff_location}, "
                f"${self.total_amount:.2f}, {self.status})")
    
    def __repr__(self) -> str:
        """详细表示"""
        return self.__str__()


class NYCRequestGenerator:
    """NYC请求生成器 - 基于真实数据模式和地理分布"""
    
    def __init__(self, data_loader=None, use_real_data: bool = False):
        """初始化生成器"""
        self.data_loader = data_loader
        self.use_real_data = use_real_data
        self.request_counter = 0
        
        # 需求热点 (基于真实NYC数据)
        self.demand_hotspots = {
            'times_square': {'center': (40.7580, -73.9855), 'weight': 0.15},
            'penn_station': {'center': (40.7505, -73.9934), 'weight': 0.12},
            'grand_central': {'center': (40.7527, -73.9772), 'weight': 0.10},
            'financial_district': {'center': (40.7074, -74.0113), 'weight': 0.10},
            'upper_east_side': {'center': (40.7794, -73.9441), 'weight': 0.08},
            'soho': {'center': (40.7230, -74.0030), 'weight': 0.08},
            'chelsea': {'center': (40.7420, -74.0063), 'weight': 0.07},
            'union_square': {'center': (40.7359, -73.9911), 'weight': 0.06}
        }
        
        # 需求模式
        self.demand_patterns = {
            "base_requests_per_hour": 50,
            "hourly_multiplier": [
                0.3, 0.2, 0.2, 0.3, 0.5, 0.8,  # 0-5 AM
                1.5, 2.0, 1.8, 1.2, 1.0, 1.1,  # 6-11 AM
                1.2, 1.1, 1.0, 1.2, 1.5, 2.2,  # 12-17 PM
                2.5, 2.0, 1.5, 1.0, 0.8, 0.5   # 18-23 PM
            ],
            "weekend_multiplier": 1.3
        }
    
    def generate_request(self, current_time: Union[datetime, int]) -> NYCRequest:
        """生成单个请求"""
        self.request_counter += 1
        
        # 处理时间格式
        if isinstance(current_time, int):
            dt = datetime(2024, 1, 1, 0, 0) + timedelta(minutes=current_time)
        else:
            dt = current_time
        
        # 选择起点热点
        pickup_hotspot = self._select_hotspot(dt)
        pickup_location = self._generate_location_near_hotspot(pickup_hotspot)
        
        # 生成终点
        dropoff_location = self._generate_dropoff_location(pickup_location, dt)
        
        # 确定行程类型
        trip_type = self._determine_trip_type(pickup_location, dropoff_location, dt)
        
        return NYCRequest(
            request_id=f"nyc_{dt.strftime('%Y%m%d_%H%M%S')}_{self.request_counter}",
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            request_time=dt,
            passenger_count=np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02]),
            trip_type=trip_type
        )
    
    def generate_batch_requests(self, current_time: Union[datetime, int], 
                              num_requests: int = None) -> list:
        """批量生成请求"""
        if num_requests is None:
            num_requests = self._calculate_request_count(current_time)
            
        requests = []
        for _ in range(num_requests):
            request = self.generate_request(current_time)
            requests.append(request)
            
        return requests
        
    def _calculate_request_count(self, current_time: Union[datetime, int]) -> int:
        """根据时间模式计算请求数量"""
        if isinstance(current_time, int):
            dt = datetime(2024, 1, 1, 0, 0) + timedelta(minutes=current_time)
        else:
            dt = current_time
            
        current_hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        base_rate = self.demand_patterns["base_requests_per_hour"]
        hourly_multiplier = self.demand_patterns["hourly_multiplier"][current_hour]
        weekend_multiplier = self.demand_patterns["weekend_multiplier"] if is_weekend else 1.0
        
        requests_per_minute = (base_rate * hourly_multiplier * weekend_multiplier) / 60
        return max(0, int(np.random.poisson(requests_per_minute)))
        
    def _select_hotspot(self, current_time: datetime) -> Dict:
        """基于时间选择热点"""
        hour = current_time.hour
        
        # 调整权重基于时间
        weights = []
        hotspots = list(self.demand_hotspots.values())
        
        for hotspot in hotspots:
            weight = hotspot['weight']
            
            # 时间调整因子
            if hour in [7, 8, 9]:  # 早高峰
                if 'penn_station' in str(hotspot) or 'grand_central' in str(hotspot):
                    weight *= 2.0
            elif hour in [17, 18, 19]:  # 晚高峰
                if 'financial_district' in str(hotspot):
                    weight *= 1.8
            elif hour in [20, 21, 22]:  # 夜间娱乐
                if 'times_square' in str(hotspot) or 'soho' in str(hotspot):
                    weight *= 1.5
                    
            weights.append(weight)
        
        # 标准化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        selected_idx = np.random.choice(len(hotspots), p=weights)
        return hotspots[selected_idx]
    
    def _generate_location_near_hotspot(self, hotspot: Dict) -> Tuple[float, float]:
        """在热点附近生成位置"""
        center_lat, center_lon = hotspot['center']
        
        # 在半径0.5km内随机生成
        radius_km = 0.5
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_km)
        
        # 转换为坐标偏移 (粗略转换)
        lat_offset = (distance / 111.0) * np.cos(angle)  # 1度纬度约111km
        lon_offset = (distance / (111.0 * np.cos(np.radians(center_lat)))) * np.sin(angle)
        
        return (center_lat + lat_offset, center_lon + lon_offset)
    
    def _generate_dropoff_location(self, pickup_location: Tuple[float, float], 
                                 current_time: datetime) -> Tuple[float, float]:
        """生成下车地点"""
        # 80%概率在曼哈顿内，20%跨区
        if np.random.random() < 0.8:
            # 曼哈顿内
            return self._generate_manhattan_location()
        else:
            # 可能去其他区域
            return self._generate_cross_borough_destination(pickup_location)
    
    def _generate_manhattan_location(self) -> Tuple[float, float]:
        """在曼哈顿内生成随机位置"""
        lat = np.random.uniform(40.7000, 40.8800)
        lon = np.random.uniform(-74.0200, -73.9300)
        return (lat, lon)
    
    def _generate_cross_borough_destination(self, pickup_location: Tuple[float, float]) -> Tuple[float, float]:
        """生成跨区域目的地"""
        destinations = [
            (40.6782, -73.9442),  # Brooklyn Heights
            (40.7282, -73.7949),  # Queens
            (40.8448, -73.8648),  # Bronx
        ]
        return destinations[np.random.randint(len(destinations))]
    
    def _determine_trip_type(self, pickup_location: Tuple[float, float], 
                           dropoff_location: Tuple[float, float],
                           current_time: datetime) -> str:
        """确定行程类型"""
        distance = geodesic(pickup_location, dropoff_location).kilometers
        
        if distance > 15:
            return "airport"  # 长距离可能是机场
        elif current_time.hour in [20, 21, 22, 23] or current_time.weekday() >= 5:
            return "premium" if np.random.random() < 0.1 else "standard"
        else:
            return "standard"


class RequestGenerator:
    """
    兼容性请求生成器 (向后兼容)
    """
    
    def __init__(self, manhattan_bounds, demand_patterns):
        self.manhattan_bounds = manhattan_bounds
        self.demand_patterns = demand_patterns
        self.request_counter = 0
        
        # 创建NYC生成器
        self.nyc_generator = NYCRequestGenerator()
        
        # 转换边界格式
        self.bounds = {
            'min_lat': manhattan_bounds[0][0],
            'max_lat': manhattan_bounds[1][0], 
            'min_lon': manhattan_bounds[0][1],
            'max_lon': manhattan_bounds[1][1]
        }
    
    def generate_request(self, current_time: Union[datetime, int]) -> NYCRequest:
        """生成单个请求"""
        self.request_counter += 1
        
        # 处理时间格式
        if isinstance(current_time, int):
            dt = datetime(2024, 1, 1, 0, 0) + timedelta(minutes=current_time)
        else:
            dt = current_time
        
        # 选择起点热点
        pickup_hotspot = self._select_hotspot(dt)
        pickup_location = self._generate_location_near_hotspot(pickup_hotspot)
        
        # 生成终点
        dropoff_location = self._generate_dropoff_location(pickup_location, dt)
        
        # 确定行程类型
        trip_type = self._determine_trip_type(pickup_location, dropoff_location, dt)
        
        return NYCRequest(
            request_id=f"nyc_{dt.strftime('%Y%m%d_%H%M%S')}_{self.request_counter}",
            pickup_location=pickup_location,
            dropoff_location=dropoff_location,
            request_time=dt,
            passenger_count=np.random.choice([1, 2, 3, 4], p=[0.7, 0.2, 0.08, 0.02]),
            trip_type=trip_type
        )
    
    def generate_batch_requests(self, current_time: Union[datetime, int], 
                              num_requests: int = None) -> list:
        """批量生成请求"""
        if num_requests is None:
            num_requests = self._calculate_request_count(current_time)
            
        requests = []
        for _ in range(num_requests):
            request = self.generate_request(current_time)
            requests.append(request)
            
        return requests
        
    def _calculate_request_count(self, current_time: Union[datetime, int]) -> int:
        """根据时间模式计算请求数量"""
        if isinstance(current_time, int):
            dt = datetime(2024, 1, 1, 0, 0) + timedelta(minutes=current_time)
        else:
            dt = current_time
            
        current_hour = dt.hour
        is_weekend = dt.weekday() >= 5
        
        base_rate = self.demand_patterns["base_requests_per_hour"]
        hourly_multiplier = self.demand_patterns["hourly_multiplier"][current_hour]
        weekend_multiplier = self.demand_patterns["weekend_multiplier"] if is_weekend else 1.0
        
        requests_per_minute = (base_rate * hourly_multiplier * weekend_multiplier) / 60
        return max(0, int(np.random.poisson(requests_per_minute)))
        
    def _select_hotspot(self, current_time: datetime) -> Dict:
        """基于时间选择热点"""
        hour = current_time.hour
        
        # 调整权重基于时间
        weights = []
        hotspots = list(self.demand_hotspots.values())
        
        for hotspot in hotspots:
            weight = hotspot['weight']
            
            # 时间调整因子
            if hour in [7, 8, 9]:  # 早高峰
                if 'penn_station' in str(hotspot) or 'grand_central' in str(hotspot):
                    weight *= 2.0
            elif hour in [17, 18, 19]:  # 晚高峰
                if 'financial_district' in str(hotspot):
                    weight *= 1.8
            elif hour in [20, 21, 22]:  # 夜间娱乐
                if 'times_square' in str(hotspot) or 'soho' in str(hotspot):
                    weight *= 1.5
                    
            weights.append(weight)
        
        # 标准化权重
        weights = np.array(weights)
        weights = weights / weights.sum()
        
        selected_idx = np.random.choice(len(hotspots), p=weights)
        return hotspots[selected_idx]
    
    def _generate_location_near_hotspot(self, hotspot: Dict) -> Tuple[float, float]:
        """在热点附近生成位置"""
        center_lat, center_lon = hotspot['center']
        
        # 在半径0.5km内随机生成
        radius_km = 0.5
        angle = np.random.uniform(0, 2 * np.pi)
        distance = np.random.uniform(0, radius_km)
        
        # 转换为坐标偏移 (粗略转换)
        lat_offset = (distance / 111.0) * np.cos(angle)  # 1度纬度约111km
        lon_offset = (distance / (111.0 * np.cos(np.radians(center_lat)))) * np.sin(angle)
        
        return (center_lat + lat_offset, center_lon + lon_offset)
    
    def _generate_dropoff_location(self, pickup_location: Tuple[float, float], 
                                 current_time: datetime) -> Tuple[float, float]:
        """生成下车地点"""
        # 80%概率在曼哈顿内，20%跨区
        if np.random.random() < 0.8:
            # 曼哈顿内
            return self._generate_manhattan_location()
        else:
            # 可能去其他区域
            return self._generate_cross_borough_destination(pickup_location)
    
    def _generate_manhattan_location(self) -> Tuple[float, float]:
        """在曼哈顿内生成随机位置"""
        lat = np.random.uniform(40.7000, 40.8800)
        lon = np.random.uniform(-74.0200, -73.9300)
        return (lat, lon)
    
    def _generate_cross_borough_destination(self, pickup_location: Tuple[float, float]) -> Tuple[float, float]:
        """生成跨区域目的地"""
        destinations = [
            (40.6782, -73.9442),  # Brooklyn Heights
            (40.7282, -73.7949),  # Queens
            (40.8448, -73.8648),  # Bronx
        ]
        return destinations[np.random.randint(len(destinations))]
    
    def _determine_trip_type(self, pickup_location: Tuple[float, float], 
                           dropoff_location: Tuple[float, float],
                           current_time: datetime) -> str:
        """确定行程类型"""
        distance = geodesic(pickup_location, dropoff_location).kilometers
        
        if distance > 15:
            return "airport"  # 长距离可能是机场
        elif current_time.hour in [20, 21, 22, 23] or current_time.weekday() >= 5:
            return "premium" if np.random.random() < 0.1 else "standard"
        else:
            return "standard"


class RequestGenerator:
    """兼容性请求生成器 (向后兼容)"""
    
    def __init__(self, manhattan_bounds, demand_patterns):
        self.manhattan_bounds = manhattan_bounds
        self.demand_patterns = demand_patterns
        self.request_counter = 0
        
        # 创建NYC生成器
        self.nyc_generator = NYCRequestGenerator()
        
        # 转换边界格式
        self.bounds = {
            'min_lat': manhattan_bounds[0][0],
            'max_lat': manhattan_bounds[1][0], 
            'min_lon': manhattan_bounds[0][1],
            'max_lon': manhattan_bounds[1][1]
        }
        
    def generate_batch_requests(self, current_time: int, 
                              num_requests: int = None) -> list:
        """批量生成请求 (兼容性方法)"""
        return self.nyc_generator.generate_batch_requests(current_time, num_requests)
        
    def _generate_single_request(self, current_time: int) -> Optional[NYCRequest]:
        """生成单个请求 (兼容性方法)"""
        return self.nyc_generator.generate_request(current_time)
# 测试代码和示例用法
if __name__ == "__main__":
    print("🚕 Testing NYC Request Class...")
    
    # 测试基本请求创建
    test_request = NYCRequest(
        request_id="test_001",
        pickup_location=(40.7580, -73.9855),  # Times Square
        dropoff_location=(40.7074, -74.0113),  # Financial District
        request_time=datetime(2024, 10, 20, 8, 30)
    )
    
    print(f"✓ Request created: {test_request}")
    print(f"   Distance: {test_request.trip_distance:.2f} km")
    print(f"   Duration: {test_request.estimated_duration} minutes")
    print(f"   Fare: ${test_request.total_amount:.2f}")
    print(f"   Borough: {test_request.pickup_borough} -> {test_request.dropoff_borough}")
    
    # 测试兼容性属性
    print(f"\n🔧 Testing compatibility attributes...")
    print(f"   pickup (grid): {test_request.pickup}")
    print(f"   dropoff (grid): {test_request.dropoff}")
    print(f"   value: ${test_request.value:.2f}")
    print(f"   final_value: ${test_request.final_value:.2f}")
    
    # 测试状态更新
    print(f"\n📝 Testing status updates...")
    print(f"   Initial status: {test_request.status}")
    test_request.update_status("assigned", vehicle_id=123)
    print(f"   After assignment: {test_request.status} (vehicle: {test_request.assigned_vehicle_id})")
    
    # 测试时间兼容性
    test_request.set_pickup(datetime.now())
    print(f"   After pickup: {test_request.status}")
    
    # 测试请求生成器
    print(f"\n🔧 Testing request generator...")
    generator = NYCRequestGenerator()
    current_time = datetime(2024, 10, 20, 9, 0)
    
    for i in range(3):
        request = generator.generate_request(current_time)
        print(f"   Generated: {request}")
        current_time += timedelta(minutes=5)
    
    # 测试批量生成
    print(f"\n📦 Testing batch generation...")
    batch_requests = generator.generate_batch_requests(current_time, num_requests=5)
    print(f"   Generated {len(batch_requests)} requests in batch")
    for req in batch_requests[:2]:  # 显示前两个
        print(f"     - {req}")
    
    # 测试数据加载 (如果可用)
    if DATA_DOWNLOAD_AVAILABLE:
        print(f"\n📥 Testing data loading...")
        try:
            real_requests = NYCRequest.load_from_data("2024-01", num_requests=5)
            print(f"   Loaded {len(real_requests)} requests from real data")
            for req in real_requests[:2]:
                print(f"     - {req}")
        except Exception as e:
            print(f"   Data loading test failed: {e}")
    else:
        print(f"\n⚠️ Real data loading not available (download module missing)")
        synthetic_requests = NYCRequest._generate_synthetic_requests(3)
        print(f"   Generated {len(synthetic_requests)} synthetic requests")
        for req in synthetic_requests:
            print(f"     - {req}")
    
    # 测试向后兼容性
    print(f"\n🔄 Testing backward compatibility...")
    # 模拟旧式边界和模式
    old_bounds = [(40.7000, -74.0200), (40.8800, -73.9300)]
    old_patterns = {
        "base_requests_per_hour": 30,
        "hourly_multiplier": [0.5] * 24,
        "weekend_multiplier": 1.2
    }
    
    old_generator = RequestGenerator(old_bounds, old_patterns)
    old_style_requests = old_generator.generate_batch_requests(current_time=540, num_requests=2)  # 9AM
    print(f"   Generated {len(old_style_requests)} requests with old interface")
    for req in old_style_requests:
        print(f"     - {req}")
    
    # 测试字典转换
    print(f"\n📋 Testing dictionary conversion...")
    request_dict = test_request.to_dict()
    print(f"   Dictionary keys: {list(request_dict.keys())[:5]}...")  # 显示前5个key
    
    print(f"\n🎉 NYC Request class test completed successfully!")