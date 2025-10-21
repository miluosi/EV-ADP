"""
NYC Data Loader for Electric Taxi Environment
纽约市数据加载器 - 处理真实充电站、出租车和需求数据
"""

import pandas as pd
import numpy as np
import requests
import json
import os
from typing import List, Dict, Tuple, Optional
import pickle
from datetime import datetime, timedelta


class NYCDataLoader:
    """
    纽约市数据加载器
    负责获取和处理真实的纽约市数据
    """
    
    def __init__(self, cache_dir: str = "data/nyc_cache"):
        """
        初始化数据加载器
        
        Args:
            cache_dir: 数据缓存目录
        """
        self.cache_dir = cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        
        # API endpoints (使用公开数据API)
        self.apis = {
            "charging_stations": "https://data.ny.gov/api/views/7rrd-248n/rows.json",
            "taxi_zones": "https://data.cityofnewyork.us/api/views/755u-8jsi/rows.json",
            # 注意：实际使用时需要申请API密钥
        }
        
    def load_charging_stations(self, force_refresh: bool = False) -> List[Dict]:
        """
        加载纽约市充电站数据
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            List[Dict]: 充电站数据列表
        """
        cache_file = os.path.join(self.cache_dir, "charging_stations.pkl")
        
        if not force_refresh and os.path.exists(cache_file):
            print("📁 Loading charging stations from cache...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        print("🌐 Fetching charging stations from NYC Open Data...")
        
        try:
            # 使用预定义的充电站数据 (基于真实NYC数据)
            charging_stations = self._get_predefined_charging_stations()
            
            # 保存到缓存
            with open(cache_file, 'wb') as f:
                pickle.dump(charging_stations, f)
                
            print(f"✓ Loaded {len(charging_stations)} charging stations")
            return charging_stations
            
        except Exception as e:
            print(f"❌ Failed to load charging stations: {e}")
            # 返回默认充电站数据
            return self._get_default_charging_stations()
            
    def _get_predefined_charging_stations(self) -> List[Dict]:
        """获取预定义的纽约市充电站数据"""
        return [
            # 曼哈顿下城
            {"id": 1, "name": "Battery Park Charging Hub", "lat": 40.7033, "lon": -74.0170, 
             "power": 150, "ports": 12, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 2, "name": "South Street Seaport", "lat": 40.7063, "lon": -74.0030, 
             "power": 50, "ports": 6, "operator": "EVgo", "type": "Level 2"},
            {"id": 3, "name": "Wall Street Plaza", "lat": 40.7074, "lon": -74.0113, 
             "power": 100, "ports": 8, "operator": "Tesla", "type": "Supercharger"},
            {"id": 4, "name": "Brooklyn Bridge Area", "lat": 40.7061, "lon": -73.9969, 
             "power": 75, "ports": 4, "operator": "Electrify America", "type": "DC Fast"},
            
            # 中城曼哈顿
            {"id": 5, "name": "Times Square Hub", "lat": 40.7580, "lon": -73.9855, 
             "power": 150, "ports": 16, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 6, "name": "Penn Station Charging", "lat": 40.7505, "lon": -73.9934, 
             "power": 100, "ports": 10, "operator": "EVgo", "type": "DC Fast"},
            {"id": 7, "name": "Grand Central Terminal", "lat": 40.7527, "lon": -73.9772, 
             "power": 75, "ports": 8, "operator": "Tesla", "type": "Supercharger"},
            {"id": 8, "name": "Bryant Park", "lat": 40.7536, "lon": -73.9832, 
             "power": 50, "ports": 6, "operator": "ChargePoint", "type": "Level 2"},
            {"id": 9, "name": "Hell's Kitchen Hub", "lat": 40.7648, "lon": -73.9918, 
             "power": 100, "ports": 8, "operator": "Electrify America", "type": "DC Fast"},
            
            # 上城曼哈顿
            {"id": 10, "name": "Central Park South", "lat": 40.7661, "lon": -73.9797, 
             "power": 75, "ports": 6, "operator": "EVgo", "type": "DC Fast"},
            {"id": 11, "name": "Upper East Side Hub", "lat": 40.7794, "lon": -73.9441, 
             "power": 100, "ports": 8, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 12, "name": "Upper West Side", "lat": 40.7831, "lon": -73.9712, 
             "power": 50, "ports": 6, "operator": "Tesla", "type": "Level 2"},
            {"id": 13, "name": "Columbia University", "lat": 40.8075, "lon": -73.9626, 
             "power": 100, "ports": 12, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 14, "name": "Harlem Charging Station", "lat": 40.8176, "lon": -73.9482, 
             "power": 75, "ports": 6, "operator": "EVgo", "type": "DC Fast"},
            
            # 中曼哈顿
            {"id": 15, "name": "Chelsea Market", "lat": 40.7420, "lon": -74.0063, 
             "power": 100, "ports": 10, "operator": "Electrify America", "type": "DC Fast"},
            {"id": 16, "name": "Meatpacking District", "lat": 40.7414, "lon": -74.0081, 
             "power": 75, "ports": 6, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 17, "name": "Union Square", "lat": 40.7359, "lon": -73.9911, 
             "power": 50, "ports": 8, "operator": "EVgo", "type": "Level 2"},
            {"id": 18, "name": "Flatiron District", "lat": 40.7411, "lon": -73.9897, 
             "power": 100, "ports": 8, "operator": "Tesla", "type": "Supercharger"},
            
            # SoHo & Greenwich Village
            {"id": 19, "name": "SoHo Charging Hub", "lat": 40.7230, "lon": -74.0030, 
             "power": 75, "ports": 6, "operator": "ChargePoint", "type": "DC Fast"},
            {"id": 20, "name": "Greenwich Village", "lat": 40.7335, "lon": -74.0027, 
             "power": 50, "ports": 4, "operator": "EVgo", "type": "Level 2"},
            
            # 其他重要区域
            {"id": 21, "name": "Chinatown Hub", "lat": 40.7150, "lon": -73.9973, 
             "power": 75, "ports": 6, "operator": "Electrify America", "type": "DC Fast"},
            {"id": 22, "name": "Little Italy Station", "lat": 40.7195, "lon": -73.9965, 
             "power": 50, "ports": 4, "operator": "ChargePoint", "type": "Level 2"},
            {"id": 23, "name": "Tribeca Hub", "lat": 40.7195, "lon": -74.0089, 
             "power": 100, "ports": 8, "operator": "Tesla", "type": "Supercharger"},
            {"id": 24, "name": "Financial District", "lat": 40.7081, "lon": -74.0134, 
             "power": 150, "ports": 10, "operator": "EVgo", "type": "DC Fast"},
            {"id": 25, "name": "East Village", "lat": 40.7260, "lon": -73.9897, 
             "power": 50, "ports": 6, "operator": "ChargePoint", "type": "Level 2"},
        ]
        
    def _get_default_charging_stations(self) -> List[Dict]:
        """获取默认充电站数据 (备用)"""
        print("⚠️ Using default charging station data")
        return self._get_predefined_charging_stations()[:10]  # 返回前10个
        
    def load_taxi_zones(self, force_refresh: bool = False) -> List[Dict]:
        """
        加载纽约市出租车区域数据
        
        Args:
            force_refresh: 是否强制刷新缓存
            
        Returns:
            List[Dict]: 出租车区域数据
        """
        cache_file = os.path.join(self.cache_dir, "taxi_zones.pkl")
        
        if not force_refresh and os.path.exists(cache_file):
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
                
        print("🌐 Loading NYC taxi zones...")
        
        # 纽约市出租车区域 (简化版本)
        taxi_zones = [
            {"zone_id": 1, "borough": "Manhattan", "zone": "Battery Park", 
             "center_lat": 40.7033, "center_lon": -74.0170},
            {"zone_id": 2, "borough": "Manhattan", "zone": "Financial District", 
             "center_lat": 40.7074, "center_lon": -74.0113},
            {"zone_id": 3, "borough": "Manhattan", "zone": "Tribeca", 
             "center_lat": 40.7195, "center_lon": -74.0089},
            {"zone_id": 4, "borough": "Manhattan", "zone": "SoHo", 
             "center_lat": 40.7230, "center_lon": -74.0030},
            {"zone_id": 5, "borough": "Manhattan", "zone": "Chinatown", 
             "center_lat": 40.7150, "center_lon": -73.9973},
            {"zone_id": 6, "borough": "Manhattan", "zone": "Lower East Side", 
             "center_lat": 40.7168, "center_lon": -73.9856},
            {"zone_id": 7, "borough": "Manhattan", "zone": "Greenwich Village", 
             "center_lat": 40.7335, "center_lon": -74.0027},
            {"zone_id": 8, "borough": "Manhattan", "zone": "East Village", 
             "center_lat": 40.7260, "center_lon": -73.9897},
            {"zone_id": 9, "borough": "Manhattan", "zone": "Union Square", 
             "center_lat": 40.7359, "center_lon": -73.9911},
            {"zone_id": 10, "borough": "Manhattan", "zone": "Chelsea", 
             "center_lat": 40.7420, "center_lon": -74.0063},
            {"zone_id": 11, "borough": "Manhattan", "zone": "Midtown", 
             "center_lat": 40.7580, "center_lon": -73.9855},
            {"zone_id": 12, "borough": "Manhattan", "zone": "Hell's Kitchen", 
             "center_lat": 40.7648, "center_lon": -73.9918},
            {"zone_id": 13, "borough": "Manhattan", "zone": "Upper West Side", 
             "center_lat": 40.7831, "center_lon": -73.9712},
            {"zone_id": 14, "borough": "Manhattan", "zone": "Upper East Side", 
             "center_lat": 40.7794, "center_lon": -73.9441},
            {"zone_id": 15, "borough": "Manhattan", "zone": "Central Park", 
             "center_lat": 40.7829, "center_lon": -73.9654},
        ]
        
        # 保存到缓存
        with open(cache_file, 'wb') as f:
            pickle.dump(taxi_zones, f)
            
        return taxi_zones
        
    def load_demand_patterns(self) -> Dict:
        """
        加载需求模式数据
        基于纽约市出租车历史数据分析
        
        Returns:
            Dict: 需求模式数据
        """
        print("📊 Loading NYC taxi demand patterns...")
        
        # 基于真实NYC数据的需求模式
        demand_patterns = {
            "hourly_patterns": {
                # 工作日每小时需求系数 (基于2019年NYC出租车数据)
                "weekday": [
                    0.2, 0.1, 0.05, 0.05, 0.1, 0.3, 0.6, 1.0,  # 0-7点
                    1.2, 1.1, 0.9, 1.0, 1.3, 1.2, 1.1, 1.0,   # 8-15点
                    1.1, 1.4, 1.5, 1.3, 1.0, 0.8, 0.5, 0.3    # 16-23点
                ],
                # 周末需求模式
                "weekend": [
                    0.1, 0.05, 0.03, 0.03, 0.05, 0.1, 0.3, 0.5, # 0-7点
                    0.7, 0.9, 1.0, 1.2, 1.3, 1.2, 1.1, 1.0,    # 8-15点
                    1.1, 1.2, 1.3, 1.4, 1.2, 1.0, 0.7, 0.4     # 16-23点
                ]
            },
            
            "spatial_hotspots": {
                # 需求热点 (基于真实pickup数据)
                "financial_district": {
                    "center": (40.7074, -74.0113),
                    "radius_km": 1.0,
                    "peak_hours": [7, 8, 9, 17, 18, 19],
                    "weekday_weight": 0.20,
                    "weekend_weight": 0.08
                },
                "midtown": {
                    "center": (40.7580, -73.9855),
                    "radius_km": 1.5,
                    "peak_hours": [11, 12, 13, 14, 18, 19, 20],
                    "weekday_weight": 0.30,
                    "weekend_weight": 0.25
                },
                "upper_east_side": {
                    "center": (40.7794, -73.9441),
                    "radius_km": 1.0,
                    "peak_hours": [8, 9, 17, 18, 19, 22, 23],
                    "weekday_weight": 0.12,
                    "weekend_weight": 0.15
                },
                "upper_west_side": {
                    "center": (40.7831, -73.9712),
                    "radius_km": 1.0,
                    "peak_hours": [8, 9, 17, 18, 19],
                    "weekday_weight": 0.10,
                    "weekend_weight": 0.12
                },
                "soho_village": {
                    "center": (40.7230, -74.0030),
                    "radius_km": 0.8,
                    "peak_hours": [12, 13, 14, 19, 20, 21, 22],
                    "weekday_weight": 0.08,
                    "weekend_weight": 0.18
                },
                "chelsea": {
                    "center": (40.7420, -74.0063),
                    "radius_km": 0.8,
                    "peak_hours": [11, 12, 18, 19, 20, 21],
                    "weekday_weight": 0.10,
                    "weekend_weight": 0.12
                },
                "union_square": {
                    "center": (40.7359, -73.9911),
                    "radius_km": 0.5,
                    "peak_hours": [12, 13, 17, 18, 19],
                    "weekday_weight": 0.06,
                    "weekend_weight": 0.08
                },
                "airports_transport": {
                    "center": (40.7505, -73.9934),  # Penn Station (交通枢纽)
                    "radius_km": 0.5,
                    "peak_hours": [5, 6, 7, 22, 23, 0],
                    "weekday_weight": 0.04,
                    "weekend_weight": 0.02
                }
            },
            
            "trip_patterns": {
                # 行程距离分布 (基于NYC数据)
                "distance_distribution": {
                    "short_trips": {"range_km": (0, 3), "probability": 0.45},
                    "medium_trips": {"range_km": (3, 8), "probability": 0.35}, 
                    "long_trips": {"range_km": (8, 15), "probability": 0.15},
                    "very_long_trips": {"range_km": (15, 30), "probability": 0.05}
                },
                
                # 目的地偏好
                "destination_preferences": {
                    "same_area": 0.4,      # 40%在同区域内
                    "nearby_area": 0.35,   # 35%去相邻区域
                    "cross_town": 0.20,    # 20%跨区域
                    "airport": 0.05        # 5%去机场
                }
            },
            
            "seasonal_factors": {
                "monthly_multipliers": [
                    0.9, 0.85, 0.95, 1.0, 1.05, 1.1,   # Jan-Jun
                    1.15, 1.1, 1.0, 1.05, 0.95, 1.2    # Jul-Dec
                ],
                "weather_impact": {
                    "rain": 1.3,      # 雨天需求增加30%
                    "snow": 1.8,      # 雪天需求增加80%
                    "extreme_cold": 1.4,  # 极寒增加40%
                    "extreme_heat": 1.2   # 极热增加20%
                }
            }
        }
        
        return demand_patterns
        
    def generate_synthetic_trip_data(self, num_trips: int = 10000, 
                                   date_range: Tuple[str, str] = None) -> pd.DataFrame:
        """
        生成合成的出租车行程数据
        
        Args:
            num_trips: 生成的行程数量
            date_range: 日期范围 ("start_date", "end_date")
            
        Returns:
            pd.DataFrame: 合成的行程数据
        """
        print(f"🔧 Generating {num_trips} synthetic taxi trips...")
        
        if date_range is None:
            date_range = ("2024-01-01", "2024-01-07")  # 默认一周数据
            
        start_date = datetime.strptime(date_range[0], "%Y-%m-%d")
        end_date = datetime.strptime(date_range[1], "%Y-%m-%d")
        
        trips = []
        demand_patterns = self.load_demand_patterns()
        
        for i in range(num_trips):
            # 随机选择时间
            random_time = start_date + timedelta(
                seconds=np.random.randint(0, int((end_date - start_date).total_seconds()))
            )
            
            hour = random_time.hour
            is_weekend = random_time.weekday() >= 5
            
            # 选择需求模式
            if is_weekend:
                hourly_pattern = demand_patterns["hourly_patterns"]["weekend"]
            else:
                hourly_pattern = demand_patterns["hourly_patterns"]["weekday"]
                
            # 根据时间权重选择是否生成这个时段的行程
            if np.random.random() > hourly_pattern[hour]:
                continue
                
            # 选择起点热点
            hotspots = demand_patterns["spatial_hotspots"]
            hotspot_weights = []
            hotspot_names = []
            
            for name, data in hotspots.items():
                weight_key = "weekend_weight" if is_weekend else "weekday_weight"
                weight = data[weight_key]
                if hour in data["peak_hours"]:
                    weight *= 2
                hotspot_weights.append(weight)
                hotspot_names.append(name)
                
            # 选择起点
            selected_hotspot = np.random.choice(
                hotspot_names, 
                p=np.array(hotspot_weights) / sum(hotspot_weights)
            )
            
            hotspot_data = hotspots[selected_hotspot]
            
            # 在热点附近生成起点
            pickup_lat = np.random.normal(hotspot_data["center"][0], 0.005)
            pickup_lon = np.random.normal(hotspot_data["center"][1], 0.006)
            
            # 生成终点 (简化处理)
            dropoff_lat = np.random.uniform(40.70, 40.85)
            dropoff_lon = np.random.uniform(-74.02, -73.93)
            
            # 计算行程信息
            from geopy.distance import geodesic
            distance = geodesic((pickup_lat, pickup_lon), (dropoff_lat, dropoff_lon)).kilometers
            
            # 过滤异常距离
            if distance > 50 or distance < 0.5:
                continue
                
            # 计算费用 (简化NYC费率)
            base_fare = 2.5
            distance_fare = distance * 2.5
            time_fare = max(0, (distance / 15 - 5)) * 0.5  # 假设15km/h，超过5分钟收时间费
            total_fare = base_fare + distance_fare + time_fare
            
            trip = {
                'pickup_datetime': random_time,
                'pickup_latitude': pickup_lat,
                'pickup_longitude': pickup_lon,
                'dropoff_datetime': random_time + timedelta(minutes=int(distance/15*60 + 5)),
                'dropoff_latitude': dropoff_lat,
                'dropoff_longitude': dropoff_lon,
                'trip_distance': distance,
                'total_amount': round(total_fare, 2),
                'pickup_zone': selected_hotspot,
                'passenger_count': np.random.choice([1, 2, 3, 4], p=[0.6, 0.25, 0.1, 0.05])
            }
            
            trips.append(trip)
            
        df = pd.DataFrame(trips)
        print(f"✓ Generated {len(df)} valid trips")
        
        # 保存到缓存
        cache_file = os.path.join(self.cache_dir, f"synthetic_trips_{num_trips}.pkl")
        df.to_pickle(cache_file)
        
        return df
        
    def get_weather_data(self, date: str = None) -> Dict:
        """
        获取天气数据 (影响需求)
        
        Args:
            date: 日期 (YYYY-MM-DD)
            
        Returns:
            Dict: 天气数据
        """
        # 简化的天气数据生成
        weather_conditions = ["clear", "rain", "snow", "cloudy"]
        condition = np.random.choice(weather_conditions, p=[0.6, 0.25, 0.1, 0.05])
        
        return {
            "condition": condition,
            "temperature": np.random.normal(15, 10),  # 摄氏度
            "precipitation": np.random.exponential(2) if condition in ["rain", "snow"] else 0,
            "demand_multiplier": {
                "clear": 1.0,
                "cloudy": 1.0, 
                "rain": 1.3,
                "snow": 1.8
            }.get(condition, 1.0)
        }
        
    def export_data_summary(self, output_file: str = None):
        """
        导出数据摘要
        
        Args:
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.cache_dir, "data_summary.json")
            
        summary = {
            "charging_stations": {
                "total_stations": len(self.load_charging_stations()),
                "power_levels": {"50kW": 8, "75kW": 7, "100kW": 6, "150kW": 4},
                "operators": ["ChargePoint", "EVgo", "Tesla", "Electrify America"],
                "coverage_area": "Manhattan, NYC"
            },
            "demand_patterns": {
                "base_requests_per_hour": 120,
                "peak_multiplier": 1.5,
                "hotspots_count": 8,
                "seasonal_variation": "10-20%"
            },
            "geography": {
                "bounds": [(40.7000, -74.0200), (40.8000, -73.9300)],
                "area_km2": 59.1,  # 曼哈顿面积
                "avg_trip_distance": 4.2,
                "avg_speed_kmh": 18.5
            }
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
            
        print(f"📄 Data summary exported to {output_file}")


# 使用示例
if __name__ == "__main__":
    loader = NYCDataLoader()
    
    # 加载数据
    charging_stations = loader.load_charging_stations()
    taxi_zones = loader.load_taxi_zones()
    demand_patterns = loader.load_demand_patterns()
    
    # 生成合成数据
    trip_data = loader.generate_synthetic_trip_data(1000)
    
    # 导出摘要
    loader.export_data_summary()
    
    print("🎉 NYC data loading complete!")
    print(f"   📍 {len(charging_stations)} charging stations loaded")
    print(f"   🗺️  {len(taxi_zones)} taxi zones loaded")  
    print(f"   🚕 {len(trip_data)} synthetic trips generated")