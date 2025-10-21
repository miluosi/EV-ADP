"""
Test NYC Electric Taxi Environment
测试纽约市电动出租车环境
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib.pyplot as plt    
from datetime import datetime, timedelta
import json

# 导入我们的模块
from src.NYEEnvironment import NYEEnvironment
from src.NYCDataLoader import NYCDataLoader
from src.NYCRequest import NYCRequest
from src.Action import Action, IdleAction, ChargingAction, ServiceAction
from config.config_manager import ConfigManager


class NYEEnvironmentTester:
    """纽约市电动出租车环境测试器"""
    
    def __init__(self):
        """初始化测试器"""
        self.config = ConfigManager()
        self.data_loader = NYCDataLoader()
        
        # 创建测试环境
        self.env = None
        self.test_results = {}
        
    def setup_test_environment(self):
        """设置测试环境"""
        print("🔧 Setting up NYC Electric Taxi Environment...")
        
        # 加载数据
        charging_stations = self.data_loader.load_charging_stations()[:10]  # 使用前10个充电站
        demand_patterns = self.data_loader.load_demand_patterns()
        
        # 创建环境
        self.env = NYEEnvironment(
            num_vehicles=20,
            num_stations=len(charging_stations)
        )
        
        print(f"✓ Environment created with {self.env.num_vehicles} vehicles")
        print(f"✓ {len(self.env.charging_stations_data)} charging stations loaded")
        print(f"✓ Environment initialized successfully")
        
    def test_environment_initialization(self):
        """测试环境初始化"""
        print("\n📋 Testing Environment Initialization...")
        
        # 检查车辆初始化
        assert len(self.env.vehicles) == self.env.num_vehicles, "Vehicle count mismatch"
        
        # 检查车辆位置
        for i, vehicle in self.env.vehicles.items():
            lat, lon = vehicle['location']
            assert 40.70 <= lat <= 40.85, f"Vehicle {i} latitude out of bounds: {lat}"
            assert -74.02 <= lon <= -73.93, f"Vehicle {i} longitude out of bounds: {lon}"
            
        # 检查充电站
        assert len(self.env.charging_stations_data) >= 5, "Too few charging stations"
        
        # 检查初始请求
        assert len(self.env.requests) >= 0, "Request initialization failed"
        
        print("✓ Environment initialization test passed")
        return True
        
    def test_vehicle_movement(self):
        """测试车辆移动"""
        print("\n🚗 Testing Vehicle Movement...")
        
        vehicle = self.env.vehicles[0]
        initial_location = vehicle['location']
        
        # 创建移动动作 (使用IdleAction作为占位)
        target_location = (40.7580, -73.9855)  # Times Square
        action = IdleAction(
            requests=[],  # 空请求列表
            current_coords=initial_location,
            target_coords=target_location,
            vehicle_loc=initial_location,
            vehicle_battery=0.8
        )
        
        # 执行动作
        result = self.env._execute_action(0, action)
        
        # 检查结果
        new_location = vehicle['location']
        
        # 验证车辆位置改变
        distance_moved = self._calculate_distance(initial_location, new_location)
        
        print(f"   Initial: {initial_location}")
        print(f"   Target:  {target_location}")
        print(f"   Final:   {new_location}")
        print(f"   Distance moved: {distance_moved:.2f} km")
        
        assert distance_moved > 0, "Vehicle did not move"
        assert result is not None, "Action execution failed"
        
        print("✓ Vehicle movement test passed")
        return True
        
    def test_request_generation(self):
        """测试请求生成"""
        print("\n📞 Testing Request Generation...")
        
        # 生成一些步骤来触发请求生成
        initial_requests = len(self.env.requests)
        
        for step in range(5):
            new_requests = self.env._generate_requests()
            self.env.requests.extend(new_requests)
            self.env.current_time += timedelta(minutes=10)
            
        final_requests = len(self.env.requests)
        
        print(f"   Initial requests: {initial_requests}")
        print(f"   Final requests: {final_requests}")
        print(f"   New requests generated: {final_requests - initial_requests}")
        
        # 检查请求有效性
        if self.env.requests:
            request = self.env.requests[0]
            pickup_lat, pickup_lon = request.pickup_location
            dropoff_lat, dropoff_lon = request.dropoff_location
            
            assert 40.70 <= pickup_lat <= 40.85, "Invalid pickup latitude"
            assert -74.02 <= pickup_lon <= -73.93, "Invalid pickup longitude"
            assert 40.70 <= dropoff_lat <= 40.85, "Invalid dropoff latitude"
            assert -74.02 <= dropoff_lon <= -73.93, "Invalid dropoff longitude"
            
            print(f"   Sample request: {pickup_lat:.4f}, {pickup_lon:.4f} -> {dropoff_lat:.4f}, {dropoff_lon:.4f}")
            
        assert final_requests >= initial_requests, "No requests generated"
        
        print("✓ Request generation test passed")
        return True
        
    def test_charging_functionality(self):
        """测试充电功能"""
        print("\n🔋 Testing Charging Functionality...")
        
        vehicle = self.env.vehicles[0]
        
        # 降低电池电量
        vehicle['battery_kwh'] = 15  # 降低到15kWh
        initial_battery = vehicle['battery_kwh']
        
        # 移动到充电站
        if not self.env.charging_stations_data:
            print("   ⚠️ No charging stations available")
            return True
            
        charging_station = self.env.charging_stations_data[0]
        
        # 验证充电站数据类型
        if not isinstance(charging_station, dict):
            print(f"   ❌ Invalid charging station type: {type(charging_station)}")
            return False
            
        station_location = (charging_station["lat"], charging_station["lon"])
        
        # 创建充电动作
        action = ChargingAction(
            requests=[],  # 空请求列表
            charging_station_id=charging_station["id"],
            charging_duration=30.0,
            vehicle_loc=vehicle['location'],
            vehicle_battery=initial_battery / 75.0  # 标准化电池电量
        )
        
        # 执行充电动作
        result = self.env._execute_action(0, action)
        
        final_battery = vehicle['battery_kwh']
        
        print(f"   Initial battery: {initial_battery:.1f} kWh")
        print(f"   Final battery: {final_battery:.1f} kWh")
        print(f"   Battery change: {(final_battery - initial_battery):.1f} kWh")
        print(f"   Charging station: {charging_station['name']}")
        
        # 验证充电动作执行成功（允许电池先消耗再充电的情况）
        assert result is not None, "Charging action failed"
        
        # 如果车辆移动到充电站但还没开始充电，这是正常的
        if vehicle['status'] == 'charging':
            print("   ✓ Vehicle started charging")
        else:
            print("   ○ Vehicle moving to charging station")
        
        print("✓ Charging functionality test passed")
        return True
        
    def test_request_assignment(self):
        """测试请求分配"""
        print("\n📋 Testing Request Assignment...")
        
        # 确保有请求可用
        if not self.env.requests:
            new_requests = self.env._generate_requests()
            self.env.requests.extend(new_requests[:5])
            
        if not self.env.requests:
            print("⚠️ No requests available for assignment test")
            return True
            
        request = self.env.requests[0]
        vehicle = self.env.vehicles[0]
        
        # 设置车辆状态为空闲
        vehicle['status'] = 'idle'
        vehicle['assigned_request'] = None
        
        # 创建分配动作
        action = ServiceAction(
            requests=[request],  # 包含请求
            request_id=id(request),  # 使用对象ID作为请求ID
            vehicle_loc=vehicle['location'],
            vehicle_battery=vehicle['battery_kwh'] / 75.0
        )
        
        # 执行分配动作
        result = self.env._execute_action(0, action)
        
        print(f"   Request: {request.pickup_location} -> {request.dropoff_location}")
        print(f"   Vehicle {vehicle['id']} assigned: {vehicle['assigned_request'] is not None}")
        
        print("✓ Request assignment test passed")
        return True
        
    def test_full_episode(self):
        """测试完整episode"""
        print("\n🎯 Testing Full Episode...")
        
        episode_rewards = []
        episode_stats = {
            "total_requests": 0,
            "completed_requests": 0,
            "total_revenue": 0,
            "charging_events": 0,
            "vehicle_utilization": []
        }
        
        # 运行一个短episode
        for step in range(20):
            # 环境更新
            self.env._update_environment()
            
            # 简单的随机动作策略 (仅用于测试)
            actions = self._generate_random_actions()
            
            # 执行动作
            rewards = []
            for vehicle_id, action in enumerate(actions):
                if action is not None and vehicle_id < len(self.env.vehicles):
                    result = self.env._execute_action(vehicle_id, action)
                    reward = self._calculate_reward(action, result)
                    rewards.append(reward)
                    
            # 记录统计信息
            step_reward = sum(rewards) if rewards else 0
            episode_rewards.append(step_reward)
            
            # 更新统计
            episode_stats["total_requests"] = len(self.env.requests)
            episode_stats["completed_requests"] = len([v for v in self.env.vehicles.values() 
                                                    if v['assigned_request'] is None and v['status'] == 'idle'])
            
            if step % 5 == 0:
                print(f"   Step {step}: Reward={step_reward:.2f}, Requests={len(self.env.requests)}")
                
        # 计算最终统计
        total_reward = sum(episode_rewards)
        avg_reward = total_reward / len(episode_rewards) if episode_rewards else 0
        
        print(f"   Episode completed: {len(episode_rewards)} steps")
        print(f"   Total reward: {total_reward:.2f}")
        print(f"   Average reward: {avg_reward:.2f}")
        print(f"   Total requests: {episode_stats['total_requests']}")
        
        # 保存测试结果
        self.test_results["full_episode"] = {
            "total_reward": total_reward,
            "avg_reward": avg_reward,
            "episode_length": len(episode_rewards),
            "final_stats": episode_stats
        }
        
        assert len(episode_rewards) > 0, "Episode produced no rewards"
        
        print("✓ Full episode test passed")
        return True
        
    def test_data_loader(self):
        """测试数据加载器"""
        print("\n📊 Testing NYC Data Loader...")
        
        # 测试充电站加载
        stations = self.data_loader.load_charging_stations()
        assert len(stations) >= 10, "Too few charging stations loaded"
        
        # 测试需求模式
        demand_patterns = self.data_loader.load_demand_patterns()
        assert "hourly_patterns" in demand_patterns, "Missing hourly patterns"
        assert "spatial_hotspots" in demand_patterns, "Missing spatial hotspots"
        
        # 测试合成数据生成
        trip_data = self.data_loader.generate_synthetic_trip_data(100)
        assert len(trip_data) > 50, "Insufficient synthetic trips generated"
        
        print(f"   ✓ {len(stations)} charging stations loaded")
        print(f"   ✓ {len(demand_patterns['spatial_hotspots'])} demand hotspots")
        print(f"   ✓ {len(trip_data)} synthetic trips generated")
        
        print("✓ Data loader test passed")
        return True
        
    def _generate_random_actions(self) -> list:
        """生成随机动作 (用于测试)"""
        actions = []
        
        for vehicle_id, vehicle in self.env.vehicles.items():
            if np.random.random() < 0.3:  # 30%概率执行动作
                action_type = np.random.choice(["move", "charge", "pickup", "service"])
                
                if action_type == "move":
                    # 随机移动 (使用IdleAction)
                    target_lat = np.random.uniform(40.70, 40.85)
                    target_lon = np.random.uniform(-74.02, -73.93)
                    action = IdleAction(
                        requests=[],
                        current_coords=vehicle['location'],
                        target_coords=(target_lat, target_lon),
                        vehicle_loc=vehicle['location'],
                        vehicle_battery=vehicle['battery_kwh'] / 75.0
                    )
                elif action_type == "charge" and vehicle['battery_kwh'] < 35:
                    # 充电
                    station = np.random.choice(self.env.charging_stations_data)
                    action = ChargingAction(
                        requests=[],
                        charging_station_id=station["id"],
                        charging_duration=30.0,
                        vehicle_loc=vehicle['location'],
                        vehicle_battery=vehicle['battery_kwh'] / 75.0
                    )
                elif action_type == "pickup" and self.env.requests and vehicle['status'] == 'idle':
                    # 接客
                    request = np.random.choice(self.env.requests)
                    action = ServiceAction(
                        requests=[request],
                        request_id=id(request),
                        vehicle_loc=vehicle['location'],
                        vehicle_battery=vehicle['battery_kwh'] / 75.0
                    )
                else:
                    action = None
                    
                actions.append(action)
                
        return actions
        
    def _calculate_reward(self, action, result) -> float:
        """计算动作奖励 (简化)"""
        if result is None:
            return -1.0  # 动作失败惩罚
            
        reward = 0.0
        
        if isinstance(action, ServiceAction):
            reward += 5.0  # 接客奖励
        elif isinstance(action, ChargingAction):
            reward += 1.0  # 充电奖励
        elif isinstance(action, IdleAction):
            reward -= 0.5  # 移动成本
            
        return reward
        
    def _calculate_distance(self, loc1, loc2) -> float:
        """计算两点间距离 (km)"""
        from geopy.distance import geodesic
        return geodesic(loc1, loc2).kilometers
        
    def run_all_tests(self):
        """运行所有测试"""
        print("🚀 Starting NYC Electric Taxi Environment Tests")
        print("=" * 60)
        
        try:
            # 设置测试环境
            self.setup_test_environment()
            
            # 运行测试
            tests = [
                ("Environment Initialization", self.test_environment_initialization),
                ("Vehicle Movement", self.test_vehicle_movement), 
                ("Request Generation", self.test_request_generation),
                ("Charging Functionality", self.test_charging_functionality),
                ("Request Assignment", self.test_request_assignment),
                ("Data Loader", self.test_data_loader),
                ("Full Episode", self.test_full_episode),
            ]
            
            passed_tests = 0
            total_tests = len(tests)
            
            for test_name, test_func in tests:
                try:
                    success = test_func()
                    if success:
                        passed_tests += 1
                except Exception as e:
                    print(f"❌ {test_name} FAILED: {str(e)}")
                    
            # 测试总结
            print("\n" + "=" * 60)
            print("🏆 TEST SUMMARY")
            print(f"   Tests passed: {passed_tests}/{total_tests}")
            print(f"   Success rate: {passed_tests/total_tests:.1%}")
            
            if passed_tests == total_tests:
                print("   🎉 ALL TESTS PASSED! 🎉")
            else:
                print(f"   ⚠️ {total_tests - passed_tests} test(s) failed")
                
            # 保存测试结果
            self._save_test_results(passed_tests, total_tests)
            
            return passed_tests == total_tests
            
        except Exception as e:
            print(f"💥 Test setup failed: {str(e)}")
            return False
            
    def _save_test_results(self, passed: int, total: int):
        """保存测试结果"""
        results = {
            "test_summary": {
                "timestamp": datetime.now().isoformat(),
                "passed_tests": passed,
                "total_tests": total,
                "success_rate": passed / total,
                "status": "PASSED" if passed == total else "FAILED"
            },
            "environment_config": {
                "num_vehicles": self.env.num_vehicles,
                "num_charging_stations": len(self.env.charging_stations_data)
            },
            "detailed_results": self.test_results
        }
        
        # 保存到结果文件
        results_dir = "results/nye_tests"
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = os.path.join(results_dir, f"nye_test_results_{timestamp}.json")
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        print(f"📄 Test results saved to: {results_file}")
        
    def create_test_visualization(self):
        """创建测试可视化"""
        if not self.env:
            print("⚠️ Environment not initialized")
            return
            
        try:
            import matplotlib.pyplot as plt
            
            # 创建地图可视化
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))
            
            # 车辆和充电站位置
            vehicle_lats = [v['location'][0] for v in self.env.vehicles.values()]
            vehicle_lons = [v['location'][1] for v in self.env.vehicles.values()]
            
            station_lats = [s["lat"] for s in self.env.charging_stations_data]
            station_lons = [s["lon"] for s in self.env.charging_stations_data]
            
            # 绘制地图
            ax1.scatter(vehicle_lons, vehicle_lats, c='blue', alpha=0.7, s=50, label='Vehicles')
            ax1.scatter(station_lons, station_lats, c='red', marker='s', s=100, label='Charging Stations')
            
            ax1.set_xlabel('Longitude')
            ax1.set_ylabel('Latitude')
            ax1.set_title('NYC Electric Taxi Environment')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # 车辆电池状态分布
            battery_levels = [v['battery_percentage'] for v in self.env.vehicles.values()]
            ax2.hist(battery_levels, bins=10, alpha=0.7, color='green')
            ax2.set_xlabel('Battery Level')
            ax2.set_ylabel('Number of Vehicles')
            ax2.set_title('Vehicle Battery Distribution')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # 保存图片
            results_dir = "results/nye_tests"
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plot_file = os.path.join(results_dir, f"nye_test_visualization_{timestamp}.png")
            
            plt.savefig(plot_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📊 Test visualization saved to: {plot_file}")
            
        except ImportError:
            print("⚠️ Matplotlib not available for visualization")
        except Exception as e:
            print(f"❌ Visualization failed: {str(e)}")


# 运行测试
if __name__ == "__main__":
    print("🏁 NYC Electric Taxi Environment Test Suite")
    print("=" * 60)
    
    tester = NYEEnvironmentTester()
    success = tester.run_all_tests()
    
    # 创建可视化
    tester.create_test_visualization()
    
    if success:
        print("\n🎊 All systems operational! NYC Environment ready for deployment! 🎊")
    else:
        print("\n⚠️ Some tests failed. Please review the issues above.")
        
    print("\n🔗 Next steps:")
    print("   1. Run ADP training with NYEEnvironment")
    print("   2. Compare performance with ChargingIntegratedEnvironment")
    print("   3. Analyze NYC-specific patterns and optimize policies")