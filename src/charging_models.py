"""
Charging and relocation probability models from rsimulation_detail.ipynb.

These models implement:
- ChargingProbabilityCalculator: Nested logit model from Yang et al. (2016)
- RelocationManager: Driver relocation model from Ashkrof et al. (2024)
"""

import numpy as np
import math


class ChargingProbabilityCalculator:
    """
    基于 Yang et al. (2016) 的嵌套 Logit 模型计算充电概率。
    适用于 ChargingIntegratedEnvironment 的网格环境。
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
        # 模型系数 (参考 Yang et al. 2016, Table 3, Model M4) 
        self.coeffs = {
            # 下层: 充电站选择 (Route with charging)
            'beta_travel_time': -0.105,   # 行驶时间系数 (假设与距离成正比)
            'beta_travel_cost': -0.127,   # 行驶成本系数
            'beta_charge_time': -0.079,   # 充电时间系数
            'beta_distance_os': -0.130,   # 起点到充电站距离系数
            'beta_aocd': -0.387,          # 角度成本系数 (AOCD)
            
            # 上层: 是否充电 (Charging decision)
            'alpha_soc': 0.163,           # 初始 SOC 系数 (正值表示 SOC 越高越倾向于不充电)
            'asc_no_charge': -11.003,     # 不充电的替代特定常数
            'mu_charge': 0.468,           # 充电巢的包容值系数 (Inclusive Value Coefficient/Scale parameter)
            
            # 假设转换因子 (将网格距离转换为分钟/成本)
            'grid_to_minutes': 2.0,       # 假设网格每移动一格需要 2 分钟
            'grid_to_cost': 0.5           # 假设网格每移动一格花费 0.5 单位
        }

    def _get_coords(self, location_id):
        """将一维 ID 转换为 (x, y) 网格坐标"""
        x = location_id % self.grid_size
        y = location_id // self.grid_size
        return np.array([x, y])

    def _calculate_manhattan_distance(self, pos1, pos2):
        """计算网格上的曼哈顿距离"""
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def _calculate_aocd(self, origin_coords, station_coords, dest_coords):
        """
        计算 AOCD (Angular formed by Origin, Charging Station, and Destination)
        返回值为弧度 (radians)
        """
        # 向量 O->S
        vec_os = station_coords - origin_coords
        # 向量 O->D
        vec_od = dest_coords - origin_coords
        
        # 如果起点和终点重合，或起点和充电站重合，角度为 0
        norm_os = np.linalg.norm(vec_os)
        norm_od = np.linalg.norm(vec_od)
        
        if norm_os == 0 or norm_od == 0:
            return 0.0
            
        # 计算余弦相似度
        cos_theta = np.dot(vec_os, vec_od) / (norm_os * norm_od)
        # 截断以防止数值误差导致 acos 报错
        cos_theta = np.clip(cos_theta, -1.0, 1.0)
        
        return np.arccos(cos_theta)

    def calculate_probabilities(self, origin_id, dest_id, current_soc, charging_stations):
        """
        计算选择各个充电站的概率以及选择不充电的概率。
        
        Args:
            origin_id (int): 起点 ID
            dest_id (int): 终点 ID
            current_soc (float): 当前 SOC (0-100)
            charging_stations (list of dict): 充电站列表，每个元素形如 
                                            {'id': int, 'estimated_time': float}
        
        Returns:
            dict: 包含 'action_no_charge', 'action_charge', 和每个站点的选择概率 'station_probs'
        """
        origin_coords = self._get_coords(origin_id)
        dest_coords = self._get_coords(dest_id)
        
        # --- 1. 计算下层：各充电站路线的效用 (Utility of routes with charging) ---
        utilities_stations = []
        station_indices = []
        
        for station in charging_stations:
            station_id = station['id']
            t_charge = station.get('estimated_time', 30.0) 
            
            station_coords = self._get_coords(station_id)
            
            # 计算路线属性: Origin -> Station -> Destination
            dist_os = self._calculate_manhattan_distance(origin_coords, station_coords)
            dist_sd = self._calculate_manhattan_distance(station_coords, dest_coords)
            total_dist = dist_os + dist_sd
            
            travel_time = total_dist * self.coeffs['grid_to_minutes']
            travel_cost = total_dist * self.coeffs['grid_to_cost']
            
            # 计算角度成本 AOCD
            angle = self._calculate_aocd(origin_coords, station_coords, dest_coords)
            
            # 应用公式: V_k = B_t*T + B_c*C + B_ct*T_charge + B_dist*Dist_OS + B_ang*Angle
            utility = (
                self.coeffs['beta_travel_time'] * travel_time +
                self.coeffs['beta_travel_cost'] * travel_cost +
                self.coeffs['beta_charge_time'] * t_charge +
                self.coeffs['beta_distance_os'] * dist_os +
                self.coeffs['beta_aocd'] * angle
            )
            
            utilities_stations.append(utility)
            station_indices.append(station_id)
        
        utilities_stations = np.array(utilities_stations)
        
        # --- 2. 计算上层：是否充电的效用 (Utility of Charging Decision) ---
        
        # 计算充电巢的包容值 (Inclusive Value): IV_charge = ln(sum(exp(V_k)))
        max_u = np.max(utilities_stations)
        sum_exp_stations = np.sum(np.exp(utilities_stations - max_u))
        iv_charge = max_u + np.log(sum_exp_stations)
        
        # 计算充电的总效用 V_charge
        v_charge = self.coeffs['mu_charge'] * iv_charge
        
        # 计算不充电的效用 V_no_charge
        v_no_charge = (self.coeffs['alpha_soc'] * current_soc) + self.coeffs['asc_no_charge']
        
        # --- 3. 计算最终概率 ---
        
        # 上层概率 P(Charge) vs P(No Charge)
        max_upper = max(v_charge, v_no_charge)
        exp_c = np.exp(v_charge - max_upper)
        exp_nc = np.exp(v_no_charge - max_upper)
        sum_upper = exp_c + exp_nc
        
        p_choose_charge = exp_c / sum_upper
        p_no_charge = exp_nc / sum_upper
        
        # 下层条件概率 P(k | Charge)
        p_station_given_charge = np.exp(utilities_stations - max_u) / sum_exp_stations
        
        # 联合概率 P(k) = P(k|C) * P(C)
        final_station_probs = p_station_given_charge * p_choose_charge
        
        result = {
            'action_no_charge': p_no_charge,
            'action_charge': p_choose_charge,
            'station_probs': dict(zip(station_indices, final_station_probs))
        }
        
        return result


class RelocationManager:
    """
    基于 Ashkrof et al. (2024) 管理司机的重定位行为。
    处理 Zone 定义、Surge 区域以及概率计算。
    """
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.surge_zones = set()
        self.high_demand_zones = set()
        self.city_center_zones = set()
        
        # 模型系数 (基于 Table 3, Scenario 1 Full Model)
        self.coeffs = {
            # Waiting (W)
            'asc_waiting': -0.283,
            'beta_wait_time': -0.022,
            'beta_loc_city': 0.322,
            'beta_parking': 0.277,
            'beta_weekend': 0.350,
            
            # Surge Area (S) & High Demand (H)
            'beta_trips_sh': 0.080,
            
            # Surge Area Only (S)
            'beta_surge_price': 0.177,
            'beta_time_to_surge': -0.020,
            
            # High Demand Only (H)
            'beta_time_to_hd': -0.037,
            
            # Cruising (C)
            'beta_familiarity_c': -0.312,
            
            # 辅助转换
            'grid_to_min': 2.0
        }

    def update_zone_info(self, surge_ids, hd_ids, city_center_ids):
        """更新网格的区域属性 (由环境每 Epoch 调用)"""
        self.surge_zones = set(surge_ids) if surge_ids else set()
        self.high_demand_zones = set(hd_ids) if hd_ids else set()
        self.city_center_zones = set(city_center_ids) if city_center_ids else set()

    def _get_coords(self, location_id):
        return np.array([location_id % self.grid_size, location_id // self.grid_size])

    def _manhattan_dist(self, id1, id2):
        pos1 = self._get_coords(id1)
        pos2 = self._get_coords(id2)
        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def _find_nearest_zone_dist(self, current_loc, target_zones):
        """找到最近的目标区域网格点的距离"""
        if not target_zones:
            return float('inf'), None
        
        min_dist = float('inf')
        nearest_id = None
        
        for zid in target_zones:
            d = self._manhattan_dist(current_loc, zid)
            if d < min_dist:
                min_dist = d
                nearest_id = zid
        return min_dist, nearest_id

    def calculate_relocation_utilities(self, agent_state, global_state):
        """
        计算四个重定位选项的效用
        
        Returns:
            tuple: (utilities_dict, targets_dict)
        """
        loc = agent_state['location']
        trips = agent_state['completed_trips']
        wait_t = agent_state['current_wait_time']
        
        # 1. Utility of Waiting (W)
        is_city = 1 if loc in self.city_center_zones else 0
        has_parking = 1
        is_weekend = global_state.get('is_weekend', 0)
        
        u_wait = (self.coeffs['asc_waiting'] + 
                  self.coeffs['beta_wait_time'] * wait_t +
                  self.coeffs['beta_loc_city'] * is_city + 
                  self.coeffs['beta_parking'] * has_parking +
                  self.coeffs['beta_weekend'] * is_weekend)

        # 2. Utility of Driving to Surge Area (S)
        dist_s, nearest_s = self._find_nearest_zone_dist(loc, self.surge_zones)
        if nearest_s is not None:
            time_s = dist_s * self.coeffs['grid_to_min']
            surge_val = global_state.get('current_surge_price', 0.0)
            
            u_surge = (self.coeffs['beta_trips_sh'] * trips +
                       self.coeffs['beta_surge_price'] * surge_val +
                       self.coeffs['beta_time_to_surge'] * time_s)
        else:
            u_surge = -999.0

        # 3. Utility of Driving to High Demand Area (H)
        dist_h, nearest_h = self._find_nearest_zone_dist(loc, self.high_demand_zones)
        if nearest_h is not None:
            time_h = dist_h * self.coeffs['grid_to_min']
            
            u_hd = (self.coeffs['beta_trips_sh'] * trips +
                    self.coeffs['beta_time_to_hd'] * time_h)
        else:
            u_hd = -999.0

        # 4. Utility of Cruising (C)
        is_familiar = 1 
        u_cruise = self.coeffs['beta_familiarity_c'] * is_familiar
        
        return {
            'Wait': u_wait,
            'Surge': u_surge,
            'HighDemand': u_hd,
            'Cruise': u_cruise
        }, {'Surge': nearest_s, 'HighDemand': nearest_h}

    def get_relocation_decision(self, agent_state, global_state):
        """
        返回具体的决策动作和目标位置
        
        Returns:
            tuple: (choice, target_loc, probs_dict)
        """
        utils, targets = self.calculate_relocation_utilities(agent_state, global_state)
        
        # 计算概率 (Softmax)
        u_vec = np.array(list(utils.values()))
        exp_u = np.exp(u_vec - np.max(u_vec))
        probs = exp_u / np.sum(exp_u)
        
        choices = list(utils.keys())
        choice = np.random.choice(choices, p=probs)
        
        target_loc = None
        if choice == 'Wait':
            target_loc = agent_state['location']
        elif choice == 'Surge':
            target_loc = targets['Surge']
        elif choice == 'HighDemand':
            target_loc = targets['HighDemand']
        elif choice == 'Cruise':
            target_loc = self._get_random_neighbor(agent_state['location'])
            
        return choice, target_loc, dict(zip(choices, probs))

    def _get_random_neighbor(self, loc_id):
        return (loc_id + 1) % (self.grid_size**2)
