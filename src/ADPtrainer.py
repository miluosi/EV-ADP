"""
ADP Training Module - 电动车充电优化训练器
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
from pathlib import Path
import time
from collections import defaultdict, deque
import pandas as pd
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from datetime import datetime
from src.ChargingIntegrationVisualization import ChargingIntegrationVisualization
# 导入配置管理器
from config.config_manager import ConfigManager, get_config, get_training_config, get_sampling_config

# 导入核心组件
from .Environment import ChargingIntegratedEnvironment
from .ValueFunction_pytorch import PyTorchChargingValueFunction
from .Action import Action, ChargingAction, ServiceAction
from .Request import Request
from .charging_station import ChargingStationManager, ChargingStation
from .CentralAgent import CentralAgent
from .SpatialVisualization import SpatialVisualization


class ADPTrainer:
    """ADP训练器类 - 负责电动车充电优化的强化学习训练"""
    
    def __init__(self, config_manager=None):
        """
        初始化训练器
        
        Args:
            config_manager: 配置管理器实例
        """
        self.config_manager = config_manager or ConfigManager()
        self.training_config = self.config_manager.get_training_config()
        self.env_config = self.config_manager.get_environment_config()
        self.sampling_config = self.config_manager.get_sampling_config()
        self.adp_value = self.training_config.get('adp_value', 0)
        self.assignmentgurobi = self.training_config.get('assignmentgurobi', True)
        # 训练状态  
        self.env = None
        self.batch_size = self.training_config.get('batch_size', 256)
        self.value_function = None
        self.training_history = {
            'episode_rewards': [],
            'training_losses': [],
            'q_values': [],
            'exploration_rates': []
        }
        
        print("🚀 ADPTrainer初始化完成")
        print(f"   - 配置加载: {self.config_manager.config_path}")
    
    def setup_environment(self, num_vehicles=None, num_stations=None):
        """
        设置训练环境
        
        Args:
            num_vehicles: 车辆数量，默认从配置获取
            num_stations: 充电站数量，默认从配置获取
        """
        num_vehicles = num_vehicles or self.env_config.get('max_vehicles', 40)
        num_stations = num_stations or self.env_config.get('max_charging_stations', 12)
        
        self.env = ChargingIntegratedEnvironment(
            num_vehicles=num_vehicles, 
            num_stations=num_stations
        )
        
        print(f"✓ 环境设置完成: {num_vehicles}辆车, {num_stations}个充电站")
        return self.env
    
    def setup_value_function(self):
        """
        设置价值函数
        
        Args:
            adp_value: ADP参数值
            use_neural_network: 是否使用神经网络
        """
        use_neural_network = self.adp_value > 0 and self.assignmentgurobi
        if use_neural_network and self.adp_value > 0:
            network_config = self.config_manager.get_network_config()
            
            self.value_function = PyTorchChargingValueFunction(
                grid_size=self.env.grid_size,
                num_vehicles=self.env.num_vehicles,
                device='cuda' if torch.cuda.is_available() else 'cpu',
                episode_length=self.env.episode_length,
                max_requests=1000,
            )
            
            # 设置价值函数到环境
            self.env.set_value_function(self.value_function)
            
            print(f"✓ 神经网络价值函数初始化完成")
            print(f"   - 网络参数数量: {sum(p.numel() for p in self.value_function.network.parameters())}")
            print(f"   - 设备: {self.value_function.device}")
        else:
            self.value_function = None
            print(f"✓ 不使用神经网络 (ADP={self.adp_value})")

        return self.value_function
    
