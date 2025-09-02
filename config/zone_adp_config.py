"""
Zone-ADP配置管理模块

此模块定义了Zone-ADP系统的所有配置参数，包括环境设置、网络超参数、训练参数等。
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
import torch

@dataclass
class ZoneEnvironmentConfig:
    """Zone环境配置"""
    num_agents: int = 10
    num_locations: int = 265
    num_zones: int = 5
    max_capacity: int = 4
    episode_length: int = 1000
    
    # 奖励函数参数
    balance_weight: float = 1.0
    demand_weight: float = 2.0
    coordination_weight: float = 0.5
    
    # 环境动态参数
    demand_noise_std: float = 0.1
    state_update_noise: float = 0.1


@dataclass
class NetworkConfig:
    """神经网络配置"""
    # Zone感知价值网络
    input_dim: int = 8
    zone_embed_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256, 128])
    dropout: float = 0.1
    
    # 路径处理
    location_embed_dim: int = 100
    lstm_hidden_size: int = 200
    max_sequence_length: int = 20
    
    # 初始化参数
    weight_init: str = "xavier"  # xavier, kaiming, normal
    bias_init: float = 0.0


@dataclass
class TrainingConfig:
    """训练配置"""
    # 基本训练参数
    learning_rate: float = 1e-3
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.001  # 软更新参数
    
    # 经验回放
    buffer_size: int = 100000
    min_buffer_size: int = 1000
    priority_alpha: float = 0.6
    priority_beta: float = 0.4
    priority_beta_schedule: bool = True
    
    # 探索策略
    epsilon_start: float = 1.0
    epsilon_end: float = 0.01
    epsilon_decay: float = 0.995
    
    # 训练调度
    target_update_freq: int = 100
    save_freq: int = 1000
    eval_freq: int = 100
    
    # 梯度控制
    max_grad_norm: float = 1.0
    weight_decay: float = 1e-5


@dataclass
class VGNNIntegrationConfig:
    """VGNN集成配置"""
    # 策略更新频率
    policy_update_freq: int = 100
    policy_blend_ratio: float = 0.7  # VGNN策略与探索的混合比例
    
    # 策略生成参数
    num_policies: int = 10
    similarity_threshold: float = 0.85
    policy_temperature: float = 1.0
    
    # 集成方式
    integration_mode: str = "weighted"  # weighted, switching, ensemble
    adaptation_rate: float = 0.1


@dataclass
class LoggingConfig:
    """日志配置"""
    log_dir: str = "logs/zone_adp"
    save_dir: str = "models/zone_adp"
    
    # TensorBoard日志
    log_scalar_freq: int = 10
    log_histogram_freq: int = 100
    log_image_freq: int = 500
    
    # 控制台输出
    console_log_level: str = "INFO"
    print_freq: int = 10
    
    # 模型保存
    save_best_only: bool = True
    monitor_metric: str = "episode_reward"


@dataclass
@dataclass
class ZoneADPConfig:
    """Zone-ADP完整配置"""
    environment: ZoneEnvironmentConfig = field(default_factory=ZoneEnvironmentConfig)
    network: NetworkConfig = field(default_factory=NetworkConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    vgnn_integration: VGNNIntegrationConfig = field(default_factory=VGNNIntegrationConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # 设备配置
    device: str = "auto"  # auto, cpu, cuda
    seed: int = 42
    
    # 实验配置
    experiment_name: str = "zone_adp_default"
    description: str = "Zone-aware ADP with VGNN integration"
    
    def __post_init__(self):
        """后处理配置"""
        # 自动检测设备
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 验证配置合理性
        self._validate_config()
    
    # 为了向后兼容，添加直接属性访问器
    @property
    def num_agents(self) -> int:
        return self.environment.num_agents
    
    @num_agents.setter
    def num_agents(self, value: int):
        self.environment.num_agents = value
    
    @property
    def num_zones(self) -> int:
        return self.environment.num_zones
    
    @num_zones.setter
    def num_zones(self, value: int):
        self.environment.num_zones = value
    
    @property
    def num_locations(self) -> int:
        return self.environment.num_locations
    
    @num_locations.setter
    def num_locations(self, value: int):
        self.environment.num_locations = value
    
    @property
    def max_capacity(self) -> int:
        return self.environment.max_capacity
    
    @max_capacity.setter
    def max_capacity(self, value: int):
        self.environment.max_capacity = value
    
    @property
    def episode_length(self) -> int:
        return self.environment.episode_length
    
    @episode_length.setter
    def episode_length(self, value: int):
        self.environment.episode_length = value
    
    @property
    def learning_rate(self) -> float:
        return self.training.learning_rate
    
    @learning_rate.setter
    def learning_rate(self, value: float):
        self.training.learning_rate = value
    
    @property
    def gamma(self) -> float:
        return self.training.gamma
    
    @gamma.setter
    def gamma(self, value: float):
        self.training.gamma = value
    
    @property
    def tau(self) -> float:
        return self.training.tau
    
    @tau.setter
    def tau(self, value: float):
        self.training.tau = value
    
    @property
    def batch_size(self) -> int:
        return self.training.batch_size
    
    @batch_size.setter
    def batch_size(self, value: int):
        self.training.batch_size = value
    
    @property
    def buffer_size(self) -> int:
        return self.training.buffer_size
    
    @buffer_size.setter
    def buffer_size(self, value: int):
        self.training.buffer_size = value
    
    @property
    def hidden_dims(self) -> List[int]:
        return self.network.hidden_dims
    
    @hidden_dims.setter
    def hidden_dims(self, value: List[int]):
        self.network.hidden_dims = value
    
    @property
    def dropout(self) -> float:
        return self.network.dropout
    
    @dropout.setter
    def dropout(self, value: float):
        self.network.dropout = value
    
    @property
    def zone_embed_dim(self) -> int:
        return self.network.zone_embed_dim
    
    @zone_embed_dim.setter
    def zone_embed_dim(self, value: int):
        self.network.zone_embed_dim = value
    
    @property
    def use_prioritized_replay(self) -> bool:
        return self.training.use_prioritized_replay
    
    @use_prioritized_replay.setter
    def use_prioritized_replay(self, value: bool):
        self.training.use_prioritized_replay = value
    
    @property
    def use_double_dqn(self) -> bool:
        return self.training.use_double_dqn
    
    @use_double_dqn.setter
    def use_double_dqn(self, value: bool):
        self.training.use_double_dqn = value
    
    @property
    def num_episodes(self) -> int:
        return self.training.num_episodes
    
    @num_episodes.setter
    def num_episodes(self, value: int):
        self.training.num_episodes = value
    
    def _validate_config(self):
        """验证配置参数的合理性"""
        # 环境参数验证
        assert self.environment.num_agents > 0, "智能体数量必须大于0"
        assert self.environment.num_zones > 0, "区域数量必须大于0"
        assert self.environment.num_locations > 0, "位置数量必须大于0"
        
        # 网络参数验证
        assert self.network.input_dim > 0, "输入维度必须大于0"
        assert len(self.network.hidden_dims) > 0, "必须至少有一个隐藏层"
        assert 0 <= self.network.dropout <= 1, "Dropout率必须在[0,1]范围内"
        
        # 训练参数验证
        assert self.training.learning_rate > 0, "学习率必须大于0"
        assert self.training.batch_size > 0, "批次大小必须大于0"
        assert 0 <= self.training.gamma <= 1, "折扣因子必须在[0,1]范围内"
        assert 0 < self.training.tau <= 1, "软更新参数必须在(0,1]范围内"
        
        # VGNN集成验证
        assert 0 <= self.vgnn_integration.policy_blend_ratio <= 1, "策略混合比例必须在[0,1]范围内"
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            'environment': self.environment.__dict__,
            'network': self.network.__dict__,
            'training': self.training.__dict__,
            'vgnn_integration': self.vgnn_integration.__dict__,
            'logging': self.logging.__dict__,
            'device': self.device,
            'seed': self.seed,
            'experiment_name': self.experiment_name,
            'description': self.description
        }
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ZoneADPConfig':
        """从字典创建配置"""
        config = cls()
        
        # 更新各个子配置
        if 'environment' in config_dict:
            for k, v in config_dict['environment'].items():
                setattr(config.environment, k, v)
        
        if 'network' in config_dict:
            for k, v in config_dict['network'].items():
                setattr(config.network, k, v)
        
        if 'training' in config_dict:
            for k, v in config_dict['training'].items():
                setattr(config.training, k, v)
        
        if 'vgnn_integration' in config_dict:
            for k, v in config_dict['vgnn_integration'].items():
                setattr(config.vgnn_integration, k, v)
        
        if 'logging' in config_dict:
            for k, v in config_dict['logging'].items():
                setattr(config.logging, k, v)
        
        # 更新顶级配置
        for key in ['device', 'seed', 'experiment_name', 'description']:
            if key in config_dict:
                setattr(config, key, config_dict[key])
        
        return config
    
    def __init__(self, 
                 # Environment parameters
                 num_agents: Optional[int] = None,
                 num_zones: Optional[int] = None,
                 num_locations: Optional[int] = None,
                 max_capacity: Optional[int] = None,
                 episode_length: Optional[int] = None,
                 # Network parameters
                 input_dim: Optional[int] = None,
                 zone_embed_dim: Optional[int] = None,
                 hidden_dims: Optional[List[int]] = None,
                 dropout: Optional[float] = None,
                 # Training parameters
                 learning_rate: Optional[float] = None,
                 gamma: Optional[float] = None,
                 tau: Optional[float] = None,
                 batch_size: Optional[int] = None,
                 buffer_size: Optional[int] = None,
                 num_episodes: Optional[int] = None,
                 use_prioritized_replay: Optional[bool] = None,
                 use_double_dqn: Optional[bool] = None,
                 # Other parameters
                 device: Optional[str] = None,
                 seed: Optional[int] = None,
                 experiment_name: Optional[str] = None,
                 description: Optional[str] = None,
                 **kwargs):
        """
        初始化Zone-ADP配置
        
        可以直接传递参数，也可以使用默认的dataclass初始化
        """
        # 初始化子配置
        self.environment = ZoneEnvironmentConfig()
        self.network = NetworkConfig()
        self.training = TrainingConfig()
        self.vgnn_integration = VGNNIntegrationConfig()
        self.logging = LoggingConfig()
        
        # 设置默认值
        self.device = device or "auto"
        self.seed = seed or 42
        self.experiment_name = experiment_name or "zone_adp_default"
        self.description = description or "Zone-aware ADP with VGNN integration"
        
        # 更新环境参数
        if num_agents is not None:
            self.environment.num_agents = num_agents
        if num_zones is not None:
            self.environment.num_zones = num_zones
        if num_locations is not None:
            self.environment.num_locations = num_locations
        if max_capacity is not None:
            self.environment.max_capacity = max_capacity
        if episode_length is not None:
            self.environment.episode_length = episode_length
            
        # 更新网络参数
        if input_dim is not None:
            self.network.input_dim = input_dim
        if zone_embed_dim is not None:
            self.network.zone_embed_dim = zone_embed_dim
        if hidden_dims is not None:
            self.network.hidden_dims = hidden_dims
        if dropout is not None:
            self.network.dropout = dropout
            
        # 更新训练参数
        if learning_rate is not None:
            self.training.learning_rate = learning_rate
        if gamma is not None:
            self.training.gamma = gamma
        if tau is not None:
            self.training.tau = tau
        if batch_size is not None:
            self.training.batch_size = batch_size
        if buffer_size is not None:
            self.training.buffer_size = buffer_size
        if num_episodes is not None:
            self.training.num_episodes = num_episodes
        if use_prioritized_replay is not None:
            self.training.use_prioritized_replay = use_prioritized_replay
        if use_double_dqn is not None:
            self.training.use_double_dqn = use_double_dqn
        
        # 处理其他关键字参数
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
        
        # 执行后处理
        self.__post_init__()


# 预定义配置模板
def get_default_config() -> ZoneADPConfig:
    """获取默认配置"""
    return ZoneADPConfig()


def get_small_test_config() -> ZoneADPConfig:
    """获取小规模测试配置"""
    config = ZoneADPConfig()
    
    # 小规模环境
    config.environment.num_agents = 3
    config.environment.num_locations = 20
    config.environment.num_zones = 2
    config.environment.episode_length = 100
    
    # 简化网络
    config.network.hidden_dims = [32, 16]
    config.network.zone_embed_dim = 16
    
    # 快速训练
    config.training.batch_size = 16
    config.training.buffer_size = 1000
    config.training.target_update_freq = 10
    
    # 频繁日志
    config.logging.print_freq = 5
    config.logging.save_freq = 50
    
    config.experiment_name = "zone_adp_small_test"
    config.description = "Small-scale test configuration"
    
    return config


def get_large_scale_config() -> ZoneADPConfig:
    """获取大规模生产配置"""
    config = ZoneADPConfig()
    
    # 大规模环境
    config.environment.num_agents = 50
    config.environment.num_locations = 1000
    config.environment.num_zones = 20
    config.environment.episode_length = 5000
    
    # 大容量网络
    config.network.hidden_dims = [512, 512, 256, 128]
    config.network.zone_embed_dim = 128
    
    # 稳定训练
    config.training.batch_size = 128
    config.training.buffer_size = 500000
    config.training.learning_rate = 5e-4
    config.training.target_update_freq = 500
    
    # VGNN深度集成
    config.vgnn_integration.num_policies = 20
    config.vgnn_integration.policy_update_freq = 200
    
    config.experiment_name = "zone_adp_large_scale"
    config.description = "Large-scale production configuration"
    
    return config


def get_research_config() -> ZoneADPConfig:
    """获取研究用配置"""
    config = ZoneADPConfig()
    
    # 中等规模
    config.environment.num_agents = 20
    config.environment.num_locations = 265  # NYC taxi zones
    config.environment.num_zones = 10
    config.environment.episode_length = 2000
    
    # 平衡的网络
    config.network.hidden_dims = [256, 256, 128]
    config.network.zone_embed_dim = 64
    
    # 研究友好的训练设置
    config.training.batch_size = 64
    config.training.buffer_size = 100000
    config.training.epsilon_decay = 0.999  # 慢速探索衰减
    
    # 详细的VGNN集成
    config.vgnn_integration.num_policies = 15
    config.vgnn_integration.integration_mode = "ensemble"
    config.vgnn_integration.policy_update_freq = 100
    
    # 详细日志
    config.logging.log_histogram_freq = 50
    config.logging.save_freq = 500
    
    config.experiment_name = "zone_adp_research"
    config.description = "Research configuration with detailed logging"
    
    return config


# 配置工厂函数
CONFIG_PRESETS = {
    'default': get_default_config,
    'small_test': get_small_test_config,
    'large_scale': get_large_scale_config,
    'research': get_research_config
}


def get_config(preset_name: str = 'default') -> ZoneADPConfig:
    """根据预设名称获取配置"""
    if preset_name not in CONFIG_PRESETS:
        raise ValueError(f"未知的配置预设: {preset_name}。可用预设: {list(CONFIG_PRESETS.keys())}")
    
    return CONFIG_PRESETS[preset_name]()


def save_config(config: ZoneADPConfig, filepath: str):
    """保存配置到文件"""
    import json
    from pathlib import Path
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(config.to_dict(), f, indent=2, ensure_ascii=False)


def load_config(filepath: str) -> ZoneADPConfig:
    """从文件加载配置"""
    import json
    from pathlib import Path
    
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"配置文件不存在: {filepath}")
    
    with open(filepath, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)
    
    return ZoneADPConfig.from_dict(config_dict)


if __name__ == "__main__":
    # 测试配置系统
    print("测试Zone-ADP配置系统")
    
    # 测试默认配置
    default_config = get_default_config()
    print(f"默认配置 - 智能体数: {default_config.environment.num_agents}")
    print(f"默认配置 - 设备: {default_config.device}")
    
    # 测试小规模配置
    small_config = get_config('small_test')
    print(f"小规模配置 - 智能体数: {small_config.environment.num_agents}")
    
    # 测试配置保存和加载
    test_path = "test_config.json"
    save_config(small_config, test_path)
    loaded_config = load_config(test_path)
    print(f"加载的配置 - 智能体数: {loaded_config.environment.num_agents}")
    
    # 清理测试文件
    import os
    if os.path.exists(test_path):
        os.remove(test_path)
    
    print("配置系统测试完成!")
