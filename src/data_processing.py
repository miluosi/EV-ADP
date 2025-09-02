"""
Data preprocessing and graph construction for OD demand data.
"""

import pandas as pd
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from torch_geometric.utils import from_networkx
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional, Dict, List
import logging

logger = logging.getLogger(__name__)


class ODDataProcessor:
    """
    Processor for Origin-Destination demand data.
    Handles data loading, preprocessing, and graph construction.
    """
    
    def __init__(self, scaling_method: str = 'standard'):
        """
        Initialize the OD data processor.
        
        Args:
            scaling_method: 'standard' or 'minmax' for feature scaling
        """
        self.scaling_method = scaling_method
        self.scaler = StandardScaler() if scaling_method == 'standard' else MinMaxScaler()
        self.node_features = None
        self.edge_features = None
        
    def load_od_data(self, file_path: str, 
                     origin_col: str = 'origin',
                     dest_col: str = 'destination', 
                     demand_col: str = 'demand',
                     **kwargs) -> pd.DataFrame:
        """
        Load OD demand data from file.
        
        Args:
            file_path: Path to the data file (CSV, Excel, etc.)
            origin_col: Name of origin column
            dest_col: Name of destination column  
            demand_col: Name of demand column
            **kwargs: Additional arguments for pandas read functions
            
        Returns:
            Loaded and validated DataFrame
        """
        logger.info(f"Loading OD data from {file_path}")
        
        # Determine file type and load accordingly
        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path, **kwargs)
        elif file_path.endswith('.xlsx') or file_path.endswith('.xls'):
            df = pd.read_excel(file_path, **kwargs)
        else:
            raise ValueError("Unsupported file format. Use CSV or Excel files.")
            
        # Validate required columns
        required_cols = [origin_col, dest_col, demand_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
            
        logger.info(f"Loaded {len(df)} OD records")
        return df
    
    def preprocess_od_data(self, df: pd.DataFrame,
                          origin_col: str = 'origin',
                          dest_col: str = 'destination',
                          demand_col: str = 'demand',
                          time_col: Optional[str] = None) -> pd.DataFrame:
        """
        Preprocess OD demand data.
        
        Args:
            df: Raw OD data
            origin_col: Origin column name
            dest_col: Destination column name
            demand_col: Demand column name
            time_col: Optional time column for temporal analysis
            
        Returns:
            Preprocessed DataFrame
        """
        logger.info("Preprocessing OD data")
        processed_df = df.copy()
        
        # Remove zero or negative demand
        processed_df = processed_df[processed_df[demand_col] > 0]
        
        # Handle missing values
        processed_df = processed_df.dropna(subset=[origin_col, dest_col, demand_col])
        
        # Aggregate duplicate OD pairs if any
        group_cols = [origin_col, dest_col]
        if time_col and time_col in processed_df.columns:
            group_cols.append(time_col)
            
        processed_df = processed_df.groupby(group_cols)[demand_col].sum().reset_index()
        
        logger.info(f"Preprocessed data shape: {processed_df.shape}")
        return processed_df
        
    def create_zone_features(self, df: pd.DataFrame,
                           zone_info_path: Optional[str] = None,
                           origin_col: str = 'origin',
                           dest_col: str = 'destination',
                           demand_col: str = 'demand') -> pd.DataFrame:
        """
        Create node features for zones based on OD patterns.
        
        Args:
            df: Preprocessed OD data
            zone_info_path: Optional path to additional zone information
            origin_col: Origin column name
            dest_col: Destination column name
            demand_col: Demand column name
            
        Returns:
            DataFrame with zone features
        """
        logger.info("Creating zone features")
        
        # Get all unique zones
        zones = set(df[origin_col].unique()) | set(df[dest_col].unique())
        
        # Calculate outflow (total demand originating from each zone)
        outflow = df.groupby(origin_col)[demand_col].sum().to_dict()
        
        # Calculate inflow (total demand destined to each zone)
        inflow = df.groupby(dest_col)[demand_col].sum().to_dict()
        
        # Calculate number of connections
        out_connections = df.groupby(origin_col)[dest_col].nunique().to_dict()
        in_connections = df.groupby(dest_col)[origin_col].nunique().to_dict()
        
        # Create feature DataFrame
        zone_features = []
        for zone in zones:
            features = {
                'zone_id': zone,
                'outflow': outflow.get(zone, 0),
                'inflow': inflow.get(zone, 0),
                'out_connections': out_connections.get(zone, 0),
                'in_connections': in_connections.get(zone, 0),
            }
            
            # Calculate derived features
            features['net_flow'] = features['outflow'] - features['inflow']
            features['total_flow'] = features['outflow'] + features['inflow']
            features['total_connections'] = features['out_connections'] + features['in_connections']
            
            zone_features.append(features)
            
        zone_df = pd.DataFrame(zone_features)
        
        # Load additional zone information if provided
        if zone_info_path:
            try:
                zone_info = pd.read_csv(zone_info_path)
                zone_df = zone_df.merge(zone_info, on='zone_id', how='left')
                logger.info("Merged with additional zone information")
            except Exception as e:
                logger.warning(f"Could not load zone info: {e}")
                
        self.node_features = zone_df
        logger.info(f"Created features for {len(zone_df)} zones")
        return zone_df
        
    def construct_graph(self, df: pd.DataFrame,
                       zone_features: pd.DataFrame,
                       origin_col: str = 'origin',
                       dest_col: str = 'destination',
                       demand_col: str = 'demand',
                       threshold: float = 0.0) -> Data:
        """
        Construct PyTorch Geometric graph from OD data.
        
        Args:
            df: Preprocessed OD data
            zone_features: Zone feature DataFrame
            origin_col: Origin column name
            dest_col: Destination column name
            demand_col: Demand column name
            threshold: Minimum demand threshold for edges
            
        Returns:
            PyTorch Geometric Data object
        """
        logger.info("Constructing graph from OD data")
        
        # Create zone ID mapping
        zones = sorted(zone_features['zone_id'].unique())
        zone_to_idx = {zone: idx for idx, zone in enumerate(zones)}
        
        # Filter edges by threshold
        filtered_df = df[df[demand_col] >= threshold].copy()
        
        # Create edge index and attributes
        edge_list = []
        edge_attr = []
        
        for _, row in filtered_df.iterrows():
            origin_idx = zone_to_idx[row[origin_col]]
            dest_idx = zone_to_idx[row[dest_col]]
            demand = row[demand_col]
            
            edge_list.append([origin_idx, dest_idx])
            edge_attr.append([demand])
            
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)
        
        # Prepare node features
        feature_cols = [col for col in zone_features.columns if col != 'zone_id']
        node_features = zone_features[feature_cols].values
        
        # Scale features
        node_features_scaled = self.scaler.fit_transform(node_features)
        x = torch.tensor(node_features_scaled, dtype=torch.float)
        
        # Create PyTorch Geometric data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(zones)
        )
        
        # Store metadata
        data.zone_mapping = zone_to_idx
        data.feature_names = feature_cols
        
        logger.info(f"Graph created: {data.num_nodes} nodes, {data.num_edges} edges")
        return data
        
    def create_spatial_graph(self, zone_coordinates: pd.DataFrame,
                           zone_features: pd.DataFrame,
                           k_neighbors: int = 5,
                           distance_threshold: Optional[float] = None) -> Data:
        """
        Create spatial graph based on zone coordinates.
        
        Args:
            zone_coordinates: DataFrame with zone_id, lat, lon columns
            zone_features: Zone feature DataFrame
            k_neighbors: Number of nearest neighbors to connect
            distance_threshold: Maximum distance for connections
            
        Returns:
            PyTorch Geometric Data object
        """
        from sklearn.neighbors import NearestNeighbors
        from geopy.distance import geodesic
        
        logger.info("Creating spatial graph")
        
        # Merge coordinates with features
        merged_df = zone_features.merge(zone_coordinates, on='zone_id', how='inner')
        
        # Create coordinate matrix
        coords = merged_df[['lat', 'lon']].values
        
        # Find k-nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=k_neighbors + 1, metric='haversine').fit(np.radians(coords))
        distances, indices = nbrs.kneighbors(np.radians(coords))
        
        # Create edge list
        edge_list = []
        edge_weights = []
        
        for i, neighbors in enumerate(indices):
            for j, neighbor_idx in enumerate(neighbors[1:]):  # Skip self
                distance_km = distances[i][j + 1] * 6371  # Convert to km
                
                if distance_threshold is None or distance_km <= distance_threshold:
                    edge_list.append([i, neighbor_idx])
                    edge_weights.append([1.0 / (1.0 + distance_km)])  # Inverse distance weight
                    
        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_weights, dtype=torch.float)
        
        # Prepare node features
        feature_cols = [col for col in merged_df.columns if col not in ['zone_id', 'lat', 'lon']]
        node_features = merged_df[feature_cols].values
        node_features_scaled = self.scaler.fit_transform(node_features)
        x = torch.tensor(node_features_scaled, dtype=torch.float)
        
        # Create data object
        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=len(merged_df)
        )
        
        data.zone_mapping = {zone_id: idx for idx, zone_id in enumerate(merged_df['zone_id'])}
        data.coordinates = coords
        data.feature_names = feature_cols
        
        logger.info(f"Spatial graph created: {data.num_nodes} nodes, {data.num_edges} edges")
        return data


def generate_synthetic_od_data(num_zones: int = 100, 
                              num_records: int = 10000,
                              seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate synthetic OD demand data for testing.
    
    Args:
        num_zones: Number of zones
        num_records: Number of OD records
        seed: Random seed
        
    Returns:
        Tuple of (OD data, zone coordinates)
    """
    np.random.seed(seed)
    
    # Generate zone coordinates (simulate a city grid)
    coords = []
    for i in range(num_zones):
        lat = 40.7 + np.random.normal(0, 0.1)  # Around NYC
        lon = -74.0 + np.random.normal(0, 0.1)
        coords.append({'zone_id': i, 'lat': lat, 'lon': lon})
    
    zone_coords = pd.DataFrame(coords)
    
    # Generate OD records
    od_records = []
    for _ in range(num_records):
        origin = np.random.randint(0, num_zones)
        
        # Bias towards nearby destinations
        dest_probs = np.exp(-0.1 * np.abs(np.arange(num_zones) - origin))
        dest_probs[origin] = 0  # No self-loops
        dest_probs /= dest_probs.sum()
        
        destination = np.random.choice(num_zones, p=dest_probs)
        demand = np.random.lognormal(2, 1)  # Log-normal demand
        
        od_records.append({
            'origin': origin,
            'destination': destination,
            'demand': demand
        })
    
    od_data = pd.DataFrame(od_records)
    
    return od_data, zone_coords
