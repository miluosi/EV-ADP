"""
Spatial Visualization Module for Request Generation and Vehicle Movement Analysis
"""
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
import pandas as pd

class SpatialVisualization:
    """Visualize spatial patterns of requests and vehicle movements"""
    
    def __init__(self, grid_size):
        self.grid_size = grid_size
        
    def plot_request_and_vehicle_distribution(self, env, save_path=None):
        """
        Create scatter plots showing:
        1. Request generation hotspots
        2. Vehicle most frequently visited locations
        3. Charging station locations
        """
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        
        # Plot 1: Request Generation Hotspots
        if hasattr(env, 'request_generation_history') and env.request_generation_history:
            pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
            dropoff_coords = [req['dropoff_coords'] for req in env.request_generation_history]
            hotspot_indices = [req['hotspot_idx'] for req in env.request_generation_history]
            
            pickup_x = [coord[0] for coord in pickup_coords]
            pickup_y = [coord[1] for coord in pickup_coords]
            dropoff_x = [coord[0] for coord in dropoff_coords]
            dropoff_y = [coord[1] for coord in dropoff_coords]
            
            # Color by hotspot
            colors = ['red', 'blue', 'green']
            hotspot_colors = [colors[idx] for idx in hotspot_indices]
            
            ax1.scatter(pickup_x, pickup_y, c=hotspot_colors, alpha=0.6, s=30, label='Pickup locations')
            ax1.scatter(dropoff_x, dropoff_y, c='orange', alpha=0.3, s=20, marker='x', label='Dropoff locations')
            
            # Mark hotspot centers
            hotspots = [
                (self.grid_size // 4, self.grid_size // 4),
                (3 * self.grid_size // 4, self.grid_size // 4),
                (self.grid_size // 2, 3 * self.grid_size // 4)
            ]
            for i, (hx, hy) in enumerate(hotspots):
                ax1.scatter(hx, hy, c=colors[i], s=200, marker='*', 
                           edgecolors='black', linewidth=2, label=f'Hotspot {i+1}')
            
            ax1.set_xlim(0, self.grid_size-1)
            ax1.set_ylim(0, self.grid_size-1)
            ax1.set_title('Request Generation Distribution\n(Pickup & Dropoff Locations)', fontsize=12)
            ax1.set_xlabel('X Coordinate')
            ax1.set_ylabel('Y Coordinate')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot 2: Vehicle Most Frequently Visited Locations
        if hasattr(env, 'vehicle_position_history') and env.vehicle_position_history:
            all_positions = []
            vehicle_types = []
            
            for vehicle_id, history in env.vehicle_position_history.items():
                vehicle_type = env.vehicles[vehicle_id]['type']
                for pos_record in history:
                    all_positions.append(pos_record['coords'])
                    vehicle_types.append(vehicle_type)
            
            if all_positions:
                # Count frequency of each position
                position_counts = Counter(all_positions)
                
                # Separate by vehicle type
                ev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'EV']
                aev_positions = [pos for pos, vtype in zip(all_positions, vehicle_types) if vtype == 'AEV']
                
                if ev_positions:
                    ev_x = [pos[0] for pos in ev_positions]
                    ev_y = [pos[1] for pos in ev_positions]
                    ax2.scatter(ev_x, ev_y, c='red', alpha=0.4, s=20, label='EV positions')
                
                if aev_positions:
                    aev_x = [pos[0] for pos in aev_positions]
                    aev_y = [pos[1] for pos in aev_positions]
                    ax2.scatter(aev_x, aev_y, c='blue', alpha=0.4, s=20, label='AEV positions')
                
                # Highlight most frequent positions
                most_frequent = position_counts.most_common(10)
                if most_frequent:
                    freq_x = [pos[0] for pos, count in most_frequent]
                    freq_y = [pos[1] for pos, count in most_frequent]
                    freq_sizes = [count * 10 for pos, count in most_frequent]  # Scale by frequency
                    
                    ax2.scatter(freq_x, freq_y, c='yellow', s=freq_sizes, 
                               alpha=0.8, edgecolors='black', linewidth=1,
                               label='High frequency locations')
            
            ax2.set_xlim(0, self.grid_size-1)
            ax2.set_ylim(0, self.grid_size-1)
            ax2.set_title('Vehicle Movement Patterns\n(Most Frequently Visited Locations)', fontsize=12)
            ax2.set_xlabel('X Coordinate')
            ax2.set_ylabel('Y Coordinate')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # Plot 3: Charging Station Distribution and Utilization
        if hasattr(env, 'charging_manager') and env.charging_manager.stations:
            station_coords = []
            station_utilizations = []
            
            for station in env.charging_manager.stations.values():
                x = station.location % self.grid_size
                y = station.location // self.grid_size
                station_coords.append((x, y))
                
                # Calculate utilization
                utilization = len(station.current_vehicles) / station.max_capacity
                station_utilizations.append(utilization)
            
            if station_coords:
                station_x = [coord[0] for coord in station_coords]
                station_y = [coord[1] for coord in station_coords]
                
                # Size by utilization
                station_sizes = [max(50, util * 300) for util in station_utilizations]
                colors_util = plt.cm.RdYlBu_r(station_utilizations)
                
                scatter = ax3.scatter(station_x, station_y, c=colors_util, s=station_sizes,
                                    alpha=0.8, edgecolors='black', linewidth=2)
                
                # Add colorbar for utilization
                cbar = plt.colorbar(scatter, ax=ax3)
                cbar.set_label('Station Utilization Rate', rotation=270, labelpad=15)
                
                # Add station IDs as labels
                for i, (x, y) in enumerate(station_coords):
                    ax3.annotate(f'S{i}', (x, y), xytext=(5, 5), textcoords='offset points',
                               fontsize=8, fontweight='bold')
            
            ax3.set_xlim(0, self.grid_size-1)
            ax3.set_ylim(0, self.grid_size-1)
            ax3.set_title('Charging Station Distribution\n(Size = Utilization Rate)', fontsize=12)
            ax3.set_xlabel('X Coordinate')
            ax3.set_ylabel('Y Coordinate')
            ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Spatial distribution plot saved to: {save_path}")
        
        return fig
    
    def analyze_spatial_patterns(self, env):
        """Analyze spatial patterns and return statistics"""
        analysis = {
            'request_patterns': {},
            'vehicle_patterns': {},
            'station_patterns': {}
        }
        
        # Analyze request patterns
        if hasattr(env, 'request_generation_history') and env.request_generation_history:
            hotspot_counts = Counter([req['hotspot_idx'] for req in env.request_generation_history])
            pickup_coords = [req['pickup_coords'] for req in env.request_generation_history]
            
            analysis['request_patterns'] = {
                'total_requests': len(env.request_generation_history),
                'hotspot_distribution': dict(hotspot_counts),
                'pickup_spread': {
                    'x_range': (min(coord[0] for coord in pickup_coords), 
                              max(coord[0] for coord in pickup_coords)),
                    'y_range': (min(coord[1] for coord in pickup_coords), 
                              max(coord[1] for coord in pickup_coords))
                }
            }
        
        # Analyze vehicle patterns
        if hasattr(env, 'vehicle_position_history') and env.vehicle_position_history:
            all_positions = []
            for history in env.vehicle_position_history.values():
                all_positions.extend([record['coords'] for record in history])
            
            if all_positions:
                position_counts = Counter(all_positions)
                most_visited = position_counts.most_common(5)
                
                analysis['vehicle_patterns'] = {
                    'total_movements': len(all_positions),
                    'unique_positions': len(position_counts),
                    'most_visited_locations': most_visited,
                    'coverage_ratio': len(position_counts) / (self.grid_size * self.grid_size)
                }
        
        # Analyze station patterns
        if hasattr(env, 'charging_manager') and env.charging_manager.stations:
            station_utilizations = []
            for station in env.charging_manager.stations.values():
                utilization = len(station.current_vehicles) / station.max_capacity
                station_utilizations.append(utilization)
            
            analysis['station_patterns'] = {
                'total_stations': len(env.charging_manager.stations),
                'avg_utilization': np.mean(station_utilizations),
                'max_utilization': max(station_utilizations),
                'min_utilization': min(station_utilizations)
            }
        
        return analysis
    
    def print_spatial_analysis(self, analysis):
        """Print spatial analysis results"""
        print("\n" + "="*60)
        print("📍 SPATIAL PATTERN ANALYSIS")
        print("="*60)
        
        # Request patterns
        if analysis['request_patterns']:
            req_patterns = analysis['request_patterns']
            print(f"\n🎯 REQUEST GENERATION PATTERNS:")
            print(f"   Total requests generated: {req_patterns['total_requests']}")
            if 'hotspot_distribution' in req_patterns:
                print(f"   Hotspot distribution:")
                for hotspot_idx, count in req_patterns['hotspot_distribution'].items():
                    percentage = (count / req_patterns['total_requests']) * 100
                    # Handle None hotspot_idx
                    hotspot_display = "Unknown" if hotspot_idx is None else f"{hotspot_idx + 1}"
                    print(f"     Hotspot {hotspot_display}: {count} requests ({percentage:.1f}%)")
        
        # Vehicle patterns
        if analysis['vehicle_patterns']:
            veh_patterns = analysis['vehicle_patterns']
            print(f"\n🚗 VEHICLE MOVEMENT PATTERNS:")
            print(f"   Total vehicle movements: {veh_patterns['total_movements']}")
            print(f"   Unique positions visited: {veh_patterns['unique_positions']}")
            print(f"   Grid coverage ratio: {veh_patterns['coverage_ratio']:.2%}")
            if 'most_visited_locations' in veh_patterns:
                print(f"   Top 5 most visited locations:")
                for i, (coords, count) in enumerate(veh_patterns['most_visited_locations']):
                    print(f"     {i+1}. {coords}: {count} visits")
        
        # Station patterns
        if analysis['station_patterns']:
            station_patterns = analysis['station_patterns']
            print(f"\n🔋 CHARGING STATION PATTERNS:")
            print(f"   Total stations: {station_patterns['total_stations']}")
            print(f"   Average utilization: {station_patterns['avg_utilization']:.1%}")
            print(f"   Utilization range: {station_patterns['min_utilization']:.1%} - {station_patterns['max_utilization']:.1%}")
