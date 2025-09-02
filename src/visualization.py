"""
Visualization utilities for GNN clustering results and OD data analysis.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import networkx as nx
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Set style
plt.style.use('default')
sns.set_palette("husl")


class ODVisualization:
    """
    Visualization toolkit for Origin-Destination data and clustering results.
    """
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize OD visualization toolkit.
        
        Args:
            figsize: Default figure size for matplotlib plots
        """
        self.figsize = figsize
        self.color_palette = sns.color_palette("husl", 10)
        
    def plot_od_flow_matrix(self, od_data: pd.DataFrame,
                           origin_col: str = 'origin',
                           dest_col: str = 'destination',
                           demand_col: str = 'demand',
                           top_zones: int = 20,
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot OD flow matrix heatmap.
        
        Args:
            od_data: OD demand data
            origin_col: Origin column name
            dest_col: Destination column name
            demand_col: Demand column name
            top_zones: Number of top zones to display
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Creating OD flow matrix visualization")
        
        # Get top zones by total flow
        zone_flows = (od_data.groupby(origin_col)[demand_col].sum() + 
                     od_data.groupby(dest_col)[demand_col].sum())
        top_zone_ids = zone_flows.nlargest(top_zones).index
        
        # Filter data for top zones
        filtered_data = od_data[
            (od_data[origin_col].isin(top_zone_ids)) & 
            (od_data[dest_col].isin(top_zone_ids))
        ]
        
        # Create pivot table
        flow_matrix = filtered_data.pivot_table(
            index=origin_col, 
            columns=dest_col, 
            values=demand_col, 
            fill_value=0
        )
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=self.figsize)
        
        sns.heatmap(
            flow_matrix, 
            annot=False, 
            cmap='YlOrRd', 
            ax=ax,
            cbar_kws={'label': 'Demand Flow'}
        )
        
        ax.set_title(f'OD Flow Matrix (Top {top_zones} Zones)')
        ax.set_xlabel('Destination Zone')
        ax.set_ylabel('Origin Zone')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved OD flow matrix to {save_path}")
        
        return fig
    
    def plot_zone_characteristics(self, zone_features: pd.DataFrame,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot zone characteristics distributions.
        
        Args:
            zone_features: Zone feature DataFrame
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting zone characteristics")
        
        # Select numeric columns
        numeric_cols = zone_features.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'zone_id']
        
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            if i < len(axes):
                zone_features[col].hist(bins=30, ax=axes[i], alpha=0.7)
                axes[i].set_title(f'{col.replace("_", " ").title()} Distribution')
                axes[i].set_xlabel(col.replace("_", " ").title())
                axes[i].set_ylabel('Frequency')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(numeric_cols), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved zone characteristics to {save_path}")
        
        return fig
    
    def plot_clustering_results(self, embeddings: np.ndarray,
                              cluster_labels: np.ndarray,
                              zone_coords: Optional[pd.DataFrame] = None,
                              method: str = 'tsne',
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot clustering results with multiple views.
        
        Args:
            embeddings: Node embeddings
            cluster_labels: Cluster assignments
            zone_coords: Optional zone coordinates for spatial view
            method: Dimensionality reduction method
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting clustering results")
        
        # Determine number of subplots
        n_plots = 2 if zone_coords is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=(12 * n_plots, 8))
        if n_plots == 1:
            axes = [axes]
        
        # Dimensionality reduction for embedding space
        if method == 'tsne':
            from sklearn.manifold import TSNE
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'pca':
            from sklearn.decomposition import PCA
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Plot embedding space clustering
        unique_labels = np.unique(cluster_labels)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
        
        for i, label in enumerate(unique_labels):
            mask = cluster_labels == label
            axes[0].scatter(
                embeddings_2d[mask, 0], 
                embeddings_2d[mask, 1],
                c=[colors[i]], 
                label=f'Cluster {label}',
                alpha=0.7, 
                s=50
            )
        
        axes[0].set_title(f'Clustering in {method.upper()} Space')
        axes[0].set_xlabel(f'{method.upper()} 1')
        axes[0].set_ylabel(f'{method.upper()} 2')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0].grid(True, alpha=0.3)
        
        # Plot spatial clustering if coordinates available
        if zone_coords is not None and n_plots > 1:
            # Merge with cluster labels
            plot_data = zone_coords.copy()
            plot_data['cluster'] = cluster_labels
            
            for i, label in enumerate(unique_labels):
                cluster_data = plot_data[plot_data['cluster'] == label]
                axes[1].scatter(
                    cluster_data['lon'], 
                    cluster_data['lat'],
                    c=[colors[i]], 
                    label=f'Cluster {label}',
                    alpha=0.7, 
                    s=50
                )
            
            axes[1].set_title('Spatial Distribution of Clusters')
            axes[1].set_xlabel('Longitude')
            axes[1].set_ylabel('Latitude')
            axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved clustering results to {save_path}")
        
        return fig
    
    def plot_cluster_statistics(self, cluster_stats: pd.DataFrame,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot cluster statistics and characteristics.
        
        Args:
            cluster_stats: Cluster statistics DataFrame
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info("Plotting cluster statistics")
        
        # Select key metrics for visualization
        key_metrics = ['size', 'embedding_mean_norm', 'intra_cluster_distance']
        available_metrics = [col for col in key_metrics if col in cluster_stats.columns]
        
        if not available_metrics:
            logger.warning("No key metrics found in cluster statistics")
            return None
        
        n_metrics = len(available_metrics)
        fig, axes = plt.subplots(1, n_metrics, figsize=(5 * n_metrics, 6))
        if n_metrics == 1:
            axes = [axes]
        
        for i, metric in enumerate(available_metrics):
            # Bar plot for each metric
            bars = axes[i].bar(
                cluster_stats['cluster_id'].astype(str), 
                cluster_stats[metric],
                alpha=0.7,
                color=self.color_palette[i % len(self.color_palette)]
            )
            
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_xlabel('Cluster ID')
            axes[i].set_ylabel(metric.replace("_", " ").title())
            axes[i].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                axes[i].text(
                    bar.get_x() + bar.get_width()/2., 
                    height,
                    f'{height:.2f}', 
                    ha='center', 
                    va='bottom'
                )
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved cluster statistics to {save_path}")
        
        return fig
    
    def create_interactive_od_map(self, od_data: pd.DataFrame,
                                zone_coords: pd.DataFrame,
                                cluster_labels: Optional[np.ndarray] = None,
                                origin_col: str = 'origin',
                                dest_col: str = 'destination',
                                demand_col: str = 'demand',
                                top_flows: int = 100,
                                save_path: Optional[str] = None) -> go.Figure:
        """
        Create interactive OD flow map using Plotly.
        
        Args:
            od_data: OD demand data
            zone_coords: Zone coordinates
            cluster_labels: Optional cluster assignments
            origin_col: Origin column name
            dest_col: Destination column name
            demand_col: Demand column name
            top_flows: Number of top flows to display
            save_path: Optional path to save HTML file
            
        Returns:
            Plotly figure
        """
        logger.info("Creating interactive OD flow map")
        
        # Get top flows
        top_od = od_data.nlargest(top_flows, demand_col)
        
        # Merge with coordinates
        top_od = top_od.merge(
            zone_coords.rename(columns={'zone_id': origin_col, 'lat': 'origin_lat', 'lon': 'origin_lon'}),
            on=origin_col
        ).merge(
            zone_coords.rename(columns={'zone_id': dest_col, 'lat': 'dest_lat', 'lon': 'dest_lon'}),
            on=dest_col
        )
        
        # Create figure
        fig = go.Figure()
        
        # Add zone points
        if cluster_labels is not None:
            zone_plot_data = zone_coords.copy()
            zone_plot_data['cluster'] = cluster_labels
            
            for cluster_id in np.unique(cluster_labels):
                cluster_zones = zone_plot_data[zone_plot_data['cluster'] == cluster_id]
                
                fig.add_trace(go.Scattermapbox(
                    lat=cluster_zones['lat'],
                    lon=cluster_zones['lon'],
                    mode='markers',
                    marker=dict(size=8, opacity=0.7),
                    name=f'Cluster {cluster_id}',
                    text=cluster_zones['zone_id'],
                    hovertemplate='Zone: %{text}<br>Cluster: ' + str(cluster_id)
                ))
        else:
            fig.add_trace(go.Scattermapbox(
                lat=zone_coords['lat'],
                lon=zone_coords['lon'],
                mode='markers',
                marker=dict(size=6, color='blue', opacity=0.7),
                name='Zones',
                text=zone_coords['zone_id'],
                hovertemplate='Zone: %{text}'
            ))
        
        # Add flow lines
        for _, row in top_od.iterrows():
            fig.add_trace(go.Scattermapbox(
                lat=[row['origin_lat'], row['dest_lat']],
                lon=[row['origin_lon'], row['dest_lon']],
                mode='lines',
                line=dict(width=max(1, min(5, row[demand_col] / top_od[demand_col].max() * 5)), color='red'),
                opacity=0.6,
                showlegend=False,
                hovertemplate=f'Flow: {row[demand_col]:.1f}<br>From: {row[origin_col]}<br>To: {row[dest_col]}'
            ))
        
        # Update layout
        fig.update_layout(
            mapbox=dict(
                style="open-street-map",
                center=dict(
                    lat=zone_coords['lat'].mean(),
                    lon=zone_coords['lon'].mean()
                ),
                zoom=10
            ),
            title="Interactive OD Flow Map",
            height=600,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved interactive map to {save_path}")
        
        return fig
    
    def plot_training_progress(self, training_history: List[Dict[str, float]],
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training progress and loss curves.
        
        Args:
            training_history: List of loss dictionaries from training
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not training_history:
            logger.warning("No training history to plot")
            return None
        
        logger.info("Plotting training progress")
        
        # Convert to DataFrame
        df = pd.DataFrame(training_history)
        
        # Create subplots
        loss_cols = [col for col in df.columns if 'loss' in col.lower() or col in ['total', 'reconstruction', 'contrastive', 'clustering']]
        n_loss_types = len(loss_cols)
        
        if n_loss_types == 0:
            logger.warning("No loss columns found in training history")
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        # Plot individual losses
        for i, loss_type in enumerate(loss_cols[:4]):
            if i < len(axes):
                axes[i].plot(df[loss_type], linewidth=2)
                axes[i].set_title(f'{loss_type.replace("_", " ").title()} Loss')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Loss')
                axes[i].grid(True, alpha=0.3)
        
        # If fewer than 4 loss types, plot all losses together in remaining subplot
        if n_loss_types < 4:
            for loss_type in loss_cols:
                axes[-1].plot(df[loss_type], label=loss_type.replace("_", " ").title(), linewidth=2)
            axes[-1].set_title('All Losses')
            axes[-1].set_xlabel('Epoch')
            axes[-1].set_ylabel('Loss')
            axes[-1].legend()
            axes[-1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved training progress to {save_path}")
        
        return fig
    
    def create_dashboard(self, results: Dict[str, Any],
                        save_path: Optional[str] = None) -> go.Figure:
        """
        Create comprehensive dashboard with all results.
        
        Args:
            results: Dictionary containing all analysis results
            save_path: Optional path to save HTML file
            
        Returns:
            Plotly figure
        """
        logger.info("Creating comprehensive dashboard")
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=2,
            subplot_titles=(
                'Cluster Distribution', 'Silhouette Scores by Method',
                'Training Loss Progression', 'Zone Feature Correlations',
                'Cluster Spatial Distribution', 'Performance Metrics'
            ),
            specs=[
                [{"type": "bar"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "heatmap"}],
                [{"type": "scattermapbox"}, {"type": "table"}]
            ]
        )
        
        # Add cluster distribution
        if 'cluster_labels' in results:
            cluster_counts = pd.Series(results['cluster_labels']).value_counts().sort_index()
            fig.add_trace(
                go.Bar(x=cluster_counts.index, y=cluster_counts.values, name="Cluster Sizes"),
                row=1, col=1
            )
        
        # Add method comparison
        if 'comparison_results' in results:
            methods = list(results['comparison_results'].keys())
            silhouette_scores = [results['comparison_results'][m]['metrics'].get('silhouette_score', 0) 
                               for m in methods]
            fig.add_trace(
                go.Bar(x=methods, y=silhouette_scores, name="Silhouette Scores"),
                row=1, col=2
            )
        
        # Add training progress
        if 'training_history' in results:
            history_df = pd.DataFrame(results['training_history'])
            epochs = list(range(len(history_df)))
            
            for loss_type in ['total', 'reconstruction', 'clustering']:
                if loss_type in history_df.columns:
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history_df[loss_type], 
                                 mode='lines', name=f'{loss_type.title()} Loss'),
                        row=2, col=1
                    )
        
        # Update layout
        fig.update_layout(
            height=1200,
            title="GNN Clustering Analysis Dashboard",
            showlegend=True
        )
        
        if save_path:
            fig.write_html(save_path)
            logger.info(f"Saved dashboard to {save_path}")
        
        return fig


def create_network_visualization(graph_data: Data, 
                               cluster_labels: np.ndarray,
                               pos: Optional[Dict] = None,
                               figsize: Tuple[int, int] = (12, 8),
                               save_path: Optional[str] = None) -> plt.Figure:
    """
    Create network visualization with cluster coloring.
    
    Args:
        graph_data: PyTorch Geometric graph data
        cluster_labels: Cluster assignments
        pos: Optional node positions
        figsize: Figure size
        save_path: Optional path to save figure
        
    Returns:
        Matplotlib figure
    """
    logger.info("Creating network visualization")
    
    # Convert to NetworkX
    G = to_networkx(graph_data, to_undirected=True)
    
    # Create layout if not provided
    if pos is None:
        pos = nx.spring_layout(G, k=1, iterations=50)
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.2, width=0.5, edge_color='gray', ax=ax)
    
    # Draw nodes by cluster
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_nodes = [j for j, c in enumerate(cluster_labels) if c == cluster_id]
        nx.draw_networkx_nodes(
            G, pos, 
            nodelist=cluster_nodes,
            node_color=[colors[i]],
            node_size=50,
            alpha=0.8,
            label=f'Cluster {cluster_id}',
            ax=ax
        )
    
    ax.set_title('Network Visualization with Clusters')
    ax.legend()
    ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved network visualization to {save_path}")
    
    return fig
