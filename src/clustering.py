"""
Clustering algorithms and evaluation utilities for GNN-based clustering.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Any
import torch
from torch_geometric.data import Data
import logging

logger = logging.getLogger(__name__)


class ClusteringEvaluator:
    """
    Comprehensive clustering evaluation and visualization toolkit.
    """
    
    def __init__(self):
        """Initialize clustering evaluator."""
        self.metrics_history = []
        
    def evaluate_clustering(self, embeddings: np.ndarray, 
                          cluster_labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None,
                          graph_data: Optional[Data] = None) -> Dict[str, float]:
        """
        Comprehensive clustering evaluation.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            cluster_labels: Predicted cluster labels [num_nodes]
            true_labels: Optional ground truth labels [num_nodes]
            graph_data: Optional graph data for graph-based metrics
            
        Returns:
            Dictionary of evaluation metrics
        """
        metrics = {}
        
        # Basic clustering metrics
        num_clusters = len(np.unique(cluster_labels))
        metrics['num_clusters'] = num_clusters
        
        if num_clusters > 1:
            # Silhouette score
            silhouette = silhouette_score(embeddings, cluster_labels)
            metrics['silhouette_score'] = silhouette
            
            # Calinski-Harabasz index
            from sklearn.metrics import calinski_harabasz_score
            ch_score = calinski_harabasz_score(embeddings, cluster_labels)
            metrics['calinski_harabasz_score'] = ch_score
            
            # Davies-Bouldin index (lower is better)
            from sklearn.metrics import davies_bouldin_score
            db_score = davies_bouldin_score(embeddings, cluster_labels)
            metrics['davies_bouldin_score'] = db_score
            
        else:
            metrics['silhouette_score'] = -1.0
            metrics['calinski_harabasz_score'] = 0.0
            metrics['davies_bouldin_score'] = float('inf')
        
        # Supervised metrics (if true labels available)
        if true_labels is not None:
            ari = adjusted_rand_score(true_labels, cluster_labels)
            nmi = normalized_mutual_info_score(true_labels, cluster_labels)
            metrics['adjusted_rand_score'] = ari
            metrics['normalized_mutual_info'] = nmi
        
        # Graph-based metrics
        if graph_data is not None:
            modularity = self._compute_modularity(graph_data, cluster_labels)
            conductance = self._compute_conductance(graph_data, cluster_labels)
            metrics['modularity'] = modularity
            metrics['conductance'] = conductance
        
        # Cluster balance metrics
        cluster_sizes = [np.sum(cluster_labels == i) for i in range(num_clusters)]
        if len(cluster_sizes) > 0:
            metrics['cluster_size_std'] = np.std(cluster_sizes)
            metrics['largest_cluster_ratio'] = max(cluster_sizes) / len(cluster_labels)
            metrics['smallest_cluster_size'] = min(cluster_sizes)
        
        return metrics
    
    def _compute_modularity(self, graph_data: Data, cluster_labels: np.ndarray) -> float:
        """
        Compute modularity of clustering.
        
        Args:
            graph_data: Graph data
            cluster_labels: Cluster assignments
            
        Returns:
            Modularity score
        """
        try:
            import networkx as nx
            from torch_geometric.utils import to_networkx
            
            # Convert to NetworkX graph
            G = to_networkx(graph_data, to_undirected=True)
            
            # Create community structure
            communities = []
            for cluster_id in np.unique(cluster_labels):
                community = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
                if len(community) > 0:
                    communities.append(community)
            
            # Compute modularity
            if len(communities) > 1:
                modularity = nx.algorithms.community.modularity(G, communities)
                return modularity
            else:
                return 0.0
                
        except Exception as e:
            logger.warning(f"Could not compute modularity: {e}")
            return 0.0
    
    def _compute_conductance(self, graph_data: Data, cluster_labels: np.ndarray) -> float:
        """
        Compute average conductance of clusters.
        
        Args:
            graph_data: Graph data
            cluster_labels: Cluster assignments
            
        Returns:
            Average conductance score
        """
        try:
            import networkx as nx
            from torch_geometric.utils import to_networkx
            
            G = to_networkx(graph_data, to_undirected=True)
            
            conductances = []
            for cluster_id in np.unique(cluster_labels):
                cluster_nodes = [i for i, c in enumerate(cluster_labels) if c == cluster_id]
                
                if len(cluster_nodes) > 0:
                    subgraph = G.subgraph(cluster_nodes)
                    if len(subgraph.edges()) > 0:
                        conductance = nx.algorithms.cuts.conductance(G, cluster_nodes)
                        conductances.append(conductance)
            
            return np.mean(conductances) if conductances else 1.0
            
        except Exception as e:
            logger.warning(f"Could not compute conductance: {e}")
            return 1.0
    
    def visualize_clusters(self, embeddings: np.ndarray, 
                          cluster_labels: np.ndarray,
                          true_labels: Optional[np.ndarray] = None,
                          method: str = 'tsne',
                          figsize: Tuple[int, int] = (12, 5),
                          save_path: Optional[str] = None) -> plt.Figure:
        """
        Visualize clustering results in 2D.
        
        Args:
            embeddings: Node embeddings [num_nodes, embedding_dim]
            cluster_labels: Predicted cluster labels
            true_labels: Optional ground truth labels
            method: Dimensionality reduction method ('tsne', 'pca')
            figsize: Figure size
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        logger.info(f"Visualizing clusters using {method}")
        
        # Reduce dimensionality to 2D
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings)-1))
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
        else:
            raise ValueError(f"Unsupported method: {method}")
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        # Create plots
        n_plots = 2 if true_labels is not None else 1
        fig, axes = plt.subplots(1, n_plots, figsize=figsize)
        if n_plots == 1:
            axes = [axes]
        
        # Plot predicted clusters
        scatter = axes[0].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                c=cluster_labels, cmap='tab10', alpha=0.7, s=30)
        axes[0].set_title('Predicted Clusters')
        axes[0].set_xlabel(f'{method.upper()} 1')
        axes[0].set_ylabel(f'{method.upper()} 2')
        plt.colorbar(scatter, ax=axes[0])
        
        # Plot true clusters if available
        if true_labels is not None:
            scatter = axes[1].scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], 
                                    c=true_labels, cmap='tab10', alpha=0.7, s=30)
            axes[1].set_title('True Clusters')
            axes[1].set_xlabel(f'{method.upper()} 1')
            axes[1].set_ylabel(f'{method.upper()} 2')
            plt.colorbar(scatter, ax=axes[1])
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved visualization to {save_path}")
        
        return fig
    
    def plot_metrics_history(self, metrics_history: List[Dict[str, float]],
                           save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot training metrics history.
        
        Args:
            metrics_history: List of metric dictionaries
            save_path: Optional path to save figure
            
        Returns:
            Matplotlib figure
        """
        if not metrics_history:
            logger.warning("No metrics history to plot")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(metrics_history)
        
        # Create subplots
        n_metrics = len(df.columns)
        n_cols = min(3, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        # Plot each metric
        for i, metric in enumerate(df.columns):
            if i < len(axes):
                axes[i].plot(df[metric])
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel('Value')
                axes[i].grid(True, alpha=0.3)
        
        # Hide unused subplots
        for i in range(len(df.columns), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Saved metrics plot to {save_path}")
        
        return fig
    
    def analyze_cluster_characteristics(self, embeddings: np.ndarray,
                                      cluster_labels: np.ndarray,
                                      node_features: Optional[np.ndarray] = None,
                                      feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Analyze characteristics of each cluster.
        
        Args:
            embeddings: Node embeddings
            cluster_labels: Cluster assignments
            node_features: Optional original node features
            feature_names: Optional feature names
            
        Returns:
            DataFrame with cluster characteristics
        """
        unique_clusters = np.unique(cluster_labels)
        cluster_stats = []
        
        for cluster_id in unique_clusters:
            mask = cluster_labels == cluster_id
            cluster_embeddings = embeddings[mask]
            
            stats = {
                'cluster_id': cluster_id,
                'size': np.sum(mask),
                'embedding_mean_norm': np.mean(np.linalg.norm(cluster_embeddings, axis=1)),
                'embedding_std_norm': np.std(np.linalg.norm(cluster_embeddings, axis=1)),
                'intra_cluster_distance': np.mean(np.linalg.norm(
                    cluster_embeddings - cluster_embeddings.mean(axis=0), axis=1
                ))
            }
            
            # Add original feature statistics if available
            if node_features is not None:
                cluster_features = node_features[mask]
                
                if feature_names is not None:
                    for i, feature_name in enumerate(feature_names):
                        stats[f'{feature_name}_mean'] = np.mean(cluster_features[:, i])
                        stats[f'{feature_name}_std'] = np.std(cluster_features[:, i])
                else:
                    for i in range(cluster_features.shape[1]):
                        stats[f'feature_{i}_mean'] = np.mean(cluster_features[:, i])
                        stats[f'feature_{i}_std'] = np.std(cluster_features[:, i])
            
            cluster_stats.append(stats)
        
        return pd.DataFrame(cluster_stats)


class BaselineClustering:
    """
    Baseline clustering methods for comparison.
    """
    
    def __init__(self):
        """Initialize baseline clustering."""
        pass
    
    def kmeans_clustering(self, embeddings: np.ndarray, 
                         n_clusters: int, 
                         random_state: int = 42) -> np.ndarray:
        """
        Perform K-means clustering.
        
        Args:
            embeddings: Node embeddings
            n_clusters: Number of clusters
            random_state: Random state
            
        Returns:
            Cluster assignments
        """
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        cluster_labels = kmeans.fit_predict(embeddings)
        return cluster_labels
    
    def dbscan_clustering(self, embeddings: np.ndarray,
                         eps: float = 0.5,
                         min_samples: int = 5) -> np.ndarray:
        """
        Perform DBSCAN clustering.
        
        Args:
            embeddings: Node embeddings
            eps: Maximum distance between samples
            min_samples: Minimum samples in neighborhood
            
        Returns:
            Cluster assignments
        """
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(embeddings)
        return cluster_labels
    
    def spectral_clustering(self, embeddings: np.ndarray,
                          n_clusters: int,
                          random_state: int = 42) -> np.ndarray:
        """
        Perform spectral clustering.
        
        Args:
            embeddings: Node embeddings  
            n_clusters: Number of clusters
            random_state: Random state
            
        Returns:
            Cluster assignments
        """
        spectral = SpectralClustering(
            n_clusters=n_clusters, 
            random_state=random_state,
            affinity='nearest_neighbors'
        )
        cluster_labels = spectral.fit_predict(embeddings)
        return cluster_labels
    
    def find_optimal_clusters(self, embeddings: np.ndarray,
                            max_clusters: int = 10,
                            method: str = 'silhouette') -> int:
        """
        Find optimal number of clusters.
        
        Args:
            embeddings: Node embeddings
            max_clusters: Maximum number of clusters to try
            method: Evaluation method ('silhouette', 'elbow')
            
        Returns:
            Optimal number of clusters
        """
        scores = []
        cluster_range = range(2, min(max_clusters + 1, len(embeddings)))
        
        for n_clusters in cluster_range:
            if method == 'silhouette':
                labels = self.kmeans_clustering(embeddings, n_clusters)
                score = silhouette_score(embeddings, labels)
                scores.append(score)
            elif method == 'elbow':
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                kmeans.fit(embeddings)
                scores.append(-kmeans.inertia_)  # Negative inertia for maximization
        
        # Find optimal number of clusters
        if method == 'silhouette':
            optimal_idx = np.argmax(scores)
        elif method == 'elbow':
            # Use elbow method (find point of maximum curvature)
            differences = np.diff(scores)
            second_differences = np.diff(differences)
            optimal_idx = np.argmax(second_differences) + 1
        
        optimal_clusters = list(cluster_range)[optimal_idx]
        return optimal_clusters


class ClusteringPipeline:
    """
    Complete clustering pipeline combining GNN and traditional methods.
    """
    
    def __init__(self, evaluator: Optional[ClusteringEvaluator] = None):
        """
        Initialize clustering pipeline.
        
        Args:
            evaluator: Optional clustering evaluator
        """
        self.evaluator = evaluator or ClusteringEvaluator()
        self.baseline = BaselineClustering()
        self.results = {}
    
    def run_comparison(self, embeddings: np.ndarray,
                      gnn_labels: np.ndarray,
                      n_clusters: int,
                      graph_data: Optional[Data] = None,
                      true_labels: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Run comparison between GNN clustering and baseline methods.
        
        Args:
            embeddings: Node embeddings
            gnn_labels: GNN clustering results
            n_clusters: Number of clusters
            graph_data: Optional graph data
            true_labels: Optional ground truth labels
            
        Returns:
            Comparison results
        """
        logger.info("Running clustering comparison")
        
        results = {}
        
        # Evaluate GNN clustering
        gnn_metrics = self.evaluator.evaluate_clustering(
            embeddings, gnn_labels, true_labels, graph_data
        )
        results['gnn'] = {
            'labels': gnn_labels,
            'metrics': gnn_metrics
        }
        
        # K-means baseline
        kmeans_labels = self.baseline.kmeans_clustering(embeddings, n_clusters)
        kmeans_metrics = self.evaluator.evaluate_clustering(
            embeddings, kmeans_labels, true_labels, graph_data
        )
        results['kmeans'] = {
            'labels': kmeans_labels,
            'metrics': kmeans_metrics
        }
        
        # Spectral clustering baseline
        try:
            spectral_labels = self.baseline.spectral_clustering(embeddings, n_clusters)
            spectral_metrics = self.evaluator.evaluate_clustering(
                embeddings, spectral_labels, true_labels, graph_data
            )
            results['spectral'] = {
                'labels': spectral_labels,
                'metrics': spectral_metrics
            }
        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}")
        
        # DBSCAN baseline (find optimal parameters)
        try:
            dbscan_labels = self._optimize_dbscan(embeddings)
            dbscan_metrics = self.evaluator.evaluate_clustering(
                embeddings, dbscan_labels, true_labels, graph_data
            )
            results['dbscan'] = {
                'labels': dbscan_labels,
                'metrics': dbscan_metrics
            }
        except Exception as e:
            logger.warning(f"DBSCAN clustering failed: {e}")
        
        self.results = results
        logger.info("Clustering comparison completed")
        return results
    
    def _optimize_dbscan(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Optimize DBSCAN parameters.
        
        Args:
            embeddings: Node embeddings
            
        Returns:
            DBSCAN cluster labels
        """
        from sklearn.neighbors import NearestNeighbors
        
        # Estimate eps using k-distance graph
        k = min(4, len(embeddings) - 1)
        nbrs = NearestNeighbors(n_neighbors=k).fit(embeddings)
        distances, _ = nbrs.kneighbors(embeddings)
        distances = np.sort(distances[:, k-1])
        
        # Use knee point as eps
        diff = np.diff(distances)
        knee_point = np.argmax(diff)
        eps = distances[knee_point]
        
        # Optimize min_samples
        best_score = -1
        best_labels = None
        
        for min_samples in [3, 5, 10]:
            labels = self.baseline.dbscan_clustering(embeddings, eps, min_samples)
            
            if len(np.unique(labels)) > 1 and -1 not in labels:
                score = silhouette_score(embeddings, labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
        
        return best_labels if best_labels is not None else np.zeros(len(embeddings))
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """
        Generate clustering comparison report.
        
        Args:
            save_path: Optional path to save report
            
        Returns:
            Report string
        """
        if not self.results:
            return "No results to report. Run comparison first."
        
        report_lines = ["# Clustering Comparison Report\n"]
        
        # Summary table
        report_lines.append("## Summary\n")
        report_lines.append("| Method | Silhouette | Modularity | Clusters | ARI | NMI |")
        report_lines.append("|--------|------------|------------|----------|-----|-----|")
        
        for method, result in self.results.items():
            metrics = result['metrics']
            line = f"| {method.upper()} |"
            line += f" {metrics.get('silhouette_score', 'N/A'):.3f} |"
            line += f" {metrics.get('modularity', 'N/A'):.3f} |"
            line += f" {metrics.get('num_clusters', 'N/A')} |"
            line += f" {metrics.get('adjusted_rand_score', 'N/A'):.3f} |"
            line += f" {metrics.get('normalized_mutual_info', 'N/A'):.3f} |"
            report_lines.append(line)
        
        report_lines.append("\n")
        
        # Detailed metrics for each method
        for method, result in self.results.items():
            report_lines.append(f"## {method.upper()} Detailed Metrics\n")
            metrics = result['metrics']
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value:.4f}")
                else:
                    report_lines.append(f"- **{metric.replace('_', ' ').title()}**: {value}")
            
            report_lines.append("\n")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w') as f:
                f.write(report)
            logger.info(f"Saved report to {save_path}")
        
        return report
