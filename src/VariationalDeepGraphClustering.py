import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import logging
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool, global_add_pool
from torch_geometric.data import Data, Batch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, normalized_mutual_info_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib.patches import FancyBboxPatch, Circle, Arrow
import matplotlib.patches as patches

# 从 gnn_models 导入现有的编码器和损失函数类
from .gnn_models import GCNEncoder, GATEncoder, GraphSAGEEncoder, ContrastiveLoss, FairnessLoss




class VariationalDeepGraphClustering(nn.Module):

    
    def __init__(self, input_dim: int, embedding_dim: int, num_clusters: int,
                 encoder_type: str = 'gcn', hidden_dims: List[int] = [64, 32],
                 dropout: float = 0.1, temperature: float = 0.1,
                 fairness_beta: float = 1.0, use_frank_wolfe: bool = False,
                 initial_lambda: float = 0.5, learning_rate: float = 0.001,
                 kl_weight: float = 0.1, use_batch_norm: bool = True):
        """
        初始化变分深度图聚类模型。
        
        Args:
            input_dim: 输入特征维度
            embedding_dim: 嵌入维度（潜在空间维度）
            num_clusters: 聚类数量
            encoder_type: 编码器类型 ('gcn', 'gat', 'sage')
            hidden_dims: 隐藏层维度列表
            dropout: Dropout率
            temperature: 对比学习温度参数
            fairness_beta: 公平性损失beta参数
            use_frank_wolfe: 是否使用Frank-Wolfe算法
            initial_lambda: 损失组合初始lambda值
            learning_rate: 学习率
            kl_weight: KL散度损失权重
            use_batch_norm: 是否使用批标准化
        """
        super(VariationalDeepGraphClustering, self).__init__()
        self.input_dim = input_dim
        self.num_clusters = num_clusters
        self.embedding_dim = embedding_dim
        self.kl_weight = kl_weight
        self.use_frank_wolfe = use_frank_wolfe
        self.learning_rate = learning_rate
        self.use_batch_norm = use_batch_norm
        self.encoder_type = encoder_type
        self.dropout = dropout
        self.hidden_dims = hidden_dims
        # Lambda参数用于损失组合
        if use_frank_wolfe:
            self.lambda_param = nn.Parameter(torch.tensor(initial_lambda, requires_grad=False))
        else:
            self.register_buffer('lambda_param', torch.tensor(initial_lambda))
        
        # 编码器选择
        if encoder_type == 'gcn':
            self.encoder = GCNEncoder(input_dim, hidden_dims, hidden_dims[-1], dropout)
        elif encoder_type == 'gat':
            heads = [4] * len(hidden_dims) + [1]
            self.encoder = GATEncoder(input_dim, hidden_dims, hidden_dims[-1], heads, dropout)
        elif encoder_type == 'sage':
            self.encoder = GraphSAGEEncoder(input_dim, hidden_dims, hidden_dims[-1], dropout)
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
        
        # 变分层：从编码器输出映射到均值和对数方差
        self.fc_mu = nn.Linear(hidden_dims[-1], embedding_dim)  # 均值层
        self.fc_logvar = nn.Linear(hidden_dims[-1], embedding_dim)  # 对数方差层
        
        # 批标准化（可选）
        if use_batch_norm:
            self.bn_mu = nn.BatchNorm1d(embedding_dim)
            self.bn_logvar = nn.BatchNorm1d(embedding_dim)
        
        # 解码器：从潜在空间重构原始特征
        decoder_layers = []
        decoder_dims = [embedding_dim] + hidden_dims[::-1] + [input_dim]
        
        for i in range(len(decoder_dims) - 1):
            decoder_layers.append(nn.Linear(decoder_dims[i], decoder_dims[i + 1]))
            if i < len(decoder_dims) - 2:  # 不在最后一层添加激活函数
                decoder_layers.append(nn.ReLU())
                decoder_layers.append(nn.Dropout(dropout))
                if use_batch_norm:
                    decoder_layers.append(nn.BatchNorm1d(decoder_dims[i + 1]))
        
        self.decoder = nn.Sequential(*decoder_layers)
        
        # 聚类层
        self.cluster_centers = nn.Parameter(torch.randn(num_clusters, embedding_dim))
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.reconstruction_loss = nn.MSELoss()
        self.fairness_loss = FairnessLoss(fairness_beta)
        
        # 优化器
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.5)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, 
               edge_weight: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        编码器：将输入特征映射到潜在空间的均值和方差
        
        Args:
            x: 节点特征 [N, input_dim]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            mu: 潜在表示的均值 [N, embedding_dim]
            logvar: 潜在表示的对数方差 [N, embedding_dim]
        """
        # 通过图神经网络编码器提取特征
        h = self.encoder(x, edge_index, edge_weight)
        
        # 映射到均值和对数方差
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        # 批标准化（可选）
        if self.use_batch_norm:
            mu = self.bn_mu(mu)
            logvar = self.bn_logvar(logvar)
        
        return mu, logvar
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        重参数化技巧：从标准正态分布采样潜在表示
        
        Args:
            mu: 均值 [N, embedding_dim]
            logvar: 对数方差 [N, embedding_dim]
            
        Returns:
            z: 采样的潜在表示 [N, embedding_dim]
        """
        if self.training:
            # 训练时使用重参数化技巧
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            # 推理时直接使用均值
            z = mu
        
        return z
    
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        解码器：从潜在空间重构原始特征
        
        Args:
            z: 潜在表示 [N, embedding_dim]
            
        Returns:
            x_recon: 重构的特征 [N, input_dim]
        """
        return self.decoder(z)
    
    def forward(self, data: Data) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        前向传播
        
        Args:
            data: PyTorch Geometric数据对象
            
        Returns:
            z: 潜在表示 [N, embedding_dim]
            mu: 均值 [N, embedding_dim] 
            logvar: 对数方差 [N, embedding_dim]
            x_recon: 重构特征 [N, input_dim]
            cluster_probs: 聚类概率 [N, num_clusters]
        """
        # 编码
        mu, logvar = self.encode(data.x, data.edge_index, 
                                getattr(data, 'edge_weight', None))
        
        # 重参数化采样
        z = self.reparameterize(mu, logvar)
        
        # 解码
        x_recon = self.decode(z)
        
        # 计算聚类概率
        cluster_probs = self._compute_cluster_probs(z)
        
        return z, mu, logvar, x_recon, cluster_probs
    
    def _compute_cluster_probs(self, z: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """
        计算聚类分配概率，使用t-分布
        
        Args:
            z: 潜在表示 [N, embedding_dim]
            alpha: t-分布自由度参数
            
        Returns:
            cluster_probs: 聚类概率 [N, num_clusters]
        """
        # 计算聚类中心的距离
        distances = torch.cdist(z, self.cluster_centers)
        
        # 使用t-分布计算概率
        q = (1.0 + distances**2 / alpha) ** (-(alpha + 1.0) / 2.0)
        q = q / q.sum(dim=1, keepdim=True)
        
        return q
    
    def compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        计算KL散度损失，约束潜在分布接近标准正态分布
        
        Args:
            mu: 均值 [N, embedding_dim]
            logvar: 对数方差 [N, embedding_dim]
            
        Returns:
            kl_loss: KL散度损失（标量）
        """
        # KL散度：KL(q(z|x) || p(z))，其中p(z) = N(0, I)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
        return kl_loss.mean()
    
    def compute_target_distribution(self, cluster_probs: torch.Tensor) -> torch.Tensor:
        """
        计算目标分布用于自训练
        
        Args:
            cluster_probs: 当前聚类概率 [N, num_clusters]
            
        Returns:
            target_probs: 目标分布 [N, num_clusters]
        """
        # 平方概率并归一化（增强置信度高的分配）
        p = cluster_probs**2 / cluster_probs.sum(dim=0, keepdim=True)
        p = p / p.sum(dim=1, keepdim=True)
        
        return p
    
    def compute_total_loss(self, data: Data, z: torch.Tensor, mu: torch.Tensor, 
                          logvar: torch.Tensor, x_recon: torch.Tensor, 
                          cluster_probs: torch.Tensor,
                          positive_pairs: Optional[torch.Tensor] = None,
                          target_probs: Optional[torch.Tensor] = None,
                          alpha_recon: float = 1.0, alpha_kl: float = 1.0,
                          alpha_cluster: float = 1.0, alpha_contrastive: float = 0.5) -> Dict[str, torch.Tensor]:
        """
        计算总损失，包括重构损失、KL损失、聚类损失和对比损失
        
        Args:
            data: PyTorch Geometric数据对象
            z: 潜在表示 [N, embedding_dim]
            mu: 均值 [N, embedding_dim]
            logvar: 对数方差 [N, embedding_dim] 
            x_recon: 重构特征 [N, input_dim]
            cluster_probs: 聚类概率 [N, num_clusters]
            positive_pairs: 正样本对 [num_pairs, 2] (可选)
            target_probs: 目标聚类分布 [N, num_clusters] (可选)
            alpha_recon: 重构损失权重
            alpha_kl: KL损失权重
            alpha_cluster: 聚类损失权重
            alpha_contrastive: 对比损失权重
            
        Returns:
            损失字典
        """
        losses = {}
        
        # 1. 重构损失
        recon_loss = self.reconstruction_loss(x_recon, data.x)
        losses['reconstruction'] = recon_loss
        
        # 2. KL散度损失
        kl_loss = self.compute_kl_loss(mu, logvar)
        losses['kl_divergence'] = kl_loss
        
        # 3. 聚类损失（如果提供目标分布）
        if target_probs is not None:
            cluster_loss = F.kl_div(torch.log(cluster_probs + 1e-8), target_probs, reduction='batchmean')
            losses['clustering'] = cluster_loss
        else:
            # 使用熵损失鼓励置信的分配
            entropy_loss = -torch.sum(cluster_probs * torch.log(cluster_probs + 1e-8), dim=1).mean()
            losses['clustering'] = -entropy_loss  # 负熵，鼓励低熵（高置信度）
        
        # 4. 对比损失（如果提供正样本对）
        if positive_pairs is not None and len(positive_pairs) > 0:
            contrastive_loss = self.contrastive_loss(z, positive_pairs)
            losses['contrastive'] = contrastive_loss
        
        # 5. 公平性损失（可选）
        # 需要 (od_matrix [N,N], cluster_assignments [N] 硬标签, charge_num [N])
        if hasattr(data, 'charge_num') and data.charge_num is not None:
            try:
                # 构建或复用 OD 矩阵
                if hasattr(data, 'od_matrix') and data.od_matrix is not None:
                    od_matrix = data.od_matrix
                else:
                    N = data.num_nodes if hasattr(data, 'num_nodes') else data.x.size(0)
                    od_matrix = torch.zeros((N, N), device=data.x.device, dtype=data.x.dtype)
                    if hasattr(data, 'edge_index') and data.edge_index is not None:
                        ei = data.edge_index
                        # 读取权重，优先 edge_attr（可能为 [E,1] 或 [E]）
                        if hasattr(data, 'edge_attr') and data.edge_attr is not None:
                            ew = data.edge_attr
                            if ew.dim() > 1:
                                ew = ew.squeeze(-1)
                        else:
                            # 若无权重，则默认 1.0
                            ew = torch.ones(ei.size(1), device=od_matrix.device, dtype=od_matrix.dtype)
                        od_matrix[ei[0], ei[1]] = ew
                    # 缓存，避免每步重建
                    data.od_matrix = od_matrix

                # 将软概率转为硬标签
                cluster_assignments = torch.argmax(cluster_probs, dim=1).long()

                fairness_loss = self.fairness_loss(od_matrix, cluster_assignments, data.charge_num)
                losses['fairness'] = fairness_loss
            except Exception as _fair_e:
                # 若公平性损失计算失败，跳过以不影响主训练
                pass
        
        # 计算总损失
        total_loss = (alpha_recon * losses['reconstruction'] + 
                     alpha_kl * self.kl_weight * losses['kl_divergence'] +
                     alpha_cluster * losses['clustering'])
        
        if 'contrastive' in losses:
            total_loss += alpha_contrastive * losses['contrastive']
        
        if 'fairness' in losses:
            total_loss += 0.1 * losses['fairness']  # 固定权重
        
        losses['total'] = total_loss
        
        return losses
    
    def pretrain_with_em(self, data: Data, max_iter: int = 100, tol: float = 1e-4,
                        init_method: str = 'kmeans', verbose: bool = True,
                        use_fairness_prior: bool = False, fairness_weight: float = 0.1,
                        charge_num: Optional[torch.Tensor] = None,
                        target_balance_ratio: float = 0.8) -> Dict[str, Any]:
        """
        使用EM算法预训练聚类中心（基于变分编码器的均值表示）
        
        Args:
            data: PyTorch Geometric数据对象
            max_iter: EM算法最大迭代次数
            tol: 收敛容忍度
            init_method: 初始化方法
            verbose: 是否打印详细信息
            use_fairness_prior: 是否使用公平性先验
            fairness_weight: 公平性权重
            charge_num: 充电桩数量
            target_balance_ratio: 目标均衡比例
            
        Returns:
            EM训练结果字典        """
        print("开始变分模型的EM预训练...")
        
        # 获取潜在表示（使用均值，不使用采样）
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(data.x, data.edge_index, 
                                   getattr(data, 'edge_weight', None))
            embeddings = mu  # 使用均值作为确定性表示
        
        # 使用K-means进行简化的聚类中心初始化
        embeddings_np = embeddings.detach().cpu().numpy()
        N, D = embeddings_np.shape
        K = self.num_clusters
        
        # 处理服务能力数据（移动到获取N和K之后）
        if use_fairness_prior and charge_num is not None:
            charge_capacities = charge_num.detach().cpu().numpy()
            total_capacity = np.sum(charge_capacities)
            # 计算理想的每个聚类的服务能力 (均匀分布)
            target_capacity_per_cluster = total_capacity / K
        else:
            charge_capacities = np.ones(N)  # 如果没有服务能力数据，假设所有节点能力相等
            total_capacity = N
            target_capacity_per_cluster = N / K
        
        # K-means初始化
        if init_method == 'kmeans':
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=K, random_state=42, n_init=10)
            kmeans.fit(embeddings_np)
            centers = kmeans.cluster_centers_
            
        elif init_method == 'random':
            indices = np.random.choice(N, K, replace=False)
            centers = embeddings_np[indices].copy()
        else:
            raise ValueError(f"不支持的初始化方法: {init_method}")
        
        # 简化的EM算法
        log_likelihood_history = []
        
        for iter_num in range(max_iter):
            # E步：计算距离和责任矩阵
            distances = np.linalg.norm(embeddings_np[:, np.newaxis] - centers, axis=2)
            responsibilities = np.exp(-distances)
            responsibilities = responsibilities / responsibilities.sum(axis=1, keepdims=True)
            
            # M步：更新聚类中心
            old_centers = centers.copy()
            for k in range(K):
                if responsibilities[:, k].sum() > 1e-6:
                    centers[k] = np.average(embeddings_np, axis=0, weights=responsibilities[:, k])
            
            # 计算似然
            log_likelihood = -np.sum(np.min(distances, axis=1))
            log_likelihood_history.append(log_likelihood)
            
            # 检查收敛
            center_change = np.linalg.norm(centers - old_centers)
            if center_change < tol:
                if verbose:
                    print(f"EM算法在第 {iter_num + 1} 轮收敛")
                break
        
        # 更新模型聚类中心
        device = next(self.parameters()).device
        self.cluster_centers.data = torch.tensor(centers, dtype=torch.float32, device=device)
        
        if verbose:
            print(f"EM预训练完成，最终似然: {log_likelihood_history[-1]:.4f}")
        
        return {
            'converged': iter_num < max_iter - 1,
            'n_iter': iter_num + 1,
            'log_likelihood': log_likelihood_history,
            'final_centers': torch.tensor(centers, dtype=torch.float32),
            'responsibilities': torch.tensor(responsibilities, dtype=torch.float32)
        }
    
    def complete_pretraining_workflow(self, data: Data,
                                    pretrain_epochs: int = 50,
                                    em_max_iter: int = 100,
                                    em_init_method: str = 'kmeans',
                                    learning_rate: float = 0.001,
                                    verbose: bool = True) -> Dict[str, Any]:
        """
        完整的预训练工作流程
        
        Args:
            data: PyTorch Geometric数据对象
            pretrain_epochs: 变分自编码器预训练轮数
            em_max_iter: EM算法最大迭代次数
            em_init_method: EM初始化方法
            learning_rate: 学习率
            verbose: 是否打印详细信息
            
        Returns:
            预训练结果字典
        """
        print("=" * 60)
        print("开始变分深度图聚类预训练工作流")
        print("=" * 60)
        
        device = next(self.parameters()).device
        data = data.to(device)
        
        # 阶段1：变分自编码器预训练
        print(f"\n阶段1: 变分自编码器预训练 ({pretrain_epochs} 轮)")
        print("-" * 40)
        
        self.train()
        reconstruction_losses = []
        kl_losses = []
        
        for epoch in range(pretrain_epochs):
            self.optimizer.zero_grad()
            
            # 前向传播
            z, mu, logvar, x_recon, _ = self.forward(data)
            
            # 计算损失
            recon_loss = self.reconstruction_loss(x_recon, data.x)
            kl_loss = self.compute_kl_loss(mu, logvar)
            total_loss = recon_loss + self.kl_weight * kl_loss
            
            # 反向传播
            total_loss.backward()
            self.optimizer.step()
            
            reconstruction_losses.append(recon_loss.item())
            kl_losses.append(kl_loss.item())
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"轮次 {epoch + 1}/{pretrain_epochs}, "
                      f"重构损失: {recon_loss.item():.6f}, "
                      f"KL损失: {kl_loss.item():.6f}")
        
        print(f"变分自编码器预训练完成")
        
        # 阶段2：EM算法初始化聚类中心
        print(f"\n阶段2: EM算法聚类中心初始化")
        print("-" * 40)
        
        em_results = self.pretrain_with_em(
            data=data,
            max_iter=em_max_iter,
            tol=0.001,
            init_method=em_init_method,
            verbose=verbose
        )
        
        # 汇总结果
        results = {
            'reconstruction_losses': reconstruction_losses,
            'kl_losses': kl_losses,
            'em_results': em_results,
            'final_cluster_centers': self.cluster_centers.detach().cpu()
        }
        
        print("\n" + "=" * 60)
        print("变分深度图聚类预训练工作流完成！")
        print("=" * 60)
        
        return results
    
    def evaluate_clustering_quality(self, data: Data, true_labels: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        评估聚类质量
        
        Args:
            data: PyTorch Geometric数据对象
            true_labels: 真实标签（可选）
            
        Returns:
            评估指标字典
        """
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
        
        self.eval()
        with torch.no_grad():
            mu, logvar = self.encode(data.x, data.edge_index, 
                                   getattr(data, 'edge_weight', None))
            z = mu  # 使用均值进行评估
            cluster_probs = self._compute_cluster_probs(z)
        
        z_np = z.cpu().numpy()
        predicted_labels = cluster_probs.argmax(dim=1).cpu().numpy()
        
        metrics = {}
        
        # 内部评估指标
        if len(np.unique(predicted_labels)) > 1:
            metrics['silhouette_score'] = silhouette_score(z_np, predicted_labels)
            metrics['calinski_harabasz_score'] = calinski_harabasz_score(z_np, predicted_labels)
            metrics['davies_bouldin_score'] = davies_bouldin_score(z_np, predicted_labels)
        
        # 外部评估指标
        if true_labels is not None:
            from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
            metrics['adjusted_rand_score'] = adjusted_rand_score(true_labels, predicted_labels)
            metrics['normalized_mutual_info'] = normalized_mutual_info_score(true_labels, predicted_labels)
        
        return metrics
    
    def generate_zoning_policies(self, n: int, similarity_threshold: float = 0.95) -> torch.Tensor:
        """
        生成 n 个不同的 zoning policies (聚类概率分布)
        
        Args:
            n: 要生成的 policy 数量
            similarity_threshold: 去重时的余弦相似度阈值，超过此值认为是重复
            
        Returns:
            torch.Tensor: shape [n, num_clusters] 的 policy 矩阵
        """
        self.eval()
        
        policies = []
        max_attempts = n * 10  # 最大尝试次数，避免无限循环
        attempts = 0
        
        with torch.no_grad():
            while len(policies) < n and attempts < max_attempts:
                attempts += 1
                
                # 从标准正态分布采样 latent 向量
                z_sample = torch.randn(1, self.embedding_dim, device=next(self.parameters()).device)
                
                # 通过聚类概率计算得到 policy
                cluster_probs = self._compute_cluster_probs(z_sample)
                policy = cluster_probs.squeeze(0)  # 移除 batch 维度，得到 [num_clusters] 的概率分布
                
                # 检查与已有 policies 的相似度
                is_unique = True
                for existing_policy in policies:
                    # 计算余弦相似度
                    similarity = F.cosine_similarity(policy.unsqueeze(0), existing_policy.unsqueeze(0), dim=-1)
                    if similarity.item() > similarity_threshold:
                        is_unique = False
                        break
                
                if is_unique:
                    policies.append(policy)
            
            if len(policies) < n:
                print(f"Warning: Only generated {len(policies)} unique policies out of {n} requested after {max_attempts} attempts")
        
        # 将 policies 堆叠成矩阵
        if policies:
            return torch.stack(policies)
        else:
            # 如果没有生成任何 policy，返回随机初始化的
            print("Warning: No unique policies generated, returning random policies")
            random_policies = torch.randn(n, self.num_clusters, device=next(self.parameters()).device)
            return F.softmax(random_policies, dim=-1)

    def generate_node_policies(self, data: Data, n: int = 1, use_mean: bool = True,
                               noise_scale: float = 1.0) -> torch.Tensor:
        """
        生成节点级的聚类 policy（每个节点一行、每个簇一列）。

        Args:
            data: 图数据（需包含 x, edge_index, 可选 edge_weight）
            n: 生成的 policy 份数；n=1 返回 [N, K]；n>1 返回 [n, N, K]
            use_mean: 是否使用均值 mu 作为潜变量；若为 False 则按重参数化采样
            noise_scale: 采样时对 std 的缩放系数

        Returns:
            Tensor: 当 n==1 时 shape [N, num_clusters]；否则 [n, N, num_clusters]

        说明：聚类概率是由潜变量 z 与可学习的 cluster_centers 通过 t-分布核计算得到，
        与解码器重构无直接关系；decoder 主要服务于重构损失。
        """
        self.eval()
        device = next(self.parameters()).device
        data = data.to(device)

        with torch.no_grad():
            mu, logvar = self.encode(data.x, data.edge_index, getattr(data, 'edge_weight', None))

            if use_mean and n == 1:
                z = mu
                probs = self._compute_cluster_probs(z)
                return probs  # [N, K]

            # 否则进行 n 次采样
            std = torch.exp(0.5 * logvar) * noise_scale
            policies = []
            for _ in range(max(1, n)):
                if use_mean:
                    z = mu
                else:
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                probs = self._compute_cluster_probs(z)  # [N, K]
                policies.append(probs.unsqueeze(0))  # [1, N, K]

            return torch.cat(policies, dim=0)  # [n, N, K]


class VariationalClusteringTrainer:
    """
    变分深度图聚类训练器
    """
    
    def __init__(self, model: VariationalDeepGraphClustering, device: str = 'cpu'):
        """
        初始化训练器
        
        Args:
            model: 变分深度图聚类模型
            device: 计算设备
        """
        self.model = model.to(device)
        self.device = device
        
    def train_epoch(self, data: Data, target_probs: Optional[torch.Tensor] = None,
                   alpha_recon: float = 1.0, alpha_kl: float = 1.0,
                   alpha_cluster: float = 1.0) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            data: 图数据
            target_probs: 目标聚类分布
            alpha_recon: 重构损失权重
            alpha_kl: KL损失权重
            alpha_cluster: 聚类损失权重
            
        Returns:
            损失字典
        """
        self.model.train()
        data = data.to(self.device)
        
        # 前向传播
        z, mu, logvar, x_recon, cluster_probs = self.model(data)
        
        # 计算损失
        losses = self.model.compute_total_loss(
            data, z, mu, logvar, x_recon, cluster_probs,
            target_probs=target_probs,
            alpha_recon=alpha_recon,
            alpha_kl=alpha_kl,
            alpha_cluster=alpha_cluster
        )
        
        # 反向传播
        self.model.optimizer.zero_grad()
        losses['total'].backward()
        self.model.optimizer.step()
        
        # 返回损失值
        return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}
    
    def train_full_workflow(self, data: Data, epochs: int = 100,
                           pretrain_epochs: int = 50, em_max_iter: int = 100,
                           update_target_freq: int = 10, verbose: bool = True,
                           use_fairness: bool = True, use_frank_wolfe: bool = True,
                           data_type: str = "vgnn", save_model: bool = True) -> Dict[str, Any]:
        """
        完整的训练工作流程
        
        Args:
            data: 图数据
            epochs: 主训练轮数
            pretrain_epochs: 预训练轮数
            em_max_iter: EM算法最大迭代次数
            update_target_freq: 更新目标分布的频率
            verbose: 是否显示详细信息
            use_fairness: 是否使用公平性损失
            use_frank_wolfe: 是否使用Frank-Wolfe算法
            data_type: 数据类型标识
            save_model: 是否保存训练好的模型
            
        Returns:
            训练结果字典
        """
        print("开始变分深度图聚类完整训练流程")
        print("=" * 60)
        
        # 阶段1：预训练
        if verbose:
            print("阶段1: 预训练阶段")
        
        pretrain_results = self.model.complete_pretraining_workflow(
            data=data,
            pretrain_epochs=pretrain_epochs,
            em_max_iter=em_max_iter,
            verbose=verbose
        )
        
        # 阶段2：主训练循环
        if verbose:
            print(f"\n阶段2: 主训练阶段 ({epochs} 轮)")
            print("-" * 40)
        
        train_losses = []
        target_probs = None
        
        for epoch in range(epochs):
            # 定期更新目标分布
            if epoch % update_target_freq == 0:
                self.model.eval()
                with torch.no_grad():
                    _, _, _, _, cluster_probs = self.model(data.to(self.device))
                    target_probs = self.model.compute_target_distribution(cluster_probs)
            
            # 训练一个epoch
            losses = self.train_epoch(data, target_probs)
            train_losses.append(losses)
            
            if verbose and (epoch + 1) % 20 == 0:
                print(f"轮次 {epoch + 1}/{epochs}, "
                      f"总损失: {losses['total']:.6f}, "
                      f"重构: {losses['reconstruction']:.6f}, "
                      f"KL: {losses['kl_divergence']:.6f}, "
                      f"聚类: {losses['clustering']:.6f}")
        
        # 最终评估
        if verbose:
            print("\n阶段3: 最终评估")
            print("-" * 40)
        
        final_metrics = self.model.evaluate_clustering_quality(data.to(self.device))
        
        if verbose:
            print("最终聚类质量:")
            for metric, value in final_metrics.items():
                print(f"  {metric}: {value:.4f}")
        
        results = {
            'pretrain_results': pretrain_results,
            'train_losses': train_losses,
            'final_metrics': final_metrics
        }
        
        # 保存模型（如果启用）
        if save_model:
            self._save_vgnn_model_with_params(
                use_fairness=use_fairness,
                use_frank_wolfe=use_frank_wolfe,
                kl_weight=self.model.kl_weight,
                # fairness_beta=self.model.fairness_beta,
                # temperature=self.model.temperature,
                learning_rate=self.model.learning_rate,
                data_type=data_type,
                num_clusters=self.model.num_clusters,
                epochs=epochs,
                pretrain_epochs=pretrain_epochs,
                results=results
            )
        
        print("\n" + "=" * 60)
        print("变分深度图聚类训练完成！")
        print("=" * 60)
        
        return results
    
    def _save_vgnn_model_with_params(self, use_fairness: bool = True, use_frank_wolfe: bool = True,
                                    kl_weight: float = 0.1, fairness_beta: float = 1.0,
                                    temperature: float = 0.1, learning_rate: float = 0.001,
                                    data_type: str = "vgnn", num_clusters: int = 5, 
                                    epochs: int = 100, pretrain_epochs: int = 50,
                                    results: Optional[Dict] = None) -> str:
        """
        保存训练好的变分GNN模型及其参数信息到pth文件夹
        
        Args:
            use_fairness: 是否使用公平性损失
            use_frank_wolfe: 是否使用Frank-Wolfe算法
            kl_weight: KL散度损失权重
            fairness_beta: 公平性损失beta参数
            temperature: 对比学习温度参数
            learning_rate: 学习率
            data_type: 数据类型标识
            num_clusters: 聚类数量
            epochs: 主训练轮数
            pretrain_epochs: 预训练轮数
            results: 训练结果字典
            
        Returns:
            保存的文件路径
        """
        import datetime
        import json
        
        # 创建pth文件夹
        pth_folder = Path("pth")
        pth_folder.mkdir(exist_ok=True)
        
        # 生成文件名（包含关键参数信息）
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fairness_tag = "fair" if use_fairness else "nofair"
        fw_tag = "fw" if use_frank_wolfe else "nofw"
        filename = f"vgnn_clustering_{data_type}_{fairness_tag}_{fw_tag}_c{num_clusters}_e{epochs}_{timestamp}"
        
        # 准备保存的数据
        save_data = {
            # 模型状态
            'model_state_dict': self.model.state_dict(),
            'model_config': {
                'input_dim': self.model.input_dim,
                'embedding_dim': self.model.embedding_dim,
                'num_clusters': self.model.num_clusters,
                'encoder_type': self.model.encoder_type,
                'hidden_dims': self.model.hidden_dims,
                'dropout': self.model.dropout,
                # 'temperature': self.model.temperature,
                # 'fairness_beta': self.model.fairness_beta,
                'use_frank_wolfe': self.model.use_frank_wolfe,
                'kl_weight': self.model.kl_weight,
                'use_batch_norm': self.model.use_batch_norm,
                'model_type': self.model.__class__.__name__
            },
            
            # 训练参数
            'training_params': {
                'use_fairness': use_fairness,
                'use_frank_wolfe': use_frank_wolfe,
                'kl_weight': kl_weight,
                'fairness_beta': fairness_beta,
                'temperature': temperature,
                'learning_rate': learning_rate,
                'epochs': epochs,
                'pretrain_epochs': pretrain_epochs,
                'data_type': data_type,
                'device': str(self.device)
            },
            
            # 训练结果（如果提供）
            'training_results': results if results else {},
            
            # 元信息
            'metadata': {
                'timestamp': timestamp,
                'torch_version': torch.__version__,
                'saved_datetime': datetime.datetime.now().isoformat(),
                'fairness_enabled': use_fairness,
                'frank_wolfe_enabled': use_frank_wolfe,
                'model_architecture': 'VariationalGNN'
            }
        }
        
        # 保存模型文件
        model_path = pth_folder / f"{filename}.pth"
        torch.save(save_data, model_path, _use_new_zipfile_serialization=False)
        
        # 保存参数信息到JSON文件（便于查看）
        params_info = {
            'model_file': f"{filename}.pth",
            'model_architecture': 'Variational Graph Neural Network',
            'training_config': {
                'fairness_loss': use_fairness,
                'frank_wolfe_algorithm': use_frank_wolfe,
                'variational_parameters': {
                    'kl_weight': kl_weight,
                    'temperature': temperature,
                    'fairness_beta': fairness_beta
                },
                'training_settings': {
                    'main_epochs': epochs,
                    'pretrain_epochs': pretrain_epochs,
                    'learning_rate': learning_rate,
                    'num_clusters': num_clusters,
                    'data_type': data_type,
                    'device': str(self.device)
                }
            },
            'model_structure': {
                'encoder_type': self.model.encoder_type,
                'input_dim': self.model.input_dim,
                'embedding_dim': self.model.embedding_dim,
                'hidden_dims': self.model.hidden_dims,
                'dropout': self.model.dropout,
                'use_batch_norm': self.model.use_batch_norm
            },
            'final_results': {
                'training_completed': True,
                'final_metrics': results.get('final_metrics', {}) if results else {},
                'pretrain_results': results.get('pretrain_results', {}) if results else {}
            },
            'metadata': save_data['metadata']
        }
        
        # 清理numpy数组，只保留基础数据类型
        if 'final_metrics' in params_info['final_results']:
            metrics = params_info['final_results']['final_metrics']
            if metrics and any(hasattr(v, 'tolist') for v in metrics.values() if v is not None):
                params_info['final_results']['metrics_summary'] = {
                    k: float(v) if hasattr(v, 'item') else v 
                    for k, v in metrics.items() if v is not None
                }
        
        # 保存JSON参数文件
        json_path = pth_folder / f"{filename}_params.json"
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(params_info, f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 变分GNN模型已保存:")
        print(f"   模型文件: {model_path}")
        print(f"   参数文件: {json_path}")
        print(f"   训练配置:")
        print(f"     - 模型架构: 变分图神经网络 ({self.model.encoder_type.upper()})")
        print(f"     - 公平性损失: {'✅' if use_fairness else '❌'}")
        print(f"     - Frank-Wolfe: {'✅' if use_frank_wolfe else '❌'}")
        print(f"     - 聚类数量: {num_clusters}")
        print(f"     - 主训练轮数: {epochs}")
        print(f"     - 预训练轮数: {pretrain_epochs}")
        print(f"     - 数据类型: {data_type}")
        print(f"     - KL权重: {kl_weight}")
        print(f"     - 温度参数: {temperature}")
        print(f"     - 公平性Beta: {fairness_beta}")
        
        return str(model_path)

    @staticmethod
    def load_vgnn_model_with_params(model_path: str, device: str = 'cpu') -> Tuple['VariationalDeepGraphClustering', Dict]:
        """
        加载保存的变分GNN模型及其参数信息
        
        Args:
            model_path: 模型文件路径
            device: 加载到的设备
            
        Returns:
            (model, params_dict): 加载的模型和参数字典
        """
        # 加载保存的数据
        saved_data = torch.load(model_path, map_location=device, weights_only=False)
        
        # 重建模型
        model_config = saved_data['model_config']
        if saved_data['model_config']['model_type'] == 'VariationalDeepGraphClustering':
            model = VariationalDeepGraphClustering(
                input_dim=model_config['input_dim'],
                embedding_dim=model_config['embedding_dim'],
                num_clusters=model_config['num_clusters'],
                encoder_type=model_config.get('encoder_type', 'gcn'),
                hidden_dims=model_config.get('hidden_dims', [64, 32]),
                dropout=model_config.get('dropout', 0.1),
                temperature=model_config.get('temperature', 0.1),
                fairness_beta=model_config.get('fairness_beta', 1.0),
                use_frank_wolfe=model_config.get('use_frank_wolfe', False),
                kl_weight=model_config.get('kl_weight', 0.1),
                use_batch_norm=model_config.get('use_batch_norm', True)
            )
        else:
            raise ValueError(f"不支持的模型类型: {model_config['model_type']}")
        
        # 加载模型状态
        model.load_state_dict(saved_data['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ 变分GNN模型已加载: {model_path}")
        print(f"   模型配置:")
        print(f"     - 架构: {model_config['encoder_type'].upper()} 变分编码器")
        print(f"     - 嵌入维度: {model_config['embedding_dim']}")
        print(f"     - 聚类数量: {model_config['num_clusters']}")
        
        training_params = saved_data['training_params']
        print(f"   训练参数:")
        print(f"     - 公平性损失: {'✅' if training_params['use_fairness'] else '❌'}")
        print(f"     - Frank-Wolfe: {'✅' if training_params['use_frank_wolfe'] else '❌'}")
        print(f"     - KL权重: {training_params.get('kl_weight', 'N/A')}")
        print(f"     - 温度参数: {training_params.get('temperature', 'N/A')}")
        print(f"     - 数据类型: {training_params.get('data_type', 'N/A')}")
        
        return model, saved_data
    
    def train_enhanced_vgnn(self, data: Data, od_data: pd.DataFrame, adj_matrix: torch.Tensor,
                           charge_num: Optional[torch.Tensor] = None, num_clusters: int = 5,
                           epochs: int = 200, pretrain_epochs: int = 50,
                           use_fairness: bool = True, use_frank_wolfe: bool = True,
                           kl_weight: float = 0.1, fairness_beta: float = 1.0,
                           temperature: float = 0.1, data_type: str = "vgnn",
                           save_model: bool = True, verbose: bool = True) -> Tuple['VariationalDeepGraphClustering', 'VariationalClusteringTrainer', Dict]:
        """
        训练增强的变分GNN聚类模型，支持全面的参数配置
        
        Args:
            data: 图数据
            od_data: OD数据
            adj_matrix: 邻接矩阵
            charge_num: 充电桩数量特征
            num_clusters: 聚类数量
            epochs: 主训练轮数
            pretrain_epochs: 预训练轮数
            use_fairness: 是否使用公平性损失
            use_frank_wolfe: 是否使用Frank-Wolfe算法
            kl_weight: KL散度损失权重
            fairness_beta: 公平性损失beta参数
            temperature: 对比学习温度参数
            data_type: 数据类型标识
            save_model: 是否保存训练好的模型
            verbose: 是否显示详细信息
            
        Returns:
            (model, trainer, results): 训练好的模型、训练器和结果
        """
        print(f"\n训练增强变分GNN聚类模型...")
        print(f"公平性: {use_fairness}, Frank-Wolfe: {use_frank_wolfe}")
        print(f"KL权重: {kl_weight}, 公平性Beta: {fairness_beta}, 温度: {temperature}")
        print(f"聚类数量: {num_clusters}, 主训练: {epochs}轮, 预训练: {pretrain_epochs}轮")
        
        # 确保数据在正确的设备上
        data = data.to(self.device)
        adj_matrix = adj_matrix.to(self.device)
        if charge_num is not None:
            charge_num = charge_num.to(self.device)
        
        # 更新模型参数
        self.model.kl_weight = kl_weight
        self.model.fairness_beta = fairness_beta
        self.model.temperature = temperature
        self.model.use_frank_wolfe = use_frank_wolfe
        
        # 运行完整的训练工作流程
        results = self.train_full_workflow(
            data=data,
            epochs=epochs,
            pretrain_epochs=pretrain_epochs,
            em_max_iter=100,
            update_target_freq=10,
            verbose=verbose,
            use_fairness=use_fairness,
            use_frank_wolfe=use_frank_wolfe,
            data_type=data_type,
            save_model=save_model
        )
        
        # 添加额外的训练信息到结果中
        results['training_config'] = {
            'use_fairness': use_fairness,
            'use_frank_wolfe': use_frank_wolfe,
            'kl_weight': kl_weight,
            'fairness_beta': fairness_beta,
            'temperature': temperature,
            'num_clusters': num_clusters,
            'epochs': epochs,
            'pretrain_epochs': pretrain_epochs,
            'data_type': data_type
        }
        
        print(f"\n✅ 增强变分GNN聚类训练完成!")
        print(f"   - 模型架构: {self.model.encoder_type.upper()} 变分编码器")
        print(f"   - 最终聚类质量指标:")
        if 'final_metrics' in results:
            for metric, value in results['final_metrics'].items():
                print(f"     • {metric}: {value:.4f}")
        
        return self.model, self, results