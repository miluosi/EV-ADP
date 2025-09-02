import torch
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import Data
from src.VariationalDeepGraphClustering import VariationalDeepGraphClustering, VariationalClusteringTrainer

def test_zoning_policy_generation():
    """
    测试 zoning policy 生成函数
    """
    print("=" * 60)
    print("测试 Zoning Policy 生成函数")
    print("=" * 60)
    
    # 创建模拟数据
    num_nodes = 50
    input_dim = 10
    embedding_dim = 16
    num_clusters = 5
    
    # 创建模拟图数据
    x = torch.randn(num_nodes, input_dim)
    edge_index = torch.randint(0, num_nodes, (2, 100))  # 随机边
    data = Data(x=x, edge_index=edge_index)
    
    print(f"创建模拟数据:")
    print(f"  节点数: {num_nodes}")
    print(f"  特征维度: {input_dim}")
    print(f"  嵌入维度: {embedding_dim}")
    print(f"  聚类数: {num_clusters}")
    
    # 创建变分聚类模型
    model = VariationalDeepGraphClustering(
        input_dim=input_dim,
        embedding_dim=embedding_dim,
        num_clusters=num_clusters,
        hidden_dims=[32, 16],
        encoder_type='gcn'
    )
    
    print(f"\n模型创建成功")
    
    # 测试1: 生成少量policies
    print(f"\n测试1: 生成 3 个 zoning policies")
    n_policies = 3
    
    try:
        # 检查模型是否有 generate_zoning_policies 方法
        if hasattr(model, 'generate_zoning_policies'):
            policies = model.generate_zoning_policies(n_policies)
            
            print(f"成功生成 {len(policies)} 个 policies")
            print(f"Policy 矩阵形状: {policies.shape}")
            print(f"预期形状: [{n_policies}, {num_clusters}]")
            
            # 验证每个policy是概率分布
            for i, policy in enumerate(policies):
                prob_sum = policy.sum().item()
                print(f"Policy {i+1} 概率和: {prob_sum:.6f} (应该接近 1.0)")
                
                # 检查是否为有效概率分布
                if abs(prob_sum - 1.0) < 0.01:
                    print(f"  ✓ Policy {i+1} 是有效的概率分布")
                else:
                    print(f"  ✗ Policy {i+1} 不是有效的概率分布")
        else:
            print("✗ 模型没有 generate_zoning_policies 方法")
            return
            
    except Exception as e:
        print(f"✗ 生成 policies 时出错: {e}")
        return
    
    # 测试2: 验证policy多样性
    print(f"\n测试2: 验证 policy 多样性")
    
    try:
        # 计算policies之间的余弦相似度
        similarities = []
        for i in range(len(policies)):
            for j in range(i+1, len(policies)):
                similarity = F.cosine_similarity(
                    policies[i].unsqueeze(0), 
                    policies[j].unsqueeze(0)
                ).item()
                similarities.append(similarity)
                print(f"Policy {i+1} 与 Policy {j+1} 的相似度: {similarity:.4f}")
        
        avg_similarity = np.mean(similarities)
        print(f"\n平均相似度: {avg_similarity:.4f}")
        
        if avg_similarity < 0.95:  # 默认阈值
            print("✓ Policies 具有良好的多样性")
        else:
            print("✗ Policies 过于相似，多样性不足")
            
    except Exception as e:
        print(f"✗ 验证多样性时出错: {e}")
    
    # 测试3: 生成大量policies测试去重功能
    print(f"\n测试3: 生成大量 policies 测试去重功能")
    
    try:
        n_large = 10
        large_policies = model.generate_zoning_policies(n_large, similarity_threshold=0.90)
        
        print(f"请求生成 {n_large} 个 policies")
        print(f"实际生成 {len(large_policies)} 个 policies")
        
        if len(large_policies) == n_large:
            print("✓ 成功生成所有请求的 policies")
        else:
            print(f"⚠ 只生成了 {len(large_policies)} 个 policies（可能由于去重）")
            
    except Exception as e:
        print(f"✗ 生成大量 policies 时出错: {e}")
    
    # 测试4: 验证policy分布特性
    print(f"\n测试4: 验证 policy 分布特性")
    
    try:
        # 分析policy的统计特性
        policies_np = policies.detach().cpu().numpy()
        
        print("Policy 统计分析:")
        print(f"  最小值: {policies_np.min():.6f}")
        print(f"  最大值: {policies_np.max():.6f}")
        print(f"  均值: {policies_np.mean():.6f}")
        print(f"  标准差: {policies_np.std():.6f}")
          # 检查每个聚类的平均概率
        cluster_avg_probs = policies_np.mean(axis=0)
        print(f"\n每个聚类的平均概率:")
        for i, prob in enumerate(cluster_avg_probs):
            print(f"  聚类 {i+1}: {prob:.4f}")
            
        # 检查是否有偏向某些聚类
        prob_variance = np.var(cluster_avg_probs)
        print(f"\n聚类概率方差: {prob_variance:.6f}")
        
        if prob_variance < 0.1:
            print("✓ 聚类概率分布相对均匀")
        else:
            print("⚠ 聚类概率分布存在偏向")
            
    except Exception as e:
        print(f"✗ 分析 policy 特性时出错: {e}")
    
    print(f"\n" + "=" * 60)
    print("Zoning Policy 生成测试完成")
    print("=" * 60)


def test_variational_clustering_trainer():
    """
    测试 VariationalClusteringTrainer 的训练流程
    """
    print("=" * 60)
    print("测试 VariationalClusteringTrainer 训练流程")
    print("=" * 60)
      # 创建实验配置
    config = VGNNExperimentConfig()
    
    # 设置较小的参数以加快测试
    config.model.embedding_dim = 16
    config.model.hidden_dims = [32, 16]
    config.training.main_epochs = 5
    config.training.pretrain_epochs = 3
    
    print("实验配置:")
    print(f"  嵌入维度: {config.model.embedding_dim}")
    print(f"  隐藏层维度: {config.model.hidden_dims}")
    print(f"  主训练轮数: {config.training.main_epochs}")
    print(f"  预训练轮数: {config.training.pretrain_epochs}")
    
    # 创建模拟数据集
    print(f"\n创建模拟数据集...")
    num_nodes = 100
    input_dim = 10
    num_clusters = 5
      # 创建多个图数据样本
    data_list = []
    for i in range(20):  # 创建20个图样本
        x = torch.randn(num_nodes, input_dim)
        # 创建更有结构的边连接
        edge_index = []
        for node in range(num_nodes):
            # 每个节点连接到几个随机邻居
            neighbors = np.random.choice(num_nodes, size=5, replace=False)
            for neighbor in neighbors:
                if neighbor != node:
                    edge_index.append([node, neighbor])
        
        edge_index = torch.tensor(edge_index).t().contiguous()
        data_list.append(Data(x=x, edge_index=edge_index))
    
    print(f"  创建了 {len(data_list)} 个图样本")
    print(f"  每个图节点数: {num_nodes}")
    print(f"  特征维度: {input_dim}")
    print(f"  目标聚类数: {num_clusters}")
    
    # 创建trainer
    print(f"\n创建 VariationalClusteringTrainer...")
    try:
        # 首先创建模型
        model = VariationalDeepGraphClustering(
            input_dim=input_dim,
            embedding_dim=config.model.embedding_dim,
            num_clusters=num_clusters,
            hidden_dims=config.model.hidden_dims,
            encoder_type=config.model.encoder_type
        )
        
        # 然后创建trainer
        trainer = VariationalClusteringTrainer(
            model=model,
            device=config.training.device
        )
        print("✓ Trainer 创建成功")
    except Exception as e:
        print(f"✗ Trainer 创建失败: {e}")
        return
      # 测试完整训练流程
    print(f"\n测试完整训练流程...")
    try:
        # 使用一个图数据样本进行训练
        sample_data = data_list[0]
        
        # 运行完整训练流程
        train_results = trainer.train_full_workflow(
            data=sample_data,
            epochs=config.training.main_epochs,
            pretrain_epochs=config.training.pretrain_epochs,
            verbose=True
        )
        
        print(f"✓ 完整训练流程完成")
        
        # 检查训练结果
        if 'pretrain_results' in train_results:
            print(f"  预训练结果: 已获取")
        if 'train_losses' in train_results:
            train_losses = train_results['train_losses']
            print(f"  训练损失记录: {len(train_losses)} 个epoch")
            if len(train_losses) > 0:
                final_loss = train_losses[-1]
                print(f"    最终总损失: {final_loss['total']:.6f}")
                print(f"    最终重构损失: {final_loss['reconstruction']:.6f}")
                print(f"    最终KL损失: {final_loss['kl_divergence']:.6f}")
                print(f"    最终聚类损失: {final_loss['clustering']:.6f}")
        
        if 'final_metrics' in train_results:
            final_metrics = train_results['final_metrics']
            print(f"  最终评估指标: {len(final_metrics)} 个")
            for metric, value in final_metrics.items():
                print(f"    {metric}: {value:.4f}")
        
        # 验证训练轮数
        train_losses = train_results.get('train_losses', [])
        if len(train_losses) == config.training.main_epochs:
            print("✓ 主训练轮数正确")
        else:
            print(f"⚠ 主训练轮数不匹配: 期望 {config.training.main_epochs}, 实际 {len(train_losses)}")
            
    except Exception as e:
        print(f"✗ 训练流程失败: {e}")
        import traceback
        traceback.print_exc()
        return
      # 测试模型评估
    print(f"\n测试模型评估...")
    try:
        # 使用一个样本进行评估
        sample_data = data_list[0]
        
        # 获取聚类结果
        with torch.no_grad():
            # 使用模型的forward方法获取所有输出
            z, mu, logvar, x_recon, cluster_probs = trainer.model(sample_data)
            cluster_assignments = torch.argmax(cluster_probs, dim=1)
        
        print(f"✓ 模型评估完成")
        print(f"  潜在表示维度: {z.shape}")
        print(f"  重构特征维度: {x_recon.shape}")
        print(f"  聚类概率维度: {cluster_probs.shape}")
        print(f"  聚类分配: {cluster_assignments.shape}")
        
        # 检查聚类分布
        unique_clusters, counts = torch.unique(cluster_assignments, return_counts=True)
        print(f"  发现的聚类: {len(unique_clusters)} 个")
        for cluster, count in zip(unique_clusters, counts):
            print(f"    聚类 {cluster.item()}: {count.item()} 个节点")
            
    except Exception as e:
        print(f"✗ 模型评估失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 测试policy生成
    print(f"\n测试训练后的 policy 生成...")
    try:
        n_policies = 3
        policies = trainer.model.generate_zoning_policies(n_policies)
        
        print(f"✓ Policy 生成成功")
        print(f"  生成 policies 数量: {len(policies)}")
        print(f"  Policy 维度: {policies.shape}")
        
        # 验证每个policy的有效性
        for i, policy in enumerate(policies):
            prob_sum = policy.sum().item()
            print(f"  Policy {i+1} 概率和: {prob_sum:.6f}")
            
    except Exception as e:
        print(f"✗ Policy 生成失败: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"\n" + "=" * 60)
    print("VariationalClusteringTrainer 测试完成")
    print("=" * 60)


def test_config_integration():
    """
    测试配置文件的集成使用
    """
    print("=" * 60)
    print("测试配置文件集成")
    print("=" * 60)
      # 测试默认配置
    print("测试默认配置...")
    try:
        config = VGNNExperimentConfig()
        print("✓ 默认配置创建成功")
        
        # 打印主要配置信息
        print("主要配置参数:")
        print(f"  模型类型: {config.model.encoder_type}")
        print(f"  嵌入维度: {config.model.embedding_dim}")
        print(f"  聚类数: {config.model.num_clusters}")
        print(f"  主训练轮数: {config.training.main_epochs}")
        print(f"  学习率: {config.model.learning_rate}")
        print(f"  KL权重: {config.model.kl_weight}")
        
    except Exception as e:
        print(f"✗ 默认配置创建失败: {e}")
        return
      # 测试配置修改
    print(f"\n测试配置修改...")
    try:
        config.model.num_clusters = 8
        config.model.learning_rate = 0.005
        
        print("✓ 配置修改成功")
        print(f"  新聚类数: {config.model.num_clusters}")
        print(f"  新学习率: {config.model.learning_rate}")
        
    except Exception as e:
        print(f"✗ 配置修改失败: {e}")
        return
    
    # 测试配置验证
    print(f"\n测试配置验证...")
    try:
        # 设置无效配置
        config.model.num_clusters = -1  # 无效值
        
        is_valid = validate_config(config)
        if not is_valid:
            print("✓ 配置验证正确识别了无效配置")
        else:
            print("⚠ 配置验证未能识别无效配置")
        
        # 恢复有效配置
        config.model.num_clusters = 5
        is_valid = validate_config(config)
        if is_valid:
            print("✓ 配置验证通过")
        else:
            print("✗ 有效配置验证失败")
            
    except Exception as e:
        print(f"✗ 配置验证测试失败: {e}")
    
    print(f"\n" + "=" * 60)
    print("配置文件集成测试完成")
    print("=" * 60)


if __name__ == "__main__":
    # 运行所有测试
    test_zoning_policy_generation()
    print("\n" + "="*80 + "\n")
    
    test_config_integration()
    print("\n" + "="*80 + "\n")
    
    test_variational_clustering_trainer()
