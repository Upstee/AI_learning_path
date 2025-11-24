"""
从零实现层次聚类算法
本示例详细展示了如何从零开始实现层次聚类算法
适合小白学习，包含大量注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs
from scipy.spatial.distance import pdist, squareform


class HierarchicalClustering:
    """
    层次聚类算法（凝聚式）
    
    这个类实现了凝聚式层次聚类算法的核心功能
    通过逐步合并最相似的簇来构建层次树
    """
    
    def __init__(self, linkage='average', metric='euclidean'):
        """
        初始化层次聚类器
        
        参数说明：
        ----------
        linkage : str, 默认='average'
            链接方法，决定如何计算簇间距离
            - 'single': 单链接（最近距离）
            - 'complete': 完全链接（最远距离）
            - 'average': 平均链接（平均距离）
            - 'centroid': 质心链接（质心距离）
        
        metric : str, 默认='euclidean'
            距离度量方法
            - 'euclidean': 欧氏距离
            - 'manhattan': 曼哈顿距离
        """
        self.linkage = linkage
        self.metric = metric
        
        # 存储合并历史
        # 每一行记录一次合并：[簇i, 簇j, 距离, 新簇的样本数]
        self.merge_history = []
        
        # 存储距离矩阵
        self.distance_matrix = None
    
    def _compute_distance_matrix(self, X):
        """
        计算距离矩阵
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            样本数据
        
        返回：
        ------
        distance_matrix : array, shape = [n_samples, n_samples]
            距离矩阵，distance_matrix[i, j]是样本i和j之间的距离
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 初始化距离矩阵
        distance_matrix = np.zeros((n_samples, n_samples))
        
        # 计算每对样本之间的距离
        # 对于每对样本(i, j)
        for i in range(n_samples):
            for j in range(i + 1, n_samples):
                if self.metric == 'euclidean':
                    # 欧氏距离：sqrt(sum((x_i - x_j)^2))
                    diff = X[i] - X[j]
                    dist = np.sqrt(np.sum(diff ** 2))
                elif self.metric == 'manhattan':
                    # 曼哈顿距离：sum(|x_i - x_j|)
                    dist = np.sum(np.abs(X[i] - X[j]))
                else:
                    raise ValueError(f"不支持的距离度量: {self.metric}")
                
                # 距离矩阵是对称的
                distance_matrix[i, j] = dist
                distance_matrix[j, i] = dist
        
        return distance_matrix
    
    def _compute_cluster_distance(self, cluster_i, cluster_j, distance_matrix):
        """
        计算两个簇之间的距离
        
        根据链接方法计算簇间距离
        
        参数说明：
        ----------
        cluster_i : set
            簇i中的样本索引集合
        
        cluster_j : set
            簇j中的样本索引集合
        
        distance_matrix : array
            距离矩阵
        
        返回：
        ------
        distance : float
            两个簇之间的距离
        """
        if self.linkage == 'single':
            # 单链接：最近距离
            # 找到两个簇中最近的一对样本
            min_dist = float('inf')
            for i in cluster_i:
                for j in cluster_j:
                    dist = distance_matrix[i, j]
                    if dist < min_dist:
                        min_dist = dist
            return min_dist
        
        elif self.linkage == 'complete':
            # 完全链接：最远距离
            # 找到两个簇中最远的一对样本
            max_dist = 0
            for i in cluster_i:
                for j in cluster_j:
                    dist = distance_matrix[i, j]
                    if dist > max_dist:
                        max_dist = dist
            return max_dist
        
        elif self.linkage == 'average':
            # 平均链接：平均距离
            # 计算两个簇中所有样本对距离的平均值
            total_dist = 0
            count = 0
            for i in cluster_i:
                for j in cluster_j:
                    total_dist += distance_matrix[i, j]
                    count += 1
            return total_dist / count if count > 0 else float('inf')
        
        elif self.linkage == 'centroid':
            # 质心链接：质心距离
            # 计算两个簇的质心，然后计算质心之间的距离
            # 这里简化处理，使用平均链接近似
            # 实际应用中需要计算真正的质心
            return self._compute_cluster_distance(cluster_i, cluster_j, distance_matrix)
        
        else:
            raise ValueError(f"不支持的链接方法: {self.linkage}")
    
    def fit(self, X):
        """
        训练层次聚类器
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练数据
        
        返回：
        ------
        self : object
            返回自身，支持链式调用
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        print("=" * 60)
        print("层次聚类训练")
        print("=" * 60)
        print(f"数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  链接方法: {self.linkage}")
        print(f"  距离度量: {self.metric}")
        
        # 1. 计算距离矩阵
        print("\n步骤1: 计算距离矩阵")
        self.distance_matrix = self._compute_distance_matrix(X)
        print(f"距离矩阵形状: {self.distance_matrix.shape}")
        
        # 2. 初始化：每个样本作为一个簇
        print("\n步骤2: 初始化簇")
        clusters = [{i} for i in range(n_samples)]  # 每个簇包含一个样本
        print(f"初始簇数: {len(clusters)}")
        
        # 3. 迭代合并
        print("\n步骤3: 开始迭代合并")
        self.merge_history = []
        next_cluster_id = n_samples  # 新簇的ID从n_samples开始
        
        for iteration in range(n_samples - 1):
            # 3.1 找到距离最近的两个簇
            min_dist = float('inf')
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    dist = self._compute_cluster_distance(
                        clusters[i], clusters[j], self.distance_matrix
                    )
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # 3.2 合并两个簇
            # 获取要合并的簇的ID（原始样本索引）
            cluster_i_id = min(clusters[merge_i])
            cluster_j_id = min(clusters[merge_j])
            
            # 合并簇
            new_cluster = clusters[merge_i].union(clusters[merge_j])
            
            # 记录合并历史
            # 格式：[簇i的ID, 簇j的ID, 距离, 新簇的样本数]
            self.merge_history.append([
                cluster_i_id,
                cluster_j_id,
                min_dist,
                len(new_cluster)
            ])
            
            # 更新簇列表
            # 删除被合并的两个簇，添加新簇
            if merge_i > merge_j:
                clusters.pop(merge_i)
                clusters.pop(merge_j)
            else:
                clusters.pop(merge_j)
                clusters.pop(merge_i)
            clusters.append(new_cluster)
            
            # 打印进度（每10次迭代打印一次）
            if (iteration + 1) % 10 == 0 or iteration < 5:
                print(f"  迭代 {iteration + 1}: 合并簇 {cluster_i_id} 和 {cluster_j_id}, "
                      f"距离={min_dist:.4f}, 剩余簇数={len(clusters)}")
        
        # 转换为numpy数组
        self.merge_history = np.array(self.merge_history)
        
        print(f"\n训练完成！")
        print(f"  总合并次数: {len(self.merge_history)}")
        print(f"  最终簇数: 1（所有样本合并为一个簇）")
        
        return self
    
    def get_labels(self, n_clusters):
        """
        获取指定数量的簇的标签
        
        参数说明：
        ----------
        n_clusters : int
            要得到的簇数
        
        返回：
        ------
        labels : array, shape = [n_samples]
            每个样本所属的簇（0到n_clusters-1）
        """
        n_samples = self.distance_matrix.shape[0]
        
        # 从合并历史中，找到需要保留的最后n_clusters-1次合并
        # 也就是说，我们需要"撤销"前面的合并
        if n_clusters > n_samples:
            raise ValueError(f"簇数不能大于样本数: {n_clusters} > {n_samples}")
        
        # 初始化：每个样本一个簇
        clusters = [{i} for i in range(n_samples)]
        cluster_labels = {i: i for i in range(n_samples)}  # 每个样本的簇标签
        
        # 执行合并，直到只剩n_clusters个簇
        # 需要执行 n_samples - n_clusters 次合并
        n_merges = n_samples - n_clusters
        
        for i in range(n_merges):
            merge_info = self.merge_history[i]
            cluster_i_id = int(merge_info[0])
            cluster_j_id = int(merge_info[1])
            
            # 找到包含这两个样本的簇
            cluster_i = None
            cluster_j = None
            for cluster in clusters:
                if cluster_i_id in cluster:
                    cluster_i = cluster
                if cluster_j_id in cluster:
                    cluster_j = cluster
            
            # 合并簇
            if cluster_i and cluster_j and cluster_i != cluster_j:
                new_cluster = cluster_i.union(cluster_j)
                clusters.remove(cluster_i)
                clusters.remove(cluster_j)
                clusters.append(new_cluster)
        
        # 分配标签
        labels = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = label
        
        return labels


# ========== 示例：使用层次聚类 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("层次聚类示例")
    print("=" * 60)
    
    # ========== 1. 生成示例数据 ==========
    print("\n1. 生成示例数据")
    print("-" * 60)
    
    # 生成聚类数据
    X, y_true = make_blobs(
        n_samples=30,  # 使用较少的样本，因为层次聚类计算慢
        centers=3,
        n_features=2,
        random_state=42
    )
    
    print(f"数据形状: {X.shape}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # ========== 2. 训练模型 ==========
    print("\n2. 训练层次聚类模型")
    print("-" * 60)
    
    # 创建层次聚类器
    # linkage='average' 使用平均链接
    hc = HierarchicalClustering(linkage='average', metric='euclidean')
    
    # 训练模型
    hc.fit(X)
    
    # ========== 3. 获取不同数量的簇 ==========
    print("\n3. 获取不同数量的簇")
    print("-" * 60)
    
    # 测试不同的簇数
    for n_clusters in [2, 3, 4, 5]:
        labels = hc.get_labels(n_clusters)
        print(f"K={n_clusters}: 簇标签={np.unique(labels)}")
    
    # ========== 4. 可视化结果 ==========
    print("\n4. 可视化结果")
    print("-" * 60)
    
    # 获取K=3的标签
    labels = hc.get_labels(3)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：真实标签
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('真实标签', fontsize=14)
    axes[0].set_xlabel('特征1', fontsize=12)
    axes[0].set_ylabel('特征2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：层次聚类结果
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('层次聚类结果 (K=3)', fontsize=14)
    axes[1].set_xlabel('特征1', fontsize=12)
    axes[1].set_ylabel('特征2', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('hierarchical_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 5. 使用scipy绘制树状图 ==========
    print("\n5. 绘制树状图")
    print("-" * 60)
    
    # 使用scipy的linkage函数（更高效）
    from scipy.cluster.hierarchy import linkage, dendrogram
    
    # 计算链接矩阵
    Z = linkage(X, method='average', metric='euclidean')
    
    # 绘制树状图
    plt.figure(figsize=(12, 8))
    dendrogram(Z, leaf_rotation=90, leaf_font_size=12)
    plt.title('层次聚类树状图', fontsize=14)
    plt.xlabel('样本索引', fontsize=12)
    plt.ylabel('距离', fontsize=12)
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
总结：
1. 层次聚类通过逐步合并最相似的簇来构建层次树
2. 不同的链接方法会产生不同的结果
3. 树状图可以可视化聚类过程和选择K值
4. 层次聚类计算复杂度高，适合小规模数据
""")

