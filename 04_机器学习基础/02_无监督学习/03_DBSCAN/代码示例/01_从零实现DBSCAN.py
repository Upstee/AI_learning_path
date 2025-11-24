"""
从零实现DBSCAN聚类算法
本示例详细展示了如何从零开始实现DBSCAN算法
适合小白学习，包含大量注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from collections import deque


class DBSCAN:
    """
    DBSCAN聚类算法
    
    这个类实现了DBSCAN算法的核心功能
    基于密度进行聚类，可以自动发现簇的数量和识别噪声点
    """
    
    def __init__(self, eps=0.5, min_samples=5):
        """
        初始化DBSCAN聚类器
        
        参数说明：
        ----------
        eps : float, 默认=0.5
            邻域半径（epsilon）
            定义"附近"的范围
            为什么需要这个参数？
            - 太小：很多点成为噪声，簇被分裂
            - 太大：所有点聚成一个簇，失去意义
            - 需要根据数据特点选择合适的值
        
        min_samples : int, 默认=5
            成为核心点所需的最少邻居数
            为什么需要这个参数？
            - 太小：噪声点可能成为核心点
            - 太大：真正的簇可能被识别为噪声
            - 经验法则：min_samples = 2 × 特征数
        """
        self.eps = eps
        self.min_samples = min_samples
        
        # 存储聚类结果
        # labels_: 每个样本的簇标签，-1表示噪声点
        self.labels_ = None
        
        # 存储核心点索引
        self.core_samples_ = None
    
    def _compute_distance(self, x1, x2):
        """
        计算两个样本之间的欧氏距离
        
        参数说明：
        ----------
        x1 : array-like
            第一个样本
        
        x2 : array-like
            第二个样本
        
        返回：
        ------
        distance : float
            欧氏距离
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        diff = x1 - x2
        return np.sqrt(np.sum(diff ** 2))
    
    def _get_neighbors(self, point_idx, X):
        """
        获取点point_idx在eps半径内的所有邻居
        
        参数说明：
        ----------
        point_idx : int
            点的索引
        
        X : array-like, shape = [n_samples, n_features]
            所有样本数据
        
        返回：
        ------
        neighbors : list
            邻居点的索引列表
        """
        neighbors = []
        point = X[point_idx]
        
        # 遍历所有其他点
        for i in range(len(X)):
            if i != point_idx:
                # 计算距离
                distance = self._compute_distance(point, X[i])
                # 如果距离小于eps，则是邻居
                if distance <= self.eps:
                    neighbors.append(i)
        
        return neighbors
    
    def _is_core_point(self, point_idx, X):
        """
        判断点point_idx是否是核心点
        
        核心点的定义：在eps半径内至少有min_samples个邻居
        
        参数说明：
        ----------
        point_idx : int
            点的索引
        
        X : array-like
            所有样本数据
        
        返回：
        ------
        is_core : bool
            是否是核心点
        """
        neighbors = self._get_neighbors(point_idx, X)
        # 如果邻居数 >= min_samples，则是核心点
        return len(neighbors) >= self.min_samples
    
    def fit(self, X):
        """
        训练DBSCAN聚类器
        
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
        print("DBSCAN聚类训练")
        print("=" * 60)
        print(f"数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  eps: {self.eps}")
        print(f"  min_samples: {self.min_samples}")
        
        # 初始化
        labels = np.full(n_samples, -1)  # -1表示未分类（可能是噪声）
        cluster_id = 0  # 当前簇的ID
        visited = set()  # 已访问的点
        core_samples = []  # 核心点索引列表
        
        print("\n开始聚类...")
        
        # 遍历每个点
        for point_idx in range(n_samples):
            # 如果已经访问过，跳过
            if point_idx in visited:
                continue
            
            # 标记为已访问
            visited.add(point_idx)
            
            # 获取邻居
            neighbors = self._get_neighbors(point_idx, X)
            
            # 检查是否是核心点
            if len(neighbors) >= self.min_samples:
                # 是核心点，创建一个新簇
                core_samples.append(point_idx)
                labels[point_idx] = cluster_id
                
                # 使用队列进行广度优先搜索，找到所有密度可达的点
                # 队列中存储需要处理的点
                queue = deque(neighbors)
                
                # 处理队列中的所有点
                while queue:
                    # 取出队列中的第一个点
                    neighbor_idx = queue.popleft()
                    
                    # 如果已经访问过，跳过
                    if neighbor_idx in visited:
                        continue
                    
                    # 标记为已访问
                    visited.add(neighbor_idx)
                    
                    # 获取该点的邻居
                    neighbor_neighbors = self._get_neighbors(neighbor_idx, X)
                    
                    # 如果该点也是核心点，将其邻居加入队列
                    # 这样可以找到所有密度可达的点
                    if len(neighbor_neighbors) >= self.min_samples:
                        core_samples.append(neighbor_idx)
                        # 将该点的邻居加入队列
                        queue.extend(neighbor_neighbors)
                    
                    # 将该点加入当前簇（如果还没有被分配到其他簇）
                    if labels[neighbor_idx] == -1:
                        labels[neighbor_idx] = cluster_id
                
                # 簇ID递增，准备创建下一个簇
                cluster_id += 1
                
                if cluster_id % 10 == 0 or cluster_id <= 3:
                    print(f"  创建簇 {cluster_id-1}, 当前已处理 {len(visited)}/{n_samples} 个点")
        
        # 保存结果
        self.labels_ = labels
        self.core_samples_ = np.array(core_samples)
        
        # 统计结果
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)  # 减去噪声簇
        n_noise = np.sum(labels == -1)
        
        print(f"\n聚类完成！")
        print(f"  发现的簇数: {n_clusters}")
        print(f"  核心点数: {len(core_samples)}")
        print(f"  噪声点数: {n_noise}")
        print(f"  各簇的样本数:")
        for cluster_id in range(n_clusters):
            count = np.sum(labels == cluster_id)
            print(f"    簇 {cluster_id}: {count} 个样本")
        
        return self


# ========== 示例：使用DBSCAN聚类 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("DBSCAN聚类示例")
    print("=" * 60)
    
    # ========== 1. 生成示例数据 ==========
    print("\n1. 生成示例数据")
    print("-" * 60)
    
    # 生成包含噪声的聚类数据
    X, y_true = make_blobs(
        n_samples=200,
        centers=3,
        n_features=2,
        cluster_std=0.6,
        random_state=42
    )
    
    # 添加一些噪声点
    noise = np.random.uniform(-6, 6, (20, 2))
    X = np.vstack([X, noise])
    y_true = np.hstack([y_true, [-1] * 20])  # -1表示噪声
    
    print(f"数据形状: {X.shape}")
    print(f"真实簇数: {len(np.unique(y_true[y_true != -1]))}")
    print(f"噪声点数: {np.sum(y_true == -1)}")
    
    # ========== 2. 训练模型 ==========
    print("\n2. 训练DBSCAN模型")
    print("-" * 60)
    
    # 创建DBSCAN聚类器
    # eps=0.5: 邻域半径
    # min_samples=5: 成为核心点所需的最少邻居数
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    
    # 训练模型
    dbscan.fit(X)
    
    # ========== 3. 可视化结果 ==========
    print("\n3. 可视化结果")
    print("-" * 60)
    
    # 获取预测标签
    labels = dbscan.labels_
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：真实标签
    scatter1 = axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('真实标签（包含噪声）', fontsize=14)
    axes[0].set_xlabel('特征1', fontsize=12)
    axes[0].set_ylabel('特征2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：DBSCAN聚类结果
    scatter2 = axes[1].scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
    # 标记核心点
    if len(dbscan.core_samples_) > 0:
        axes[1].scatter(X[dbscan.core_samples_, 0], X[dbscan.core_samples_, 1],
                       c='red', marker='x', s=100, linewidths=2, label='核心点')
    axes[1].set_title('DBSCAN聚类结果', fontsize=14)
    axes[1].set_xlabel('特征1', fontsize=12)
    axes[1].set_ylabel('特征2', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('dbscan_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 4. 分析噪声点 ==========
    print("\n4. 分析噪声点")
    print("-" * 60)
    
    noise_points = X[labels == -1]
    print(f"识别出的噪声点数: {len(noise_points)}")
    print(f"噪声点位置（前5个）:")
    for i, point in enumerate(noise_points[:5]):
        print(f"  噪声点 {i+1}: {point}")
    
    # ========== 5. 测试不同参数 ==========
    print("\n5. 测试不同参数")
    print("-" * 60)
    
    # 测试不同的eps值
    eps_values = [0.3, 0.5, 0.7, 1.0]
    print("\n测试不同的eps值（min_samples=5）:")
    for eps in eps_values:
        dbscan_eps = DBSCAN(eps=eps, min_samples=5)
        dbscan_eps.fit(X)
        n_clusters = len(set(dbscan_eps.labels_)) - (1 if -1 in dbscan_eps.labels_ else 0)
        n_noise = np.sum(dbscan_eps.labels_ == -1)
        print(f"  eps={eps}: 簇数={n_clusters}, 噪声点数={n_noise}")
    
    # 测试不同的min_samples值
    min_samples_values = [3, 5, 7, 10]
    print("\n测试不同的min_samples值（eps=0.5）:")
    for min_samples in min_samples_values:
        dbscan_min = DBSCAN(eps=0.5, min_samples=min_samples)
        dbscan_min.fit(X)
        n_clusters = len(set(dbscan_min.labels_)) - (1 if -1 in dbscan_min.labels_ else 0)
        n_noise = np.sum(dbscan_min.labels_ == -1)
        print(f"  min_samples={min_samples}: 簇数={n_clusters}, 噪声点数={n_noise}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
总结：
1. DBSCAN基于密度进行聚类
2. 可以自动发现簇的数量
3. 可以自动识别噪声点（离群点）
4. 参数eps和min_samples的选择很重要
5. 适合处理任意形状的簇和包含噪声的数据
""")

