"""
从零实现K-means聚类算法
本示例详细展示了如何从零开始实现K-means算法
适合小白学习，包含大量注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import random


class KMeans:
    """
    K-means聚类算法
    
    这个类实现了K-means算法的核心功能
    通过迭代优化将数据分成K个簇
    """
    
    def __init__(self, n_clusters=3, max_iters=100, random_state=42):
        """
        初始化K-means聚类器
        
        参数说明：
        ----------
        n_clusters : int, 默认=3
            K值，即要分成几个簇
            为什么选择3？
            - 通常根据数据特点或业务需求选择
            - 可以使用肘部法则或轮廓系数选择
        
        max_iters : int, 默认=100
            最大迭代次数
            为什么需要限制迭代次数？
            - 防止算法无限循环
            - 如果算法不收敛，可以提前停止
        
        random_state : int, 默认=42
            随机种子，确保结果可复现
        """
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        
        # 存储聚类中心
        # 形状: (n_clusters, n_features)
        self.centroids = None
        
        # 存储每个样本所属的簇
        # 形状: (n_samples,)
        self.labels_ = None
        
        # 存储每次迭代的损失（WCSS）
        self.inertia_history = []
    
    def _initialize_centroids(self, X):
        """
        初始化聚类中心
        
        方法：随机选择K个样本作为初始中心
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练数据
        
        返回：
        ------
        centroids : array, shape = [n_clusters, n_features]
            初始聚类中心
        """
        # 设置随机种子，确保结果可复现
        np.random.seed(self.random_state)
        
        # 获取样本数和特征数
        n_samples, n_features = X.shape
        
        # 随机选择K个样本的索引
        # np.random.choice() 从0到n_samples-1中随机选择n_clusters个不重复的数
        indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        
        # 选择对应的样本作为初始中心
        centroids = X[indices].copy()
        
        print(f"初始化聚类中心:")
        print(f"  选择的样本索引: {indices}")
        print(f"  初始中心形状: {centroids.shape}")
        
        return centroids
    
    def _compute_distance(self, X, centroids):
        """
        计算每个样本到每个聚类中心的距离
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            样本数据
        
        centroids : array-like, shape = [n_clusters, n_features]
            聚类中心
        
        返回：
        ------
        distances : array, shape = [n_samples, n_clusters]
            每个样本到每个中心的距离
        """
        # 转换为numpy数组
        X = np.array(X)
        centroids = np.array(centroids)
        
        # 初始化距离矩阵
        n_samples = X.shape[0]
        n_clusters = centroids.shape[0]
        distances = np.zeros((n_samples, n_clusters))
        
        # 计算每个样本到每个中心的欧氏距离
        # 对于每个样本
        for i in range(n_samples):
            # 对于每个聚类中心
            for j in range(n_clusters):
                # 计算欧氏距离
                # ||x - c|| = sqrt(sum((x_i - c_i)^2))
                diff = X[i] - centroids[j]
                distances[i, j] = np.sqrt(np.sum(diff ** 2))
        
        return distances
    
    def _assign_clusters(self, X, centroids):
        """
        将每个样本分配到最近的聚类中心
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            样本数据
        
        centroids : array-like, shape = [n_clusters, n_features]
            聚类中心
        
        返回：
        ------
        labels : array, shape = [n_samples]
            每个样本所属的簇（0到K-1）
        """
        # 计算距离
        distances = self._compute_distance(X, centroids)
        
        # 找到每个样本最近的聚类中心
        # np.argmin() 返回最小值的索引
        # axis=1 表示在每一行（每个样本）中找最小值
        labels = np.argmin(distances, axis=1)
        
        return labels
    
    def _update_centroids(self, X, labels):
        """
        更新聚类中心
        
        方法：计算每个簇内样本的均值作为新的中心
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            样本数据
        
        labels : array-like, shape = [n_samples]
            每个样本所属的簇
        
        返回：
        ------
        new_centroids : array, shape = [n_clusters, n_features]
            新的聚类中心
        """
        # 转换为numpy数组
        X = np.array(X)
        labels = np.array(labels)
        
        # 初始化新的聚类中心
        n_features = X.shape[1]
        new_centroids = np.zeros((self.n_clusters, n_features))
        
        # 对于每个簇
        for k in range(self.n_clusters):
            # 找到属于簇k的所有样本
            # labels == k 返回一个布尔数组，True表示该样本属于簇k
            cluster_samples = X[labels == k]
            
            # 如果簇k中有样本
            if len(cluster_samples) > 0:
                # 计算簇内样本的均值作为新的中心
                # np.mean(axis=0) 对每一列（每个特征）求均值
                new_centroids[k] = np.mean(cluster_samples, axis=0)
            else:
                # 如果簇k为空（没有样本），保持原中心不变
                # 这种情况很少见，但需要处理
                new_centroids[k] = self.centroids[k]
        
        return new_centroids
    
    def _compute_inertia(self, X, labels, centroids):
        """
        计算簇内平方和（WCSS）
        
        公式：J = sum(sum(||x - μ_k||^2))
        其中x属于簇k，μ_k是簇k的中心
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            样本数据
        
        labels : array-like, shape = [n_samples]
            每个样本所属的簇
        
        centroids : array-like, shape = [n_clusters, n_features]
            聚类中心
        
        返回：
        ------
        inertia : float
            簇内平方和
        """
        X = np.array(X)
        labels = np.array(labels)
        centroids = np.array(centroids)
        
        inertia = 0.0
        
        # 对于每个样本
        for i in range(len(X)):
            # 获取样本所属的簇
            k = labels[i]
            # 计算样本到其所属簇中心的距离的平方
            diff = X[i] - centroids[k]
            inertia += np.sum(diff ** 2)
        
        return inertia
    
    def fit(self, X):
        """
        训练K-means聚类器
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练数据
        
        返回：
        ------
        self : object
            返回自身，支持链式调用
        """
        # 转换为numpy数组
        X = np.array(X)
        n_samples, n_features = X.shape
        
        print("=" * 60)
        print("K-means聚类训练")
        print("=" * 60)
        print(f"数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {n_features}")
        print(f"  簇数: {self.n_clusters}")
        print(f"  最大迭代次数: {self.max_iters}")
        
        # 1. 初始化聚类中心
        print("\n步骤1: 初始化聚类中心")
        self.centroids = self._initialize_centroids(X)
        
        # 2. 迭代优化
        print("\n步骤2: 开始迭代优化")
        for iteration in range(self.max_iters):
            # 2.1 分配样本到最近的簇
            labels = self._assign_clusters(X, self.centroids)
            
            # 2.2 更新聚类中心
            new_centroids = self._update_centroids(X, labels)
            
            # 2.3 计算损失
            inertia = self._compute_inertia(X, labels, new_centroids)
            self.inertia_history.append(inertia)
            
            # 2.4 检查是否收敛
            # 如果聚类中心变化很小，认为已收敛
            if np.allclose(self.centroids, new_centroids, atol=1e-4):
                print(f"  迭代 {iteration + 1}: 已收敛！")
                break
            
            # 2.5 更新聚类中心
            self.centroids = new_centroids
            
            # 打印进度（每10次迭代打印一次）
            if (iteration + 1) % 10 == 0 or iteration == 0:
                print(f"  迭代 {iteration + 1}: 损失 = {inertia:.4f}")
        
        # 3. 最终分配
        self.labels_ = self._assign_clusters(X, self.centroids)
        final_inertia = self._compute_inertia(X, self.labels_, self.centroids)
        
        print(f"\n训练完成！")
        print(f"  总迭代次数: {iteration + 1}")
        print(f"  最终损失: {final_inertia:.4f}")
        
        return self
    
    def predict(self, X):
        """
        预测新样本所属的簇
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            新样本
        
        返回：
        ------
        labels : array, shape = [n_samples]
            每个样本所属的簇
        """
        # 将新样本分配到最近的聚类中心
        labels = self._assign_clusters(X, self.centroids)
        return labels


# ========== 示例：使用K-means聚类 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("K-means聚类示例")
    print("=" * 60)
    
    # ========== 1. 生成示例数据 ==========
    print("\n1. 生成示例数据")
    print("-" * 60)
    
    # 使用make_blobs生成聚类数据
    # n_samples: 样本数
    # centers: 簇的中心位置
    # n_features: 特征数（2维，方便可视化）
    # random_state: 随机种子
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=2,
        random_state=42
    )
    
    print(f"数据形状: {X.shape}")
    print(f"真实簇数: {len(np.unique(y_true))}")
    
    # ========== 2. 训练模型 ==========
    print("\n2. 训练K-means模型")
    print("-" * 60)
    
    # 创建K-means聚类器
    kmeans = KMeans(n_clusters=3, max_iters=100, random_state=42)
    
    # 训练模型
    kmeans.fit(X)
    
    # ========== 3. 可视化结果 ==========
    print("\n3. 可视化结果")
    print("-" * 60)
    
    # 获取预测标签
    y_pred = kmeans.labels_
    
    # 创建图形
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：真实标签
    axes[0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('真实标签', fontsize=14)
    axes[0].set_xlabel('特征1', fontsize=12)
    axes[0].set_ylabel('特征2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：K-means预测标签
    scatter = axes[1].scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6, s=50)
    # 绘制聚类中心
    axes[1].scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
                   c='red', marker='x', s=200, linewidths=3, label='聚类中心')
    axes[1].set_title('K-means聚类结果', fontsize=14)
    axes[1].set_xlabel('特征1', fontsize=12)
    axes[1].set_ylabel('特征2', fontsize=12)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('kmeans_clustering.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 4. 可视化损失变化 ==========
    print("\n4. 可视化损失变化")
    print("-" * 60)
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(kmeans.inertia_history) + 1), kmeans.inertia_history, 'o-', linewidth=2)
    plt.xlabel('迭代次数', fontsize=12)
    plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
    plt.title('K-means损失变化', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('kmeans_inertia.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 5. 评估聚类质量 ==========
    print("\n5. 评估聚类质量")
    print("-" * 60)
    
    # 计算每个簇的样本数
    unique, counts = np.unique(y_pred, return_counts=True)
    print("\n每个簇的样本数:")
    for k, count in zip(unique, counts):
        print(f"  簇 {k}: {count} 个样本")
    
    # 计算簇内平方和
    print(f"\n簇内平方和 (WCSS): {kmeans.inertia_history[-1]:.4f}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
总结：
1. K-means通过迭代优化将数据分成K个簇
2. 每次迭代包括：分配样本和更新中心
3. 算法会收敛，但可能收敛到局部最优
4. 初始化对结果有重要影响
""")

