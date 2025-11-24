"""
从零实现t-SNE算法（简化版）
本示例详细展示了如何从零开始实现t-SNE算法
适合小白学习，包含大量注释和解释

注意：这是简化版实现，完整实现更复杂
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


class TSNE:
    """
    t-SNE降维算法（简化版）
    
    这个类实现了t-SNE算法的核心功能
    通过匹配高维和低维空间的概率分布来实现非线性降维
    """
    
    def __init__(self, n_components=2, perplexity=30, learning_rate=200, n_iter=1000):
        """
        初始化t-SNE
        
        参数说明：
        ----------
        n_components : int, 默认=2
            降维后的维度（通常是2或3，用于可视化）
            为什么通常是2或3？
            - 2维或3维可以可视化
            - 更高维度计算更慢，效果不一定更好
        
        perplexity : float, 默认=30
            困惑度，控制每个点的有效邻居数
            为什么需要这个参数？
            - 控制邻域大小
            - 通常选择5-50之间
            - 较大的perplexity保留更多全局结构
        
        learning_rate : float, 默认=200
            学习率，控制优化的步长
            为什么需要这个参数？
            - 控制梯度下降的步长
            - 通常选择100-1000
            - 较大的学习率可能导致不稳定
        
        n_iter : int, 默认=1000
            迭代次数
            为什么需要多次迭代？
            - t-SNE需要迭代优化
            - 通常需要1000次以上
            - 更多迭代通常效果更好，但计算更慢
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        
        # 存储结果
        self.embedding_ = None
    
    def _compute_pairwise_distances(self, X):
        """
        计算所有点对之间的欧氏距离
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            数据
        
        返回：
        ------
        distances : array, shape = [n_samples, n_samples]
            距离矩阵
        """
        # 计算所有点对之间的欧氏距离
        distances = squareform(pdist(X, metric='euclidean'))
        return distances
    
    def _binary_search_perplexity(self, distances, target_perplexity, tol=1e-5):
        """
        使用二分搜索找到合适的带宽参数sigma
        
        参数说明：
        ----------
        distances : array
            距离矩阵
        
        target_perplexity : float
            目标困惑度
        
        tol : float
            容差
        
        返回：
        ------
        sigmas : array
            每个点的带宽参数
        """
        n_samples = distances.shape[0]
        sigmas = np.ones(n_samples)
        
        # 对每个点，使用二分搜索找到合适的sigma
        for i in range(n_samples):
            # 排除自身
            dist_i = distances[i, np.concatenate((np.arange(i), np.arange(i+1, n_samples)))]
            
            # 二分搜索
            sigma_min = 0
            sigma_max = np.inf
            sigma = 1.0
            
            for _ in range(50):  # 最多50次迭代
                # 计算条件概率
                p_i = np.exp(-dist_i ** 2 / (2 * sigma ** 2))
                p_i = p_i / np.sum(p_i)
                
                # 计算困惑度
                entropy = -np.sum(p_i * np.log2(p_i + 1e-10))
                perplexity_i = 2 ** entropy
                
                # 调整sigma
                if perplexity_i < target_perplexity:
                    sigma_min = sigma
                    if sigma_max == np.inf:
                        sigma *= 2
                    else:
                        sigma = (sigma_min + sigma_max) / 2
                else:
                    sigma_max = sigma
                    if sigma_min == 0:
                        sigma /= 2
                    else:
                        sigma = (sigma_min + sigma_max) / 2
                
                # 检查是否收敛
                if abs(perplexity_i - target_perplexity) < tol:
                    break
            
            sigmas[i] = sigma
        
        return sigmas
    
    def _compute_high_dimensional_probabilities(self, X):
        """
        计算高维空间中的概率分布
        
        参数说明：
        ----------
        X : array-like
            高维数据
        
        返回：
        ------
        P : array
            高维空间中的概率分布
        """
        n_samples = X.shape[0]
        
        # 计算距离矩阵
        distances = self._compute_pairwise_distances(X)
        
        # 找到合适的带宽参数
        sigmas = self._binary_search_perplexity(distances, self.perplexity)
        
        # 计算条件概率 p_{j|i}
        P = np.zeros((n_samples, n_samples))
        
        for i in range(n_samples):
            # 计算条件概率
            p_i = np.exp(-distances[i, :] ** 2 / (2 * sigmas[i] ** 2))
            p_i[i] = 0  # 排除自身
            p_i = p_i / np.sum(p_i)
            P[i, :] = p_i
        
        # 对称化
        P = (P + P.T) / (2 * n_samples)
        
        # 防止数值问题
        P = np.maximum(P, 1e-12)
        
        return P
    
    def _compute_low_dimensional_probabilities(self, Y):
        """
        计算低维空间中的概率分布
        
        参数说明：
        ----------
        Y : array-like
            低维数据
        
        返回：
        ------
        Q : array
            低维空间中的概率分布
        """
        n_samples = Y.shape[0]
        
        # 计算距离矩阵
        distances = self._compute_pairwise_distances(Y)
        
        # 使用t分布计算概率
        # q_{ij} = (1 + ||y_i - y_j||^2)^{-1} / sum
        Q = (1 + distances ** 2) ** (-1)
        np.fill_diagonal(Q, 0)  # 排除自身
        Q = Q / np.sum(Q)
        
        # 防止数值问题
        Q = np.maximum(Q, 1e-12)
        
        return Q
    
    def _compute_gradient(self, P, Q, Y):
        """
        计算梯度
        
        参数说明：
        ----------
        P : array
            高维空间中的概率分布
        
        Q : array
            低维空间中的概率分布
        
        Y : array
            低维数据
        
        返回：
        ------
        gradient : array
            梯度
        """
        n_samples = Y.shape[0]
        gradient = np.zeros_like(Y)
        
        # 计算梯度
        for i in range(n_samples):
            # 计算点i的梯度
            grad_i = np.zeros(self.n_components)
            
            for j in range(n_samples):
                if i != j:
                    # 计算梯度项
                    diff = Y[i] - Y[j]
                    dist_sq = np.sum(diff ** 2)
                    
                    # 梯度公式
                    grad_term = (P[i, j] - Q[i, j]) * diff * (1 + dist_sq) ** (-1)
                    grad_i += grad_term
            
            gradient[i] = 4 * grad_i
        
        return gradient
    
    def fit_transform(self, X):
        """
        训练模型并转换数据
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            高维数据
        
        返回：
        ------
        Y : array, shape = [n_samples, n_components]
            低维数据
        """
        X = np.array(X)
        n_samples = X.shape[0]
        
        print("=" * 60)
        print("t-SNE训练")
        print("=" * 60)
        print(f"数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {X.shape[1]}")
        print(f"  降维到: {self.n_components}维")
        print(f"  困惑度: {self.perplexity}")
        print(f"  学习率: {self.learning_rate}")
        print(f"  迭代次数: {self.n_iter}")
        
        # 1. 计算高维空间中的概率分布
        print("\n步骤1: 计算高维空间中的概率分布")
        P = self._compute_high_dimensional_probabilities(X)
        print(f"概率分布矩阵形状: {P.shape}")
        
        # 2. 随机初始化低维空间
        print("\n步骤2: 随机初始化低维空间")
        np.random.seed(42)
        Y = np.random.randn(n_samples, self.n_components) * 1e-4
        print(f"低维数据形状: {Y.shape}")
        
        # 3. 迭代优化
        print("\n步骤3: 开始迭代优化")
        momentum = 0.5  # 动量参数
        Y_prev = Y.copy()
        
        for iteration in range(self.n_iter):
            # 计算低维空间中的概率分布
            Q = self._compute_low_dimensional_probabilities(Y)
            
            # 计算梯度
            gradient = self._compute_gradient(P, Q, Y)
            
            # 更新位置（使用动量）
            Y_new = Y - self.learning_rate * gradient + momentum * (Y - Y_prev)
            Y_prev = Y.copy()
            Y = Y_new
            
            # 打印进度（每100次迭代打印一次）
            if (iteration + 1) % 100 == 0 or iteration < 5:
                # 计算KL散度（损失）
                kl_div = np.sum(P * np.log(P / Q + 1e-10))
                print(f"  迭代 {iteration + 1}/{self.n_iter}: KL散度 = {kl_div:.4f}")
        
        # 保存结果
        self.embedding_ = Y
        
        print(f"\n训练完成！")
        print(f"  最终低维数据形状: {Y.shape}")
        
        return Y


# ========== 示例：使用t-SNE降维 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("t-SNE降维示例")
    print("=" * 60)
    
    # ========== 1. 加载数据 ==========
    print("\n1. 加载数据")
    print("-" * 60)
    
    # 加载鸢尾花数据集
    iris = load_iris()
    X = iris.data
    y = iris.target
    
    print(f"数据形状: {X.shape}")
    print(f"特征数: {X.shape[1]}")
    print(f"类别数: {len(np.unique(y))}")
    
    # 标准化数据
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n数据标准化完成！")
    
    # ========== 2. 训练t-SNE模型 ==========
    print("\n2. 训练t-SNE模型")
    print("-" * 60)
    
    # 创建t-SNE模型
    # 注意：这是简化版实现，完整实现更复杂
    # 对于实际应用，建议使用sklearn的TSNE
    tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, n_iter=1000)
    
    # 训练模型并转换数据
    X_tsne = tsne.fit_transform(X_scaled)
    
    # ========== 3. 可视化结果 ==========
    print("\n3. 可视化结果")
    print("-" * 60)
    
    # 可视化降维后的数据
    plt.figure(figsize=(12, 6))
    
    # 左图：原始数据（前两个特征）
    plt.subplot(1, 2, 1)
    plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
    plt.title('原始数据（前两个特征）', fontsize=14)
    plt.xlabel('特征1', fontsize=12)
    plt.ylabel('特征2', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # 右图：t-SNE降维后的数据
    plt.subplot(1, 2, 2)
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
    plt.title('t-SNE降维后的数据（2维）', fontsize=14)
    plt.xlabel('第一维度', fontsize=12)
    plt.ylabel('第二维度', fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tsne_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
注意：
1. 这是t-SNE的简化版实现
2. 完整实现更复杂，包括更多优化技巧
3. 对于实际应用，建议使用sklearn的TSNE
4. t-SNE主要用于数据可视化
5. 结果有随机性，每次运行可能不同
""")

