"""
从零实现PCA主成分分析算法
本示例详细展示了如何从零开始实现PCA算法
适合小白学习，包含大量注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, make_blobs
from sklearn.preprocessing import StandardScaler


class PCA:
    """
    PCA主成分分析算法
    
    这个类实现了PCA算法的核心功能
    通过找到数据中方差最大的方向来降低数据的维度
    """
    
    def __init__(self, n_components=None):
        """
        初始化PCA
        
        参数说明：
        ----------
        n_components : int or None, 默认=None
            要保留的主成分数量
            - None: 保留所有主成分
            - int: 保留前n_components个主成分
            为什么需要这个参数？
            - 降维：通过减少主成分数量来降低数据维度
            - 信息保留：保留主要信息，去除次要信息
        """
        self.n_components = n_components
        
        # 存储主成分（特征向量）
        self.components_ = None
        
        # 存储特征值（方差）
        self.explained_variance_ = None
        
        # 存储累计方差贡献率
        self.explained_variance_ratio_ = None
        
        # 存储均值（用于中心化）
        self.mean_ = None
    
    def fit(self, X):
        """
        训练PCA模型
        
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
        n_samples, n_features = X.shape
        
        print("=" * 60)
        print("PCA训练")
        print("=" * 60)
        print(f"数据信息:")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {n_features}")
        
        # 1. 数据中心化
        print("\n步骤1: 数据中心化")
        self.mean_ = np.mean(X, axis=0)  # 计算每个特征的均值
        X_centered = X - self.mean_  # 中心化：减去均值
        print(f"均值: {self.mean_}")
        print(f"中心化后的数据形状: {X_centered.shape}")
        
        # 2. 计算协方差矩阵
        print("\n步骤2: 计算协方差矩阵")
        # 协方差矩阵 C = (1/(n-1)) * X^T * X
        # 其中X是中心化后的数据
        cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)
        print(f"协方差矩阵形状: {cov_matrix.shape}")
        print(f"协方差矩阵（前3x3）:")
        print(cov_matrix[:3, :3])
        
        # 3. 特征值分解
        print("\n步骤3: 特征值分解")
        # 计算协方差矩阵的特征值和特征向量
        # eigenvalues: 特征值（方差）
        # eigenvectors: 特征向量（主成分）
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
        
        # 特征值可能是复数（由于数值误差），取实部
        eigenvalues = np.real(eigenvalues)
        eigenvectors = np.real(eigenvectors)
        
        print(f"特征值数量: {len(eigenvalues)}")
        print(f"前5个特征值: {eigenvalues[:5]}")
        
        # 4. 按特征值从大到小排序
        print("\n步骤4: 排序主成分")
        # 获取排序索引（从大到小）
        idx = np.argsort(eigenvalues)[::-1]
        
        # 排序特征值和特征向量
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        print(f"排序后的前5个特征值: {eigenvalues[:5]}")
        
        # 5. 选择主成分
        print("\n步骤5: 选择主成分")
        if self.n_components is None:
            # 保留所有主成分
            self.n_components = n_features
        elif self.n_components > n_features:
            # 不能超过特征数
            self.n_components = n_features
        
        # 选择前n_components个主成分
        self.components_ = eigenvectors[:, :self.n_components]
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        # 计算累计方差贡献率
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        
        print(f"保留的主成分数: {self.n_components}")
        print(f"前5个主成分的方差贡献率: {self.explained_variance_ratio_[:5]}")
        print(f"累计方差贡献率: {np.sum(self.explained_variance_ratio_):.4f}")
        
        return self
    
    def transform(self, X):
        """
        将数据投影到主成分空间
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            要转换的数据
        
        返回：
        ------
        X_transformed : array, shape = [n_samples, n_components]
            转换后的数据
        """
        X = np.array(X)
        
        # 中心化
        X_centered = X - self.mean_
        
        # 投影到主成分空间
        # Y = X * V
        # 其中V是主成分矩阵
        X_transformed = np.dot(X_centered, self.components_)
        
        return X_transformed
    
    def fit_transform(self, X):
        """
        训练模型并转换数据
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练数据
        
        返回：
        ------
        X_transformed : array, shape = [n_samples, n_components]
            转换后的数据
        """
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_transformed):
        """
        将降维后的数据还原到原始空间（近似）
        
        参数说明：
        ----------
        X_transformed : array-like, shape = [n_samples, n_components]
            降维后的数据
        
        返回：
        ------
        X_reconstructed : array, shape = [n_samples, n_features]
            还原后的数据（近似）
        """
        X_transformed = np.array(X_transformed)
        
        # 还原到原始空间
        # X = Y * V^T + mean
        # 其中Y是降维后的数据，V是主成分矩阵
        X_reconstructed = np.dot(X_transformed, self.components_.T) + self.mean_
        
        return X_reconstructed


# ========== 示例：使用PCA降维 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("PCA降维示例")
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
    
    # 标准化数据（PCA对特征尺度敏感）
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    print("\n数据标准化完成！")
    
    # ========== 2. 训练PCA模型 ==========
    print("\n2. 训练PCA模型")
    print("-" * 60)
    
    # 创建PCA模型，降维到2维
    pca = PCA(n_components=2)
    
    # 训练模型并转换数据
    X_pca = pca.fit_transform(X_scaled)
    
    print(f"\n降维后的数据形状: {X_pca.shape}")
    print(f"累计方差贡献率: {np.sum(pca.explained_variance_ratio_):.4f}")
    
    # ========== 3. 可视化结果 ==========
    print("\n3. 可视化结果")
    print("-" * 60)
    
    # 可视化原始数据（前两个特征）
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # 左图：原始数据（前两个特征）
    scatter1 = axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
    axes[0].set_title('原始数据（前两个特征）', fontsize=14)
    axes[0].set_xlabel('特征1', fontsize=12)
    axes[0].set_ylabel('特征2', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    # 右图：PCA降维后的数据
    scatter2 = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
    axes[1].set_title('PCA降维后的数据（2维）', fontsize=14)
    axes[1].set_xlabel('第一主成分', fontsize=12)
    axes[1].set_ylabel('第二主成分', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_2d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 4. 分析主成分 ==========
    print("\n4. 分析主成分")
    print("-" * 60)
    
    print(f"\n前2个主成分的方差贡献率:")
    for i in range(2):
        print(f"  主成分 {i+1}: {pca.explained_variance_ratio_[i]:.4f} "
              f"({pca.explained_variance_ratio_[i]*100:.2f}%)")
    
    print(f"\n前2个主成分的系数（权重）:")
    for i in range(2):
        print(f"  主成分 {i+1}: {pca.components_[i]}")
    
    # ========== 5. 累计方差贡献率 ==========
    print("\n5. 累计方差贡献率")
    print("-" * 60)
    
    # 计算不同主成分数的累计方差贡献率
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    print(f"\n不同主成分数的累计方差贡献率:")
    for i in range(min(4, len(cumulative_variance))):
        print(f"  前{i+1}个主成分: {cumulative_variance[i]:.4f} "
              f"({cumulative_variance[i]*100:.2f}%)")
    
    # 可视化累计方差贡献率
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2, markersize=8)
    plt.axhline(y=0.85, color='r', linestyle='--', label='85%阈值')
    plt.axhline(y=0.90, color='g', linestyle='--', label='90%阈值')
    plt.xlabel('主成分数', fontsize=12)
    plt.ylabel('累计方差贡献率', fontsize=12)
    plt.title('累计方差贡献率', fontsize=14)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('pca_cumulative_variance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # ========== 6. 数据还原 ==========
    print("\n6. 数据还原")
    print("-" * 60)
    
    # 将降维后的数据还原到原始空间
    X_reconstructed = pca.inverse_transform(X_pca)
    
    print(f"原始数据形状: {X_scaled.shape}")
    print(f"还原后的数据形状: {X_reconstructed.shape}")
    
    # 计算还原误差
    reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
    print(f"还原误差（MSE）: {reconstruction_error:.6f}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
总结：
1. PCA通过找到数据中方差最大的方向来降维
2. 主成分是协方差矩阵的特征向量
3. 特征值表示该主成分的方差
4. 累计方差贡献率表示保留的信息量
5. 降维会丢失一些信息，但保留主要信息
""")

