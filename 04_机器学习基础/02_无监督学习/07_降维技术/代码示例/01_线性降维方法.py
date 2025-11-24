"""
线性降维方法
本示例展示如何使用线性降维方法（PCA、LDA、ICA）
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA, FastICA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.datasets import make_classification, load_iris
from sklearn.preprocessing import StandardScaler

# ========== 1. PCA（主成分分析） ==========
print("=" * 60)
print("1. PCA（主成分分析）")
print("=" * 60)
print("""
PCA原理：
- 找到数据方差最大的方向（主成分）
- 将数据投影到这些方向上
- 保留最多的信息
""")

# 生成数据
X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                          n_redundant=2, n_classes=3, random_state=42)

print(f"原始数据:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"\nPCA降维后:")
print(f"  降维后特征数: {X_pca.shape[1]}")
print(f"  解释方差比: {pca.explained_variance_ratio_}")
print(f"  累计解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：原始数据（前两个特征）
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.title('原始数据（前两个特征）', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

# 右图：PCA降维后
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.title('PCA降维后', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. LDA（线性判别分析） ==========
print("\n" + "=" * 60)
print("2. LDA（线性判别分析）")
print("=" * 60)
print("""
LDA原理：
- 找到使类间距离最大、类内距离最小的方向
- 专门为分类任务设计
- 使用标签信息，效果更好
""")

# 使用LDA降维到2维
# 注意：LDA需要标签信息，最多降到（类别数-1）维
lda = LDA(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

print(f"\nLDA降维后:")
print(f"  降维后特征数: {X_lda.shape[1]}")
print(f"  解释方差比: {lda.explained_variance_ratio_}")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：PCA降维后
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.title('PCA降维后', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

# 右图：LDA降维后
plt.subplot(1, 2, 2)
plt.scatter(X_lda[:, 0], X_lda[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('第一判别方向', fontsize=12)
plt.ylabel('第二判别方向', fontsize=12)
plt.title('LDA降维后', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lda_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. ICA（独立成分分析） ==========
print("\n" + "=" * 60)
print("3. ICA（独立成分分析）")
print("=" * 60)
print("""
ICA原理：
- 找到统计独立的成分
- 可以分离混合信号
- 常用于信号处理
""")

# 使用ICA降维到2维
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_scaled)

print(f"\nICA降维后:")
print(f"  降维后特征数: {X_ica.shape[1]}")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：PCA降维后
plt.subplot(1, 2, 1)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.title('PCA降维后', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

# 右图：ICA降维后
plt.subplot(1, 2, 2)
plt.scatter(X_ica[:, 0], X_ica[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.xlabel('第一独立成分', fontsize=12)
plt.ylabel('第二独立成分', fontsize=12)
plt.title('ICA降维后', fontsize=14)
plt.colorbar(label='类别')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ica_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 对比三种方法 ==========
print("\n" + "=" * 60)
print("4. 对比三种方法")
print("=" * 60)

# 使用鸢尾花数据集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

# 标准化
X_iris_scaled = scaler.fit_transform(X_iris)

# 三种方法降维
pca_iris = PCA(n_components=2)
X_iris_pca = pca_iris.fit_transform(X_iris_scaled)

lda_iris = LDA(n_components=2)
X_iris_lda = lda_iris.fit_transform(X_iris_scaled, y_iris)

ica_iris = FastICA(n_components=2, random_state=42)
X_iris_ica = ica_iris.fit_transform(X_iris_scaled)

# 可视化对比
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# PCA
axes[0].scatter(X_iris_pca[:, 0], X_iris_pca[:, 1], c=y_iris, cmap='viridis', alpha=0.7, s=60)
axes[0].set_xlabel('第一主成分', fontsize=12)
axes[0].set_ylabel('第二主成分', fontsize=12)
axes[0].set_title(f'PCA (解释方差比: {np.sum(pca_iris.explained_variance_ratio_):.3f})', fontsize=14)
axes[0].grid(True, alpha=0.3)

# LDA
axes[1].scatter(X_iris_lda[:, 0], X_iris_lda[:, 1], c=y_iris, cmap='viridis', alpha=0.7, s=60)
axes[1].set_xlabel('第一判别方向', fontsize=12)
axes[1].set_ylabel('第二判别方向', fontsize=12)
axes[1].set_title(f'LDA (解释方差比: {np.sum(lda_iris.explained_variance_ratio_):.3f})', fontsize=14)
axes[1].grid(True, alpha=0.3)

# ICA
axes[2].scatter(X_iris_ica[:, 0], X_iris_ica[:, 1], c=y_iris, cmap='viridis', alpha=0.7, s=60)
axes[2].set_xlabel('第一独立成分', fontsize=12)
axes[2].set_ylabel('第二独立成分', fontsize=12)
axes[2].set_title('ICA', fontsize=14)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_dim_reduction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. PCA：无监督，找到方差最大的方向
2. LDA：有监督，找到使类间距离最大、类内距离最小的方向
3. ICA：找到统计独立的成分
4. 不同方法适用于不同的任务
5. 需要根据数据特点选择合适的方法
""")

