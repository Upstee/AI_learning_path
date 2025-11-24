"""
简单项目2：非球形簇聚类
使用DBSCAN对非球形簇进行聚类

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN, KMeans
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler

# ========== 1. 准备非球形数据 ==========
print("=" * 60)
print("1. 准备非球形数据")
print("=" * 60)

# 生成月牙形数据
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

print(f"数据信息:")
print(f"  样本数: {X_moons.shape[0]}")
print(f"  特征数: {X_moons.shape[1]}")
print(f"  真实簇数: {len(np.unique(y_moons))}")

# 标准化数据
scaler = StandardScaler()
X_moons_scaled = scaler.fit_transform(X_moons)

# ========== 2. DBSCAN聚类 ==========
print("\n" + "=" * 60)
print("2. DBSCAN聚类")
print("=" * 60)

# 使用DBSCAN聚类
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels_dbscan = dbscan.fit_predict(X_moons_scaled)

n_clusters_dbscan = len(set(labels_dbscan)) - (1 if -1 in labels_dbscan else 0)
n_noise_dbscan = np.sum(labels_dbscan == -1)

print(f"\nDBSCAN结果:")
print(f"  簇数: {n_clusters_dbscan}")
print(f"  噪声点数: {n_noise_dbscan}")

# ========== 3. K-means聚类（对比） ==========
print("\n" + "=" * 60)
print("3. K-means聚类（对比）")
print("=" * 60)

# 使用K-means聚类（假设知道K=2）
kmeans = KMeans(n_clusters=2, random_state=42)
labels_kmeans = kmeans.fit_predict(X_moons_scaled)

print(f"\nK-means结果:")
print(f"  簇数: {len(np.unique(labels_kmeans))}")

# ========== 4. 可视化对比 ==========
print("\n" + "=" * 60)
print("4. 可视化对比")
print("=" * 60)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 左图：真实标签
axes[0].scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('真实标签（月牙形）', fontsize=14)
axes[0].set_xlabel('特征1', fontsize=12)
axes[0].set_ylabel('特征2', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 中图：DBSCAN结果
scatter = axes[1].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_dbscan, cmap='viridis', alpha=0.6, s=50)
if len(dbscan.core_sample_indices_) > 0:
    axes[1].scatter(X_moons[dbscan.core_sample_indices_, 0],
                   X_moons[dbscan.core_sample_indices_, 1],
                   c='red', marker='x', s=100, linewidths=2, label='核心点')
axes[1].set_title('DBSCAN聚类结果', fontsize=14)
axes[1].set_xlabel('特征1', fontsize=12)
axes[1].set_ylabel('特征2', fontsize=12)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 右图：K-means结果
axes[2].scatter(X_moons[:, 0], X_moons[:, 1], c=labels_kmeans, cmap='viridis', alpha=0.6, s=50)
axes[2].scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
               c='red', marker='x', s=200, linewidths=3, label='聚类中心')
axes[2].set_title('K-means聚类结果（效果不好）', fontsize=14)
axes[2].set_xlabel('特征1', fontsize=12)
axes[2].set_ylabel('特征2', fontsize=12)
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_vs_kmeans.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n结论：DBSCAN可以处理非球形簇，而K-means不能很好地处理")

# ========== 5. 测试环形数据 ==========
print("\n" + "=" * 60)
print("5. 测试环形数据")
print("=" * 60)

# 生成环形数据
X_circles, y_circles = make_circles(n_samples=200, noise=0.1, factor=0.3, random_state=42)

# 标准化
X_circles_scaled = scaler.fit_transform(X_circles)

# DBSCAN聚类
dbscan_circles = DBSCAN(eps=0.3, min_samples=5)
labels_circles = dbscan_circles.fit_predict(X_circles_scaled)

# K-means聚类
kmeans_circles = KMeans(n_clusters=2, random_state=42)
labels_kmeans_circles = kmeans_circles.fit_predict(X_circles_scaled)

# 可视化
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('真实标签（环形）', fontsize=14)
axes[0].set_xlabel('特征1', fontsize=12)
axes[0].set_ylabel('特征2', fontsize=12)
axes[0].grid(True, alpha=0.3)

axes[1].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_circles, cmap='viridis', alpha=0.6, s=50)
axes[1].set_title('DBSCAN聚类结果', fontsize=14)
axes[1].set_xlabel('特征1', fontsize=12)
axes[1].set_ylabel('特征2', fontsize=12)
axes[1].grid(True, alpha=0.3)

axes[2].scatter(X_circles[:, 0], X_circles[:, 1], c=labels_kmeans_circles, cmap='viridis', alpha=0.6, s=50)
axes[2].scatter(kmeans_circles.cluster_centers_[:, 0], kmeans_circles.cluster_centers_[:, 1],
               c='red', marker='x', s=200, linewidths=3)
axes[2].set_title('K-means聚类结果（效果不好）', fontsize=14)
axes[2].set_xlabel('特征1', fontsize=12)
axes[2].set_ylabel('特征2', fontsize=12)
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_circles.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. DBSCAN可以处理非球形簇（月牙形、环形等）
2. K-means假设簇是球形的，对非球形簇效果不好
3. DBSCAN基于密度，可以发现任意形状的簇
4. 这是DBSCAN相比K-means的主要优势
""")

