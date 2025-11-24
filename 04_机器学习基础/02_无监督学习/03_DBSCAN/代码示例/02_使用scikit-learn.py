"""
使用scikit-learn实现DBSCAN
本示例展示如何使用scikit-learn库快速实现DBSCAN聚类
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import seaborn as sns

# ========== 1. 基础DBSCAN聚类 ==========
print("=" * 60)
print("1. 基础DBSCAN聚类")
print("=" * 60)
print("""
DBSCAN聚类：
- 基于密度进行聚类
- 自动发现簇的数量
- 可以识别噪声点（离群点）
""")

# 生成包含噪声的聚类数据
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    cluster_std=0.6,
    random_state=42
)

# 添加噪声点
noise = np.random.uniform(-10, 10, (30, 2))
X = np.vstack([X, noise])
y_true = np.hstack([y_true, [-1] * 30])

print(f"\n数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  真实簇数: {len(np.unique(y_true[y_true != -1]))}")
print(f"  噪声点数: {np.sum(y_true == -1)}")

# 标准化数据（DBSCAN对特征尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建DBSCAN聚类器
# eps: 邻域半径
# min_samples: 成为核心点所需的最少邻居数
# metric: 距离度量（默认'euclidean'）
dbscan = DBSCAN(
    eps=0.3,
    min_samples=5,
    metric='euclidean'
)

# 训练模型
print("\n训练模型...")
labels = dbscan.fit_predict(X_scaled)

# 统计结果
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = np.sum(labels == -1)
core_samples = dbscan.core_sample_indices_

print(f"\n聚类结果:")
print(f"  发现的簇数: {n_clusters}")
print(f"  核心点数: {len(core_samples)}")
print(f"  噪声点数: {n_noise}")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签（包含噪声）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：DBSCAN聚类结果
plt.subplot(1, 2, 2)
scatter = plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
# 标记核心点
if len(core_samples) > 0:
    plt.scatter(X[core_samples, 0], X[core_samples, 1],
               c='red', marker='x', s=100, linewidths=2, label='核心点')
plt.title('DBSCAN聚类结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. K距离图选择eps ==========
print("\n" + "=" * 60)
print("2. K距离图选择eps")
print("=" * 60)
print("""
K距离图：
- 绘制每个点到第K近邻的距离
- 找到距离突然增大的点（肘部点）
- 这个距离可以作为eps的参考值
""")

# 计算每个点到第min_samples近邻的距离
k = 5  # 使用min_samples作为K值
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# 获取每个点到第K近邻的距离（第k-1个，因为索引从0开始）
k_distances = distances[:, k-1]
k_distances_sorted = np.sort(k_distances)[::-1]  # 从大到小排序

# 绘制K距离图
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances_sorted)), k_distances_sorted, linewidth=2)
plt.xlabel('样本索引（按距离排序）', fontsize=12)
plt.ylabel(f'到第{k}近邻的距离', fontsize=12)
plt.title('K距离图（用于选择eps）', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axhline(y=k_distances_sorted[int(len(k_distances_sorted) * 0.1)], 
           color='r', linestyle='--', label='建议的eps（10%分位数）')
plt.legend()
plt.tight_layout()
plt.savefig('dbscan_k_distance.png', dpi=300, bbox_inches='tight')
plt.show()

# 建议的eps值（使用10%分位数）
suggested_eps = np.percentile(k_distances, 10)
print(f"\n建议的eps值（10%分位数）: {suggested_eps:.4f}")

# ========== 3. 处理非球形簇 ==========
print("\n" + "=" * 60)
print("3. DBSCAN处理非球形簇")
print("=" * 60)
print("""
DBSCAN的优势：
- 可以处理任意形状的簇
- 不需要假设簇是球形的
- 适合处理复杂的数据结构
""")

# 生成月牙形数据
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)

# 标准化
scaler_moons = StandardScaler()
X_moons_scaled = scaler_moons.fit_transform(X_moons)

# 使用DBSCAN聚类
dbscan_moons = DBSCAN(eps=0.3, min_samples=5)
labels_moons = dbscan_moons.fit_predict(X_moons_scaled)

# 可视化
plt.figure(figsize=(14, 6))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签（月牙形）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：DBSCAN结果
plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', alpha=0.6, s=50)
if len(dbscan_moons.core_sample_indices_) > 0:
    plt.scatter(X_moons[dbscan_moons.core_sample_indices_, 0],
               X_moons[dbscan_moons.core_sample_indices_, 1],
               c='red', marker='x', s=100, linewidths=2, label='核心点')
plt.title('DBSCAN聚类结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_non_spherical.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nDBSCAN结果: 簇数={len(set(labels_moons)) - (1 if -1 in labels_moons else 0)}, "
      f"噪声点数={np.sum(labels_moons == -1)}")

# ========== 4. 不同参数对比 ==========
print("\n" + "=" * 60)
print("4. 不同参数对比")
print("=" * 60)

# 测试不同的eps值
eps_values = [0.2, 0.3, 0.4, 0.5]
results = []

print("\n测试不同的eps值（min_samples=5）:")
for eps in eps_values:
    dbscan_eps = DBSCAN(eps=eps, min_samples=5)
    labels_eps = dbscan_eps.fit_predict(X_scaled)
    n_clusters = len(set(labels_eps)) - (1 if -1 in labels_eps else 0)
    n_noise = np.sum(labels_eps == -1)
    results.append({'eps': eps, 'clusters': n_clusters, 'noise': n_noise})
    print(f"  eps={eps}: 簇数={n_clusters}, 噪声点数={n_noise}")

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
for idx, eps in enumerate(eps_values):
    dbscan_eps = DBSCAN(eps=eps, min_samples=5)
    labels_eps = dbscan_eps.fit_predict(X_scaled)
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_eps, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(f'eps={eps}', fontsize=14)
    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_eps_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 异常检测应用 ==========
print("\n" + "=" * 60)
print("5. DBSCAN用于异常检测")
print("=" * 60)
print("""
DBSCAN的噪声点就是异常点：
- 密度低的点被识别为噪声
- 这些点通常是异常值
- 可以用于异常检测任务
""")

# 使用DBSCAN识别异常点
dbscan_anomaly = DBSCAN(eps=0.3, min_samples=5)
labels_anomaly = dbscan_anomaly.fit_predict(X_scaled)

# 识别异常点（噪声点）
anomalies = X[labels_anomaly == -1]

print(f"\n异常检测结果:")
print(f"  识别出的异常点数: {len(anomalies)}")
print(f"  异常点比例: {len(anomalies)/len(X)*100:.2f}%")

# 可视化异常点
plt.figure(figsize=(12, 6))

# 左图：正常点和异常点
plt.subplot(1, 2, 1)
normal_points = X[labels_anomaly != -1]
anomaly_points = X[labels_anomaly == -1]
plt.scatter(normal_points[:, 0], normal_points[:, 1], 
           c='blue', alpha=0.6, s=50, label='正常点')
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
           c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='异常点')
plt.title('异常检测结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：异常点分布
plt.subplot(1, 2, 2)
plt.hist([np.sum(labels_anomaly != -1), np.sum(labels_anomaly == -1)], 
         bins=2, color=['blue', 'red'], alpha=0.7, edgecolor='black')
plt.xlabel('类别', fontsize=12)
plt.ylabel('数量', fontsize=12)
plt.title('正常点 vs 异常点', fontsize=14)
plt.xticks([0.25, 0.75], ['正常点', '异常点'])
plt.grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('dbscan_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. DBSCAN基于密度进行聚类
2. 可以自动发现簇的数量
3. 可以自动识别噪声点（异常点）
4. 使用K距离图可以帮助选择eps参数
5. 适合处理任意形状的簇和包含噪声的数据
6. 可以用于异常检测任务
""")

