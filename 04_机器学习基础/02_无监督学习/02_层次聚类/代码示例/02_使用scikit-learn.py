"""
使用scikit-learn实现层次聚类
本示例展示如何使用scikit-learn库快速实现层次聚类
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs, make_circles
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform
import seaborn as sns

# ========== 1. 基础层次聚类 ==========
print("=" * 60)
print("1. 基础层次聚类")
print("=" * 60)
print("""
层次聚类：
- 通过构建层次树进行聚类
- 不需要预先指定K值
- 可以可视化树状图
""")

# 生成示例数据
X, y_true = make_blobs(
    n_samples=50,  # 使用较少的样本，因为层次聚类计算慢
    centers=3,
    n_features=2,
    random_state=42
)

print(f"\n数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  真实簇数: {len(np.unique(y_true))}")

# 创建层次聚类器
# n_clusters: 要得到的簇数
# linkage: 链接方法
#   - 'ward': Ward链接（最小化簇内方差）
#   - 'complete': 完全链接（最远距离）
#   - 'average': 平均链接（平均距离）
#   - 'single': 单链接（最近距离）
agglomerative = AgglomerativeClustering(
    n_clusters=3,
    linkage='average'  # 平均链接
)

# 训练模型
print("\n训练模型...")
labels = agglomerative.fit_predict(X)

print(f"\n聚类结果:")
print(f"  簇数: {len(np.unique(labels))}")
print(f"  簇标签: {np.unique(labels)}")

# 可视化
plt.figure(figsize=(12, 5))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：层次聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.title('层次聚类结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 绘制树状图 ==========
print("\n" + "=" * 60)
print("2. 绘制树状图")
print("=" * 60)
print("""
树状图：
- 可视化层次聚类过程
- 可以在不同高度切割，得到不同数量的簇
- 高度表示合并时的距离
""")

# 使用scipy的linkage函数计算链接矩阵
# linkage函数返回一个链接矩阵，记录每次合并的信息
Z = linkage(X, method='average', metric='euclidean')

print(f"\n链接矩阵形状: {Z.shape}")
print(f"链接矩阵前5行:")
print(Z[:5])

# 绘制树状图
plt.figure(figsize=(12, 8))
dendrogram(
    Z,
    leaf_rotation=90,      # 叶子节点旋转角度
    leaf_font_size=12,     # 叶子节点字体大小
    truncate_mode='level', # 截断模式
    p=5                    # 只显示最后5层
)
plt.title('层次聚类树状图', fontsize=14)
plt.xlabel('样本索引或（簇大小）', fontsize=12)
plt.ylabel('距离', fontsize=12)
plt.tight_layout()
plt.savefig('dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 不同链接方法对比 ==========
print("\n" + "=" * 60)
print("3. 不同链接方法对比")
print("=" * 60)

linkage_methods = ['ward', 'complete', 'average', 'single']
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for idx, method in enumerate(linkage_methods):
    # 训练模型
    if method == 'ward':
        # Ward链接只能用于欧氏距离
        clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
    else:
        clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
    
    labels_method = clustering.fit_predict(X)
    
    # 可视化
    ax = axes[idx // 2, idx % 2]
    scatter = ax.scatter(X[:, 0], X[:, 1], c=labels_method, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(f'{method}链接', fontsize=14)
    ax.set_xlabel('特征1', fontsize=12)
    ax.set_ylabel('特征2', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_linkage_methods.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n不同链接方法的特点:")
print("  - ward: 最小化簇内方差，适合球形簇")
print("  - complete: 最远距离，产生紧凑的簇")
print("  - average: 平均距离，平衡单链接和完全链接")
print("  - single: 最近距离，容易产生链式效应")

# ========== 4. 从树状图选择K值 ==========
print("\n" + "=" * 60)
print("4. 从树状图选择K值")
print("=" * 60)
print("""
选择K值的方法：
- 观察树状图，找到距离变化大的地方
- 在距离变化大的地方切割，得到合理的簇数
- 可以使用不同的距离阈值
""")

# 计算不同距离阈值下的簇数
distance_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
n_clusters_list = []

for threshold in distance_thresholds:
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=threshold,
        linkage='average'
    )
    labels_thresh = clustering.fit_predict(X)
    n_clusters = len(np.unique(labels_thresh))
    n_clusters_list.append(n_clusters)
    print(f"  距离阈值={threshold:.1f}: 簇数={n_clusters}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(distance_thresholds, n_clusters_list, 'o-', linewidth=2, markersize=8)
plt.xlabel('距离阈值', fontsize=12)
plt.ylabel('簇数', fontsize=12)
plt.title('距离阈值对簇数的影响', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hierarchical_threshold.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 处理非球形簇 ==========
print("\n" + "=" * 60)
print("5. 层次聚类处理非球形簇")
print("=" * 60)
print("""
层次聚类的优势：
- 可以处理非球形簇
- 不需要假设簇的形状
- 适合复杂的数据结构
""")

# 生成非球形数据（环形数据）
X_circles, y_circles = make_circles(n_samples=100, noise=0.1, factor=0.3, random_state=42)

# 使用层次聚类
clustering_circles = AgglomerativeClustering(n_clusters=2, linkage='average')
labels_circles = clustering_circles.fit_predict(X_circles)

# 可视化
plt.figure(figsize=(12, 5))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=y_circles, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签（环形数据）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：层次聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X_circles[:, 0], X_circles[:, 1], c=labels_circles, cmap='viridis', alpha=0.6, s=50)
plt.title('层次聚类结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hierarchical_non_spherical.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n注意：层次聚类可以处理非球形簇，但K-means不能很好地处理")

# ========== 6. 性能对比 ==========
print("\n" + "=" * 60)
print("6. 层次聚类 vs K-means")
print("=" * 60)

from sklearn.cluster import KMeans
import time

# 测试不同数据规模下的性能
sample_sizes = [50, 100, 200, 500]
hierarchical_times = []
kmeans_times = []

print("\n性能对比:")
for n_samples in sample_sizes:
    X_test, _ = make_blobs(n_samples=n_samples, centers=3, n_features=2, random_state=42)
    
    # 层次聚类
    start = time.time()
    hc = AgglomerativeClustering(n_clusters=3, linkage='average')
    hc.fit(X_test)
    hierarchical_times.append(time.time() - start)
    
    # K-means
    start = time.time()
    km = KMeans(n_clusters=3, random_state=42)
    km.fit(X_test)
    kmeans_times.append(time.time() - start)
    
    print(f"  样本数={n_samples}: 层次聚类={hierarchical_times[-1]:.4f}秒, "
          f"K-means={kmeans_times[-1]:.4f}秒")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(sample_sizes, hierarchical_times, 'o-', label='层次聚类', linewidth=2, markersize=8)
plt.plot(sample_sizes, kmeans_times, 's-', label='K-means', linewidth=2, markersize=8)
plt.xlabel('样本数', fontsize=12)
plt.ylabel('训练时间（秒）', fontsize=12)
plt.title('层次聚类 vs K-means 性能对比', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('hierarchical_vs_kmeans.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n结论：层次聚类计算复杂度高，适合小规模数据；K-means适合大规模数据")

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. 层次聚类通过构建层次树进行聚类
2. 不需要预先指定K值，可以从树状图选择
3. 不同的链接方法会产生不同的结果
4. 可以处理非球形簇，但计算复杂度高
5. 适合小规模数据，大规模数据用K-means
""")

