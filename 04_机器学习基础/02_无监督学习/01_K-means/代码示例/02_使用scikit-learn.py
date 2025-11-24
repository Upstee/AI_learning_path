"""
使用scikit-learn实现K-means
本示例展示如何使用scikit-learn库快速实现K-means聚类
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ========== 1. 基础K-means聚类 ==========
print("=" * 60)
print("1. 基础K-means聚类")
print("=" * 60)
print("""
K-means聚类：
- 将数据分成K个簇
- 每个簇有一个中心（centroid）
- 通过迭代优化最小化簇内平方和
""")

# 生成示例数据
X, y_true = make_blobs(
    n_samples=300,
    centers=4,
    n_features=2,
    random_state=42
)

print(f"\n数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  真实簇数: {len(np.unique(y_true))}")

# 创建K-means聚类器
# n_clusters: K值，要分成几个簇
# init: 初始化方法
#   - 'k-means++': 改进的初始化方法（默认）
#   - 'random': 随机初始化
# max_iter: 最大迭代次数
# random_state: 随机种子
kmeans = KMeans(
    n_clusters=4,
    init='k-means++',
    max_iter=300,
    random_state=42
)

# 训练模型
print("\n训练模型...")
kmeans.fit(X)

# 获取聚类结果
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
inertia = kmeans.inertia_

print(f"\n聚类结果:")
print(f"  簇数: {len(np.unique(labels))}")
print(f"  簇内平方和: {inertia:.4f}")

# 可视化
plt.figure(figsize=(12, 5))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：K-means聚类结果
plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6, s=50)
plt.scatter(centroids[:, 0], centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='聚类中心')
plt.title('K-means聚类结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 肘部法则选择K值 ==========
print("\n" + "=" * 60)
print("2. 肘部法则选择K值")
print("=" * 60)
print("""
肘部法则：
- 绘制K值与簇内平方和的关系图
- 选择"肘部"点作为最优K值
- "肘部"点是WCSS减少变慢的转折点
""")

# 测试不同的K值
k_range = range(1, 11)
inertias = []

print("\n测试不同的K值:")
for k in k_range:
    kmeans_k = KMeans(n_clusters=k, random_state=42)
    kmeans_k.fit(X)
    inertias.append(kmeans_k.inertia_)
    if k <= 5 or k % 2 == 0:
        print(f"  K={k}: 簇内平方和={kmeans_k.inertia_:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
plt.xlabel('K值', fontsize=12)
plt.ylabel('簇内平方和 (WCSS)', fontsize=12)
plt.title('肘部法则 - 选择最优K值', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('kmeans_elbow.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 轮廓系数评估 ==========
print("\n" + "=" * 60)
print("3. 轮廓系数评估")
print("=" * 60)
print("""
轮廓系数：
- 评估聚类质量
- 范围：[-1, 1]
- 越接近1，聚类质量越好
""")

# 计算不同K值的轮廓系数
silhouette_scores = []

print("\n计算不同K值的轮廓系数:")
for k in k_range[1:]:  # K=1时无法计算轮廓系数
    kmeans_k = KMeans(n_clusters=k, random_state=42)
    labels_k = kmeans_k.fit_predict(X)
    score = silhouette_score(X, labels_k)
    silhouette_scores.append(score)
    if k <= 5 or k % 2 == 0:
        print(f"  K={k}: 轮廓系数={score:.4f}")

# 找到最佳K值
best_k_silhouette = k_range[1:][np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
print(f"\n最佳K值（轮廓系数）: {best_k_silhouette}, 轮廓系数: {best_score:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range[1:], silhouette_scores, 'o-', linewidth=2, markersize=8, color='green')
plt.xlabel('K值', fontsize=12)
plt.ylabel('轮廓系数', fontsize=12)
plt.title('轮廓系数 - 选择最优K值', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k_silhouette, color='r', linestyle='--', linewidth=2, label=f'最佳K={best_k_silhouette}')
plt.legend()
plt.tight_layout()
plt.savefig('kmeans_silhouette.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. Mini-batch K-means ==========
print("\n" + "=" * 60)
print("4. Mini-batch K-means")
print("=" * 60)
print("""
Mini-batch K-means：
- 每次只使用部分样本更新聚类中心
- 计算更快，适合大规模数据
- 结果可能不如标准K-means精确
""")

# 创建Mini-batch K-means
mb_kmeans = MiniBatchKMeans(
    n_clusters=4,
    batch_size=50,  # 每次使用的样本数
    random_state=42
)

# 训练
print("\n训练Mini-batch K-means...")
mb_kmeans.fit(X)

# 对比结果
mb_labels = mb_kmeans.labels_
mb_inertia = mb_kmeans.inertia_

print(f"\n对比结果:")
print(f"  标准K-means簇内平方和: {inertia:.4f}")
print(f"  Mini-batch簇内平方和: {mb_inertia:.4f}")

# ========== 5. 不同初始化方法对比 ==========
print("\n" + "=" * 60)
print("5. 不同初始化方法对比")
print("=" * 60)

# 测试不同的初始化方法
init_methods = ['k-means++', 'random']
results = []

for init_method in init_methods:
    kmeans_init = KMeans(n_clusters=4, init=init_method, n_init=10, random_state=42)
    kmeans_init.fit(X)
    results.append({
        'method': init_method,
        'inertia': kmeans_init.inertia_,
        'labels': kmeans_init.labels_
    })
    print(f"  {init_method}: 簇内平方和={kmeans_init.inertia_:.4f}")

# ========== 6. 处理非球形簇 ==========
print("\n" + "=" * 60)
print("6. K-means的局限性：非球形簇")
print("=" * 60)
print("""
K-means假设簇是球形的：
- 对于非球形簇，K-means效果不好
- 例如：月牙形簇、长条形簇
""")

# 生成月牙形数据
X_moons, y_moons = make_moons(n_samples=300, noise=0.1, random_state=42)

# 使用K-means聚类
kmeans_moons = KMeans(n_clusters=2, random_state=42)
labels_moons = kmeans_moons.fit_predict(X_moons)

# 可视化
plt.figure(figsize=(12, 5))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='viridis', alpha=0.6, s=50)
plt.title('真实标签（月牙形）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：K-means结果
plt.subplot(1, 2, 2)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=labels_moons, cmap='viridis', alpha=0.6, s=50)
plt.scatter(kmeans_moons.cluster_centers_[:, 0], kmeans_moons.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='聚类中心')
plt.title('K-means聚类结果（效果不好）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_limitation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n注意：K-means对非球形簇效果不好，需要使用其他算法（如DBSCAN）")

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. K-means是最常用的聚类算法
2. 使用肘部法则或轮廓系数选择K值
3. K-means++初始化通常效果更好
4. Mini-batch K-means适合大规模数据
5. K-means假设簇是球形的，对非球形簇效果不好
""")

