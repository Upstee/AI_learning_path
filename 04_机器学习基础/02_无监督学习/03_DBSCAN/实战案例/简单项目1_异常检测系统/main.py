"""
简单项目1：异常检测系统
使用DBSCAN进行异常检测

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs
from sklearn.neighbors import NearestNeighbors

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成正常数据
X_normal, _ = make_blobs(
    n_samples=200,
    centers=2,
    n_features=2,
    cluster_std=0.5,
    random_state=42
)

# 生成异常数据（远离正常数据的点）
np.random.seed(42)
X_anomaly = np.random.uniform(-8, 8, (20, 2))

# 合并数据
X = np.vstack([X_normal, X_anomaly])
y_true = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])  # 1表示异常

print(f"数据信息:")
print(f"  总样本数: {len(X)}")
print(f"  正常样本: {len(X_normal)}")
print(f"  异常样本: {len(X_anomaly)}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 2. 选择参数 ==========
print("\n" + "=" * 60)
print("2. 选择参数")
print("=" * 60)

# 使用K距离图选择eps
k = 5  # 使用5近邻
neighbors = NearestNeighbors(n_neighbors=k)
neighbors.fit(X_scaled)
distances, indices = neighbors.kneighbors(X_scaled)

# 获取每个点到第K近邻的距离
k_distances = distances[:, k-1]
k_distances_sorted = np.sort(k_distances)[::-1]

# 绘制K距离图
plt.figure(figsize=(10, 6))
plt.plot(range(len(k_distances_sorted)), k_distances_sorted, linewidth=2)
plt.xlabel('样本索引（按距离排序）', fontsize=12)
plt.ylabel(f'到第{k}近邻的距离', fontsize=12)
plt.title('K距离图（用于选择eps）', fontsize=14)
plt.grid(True, alpha=0.3)

# 建议的eps值（使用10%分位数）
suggested_eps = np.percentile(k_distances, 10)
plt.axhline(y=suggested_eps, color='r', linestyle='--', 
           label=f'建议的eps={suggested_eps:.4f}')
plt.legend()
plt.tight_layout()
plt.savefig('anomaly_detection_k_distance.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"建议的eps值: {suggested_eps:.4f}")

# ========== 3. 异常检测 ==========
print("\n" + "=" * 60)
print("3. 异常检测")
print("=" * 60)

# 使用DBSCAN进行异常检测
# 噪声点（标签为-1）就是异常点
dbscan = DBSCAN(eps=suggested_eps, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

# 识别异常点
anomalies = X[labels == -1]
normal = X[labels != -1]

print(f"\n异常检测结果:")
print(f"  识别出的异常点数: {len(anomalies)}")
print(f"  正常点数: {len(normal)}")
print(f"  异常点比例: {len(anomalies)/len(X)*100:.2f}%")

# ========== 4. 评估结果 ==========
print("\n" + "=" * 60)
print("4. 评估结果")
print("=" * 60)

# 计算准确率
# 将DBSCAN的标签转换为异常/正常标签
y_pred = (labels == -1).astype(int)

# 计算混淆矩阵
from sklearn.metrics import confusion_matrix, classification_report

cm = confusion_matrix(y_true, y_pred)
print(f"\n混淆矩阵:")
print(cm)

print(f"\n分类报告:")
print(classification_report(y_true, y_pred, target_names=['正常', '异常']))

# ========== 5. 可视化结果 ==========
print("\n" + "=" * 60)
print("5. 可视化结果")
print("=" * 60)

plt.figure(figsize=(14, 6))

# 左图：真实标签
plt.subplot(1, 2, 1)
normal_points = X[y_true == 0]
anomaly_points = X[y_true == 1]
plt.scatter(normal_points[:, 0], normal_points[:, 1], 
           c='blue', alpha=0.6, s=50, label='正常')
plt.scatter(anomaly_points[:, 0], anomaly_points[:, 1], 
           c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='异常（真实）')
plt.title('真实标签', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：DBSCAN检测结果
plt.subplot(1, 2, 2)
plt.scatter(normal[:, 0], normal[:, 1], 
           c='blue', alpha=0.6, s=50, label='正常')
plt.scatter(anomalies[:, 0], anomalies[:, 1], 
           c='red', alpha=0.8, s=100, marker='x', linewidths=2, label='异常（检测）')
plt.title('DBSCAN异常检测结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('anomaly_detection_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 分析异常点特征 ==========
print("\n" + "=" * 60)
print("6. 分析异常点特征")
print("=" * 60)

if len(anomalies) > 0:
    print(f"\n异常点特征统计:")
    print(f"  异常点数量: {len(anomalies)}")
    print(f"  特征1范围: [{anomalies[:, 0].min():.2f}, {anomalies[:, 0].max():.2f}]")
    print(f"  特征2范围: [{anomalies[:, 1].min():.2f}, {anomalies[:, 1].max():.2f}]")
    print(f"  特征1均值: {anomalies[:, 0].mean():.2f}")
    print(f"  特征2均值: {anomalies[:, 1].mean():.2f}")
    
    print(f"\n正常点特征统计:")
    print(f"  正常点数量: {len(normal)}")
    print(f"  特征1范围: [{normal[:, 0].min():.2f}, {normal[:, 0].max():.2f}]")
    print(f"  特征2范围: [{normal[:, 1].min():.2f}, {normal[:, 1].max():.2f}]")
    print(f"  特征1均值: {normal[:, 0].mean():.2f}")
    print(f"  特征2均值: {normal[:, 1].mean():.2f}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. DBSCAN可以用于异常检测
2. 噪声点（标签为-1）就是异常点
3. 使用K距离图可以帮助选择eps参数
4. 可以分析异常点的特征，理解异常的原因
""")

