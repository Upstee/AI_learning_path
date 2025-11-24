"""
机器学习方法异常检测
本示例展示如何使用机器学习方法进行异常检测
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_blobs, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ========== 1. Isolation Forest ==========
print("=" * 60)
print("1. Isolation Forest")
print("=" * 60)
print("""
Isolation Forest：
- 异常数据更容易被隔离（需要更少的切分）
- 使用随机树隔离异常
- 计算效率高
""")

# 生成数据
np.random.seed(42)
X_normal, _ = make_blobs(n_samples=200, centers=1, n_features=2, random_state=42)
X_anomaly = np.random.uniform(-10, 10, (20, 2))
X = np.vstack([X_normal, X_anomaly])
y_true = np.hstack([np.zeros(len(X_normal)), np.ones(len(X_anomaly))])

print(f"数据信息:")
print(f"  正常样本数: {len(X_normal)}")
print(f"  异常样本数: {len(X_anomaly)}")
print(f"  总样本数: {len(X)}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 使用Isolation Forest
iso_forest = IsolationForest(contamination=0.1, random_state=42)
y_pred_iso = iso_forest.fit_predict(X_scaled)
# Isolation Forest返回-1表示异常，1表示正常
y_pred_iso = (y_pred_iso == -1).astype(int)

print(f"\nIsolation Forest结果:")
print(f"  识别出的异常数: {np.sum(y_pred_iso)}")

# ========== 2. One-Class SVM ==========
print("\n" + "=" * 60)
print("2. One-Class SVM")
print("=" * 60)
print("""
One-Class SVM：
- 训练只包含正常数据的SVM
- 学习正常数据的边界
- 边界外的数据视为异常
""")

# 使用One-Class SVM
# 只使用正常数据训练
oc_svm = OneClassSVM(nu=0.1, kernel='rbf', gamma='scale')
oc_svm.fit(X_scaled[:len(X_normal)])  # 只用正常数据训练
y_pred_svm = oc_svm.predict(X_scaled)
# One-Class SVM返回-1表示异常，1表示正常
y_pred_svm = (y_pred_svm == -1).astype(int)

print(f"\nOne-Class SVM结果:")
print(f"  识别出的异常数: {np.sum(y_pred_svm)}")

# ========== 3. Local Outlier Factor (LOF) ==========
print("\n" + "=" * 60)
print("3. Local Outlier Factor (LOF)")
print("=" * 60)
print("""
LOF：
- 计算样本的局部密度
- 密度低的样本异常
- 考虑局部性，更准确
""")

# 使用LOF
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
y_pred_lof = lof.fit_predict(X_scaled)
# LOF返回-1表示异常，1表示正常
y_pred_lof = (y_pred_lof == -1).astype(int)

print(f"\nLOF结果:")
print(f"  识别出的异常数: {np.sum(y_pred_lof)}")

# ========== 4. 可视化对比 ==========
print("\n" + "=" * 60)
print("4. 可视化对比")
print("=" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 左上：真实标签
axes[0, 0].scatter(X[:, 0], X[:, 1], c=y_true, cmap='RdYlGn', alpha=0.6, s=50)
axes[0, 0].set_title('真实标签', fontsize=14)
axes[0, 0].set_xlabel('特征1', fontsize=12)
axes[0, 0].set_ylabel('特征2', fontsize=12)
axes[0, 0].grid(True, alpha=0.3)

# 右上：Isolation Forest
axes[0, 1].scatter(X[:, 0], X[:, 1], c=y_pred_iso, cmap='RdYlGn', alpha=0.6, s=50)
axes[0, 1].set_title('Isolation Forest', fontsize=14)
axes[0, 1].set_xlabel('特征1', fontsize=12)
axes[0, 1].set_ylabel('特征2', fontsize=12)
axes[0, 1].grid(True, alpha=0.3)

# 左下：One-Class SVM
axes[1, 0].scatter(X[:, 0], X[:, 1], c=y_pred_svm, cmap='RdYlGn', alpha=0.6, s=50)
axes[1, 0].set_title('One-Class SVM', fontsize=14)
axes[1, 0].set_xlabel('特征1', fontsize=12)
axes[1, 0].set_ylabel('特征2', fontsize=12)
axes[1, 0].grid(True, alpha=0.3)

# 右下：LOF
axes[1, 1].scatter(X[:, 0], X[:, 1], c=y_pred_lof, cmap='RdYlGn', alpha=0.6, s=50)
axes[1, 1].set_title('LOF', fontsize=14)
axes[1, 1].set_xlabel('特征1', fontsize=12)
axes[1, 1].set_ylabel('特征2', fontsize=12)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('ml_anomaly_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 性能评估 ==========
print("\n" + "=" * 60)
print("5. 性能评估")
print("=" * 60)

methods = ['Isolation Forest', 'One-Class SVM', 'LOF']
predictions = [y_pred_iso, y_pred_svm, y_pred_lof]

for method, y_pred in zip(methods, predictions):
    print(f"\n{method}:")
    print(classification_report(y_true, y_pred, target_names=['正常', '异常']))
    cm = confusion_matrix(y_true, y_pred)
    print(f"混淆矩阵:")
    print(cm)

# ========== 6. 处理非线性数据 ==========
print("\n" + "=" * 60)
print("6. 处理非线性数据")
print("=" * 60)
print("""
非线性数据：
- 正常数据可能不是线性分布的
- 需要使用非线性方法
""")

# 生成非线性数据（月牙形）
X_moons, y_moons = make_moons(n_samples=200, noise=0.1, random_state=42)
X_moons_anomaly = np.random.uniform(-2, 3, (20, 2))
X_nonlinear = np.vstack([X_moons, X_moons_anomaly])
y_nonlinear = np.hstack([np.zeros(len(X_moons)), np.ones(len(X_moons_anomaly))])

# 标准化
X_nonlinear_scaled = scaler.fit_transform(X_nonlinear)

# 使用Isolation Forest
iso_forest_nonlinear = IsolationForest(contamination=0.1, random_state=42)
y_pred_nonlinear = iso_forest_nonlinear.fit_predict(X_nonlinear_scaled)
y_pred_nonlinear = (y_pred_nonlinear == -1).astype(int)

# 可视化
plt.figure(figsize=(12, 6))

# 左图：真实标签
plt.subplot(1, 2, 1)
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_nonlinear, cmap='RdYlGn', alpha=0.6, s=50)
plt.title('真实标签（非线性数据）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：Isolation Forest结果
plt.subplot(1, 2, 2)
plt.scatter(X_nonlinear[:, 0], X_nonlinear[:, 1], c=y_pred_nonlinear, cmap='RdYlGn', alpha=0.6, s=50)
plt.title('Isolation Forest结果', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nonlinear_anomaly.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. Isolation Forest：计算效率高，适合大规模数据
2. One-Class SVM：可以处理非线性数据，需要核函数
3. LOF：考虑局部性，更准确
4. 不同方法适用于不同的数据分布
5. 需要根据数据特点选择合适的方法
""")


