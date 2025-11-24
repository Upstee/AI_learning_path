"""
简单项目2：非线性分类
使用RBF核SVM进行非线性分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_circles, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

# 生成非线性数据（同心圆）
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)

print("=" * 50)
print("非线性SVM分类（RBF核）")
print("=" * 50)

# 数据标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 尝试不同的参数
C_values = [0.1, 1, 10, 100]
gamma_values = [0.1, 1, 10, 100]

print("\n不同参数组合的效果:")
best_acc = 0
best_params = None

for C in C_values:
    for gamma in gamma_values:
        rbf_svm = svm.SVC(kernel='rbf', C=C, gamma=gamma, random_state=42)
        rbf_svm.fit(X_train, y_train)
        y_pred = rbf_svm.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        if acc > best_acc:
            best_acc = acc
            best_params = (C, gamma)
        print(f"  C={C}, gamma={gamma}: 准确率={acc:.4f}")

print(f"\n最佳参数: C={best_params[0]}, gamma={best_params[1]}")
print(f"最佳准确率: {best_acc:.4f}")

# 使用最佳参数训练
best_svm = svm.SVC(kernel='rbf', C=best_params[0], gamma=best_params[1], random_state=42)
best_svm.fit(X_train, y_train)

# 可视化
plt.figure(figsize=(15, 5))

# 原始数据
plt.subplot(1, 3, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('原始数据（非线性）')
plt.grid(True, alpha=0.3)

# 决策边界
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = best_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.subplot(1, 3, 2)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.scatter(best_svm.support_vectors_[:, 0], best_svm.support_vectors_[:, 1],
           s=200, facecolors='none', edgecolors='red', linewidths=2, label='支持向量')
plt.xlabel('特征1（标准化）')
plt.ylabel('特征2（标准化）')
plt.title(f'决策边界 (C={best_params[0]}, γ={best_params[1]})')
plt.legend()
plt.grid(True, alpha=0.3)

# 支持向量
plt.subplot(1, 3, 3)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c='lightgray', alpha=0.5)
plt.scatter(best_svm.support_vectors_[:, 0], best_svm.support_vectors_[:, 1],
           s=200, c=y[best_svm.support_], cmap=plt.cm.RdYlBu, edgecolors='red', linewidths=2)
plt.xlabel('特征1（标准化）')
plt.ylabel('特征2（标准化）')
plt.title(f'支持向量 (共{len(best_svm.support_vectors_)}个)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('nonlinear_svm_classification.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

