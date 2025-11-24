"""
简单项目1：线性分类
使用线性SVM进行二分类
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

# 生成数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          random_state=42)

print("=" * 50)
print("线性SVM分类")
print("=" * 50)

# 数据标准化（SVM对尺度敏感）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# 创建线性SVM
linear_svm = svm.SVC(kernel='linear', C=1.0, random_state=42)

# 训练
print("\n训练线性SVM...")
linear_svm.fit(X_train, y_train)

# 预测
y_pred = linear_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n模型性能:")
print(f"准确率: {accuracy:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 支持向量
print(f"\n支持向量数量: {len(linear_svm.support_vectors_)}")
print(f"支持向量比例: {len(linear_svm.support_vectors_)/len(X_train)*100:.2f}%")

# 可视化
plt.figure(figsize=(12, 5))

# 原始数据
plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('原始数据')
plt.grid(True, alpha=0.3)

# 决策边界
plt.subplot(1, 2, 2)
h = 0.02
x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

Z = linear_svm.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.scatter(linear_svm.support_vectors_[:, 0], linear_svm.support_vectors_[:, 1],
           s=200, facecolors='none', edgecolors='red', linewidths=2, label='支持向量')
plt.xlabel('特征1（标准化）')
plt.ylabel('特征2（标准化）')
plt.title('决策边界和支持向量')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_svm_classification.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

