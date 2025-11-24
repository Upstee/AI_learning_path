"""
使用scikit-learn实现SVM
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ========== 1. 线性SVM ==========
print("=" * 50)
print("1. 线性SVM")
print("=" * 50)

# 生成线性可分数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建线性SVM
linear_svm = svm.SVC(kernel='linear', C=1.0, random_state=42)
linear_svm.fit(X_train, y_train)

# 预测
y_pred = linear_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"线性SVM准确率: {accuracy:.4f}")

# 支持向量
print(f"支持向量数量: {len(linear_svm.support_vectors_)}")
print(f"支持向量索引: {linear_svm.support_}")

# ========== 2. 非线性SVM（RBF核） ==========
print("\n" + "=" * 50)
print("2. 非线性SVM（RBF核）")
print("=" * 50)

# 生成非线性数据（同心圆）
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建RBF核SVM
rbf_svm = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
rbf_svm.fit(X_train, y_train)

# 预测
y_pred = rbf_svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"RBF核SVM准确率: {accuracy:.4f}")

# ========== 3. 不同核函数对比 ==========
print("\n" + "=" * 50)
print("3. 不同核函数对比")
print("=" * 50)

# 使用非线性数据
X, y = make_circles(n_samples=200, noise=0.1, factor=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

kernels = ['linear', 'poly', 'rbf', 'sigmoid']
print("不同核函数的效果:")
for kernel in kernels:
    clf = svm.SVC(kernel=kernel, C=1.0, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"  {kernel}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# ========== 4. 参数调优 ==========
print("\n" + "=" * 50)
print("4. 参数调优")
print("=" * 50)

# 使用网格搜索调优
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义参数网格
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.1, 1, 10],
    'kernel': ['rbf', 'poly']
}

# 网格搜索
grid_search = GridSearchCV(
    svm.SVC(random_state=42),
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 使用最佳参数预测
best_svm = grid_search.best_estimator_
y_pred = best_svm.predict(X_test)
test_acc = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {test_acc:.4f}")

# ========== 5. 软间隔SVM ==========
print("\n" + "=" * 50)
print("5. 软间隔SVM（不同C值）")
print("=" * 50)

# 生成有噪声的数据
X, y = make_classification(n_samples=200, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1,
                          class_sep=0.5, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

C_values = [0.1, 1, 10, 100]
print("不同C值的效果:")
for C in C_values:
    clf = svm.SVC(kernel='linear', C=C, random_state=42)
    clf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, clf.predict(X_train))
    test_acc = accuracy_score(y_test, clf.predict(X_test))
    n_support = len(clf.support_vectors_)
    print(f"  C={C}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}, 支持向量数={n_support}")

# ========== 6. 可视化决策边界 ==========
print("\n" + "=" * 50)
print("6. 可视化决策边界")
print("=" * 50)

# 生成数据
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                          n_informative=2, n_clusters_per_class=1, random_state=42)

# 训练SVM
clf = svm.SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
clf.fit(X, y)

# 创建网格
h = 0.02
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# 预测网格点
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# 绘制
plt.figure(figsize=(10, 8))
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.scatter(clf.support_vectors_[:, 0], clf.support_vectors_[:, 1],
           s=200, facecolors='none', edgecolors='red', linewidths=2, label='支持向量')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('SVM决策边界和支持向量')
plt.legend()
plt.savefig('svm_decision_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("SVM示例完成！")
print("=" * 50)

