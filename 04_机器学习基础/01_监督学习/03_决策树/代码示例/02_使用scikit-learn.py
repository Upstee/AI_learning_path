"""
使用scikit-learn实现决策树
"""

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# ========== 1. 分类任务 ==========
print("=" * 50)
print("1. 决策树分类")
print("=" * 50)

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树分类器
dt_classifier = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='gini',
    random_state=42
)

# 训练
dt_classifier.fit(X_train, y_train)

# 预测
y_pred = dt_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 特征重要性
print(f"\n特征重要性:")
for i, importance in enumerate(dt_classifier.feature_importances_):
    print(f"  特征 {i}: {importance:.4f}")

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt_classifier, filled=True, feature_names=iris.feature_names,
          class_names=iris.target_names, fontsize=10)
plt.title("决策树可视化")
plt.savefig('decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 回归任务 ==========
print("\n" + "=" * 50)
print("2. 决策树回归")
print("=" * 50)

# 生成回归数据
X, y = make_regression(n_samples=200, n_features=4, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建决策树回归器
dt_regressor = DecisionTreeRegressor(
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

# 训练
dt_regressor.fit(X_train, y_train)

# 预测
y_pred = dt_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"均方误差: {mse:.4f}")
print(f"RMSE: {np.sqrt(mse):.4f}")

# ========== 3. 不同参数对比 ==========
print("\n" + "=" * 50)
print("3. 不同参数对比")
print("=" * 50)

# 生成数据
X, y = make_classification(n_samples=200, n_features=4, n_informative=2,
                          n_redundant=0, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 不同最大深度
depths = [2, 5, 10, 20]
print("不同最大深度的效果:")
for depth in depths:
    dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"  深度={depth}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# ========== 4. 不同分割准则 ==========
print("\n" + "=" * 50)
print("4. 不同分割准则")
print("=" * 50)

criteria = ['gini', 'entropy']
print("不同分割准则的效果:")
for criterion in criteria:
    dt = DecisionTreeClassifier(criterion=criterion, max_depth=5, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"  {criterion}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# ========== 5. 剪枝效果 ==========
print("\n" + "=" * 50)
print("5. 剪枝效果")
print("=" * 50)

# 不同最小样本分割数
min_splits = [2, 5, 10, 20]
print("不同最小样本分割数的效果:")
for min_split in min_splits:
    dt = DecisionTreeClassifier(min_samples_split=min_split, random_state=42)
    dt.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    print(f"  min_samples_split={min_split}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

print("\n" + "=" * 50)
print("决策树示例完成！")
print("=" * 50)

