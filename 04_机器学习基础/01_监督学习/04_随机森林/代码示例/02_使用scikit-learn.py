"""
使用scikit-learn实现随机森林
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.datasets import load_iris, make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ========== 1. 分类任务 ==========
print("=" * 50)
print("1. 随机森林分类")
print("=" * 50)

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林分类器
rf_classifier = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1  # 使用所有CPU核心
)

# 训练
print("训练随机森林分类器...")
rf_classifier.fit(X_train, y_train)

# 预测
y_pred = rf_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 特征重要性
print(f"\n特征重要性:")
for i, (name, importance) in enumerate(zip(iris.feature_names, rf_classifier.feature_importances_)):
    print(f"  {name}: {importance:.4f}")

# 可视化特征重要性
plt.figure(figsize=(10, 6))
plt.barh(iris.feature_names, rf_classifier.feature_importances_)
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性')
plt.tight_layout()
plt.savefig('rf_feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 回归任务 ==========
print("\n" + "=" * 50)
print("2. 随机森林回归")
print("=" * 50)

# 生成回归数据
X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 创建随机森林回归器
rf_regressor = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    min_samples_split=2,
    min_samples_leaf=1,
    max_features='sqrt',
    random_state=42,
    n_jobs=-1
)

# 训练
print("训练随机森林回归器...")
rf_regressor.fit(X_train, y_train)

# 预测
y_pred = rf_regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"均方误差: {mse:.4f}")
print(f"R²: {r2:.4f}")

# ========== 3. 不同参数对比 ==========
print("\n" + "=" * 50)
print("3. 不同参数对比")
print("=" * 50)

# 生成数据
X, y = make_classification(n_samples=500, n_features=20, n_informative=10,
                          n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 不同树数量
n_trees = [10, 50, 100, 200]
print("不同树数量的效果:")
for n in n_trees:
    rf = RandomForestClassifier(n_estimators=n, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  n_estimators={n}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# 不同最大特征数
max_features_list = ['sqrt', 'log2', None]
print("\n不同最大特征数的效果:")
for mf in max_features_list:
    rf = RandomForestClassifier(n_estimators=100, max_features=mf, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, rf.predict(X_train))
    test_acc = accuracy_score(y_test, rf.predict(X_test))
    print(f"  max_features={mf}: 训练准确率={train_acc:.4f}, 测试准确率={test_acc:.4f}")

# ========== 4. OOB误差 ==========
print("\n" + "=" * 50)
print("4. OOB误差")
print("=" * 50)

# 使用OOB样本评估
rf_oob = RandomForestClassifier(
    n_estimators=100,
    oob_score=True,  # 启用OOB评分
    random_state=42,
    n_jobs=-1
)
rf_oob.fit(X_train, y_train)
print(f"OOB准确率: {rf_oob.oob_score_:.4f}")

# 对比OOB和测试集准确率
test_acc = accuracy_score(y_test, rf_oob.predict(X_test))
print(f"测试集准确率: {test_acc:.4f}")
print(f"差异: {abs(rf_oob.oob_score_ - test_acc):.4f}")

# ========== 5. 与单棵决策树对比 ==========
print("\n" + "=" * 50)
print("5. 与单棵决策树对比")
print("=" * 50)

from sklearn.tree import DecisionTreeClassifier

# 单棵决策树
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_train_acc = accuracy_score(y_train, dt.predict(X_train))
dt_test_acc = accuracy_score(y_test, dt.predict(X_test))

# 随机森林
rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
rf_train_acc = accuracy_score(y_train, rf.predict(X_train))
rf_test_acc = accuracy_score(y_test, rf.predict(X_test))

print("单棵决策树:")
print(f"  训练准确率: {dt_train_acc:.4f}")
print(f"  测试准确率: {dt_test_acc:.4f}")
print(f"  过拟合程度: {dt_train_acc - dt_test_acc:.4f}")

print("\n随机森林:")
print(f"  训练准确率: {rf_train_acc:.4f}")
print(f"  测试准确率: {rf_test_acc:.4f}")
print(f"  过拟合程度: {rf_train_acc - rf_test_acc:.4f}")

print("\n" + "=" * 50)
print("随机森林示例完成！")
print("=" * 50)

