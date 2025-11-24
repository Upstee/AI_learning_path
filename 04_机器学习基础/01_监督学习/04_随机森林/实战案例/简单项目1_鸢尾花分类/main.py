"""
简单项目1：鸢尾花分类
使用随机森林对鸢尾花进行分类
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 50)
print("鸢尾花分类 - 随机森林 vs 决策树")
print("=" * 50)

# 数据探索
print(f"\n数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 训练单棵决策树
print("\n训练单棵决策树...")
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"决策树准确率: {dt_acc:.4f}")

# 训练随机森林
print("\n训练随机森林...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    random_state=42,
    oob_score=True,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)
print(f"随机森林准确率: {rf_acc:.4f}")
print(f"OOB准确率: {rf.oob_score_:.4f}")

# 特征重要性
print(f"\n特征重要性（随机森林）:")
for i, (name, importance) in enumerate(zip(feature_names, rf.feature_importances_)):
    print(f"  {name}: {importance:.4f}")

# 可视化特征重要性
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.barh(feature_names, rf.feature_importances_)
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性')
plt.grid(True, axis='x', alpha=0.3)

plt.subplot(1, 2, 2)
plt.barh(feature_names, dt.feature_importances_)
plt.xlabel('特征重要性')
plt.title('决策树特征重要性')
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 性能对比
print(f"\n性能对比:")
print(f"  决策树: {dt_acc:.4f}")
print(f"  随机森林: {rf_acc:.4f}")
print(f"  提升: {rf_acc - dt_acc:.4f}")

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

