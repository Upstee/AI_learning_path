"""
简单项目1：鸢尾花分类
使用决策树对鸢尾花进行分类
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target
feature_names = iris.feature_names
target_names = iris.target_names

print("=" * 50)
print("鸢尾花分类 - 决策树")
print("=" * 50)

# 数据探索
print(f"\n数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"类别名称: {target_names}")
print(f"类别分布: {np.bincount(y)}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 创建决策树分类器
dt = DecisionTreeClassifier(
    max_depth=3,
    min_samples_split=2,
    min_samples_leaf=1,
    criterion='gini',
    random_state=42
)

# 训练模型
print("\n训练决策树...")
dt.fit(X_train, y_train)

# 预测
y_pred = dt.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n模型性能:")
print(f"准确率: {accuracy:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=target_names))

# 特征重要性
print(f"\n特征重要性:")
for i, (name, importance) in enumerate(zip(feature_names, dt.feature_importances_)):
    print(f"  {name}: {importance:.4f}")

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(dt, filled=True, feature_names=feature_names,
          class_names=target_names, fontsize=10)
plt.title("决策树可视化", fontsize=16)
plt.savefig('iris_decision_tree.png', dpi=300, bbox_inches='tight')
plt.show()

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=target_names, yticklabels=target_names)
plt.title('混淆矩阵', fontsize=14)
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)
plt.savefig('iris_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

