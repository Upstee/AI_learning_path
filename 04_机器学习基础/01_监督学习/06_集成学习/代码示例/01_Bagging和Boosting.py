"""
Bagging和Boosting示例
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

# ========== 1. Bagging ==========
print("=" * 50)
print("1. Bagging（装袋）")
print("=" * 50)

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 单棵决策树
dt = DecisionTreeClassifier(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print(f"单棵决策树准确率: {dt_acc:.4f}")

# Bagging
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=10),
    n_estimators=100,
    random_state=42,
    n_jobs=-1
)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_acc = accuracy_score(y_test, bagging_pred)
print(f"Bagging准确率: {bagging_acc:.4f}")
print(f"提升: {bagging_acc - dt_acc:.4f}")

# ========== 2. AdaBoost ==========
print("\n" + "=" * 50)
print("2. AdaBoost（自适应提升）")
print("=" * 50)

# AdaBoost
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=100,
    learning_rate=1.0,
    random_state=42
)
adaboost.fit(X_train, y_train)
adaboost_pred = adaboost.predict(X_test)
adaboost_acc = accuracy_score(y_test, adaboost_pred)
print(f"AdaBoost准确率: {adaboost_acc:.4f}")
print(f"提升: {adaboost_acc - dt_acc:.4f}")

# 学习器权重
print(f"\n前10个学习器的权重:")
for i, weight in enumerate(adaboost.estimator_weights_[:10]):
    print(f"  学习器 {i+1}: {weight:.4f}")

# ========== 3. 梯度提升 ==========
print("\n" + "=" * 50)
print("3. 梯度提升（Gradient Boosting）")
print("=" * 50)

# 梯度提升
gbdt = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)
gbdt.fit(X_train, y_train)
gbdt_pred = gbdt.predict(X_test)
gbdt_acc = accuracy_score(y_test, gbdt_pred)
print(f"GBDT准确率: {gbdt_acc:.4f}")
print(f"提升: {gbdt_acc - dt_acc:.4f}")

# 特征重要性
print(f"\n特征重要性（前10个）:")
feature_importance = pd.DataFrame({
    'feature': [f'特征{i}' for i in range(20)],
    'importance': gbdt.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance.head(10))

# ========== 4. 不同方法对比 ==========
print("\n" + "=" * 50)
print("4. 不同集成方法对比")
print("=" * 50)

methods = {
    '单棵决策树': dt,
    'Bagging': bagging,
    'AdaBoost': adaboost,
    'GBDT': gbdt
}

results = []
for name, model in methods.items():
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, model.predict(X_test))
    results.append({
        '方法': name,
        '训练准确率': train_acc,
        '测试准确率': test_acc,
        '过拟合程度': train_acc - test_acc
    })
    print(f"{name}: 训练={train_acc:.4f}, 测试={test_acc:.4f}, 过拟合={train_acc-test_acc:.4f}")

# ========== 5. 学习曲线 ==========
print("\n" + "=" * 50)
print("5. 学习曲线（GBDT）")
print("=" * 50)

# 不同树数量的效果
n_estimators_list = [10, 50, 100, 200, 500]
train_scores = []
test_scores = []

for n in n_estimators_list:
    gb = GradientBoostingClassifier(n_estimators=n, learning_rate=0.1, max_depth=3, random_state=42)
    gb.fit(X_train, y_train)
    train_scores.append(accuracy_score(y_train, gb.predict(X_train)))
    test_scores.append(accuracy_score(y_test, gb.predict(X_test)))

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_list, train_scores, 'o-', label='训练集')
plt.plot(n_estimators_list, test_scores, 's-', label='测试集')
plt.xlabel('树的数量')
plt.ylabel('准确率')
plt.title('GBDT学习曲线')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('gbdt_learning_curve.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("集成学习示例完成！")
print("=" * 50)

