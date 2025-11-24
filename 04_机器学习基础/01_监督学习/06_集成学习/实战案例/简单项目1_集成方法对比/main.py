"""
简单项目1：集成方法对比
对比Bagging、AdaBoost、GBDT等集成方法
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import (BaggingClassifier, AdaBoostClassifier, 
                              GradientBoostingClassifier, RandomForestClassifier)
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import time

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("=" * 50)
print("集成方法对比")
print("=" * 50)

# 定义模型
models = {
    '单棵决策树': DecisionTreeClassifier(max_depth=10, random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, n_jobs=-1),
    'Bagging': BaggingClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=10),
        n_estimators=100, random_state=42, n_jobs=-1
    ),
    'AdaBoost': AdaBoostClassifier(
        base_estimator=DecisionTreeClassifier(max_depth=1),
        n_estimators=100, learning_rate=1.0, random_state=42
    ),
    'GBDT': GradientBoostingClassifier(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )
}

# 训练和评估
results = []
for name, model in models.items():
    print(f"\n训练 {name}...")
    start = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start
    
    start = time.time()
    y_pred = model.predict(X_test)
    pred_time = time.time() - start
    
    train_acc = accuracy_score(y_train, model.predict(X_train))
    test_acc = accuracy_score(y_test, y_pred)
    
    results.append({
        '方法': name,
        '训练准确率': train_acc,
        '测试准确率': test_acc,
        '过拟合程度': train_acc - test_acc,
        '训练时间(秒)': train_time,
        '预测时间(秒)': pred_time
    })
    
    print(f"  训练准确率: {train_acc:.4f}")
    print(f"  测试准确率: {test_acc:.4f}")
    print(f"  过拟合程度: {train_acc - test_acc:.4f}")
    print(f"  训练时间: {train_time:.2f}秒")

# 结果汇总
df_results = pd.DataFrame(results)
print("\n" + "=" * 50)
print("结果汇总")
print("=" * 50)
print(df_results.to_string(index=False))

# 可视化
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 准确率对比
axes[0, 0].barh(df_results['方法'], df_results['测试准确率'])
axes[0, 0].set_xlabel('测试准确率')
axes[0, 0].set_title('测试准确率对比')
axes[0, 0].grid(True, axis='x', alpha=0.3)

# 过拟合程度
axes[0, 1].barh(df_results['方法'], df_results['过拟合程度'])
axes[0, 1].set_xlabel('过拟合程度')
axes[0, 1].set_title('过拟合程度对比')
axes[0, 1].grid(True, axis='x', alpha=0.3)

# 训练时间
axes[1, 0].barh(df_results['方法'], df_results['训练时间(秒)'])
axes[1, 0].set_xlabel('训练时间(秒)')
axes[1, 0].set_title('训练时间对比')
axes[1, 0].grid(True, axis='x', alpha=0.3)

# 预测时间
axes[1, 1].barh(df_results['方法'], df_results['预测时间(秒)'])
axes[1, 1].set_xlabel('预测时间(秒)')
axes[1, 1].set_title('预测时间对比')
axes[1, 1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

