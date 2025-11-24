"""
交叉验证
本示例展示如何使用交叉验证评估模型
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import (cross_val_score, KFold, StratifiedKFold,
                                    LeaveOneOut, cross_validate)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成分类数据
X, y = make_classification(n_samples=200, n_features=10, n_informative=5,
                          n_redundant=2, n_classes=2, random_state=42)

print(f"数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")

# ========== 2. K折交叉验证 ==========
print("\n" + "=" * 60)
print("2. K折交叉验证")
print("=" * 60)
print("""
K折交叉验证：
- 将数据分成K份
- 每次用K-1份训练，1份验证
- 重复K次，得到K个评估结果
- 计算平均值
""")

# 创建模型
model = LogisticRegression(random_state=42, max_iter=1000)

# 使用5折交叉验证
kfold = KFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')

print(f"\n5折交叉验证结果:")
for i, score in enumerate(scores, 1):
    print(f"  折 {i}: {score:.4f}")
print(f"  平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# ========== 3. 分层K折交叉验证 ==========
print("\n" + "=" * 60)
print("3. 分层K折交叉验证")
print("=" * 60)
print("""
分层K折交叉验证：
- 保持每折中各类别的比例与原始数据相同
- 在不平衡数据中更准确
""")

# 使用分层5折交叉验证
skfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_stratified = cross_val_score(model, X, y, cv=skfold, scoring='accuracy')

print(f"\n分层5折交叉验证结果:")
for i, score in enumerate(scores_stratified, 1):
    print(f"  折 {i}: {score:.4f}")
print(f"  平均准确率: {scores_stratified.mean():.4f} ± {scores_stratified.std():.4f}")

# ========== 4. 留一法（LOOCV） ==========
print("\n" + "=" * 60)
print("4. 留一法（LOOCV）")
print("=" * 60)
print("""
留一法：
- K折交叉验证的特殊情况，K等于样本数
- 每次只用1个样本验证
- 计算量大，但评估结果无偏
""")

# 注意：留一法计算量大，这里只用前50个样本演示
X_small = X[:50]
y_small = y[:50]

loocv = LeaveOneOut()
scores_loocv = cross_val_score(model, X_small, y_small, cv=loocv, scoring='accuracy')

print(f"\n留一法交叉验证结果（前50个样本）:")
print(f"  平均准确率: {scores_loocv.mean():.4f} ± {scores_loocv.std():.4f}")
print(f"  评估次数: {len(scores_loocv)}")

# ========== 5. 多指标交叉验证 ==========
print("\n" + "=" * 60)
print("5. 多指标交叉验证")
print("=" * 60)
print("""
多指标交叉验证：
- 同时计算多个评估指标
- 更全面地评估模型性能
""")

# 定义多个评估指标
scoring = {
    'accuracy': make_scorer(accuracy_score),
    'precision': make_scorer(precision_score),
    'recall': make_scorer(recall_score),
    'f1': make_scorer(f1_score)
}

# 使用交叉验证计算多个指标
cv_results = cross_validate(model, X, y, cv=5, scoring=scoring)

print(f"\n多指标交叉验证结果:")
for metric in scoring.keys():
    scores = cv_results[f'test_{metric}']
    print(f"  {metric}: {scores.mean():.4f} ± {scores.std():.4f}")

# ========== 6. 模型比较 ==========
print("\n" + "=" * 60)
print("6. 模型比较")
print("=" * 60)
print("""
模型比较：
- 使用交叉验证比较不同模型
- 选择性能最好的模型
""")

# 创建多个模型
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=10, random_state=42)
}

# 使用交叉验证评估每个模型
results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
    results[name] = scores
    print(f"\n{name}:")
    print(f"  平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 可视化模型比较
plt.figure(figsize=(10, 6))
positions = np.arange(len(models))
means = [results[name].mean() for name in models.keys()]
stds = [results[name].std() for name in models.keys()]

plt.bar(positions, means, yerr=stds, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'],
        edgecolor='black', capsize=5)
plt.xticks(positions, models.keys(), fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('模型比较（5折交叉验证）', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 添加数值标签
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 7. 不同K值的影响 ==========
print("\n" + "=" * 60)
print("7. 不同K值的影响")
print("=" * 60)
print("""
不同K值的影响：
- K值小：计算快，但方差大
- K值大：计算慢，但方差小
- 通常K取5-10
""")

# 测试不同的K值
k_values = [3, 5, 10, 20]
k_results = {}

for k in k_values:
    kfold = KFold(n_splits=k, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, y, cv=kfold, scoring='accuracy')
    k_results[k] = scores
    print(f"\nK={k}:")
    print(f"  平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 可视化
plt.figure(figsize=(10, 6))
means = [k_results[k].mean() for k in k_values]
stds = [k_results[k].std() for k in k_values]

plt.plot(k_values, means, marker='o', linewidth=2, markersize=8, label='平均准确率')
plt.fill_between(k_values, 
                 [m - s for m, s in zip(means, stds)],
                 [m + s for m, s in zip(means, stds)],
                 alpha=0.3, label='±1标准差')
plt.xlabel('K值', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('不同K值的交叉验证结果', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('k_value_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. K折交叉验证充分利用数据
2. 分层交叉验证在不平衡数据中更准确
3. 留一法计算量大但无偏
4. 可以使用多指标交叉验证全面评估
5. 交叉验证可以用于模型比较和选择
""")

