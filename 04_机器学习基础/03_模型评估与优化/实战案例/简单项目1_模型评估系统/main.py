"""
简单项目1：模型评估系统
构建一个完整的模型评估系统，包括评估指标计算、交叉验证、模型比较等功能

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, auc, classification_report)
from sklearn.datasets import make_classification

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成分类数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                          n_redundant=5, n_classes=2, random_state=42)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"数据信息:")
print(f"  训练集样本数: {len(X_train)}")
print(f"  测试集样本数: {len(X_test)}")
print(f"  特征数: {X.shape[1]}")

# ========== 2. 训练多个模型 ==========
print("\n" + "=" * 60)
print("2. 训练多个模型")
print("=" * 60)

# 创建模型
models = {
    '逻辑回归': LogisticRegression(random_state=42, max_iter=1000),
    '决策树': DecisionTreeClassifier(random_state=42),
    '随机森林': RandomForestClassifier(n_estimators=10, random_state=42)
}

# 训练模型
trained_models = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    trained_models[name] = model
    print(f"{name} 训练完成")

# ========== 3. 评估模型 ==========
print("\n" + "=" * 60)
print("3. 评估模型")
print("=" * 60)

# 评估每个模型
results = {}
for name, model in trained_models.items():
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # 计算评估指标
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    results[name] = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    
    print(f"\n{name} 评估结果:")
    print(f"  准确率: {accuracy:.4f}")
    print(f"  精确率: {precision:.4f}")
    print(f"  召回率: {recall:.4f}")
    print(f"  F1分数: {f1:.4f}")

# ========== 4. 可视化评估结果 ==========
print("\n" + "=" * 60)
print("4. 可视化评估结果")
print("=" * 60)

# 创建对比图
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# 左上：评估指标对比
ax1 = axes[0, 0]
metrics = ['准确率', '精确率', '召回率', 'F1分数']
x = np.arange(len(metrics))
width = 0.25

for i, (name, result) in enumerate(results.items()):
    values = [result['accuracy'], result['precision'], result['recall'], result['f1']]
    ax1.bar(x + i * width, values, width, label=name, alpha=0.7)

ax1.set_xlabel('评估指标', fontsize=12)
ax1.set_ylabel('分数', fontsize=12)
ax1.set_title('模型评估指标对比', fontsize=14)
ax1.set_xticks(x + width)
ax1.set_xticklabels(metrics)
ax1.legend()
ax1.grid(True, axis='y', alpha=0.3)

# 右上：混淆矩阵（逻辑回归）
ax2 = axes[0, 1]
cm = confusion_matrix(y_test, results['逻辑回归']['y_pred'])
im = ax2.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax2.figure.colorbar(im, ax=ax2)
ax2.set(xticks=np.arange(2), yticks=np.arange(2),
        xticklabels=['负例', '正例'], yticklabels=['负例', '正例'],
        title='混淆矩阵（逻辑回归）',
        ylabel='真实标签', xlabel='预测标签')

# 添加数值标签
thresh = cm.max() / 2.
for i in range(2):
    for j in range(2):
        ax2.text(j, i, format(cm[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

# 左下：ROC曲线
ax3 = axes[1, 0]
for name, result in results.items():
    if result['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test, result['y_pred_proba'])
        roc_auc = auc(fpr, tpr)
        ax3.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.3f})', linewidth=2)

ax3.plot([0, 1], [0, 1], 'k--', label='随机分类器')
ax3.set_xlabel('假正例率 (FPR)', fontsize=12)
ax3.set_ylabel('真正例率 (TPR)', fontsize=12)
ax3.set_title('ROC曲线', fontsize=14)
ax3.legend()
ax3.grid(True, alpha=0.3)

# 右下：模型排名
ax4 = axes[1, 1]
model_names = list(results.keys())
f1_scores = [results[name]['f1'] for name in model_names]
sorted_indices = np.argsort(f1_scores)[::-1]

ax4.barh(range(len(model_names)), [f1_scores[i] for i in sorted_indices],
         alpha=0.7, color='skyblue', edgecolor='black')
ax4.set_yticks(range(len(model_names)))
ax4.set_yticklabels([model_names[i] for i in sorted_indices])
ax4.set_xlabel('F1分数', fontsize=12)
ax4.set_title('模型排名（按F1分数）', fontsize=14)
ax4.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('model_evaluation_system.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 交叉验证 ==========
print("\n" + "=" * 60)
print("5. 交叉验证")
print("=" * 60)

# 使用5折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

cv_results = {}
for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
    cv_results[name] = scores
    print(f"\n{name} 交叉验证结果:")
    print(f"  平均准确率: {scores.mean():.4f} ± {scores.std():.4f}")

# 可视化交叉验证结果
plt.figure(figsize=(10, 6))
positions = np.arange(len(models))
means = [cv_results[name].mean() for name in models.keys()]
stds = [cv_results[name].std() for name in models.keys()]

plt.bar(positions, means, yerr=stds, alpha=0.7, color=['skyblue', 'lightgreen', 'lightcoral'],
        edgecolor='black', capsize=5)
plt.xticks(positions, models.keys(), fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('交叉验证结果（5折）', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 添加数值标签
for i, (mean, std) in enumerate(zip(means, stds)):
    plt.text(i, mean + std + 0.01, f'{mean:.3f}±{std:.3f}',
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('cross_validation_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 生成评估报告 ==========
print("\n" + "=" * 60)
print("6. 生成评估报告")
print("=" * 60)

print("\n详细分类报告（逻辑回归）:")
print(classification_report(y_test, results['逻辑回归']['y_pred'],
                          target_names=['负例', '正例']))

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. 模型评估系统可以全面评估模型性能
2. 可以使用多个评估指标综合评估
3. 交叉验证可以更准确地评估模型
4. 可视化可以帮助理解评估结果
5. 可以根据评估结果选择最佳模型
""")

