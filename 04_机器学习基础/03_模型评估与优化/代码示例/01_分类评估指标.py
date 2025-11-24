"""
分类评估指标
本示例展示如何计算和使用分类评估指标
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            confusion_matrix, roc_curve, auc, classification_report)
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

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

# 训练模型
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # 正类的概率

print("\n模型训练完成！")

# ========== 2. 混淆矩阵 ==========
print("\n" + "=" * 60)
print("2. 混淆矩阵")
print("=" * 60)
print("""
混淆矩阵：
- TP（True Positive）：真正例，预测为正，实际为正
- TN（True Negative）：真负例，预测为负，实际为负
- FP（False Positive）：假正例，预测为正，实际为负（误报）
- FN（False Negative）：假负例，预测为负，实际为正（漏报）
""")

# 计算混淆矩阵
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\n混淆矩阵:")
print(f"               预测正例    预测负例")
print(f"实际正例       {tp:4d}      {fn:4d}")
print(f"实际负例       {fp:4d}      {tn:4d}")

print(f"\n详细统计:")
print(f"  TP (真正例): {tp}")
print(f"  TN (真负例): {tn}")
print(f"  FP (假正例): {fp}")
print(f"  FN (假负例): {fn}")

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵', fontsize=14)
plt.colorbar()
tick_marks = np.arange(2)
plt.xticks(tick_marks, ['负例', '正例'])
plt.yticks(tick_marks, ['负例', '正例'])
plt.ylabel('真实标签', fontsize=12)
plt.xlabel('预测标签', fontsize=12)

# 添加数值标签
thresh = cm.max() / 2.
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], 'd'),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black",
             fontsize=14)

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 基本评估指标 ==========
print("\n" + "=" * 60)
print("3. 基本评估指标")
print("=" * 60)

# 计算评估指标
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"\n评估指标:")
print(f"  准确率 (Accuracy):  {accuracy:.4f}")
print(f"  精确率 (Precision): {precision:.4f}")
print(f"  召回率 (Recall):    {recall:.4f}")
print(f"  F1分数 (F1-Score):  {f1:.4f}")

# 手动计算验证
print(f"\n手动计算验证:")
print(f"  准确率 = (TP + TN) / (TP + TN + FP + FN) = ({tp} + {tn}) / ({tp} + {tn} + {fp} + {fn}) = {accuracy:.4f}")
print(f"  精确率 = TP / (TP + FP) = {tp} / ({tp} + {fp}) = {precision:.4f}")
print(f"  召回率 = TP / (TP + FN) = {tp} / ({tp} + {fn}) = {recall:.4f}")
print(f"  F1分数 = 2 * (Precision * Recall) / (Precision + Recall) = {f1:.4f}")

# ========== 4. ROC曲线和AUC ==========
print("\n" + "=" * 60)
print("4. ROC曲线和AUC")
print("=" * 60)
print("""
ROC曲线：
- 横轴：假正例率 (FPR = FP / (FP + TN))
- 纵轴：真正例率 (TPR = TP / (TP + FN) = Recall)
- AUC：ROC曲线下的面积，值越大越好（0.5-1.0）
""")

# 计算ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
roc_auc = auc(fpr, tpr)

print(f"\nROC曲线统计:")
print(f"  AUC: {roc_auc:.4f}")

# 可视化ROC曲线
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC曲线 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='随机分类器 (AUC = 0.5)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('假正例率 (FPR)', fontsize=12)
plt.ylabel('真正例率 (TPR)', fontsize=12)
plt.title('ROC曲线', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('roc_curve.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 分类报告 ==========
print("\n" + "=" * 60)
print("5. 分类报告")
print("=" * 60)

# 生成分类报告
report = classification_report(y_test, y_pred, target_names=['负例', '正例'])
print("\n分类报告:")
print(report)

# ========== 6. 不同阈值的影响 ==========
print("\n" + "=" * 60)
print("6. 不同阈值的影响")
print("=" * 60)
print("""
阈值选择：
- 默认阈值：0.5
- 提高阈值：精确率提高，召回率降低
- 降低阈值：精确率降低，召回率提高
""")

# 测试不同阈值
thresholds = np.arange(0.1, 1.0, 0.1)
precisions = []
recalls = []
f1_scores = []

for threshold in thresholds:
    y_pred_thresh = (y_pred_proba >= threshold).astype(int)
    precisions.append(precision_score(y_test, y_pred_thresh))
    recalls.append(recall_score(y_test, y_pred_thresh))
    f1_scores.append(f1_score(y_test, y_pred_thresh))

# 可视化
plt.figure(figsize=(12, 6))
plt.plot(thresholds, precisions, marker='o', label='精确率', linewidth=2)
plt.plot(thresholds, recalls, marker='s', label='召回率', linewidth=2)
plt.plot(thresholds, f1_scores, marker='^', label='F1分数', linewidth=2)
plt.xlabel('阈值', fontsize=12)
plt.ylabel('分数', fontsize=12)
plt.title('不同阈值下的评估指标', fontsize=14)
plt.legend(fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('threshold_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. 混淆矩阵是评估的基础
2. 准确率、精确率、召回率、F1分数是常用指标
3. ROC曲线和AUC用于评估分类性能
4. 不同阈值会影响精确率和召回率
5. 需要根据业务需求选择合适的指标和阈值
""")

