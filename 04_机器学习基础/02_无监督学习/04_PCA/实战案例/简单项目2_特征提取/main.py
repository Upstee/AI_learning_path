"""
简单项目2：特征提取
使用PCA从原始特征中提取主成分作为新特征

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  原始特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ========== 2. PCA特征提取 ==========
print("\n" + "=" * 60)
print("2. PCA特征提取")
print("=" * 60)

# 使用PCA提取主成分（保留95%的方差）
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_scaled)

print(f"提取的主成分数: {pca.n_components_}")
print(f"累计方差贡献率: {np.sum(pca.explained_variance_ratio_):.4f}")
print(f"新特征形状: {X_pca.shape}")

# ========== 3. 对比原始特征和主成分特征 ==========
print("\n" + "=" * 60)
print("3. 对比原始特征和主成分特征")
print("=" * 60)

# 使用原始特征进行分类
X_train_orig, X_test_orig, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42
)

clf_orig = LogisticRegression(random_state=42)
clf_orig.fit(X_train_orig, y_train)
y_pred_orig = clf_orig.predict(X_test_orig)
acc_orig = accuracy_score(y_test, y_pred_orig)

print(f"\n使用原始特征:")
print(f"  特征数: {X_train_orig.shape[1]}")
print(f"  准确率: {acc_orig:.4f}")

# 使用主成分特征进行分类
X_train_pca, X_test_pca, _, _ = train_test_split(
    X_pca, y, test_size=0.3, random_state=42
)

clf_pca = LogisticRegression(random_state=42)
clf_pca.fit(X_train_pca, y_train)
y_pred_pca = clf_pca.predict(X_test_pca)
acc_pca = accuracy_score(y_test, y_pred_pca)

print(f"\n使用主成分特征:")
print(f"  特征数: {X_train_pca.shape[1]}")
print(f"  准确率: {acc_pca:.4f}")

# ========== 4. 可视化对比 ==========
print("\n" + "=" * 60)
print("4. 可视化对比")
print("=" * 60)

# 可视化准确率对比
plt.figure(figsize=(10, 6))
categories = ['原始特征', '主成分特征']
accuracies = [acc_orig, acc_pca]
colors = ['skyblue', 'lightgreen']

bars = plt.bar(categories, accuracies, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('准确率', fontsize=12)
plt.title('原始特征 vs 主成分特征 分类准确率对比', fontsize=14)
plt.ylim([0, 1])
plt.grid(True, axis='y', alpha=0.3)

# 添加数值标签
for bar, acc in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{acc:.4f}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('pca_feature_extraction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. PCA可以从原始特征中提取主成分作为新特征
2. 主成分特征通常更少，但保留了主要信息
3. 使用主成分特征可能获得相似的性能
4. 降维可以减少计算复杂度
""")

