"""
简单项目1：数据可视化探索
使用降维技术将高维数据降维到2-3维，进行可视化探索

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.datasets import make_classification, load_wine
from sklearn.preprocessing import StandardScaler

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 使用葡萄酒数据集
wine = load_wine()
X = wine.data
y = wine.target
feature_names = wine.feature_names

print(f"数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n数据标准化完成！")

# ========== 2. PCA降维 ==========
print("\n" + "=" * 60)
print("2. PCA降维")
print("=" * 60)

# 使用PCA降维到2维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print(f"PCA降维后:")
print(f"  降维后特征数: {X_pca.shape[1]}")
print(f"  第一主成分解释方差比: {pca.explained_variance_ratio_[0]:.4f}")
print(f"  第二主成分解释方差比: {pca.explained_variance_ratio_[1]:.4f}")
print(f"  累计解释方差比: {np.sum(pca.explained_variance_ratio_):.4f}")

# ========== 3. t-SNE降维 ==========
print("\n" + "=" * 60)
print("3. t-SNE降维")
print("=" * 60)

# 使用t-SNE降维到2维
print("正在运行t-SNE（可能需要一些时间）...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_scaled)

print(f"t-SNE降维后:")
print(f"  降维后特征数: {X_tsne.shape[1]}")

# ========== 4. 可视化 ==========
print("\n" + "=" * 60)
print("4. 可视化")
print("=" * 60)

# 创建对比图
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# 左图：PCA降维
scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.7, s=60)
axes[0].set_xlabel('第一主成分', fontsize=12)
axes[0].set_ylabel('第二主成分', fontsize=12)
axes[0].set_title(f'PCA降维（解释方差比: {np.sum(pca.explained_variance_ratio_):.3f}）', fontsize=14)
axes[0].grid(True, alpha=0.3)
plt.colorbar(scatter1, ax=axes[0], label='类别')

# 右图：t-SNE降维
scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.7, s=60)
axes[1].set_xlabel('t-SNE维度1', fontsize=12)
axes[1].set_ylabel('t-SNE维度2', fontsize=12)
axes[1].set_title('t-SNE降维（保留局部结构）', fontsize=14)
axes[1].grid(True, alpha=0.3)
plt.colorbar(scatter2, ax=axes[1], label='类别')

plt.tight_layout()
plt.savefig('dim_reduction_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 分析主成分 ==========
print("\n" + "=" * 60)
print("5. 分析主成分")
print("=" * 60)

# 分析前两个主成分的贡献
print("\n第一主成分的主要贡献特征:")
pc1_contributions = np.abs(pca.components_[0])
top_features_pc1 = np.argsort(pc1_contributions)[-5:][::-1]
for i, idx in enumerate(top_features_pc1, 1):
    print(f"  {i}. {feature_names[idx]}: {pc1_contributions[idx]:.4f}")

print("\n第二主成分的主要贡献特征:")
pc2_contributions = np.abs(pca.components_[1])
top_features_pc2 = np.argsort(pc2_contributions)[-5:][::-1]
for i, idx in enumerate(top_features_pc2, 1):
    print(f"  {i}. {feature_names[idx]}: {pc2_contributions[idx]:.4f}")

# 可视化主成分贡献
plt.figure(figsize=(14, 6))

# 左图：第一主成分贡献
plt.subplot(1, 2, 1)
plt.barh(range(len(feature_names)), pc1_contributions, alpha=0.7, color='skyblue', edgecolor='black')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('贡献度', fontsize=12)
plt.title('第一主成分的特征贡献', fontsize=14)
plt.grid(True, axis='x', alpha=0.3)

# 右图：第二主成分贡献
plt.subplot(1, 2, 2)
plt.barh(range(len(feature_names)), pc2_contributions, alpha=0.7, color='lightcoral', edgecolor='black')
plt.yticks(range(len(feature_names)), feature_names)
plt.xlabel('贡献度', fontsize=12)
plt.title('第二主成分的特征贡献', fontsize=14)
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('pca_contributions.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 解释方差比分析 ==========
print("\n" + "=" * 60)
print("6. 解释方差比分析")
print("=" * 60)

# 计算所有主成分的解释方差比
pca_full = PCA()
pca_full.fit(X_scaled)

# 计算累计解释方差比
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

# 可视化
plt.figure(figsize=(12, 6))

# 左图：各主成分解释方差比
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.7, color='skyblue', edgecolor='black')
plt.xlabel('主成分', fontsize=12)
plt.ylabel('解释方差比', fontsize=12)
plt.title('各主成分的解释方差比', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 右图：累计解释方差比
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance,
         marker='o', linewidth=2, markersize=6, color='lightcoral')
plt.axhline(0.95, color='r', linestyle='--', label='95%阈值')
plt.xlabel('主成分数', fontsize=12)
plt.ylabel('累计解释方差比', fontsize=12)
plt.title('累计解释方差比', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('variance_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# 找到保留95%方差所需的主成分数
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"\n保留95%方差所需的主成分数: {n_components_95}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. PCA可以用于数据可视化，保留主要信息
2. t-SNE可以保留局部结构，适合可视化
3. 可以通过解释方差比分析降维效果
4. 可以根据业务需求选择合适的降维维度
5. 降维后的数据可以用于进一步的分析和建模
""")

