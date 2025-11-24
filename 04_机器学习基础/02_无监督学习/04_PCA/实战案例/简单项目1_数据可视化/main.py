"""
简单项目1：数据可视化
使用PCA将高维数据降到2维进行可视化

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_wine
from sklearn.preprocessing import StandardScaler

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# 加载葡萄酒数据集（13维特征）
wine = load_wine()
X = wine.data
y = wine.target

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

print(f"降维后的数据形状: {X_pca.shape}")
print(f"方差贡献率: {pca.explained_variance_ratio_}")
print(f"累计方差贡献率: {np.sum(pca.explained_variance_ratio_):.4f}")

# ========== 3. 可视化结果 ==========
print("\n" + "=" * 60)
print("3. 可视化结果")
print("=" * 60)

# 可视化降维后的数据
plt.figure(figsize=(12, 6))

# 左图：原始数据（前两个特征）
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('原始数据（前两个特征）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：PCA降维后的数据
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.colorbar(scatter, label='类别')
plt.title('PCA降维后的数据（2维）', fontsize=14)
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 分析数据分布 ==========
print("\n" + "=" * 60)
print("4. 分析数据分布")
print("=" * 60)

print(f"\n各类别在降维后的分布:")
for class_id in np.unique(y):
    class_data = X_pca[y == class_id]
    print(f"  类别 {class_id}:")
    print(f"    样本数: {len(class_data)}")
    print(f"    第一主成分范围: [{class_data[:, 0].min():.2f}, {class_data[:, 0].max():.2f}]")
    print(f"    第二主成分范围: [{class_data[:, 1].min():.2f}, {class_data[:, 1].max():.2f}]")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. PCA可以将高维数据降到2维或3维进行可视化
2. 降维后的数据保留了主要信息
3. 可以用于分析数据的分布和模式
4. 累计方差贡献率表示保留的信息量
""")

