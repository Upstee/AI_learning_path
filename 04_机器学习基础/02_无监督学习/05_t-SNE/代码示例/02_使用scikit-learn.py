"""
使用scikit-learn实现t-SNE
本示例展示如何使用scikit-learn库快速实现t-SNE降维
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_iris, load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler
import time

# ========== 1. 基础t-SNE降维 ==========
print("=" * 60)
print("1. 基础t-SNE降维")
print("=" * 60)
print("""
t-SNE降维：
- 非线性降维
- 保留局部结构
- 主要用于数据可视化
""")

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

print(f"\n数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  类别数: {len(np.unique(y))}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 创建t-SNE模型
# n_components: 降维后的维度（通常是2或3）
# perplexity: 困惑度，控制每个点的有效邻居数（通常5-50）
# learning_rate: 学习率（通常100-1000）
# n_iter: 迭代次数（通常1000以上）
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

# 训练模型并转换数据
print("\n训练模型（这可能需要一些时间）...")
start_time = time.time()
X_tsne = tsne.fit_transform(X_scaled)
elapsed_time = time.time() - start_time

print(f"\n降维结果:")
print(f"  降维后的数据形状: {X_tsne.shape}")
print(f"  训练时间: {elapsed_time:.2f}秒")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：原始数据（前两个特征）
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('原始数据（前两个特征）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：t-SNE降维后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('t-SNE降维后的数据（2维）', fontsize=14)
plt.xlabel('第一维度', fontsize=12)
plt.ylabel('第二维度', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 不同perplexity的影响 ==========
print("\n" + "=" * 60)
print("2. 不同perplexity的影响")
print("=" * 60)
print("""
perplexity的影响：
- 较小的perplexity：保留更多局部结构
- 较大的perplexity：保留更多全局结构
- 通常选择5-50之间
""")

perplexity_values = [5, 30, 50]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, perplexity in enumerate(perplexity_values):
    # 训练t-SNE
    tsne_perp = TSNE(n_components=2, perplexity=perplexity, random_state=42, n_iter=1000)
    X_tsne_perp = tsne_perp.fit_transform(X_scaled)
    
    # 可视化
    ax = axes[idx]
    ax.scatter(X_tsne_perp[:, 0], X_tsne_perp[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
    ax.set_title(f'perplexity={perplexity}', fontsize=14)
    ax.set_xlabel('第一维度', fontsize=12)
    ax.set_ylabel('第二维度', fontsize=12)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_perplexity.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n不同perplexity的效果:")
print("  - perplexity=5: 保留更多局部结构")
print("  - perplexity=30: 平衡局部和全局结构")
print("  - perplexity=50: 保留更多全局结构")

# ========== 3. 高维数据可视化 ==========
print("\n" + "=" * 60)
print("3. 高维数据可视化")
print("=" * 60)
print("""
t-SNE用于高维数据可视化：
- 将高维数据降到2维或3维
- 保留数据的局部结构
- 特别适合可视化
""")

# 加载手写数字数据集（8x8图像，64维）
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 使用前1000个样本（t-SNE计算慢）
X_digits_subset = X_digits[:1000]
y_digits_subset = y_digits[:1000]

print(f"\n手写数字数据:")
print(f"  样本数: {X_digits_subset.shape[0]}")
print(f"  特征数: {X_digits_subset.shape[1]} (8x8图像)")

# 标准化
scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits_subset)

# t-SNE降维到2维
print("\n训练t-SNE（这可能需要一些时间）...")
tsne_digits = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
start_time = time.time()
X_digits_tsne = tsne_digits.fit_transform(X_digits_scaled)
elapsed_time = time.time() - start_time

print(f"降维后的数据形状: {X_digits_tsne.shape}")
print(f"训练时间: {elapsed_time:.2f}秒")

# 可视化
plt.figure(figsize=(12, 6))
scatter = plt.scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1], c=y_digits_subset, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='数字')
plt.xlabel('第一维度', fontsize=12)
plt.ylabel('第二维度', fontsize=12)
plt.title('手写数字数据t-SNE降维可视化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_digits.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. t-SNE vs PCA对比 ==========
print("\n" + "=" * 60)
print("4. t-SNE vs PCA对比")
print("=" * 60)
print("""
t-SNE vs PCA：
- PCA: 线性降维，保留全局结构
- t-SNE: 非线性降维，保留局部结构
""")

from sklearn.decomposition import PCA

# PCA降维
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# t-SNE降维
X_tsne = tsne.fit_transform(X_scaled)

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：PCA结果
axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('PCA降维结果', fontsize=14)
axes[0].set_xlabel('第一主成分', fontsize=12)
axes[0].set_ylabel('第二主成分', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 右图：t-SNE结果
axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
axes[1].set_title('t-SNE降维结果', fontsize=14)
axes[1].set_xlabel('第一维度', fontsize=12)
axes[1].set_ylabel('第二维度', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('tsne_vs_pca.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n对比结果:")
print("  - PCA: 保留全局结构，计算快")
print("  - t-SNE: 保留局部结构，计算慢，但可视化效果通常更好")

# ========== 5. 3维可视化 ==========
print("\n" + "=" * 60)
print("5. 3维可视化")
print("=" * 60)
print("""
t-SNE也可以降到3维：
- 可以用于3维可视化
- 保留更多信息
- 计算更慢
""")

# t-SNE降维到3维
tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42, n_iter=1000)
X_tsne_3d = tsne_3d.fit_transform(X_scaled)

print(f"3维降维后的数据形状: {X_tsne_3d.shape}")

# 3维可视化
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
scatter = ax.scatter(X_tsne_3d[:, 0], X_tsne_3d[:, 1], X_tsne_3d[:, 2], 
                    c=y, cmap='viridis', alpha=0.6, s=50)
ax.set_xlabel('第一维度', fontsize=12)
ax.set_ylabel('第二维度', fontsize=12)
ax.set_zlabel('第三维度', fontsize=12)
ax.set_title('t-SNE降维到3维', fontsize=14)
plt.colorbar(scatter, label='类别')
plt.tight_layout()
plt.savefig('tsne_3d.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. t-SNE是强大的非线性降维工具
2. 特别适合数据可视化
3. 保留数据的局部结构
4. 计算较慢，但可视化效果通常很好
5. 需要选择合适的perplexity参数
6. 结果有随机性，每次运行可能不同
""")

