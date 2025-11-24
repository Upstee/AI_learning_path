"""
非线性降维方法
本示例展示如何使用非线性降维方法（t-SNE、UMAP、Isomap）
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE, Isomap
from sklearn.datasets import make_swiss_roll, load_digits
from sklearn.preprocessing import StandardScaler

# 注意：UMAP需要单独安装：pip install umap-learn
try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False
    print("警告：UMAP未安装，将跳过UMAP示例。安装命令：pip install umap-learn")

# ========== 1. t-SNE ==========
print("=" * 60)
print("1. t-SNE")
print("=" * 60)
print("""
t-SNE原理：
- 使用t分布保留数据的局部结构
- 非线性降维
- 适合可视化
""")

# 使用手写数字数据集
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

# 为了加快速度，只使用前1000个样本
n_samples = 1000
X_digits_subset = X_digits[:n_samples]
y_digits_subset = y_digits[:n_samples]

print(f"数据信息:")
print(f"  样本数: {X_digits_subset.shape[0]}")
print(f"  特征数: {X_digits_subset.shape[1]}")

# 标准化
scaler = StandardScaler()
X_digits_scaled = scaler.fit_transform(X_digits_subset)

# 使用t-SNE降维到2维
print("\n正在运行t-SNE（可能需要一些时间）...")
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X_digits_scaled)

print(f"\nt-SNE降维后:")
print(f"  降维后特征数: {X_tsne.shape[1]}")

# 可视化
plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_digits_subset, cmap='tab10', alpha=0.6, s=30)
plt.xlabel('t-SNE维度1', fontsize=12)
plt.ylabel('t-SNE维度2', fontsize=12)
plt.title('t-SNE降维结果（手写数字）', fontsize=14)
plt.colorbar(scatter, label='数字')
plt.grid(True, alpha=0.3)
plt.savefig('tsne_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. UMAP ==========
if HAS_UMAP:
    print("\n" + "=" * 60)
    print("2. UMAP")
    print("=" * 60)
    print("""
    UMAP原理：
    - 使用流形学习和拓扑数据分析
    - 非线性降维
    - 比t-SNE更快
    """)
    
    # 使用UMAP降维到2维
    print("\n正在运行UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X_digits_scaled)
    
    print(f"\nUMAP降维后:")
    print(f"  降维后特征数: {X_umap.shape[1]}")
    
    # 可视化
    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y_digits_subset, cmap='tab10', alpha=0.6, s=30)
    plt.xlabel('UMAP维度1', fontsize=12)
    plt.ylabel('UMAP维度2', fontsize=12)
    plt.title('UMAP降维结果（手写数字）', fontsize=14)
    plt.colorbar(scatter, label='数字')
    plt.grid(True, alpha=0.3)
    plt.savefig('umap_demo.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========== 3. Isomap ==========
print("\n" + "=" * 60)
print("3. Isomap")
print("=" * 60)
print("""
Isomap原理：
- 使用测地距离保留数据的全局结构
- 非线性降维
- 保留全局结构
""")

# 生成瑞士卷数据（非线性流形）
print("\n生成瑞士卷数据...")
X_swiss, color = make_swiss_roll(n_samples=1000, random_state=42)

print(f"数据信息:")
print(f"  样本数: {X_swiss.shape[0]}")
print(f"  特征数: {X_swiss.shape[1]}")

# 标准化
X_swiss_scaled = scaler.fit_transform(X_swiss)

# 使用Isomap降维到2维
print("\n正在运行Isomap...")
isomap = Isomap(n_components=2, n_neighbors=10)
X_isomap = isomap.fit_transform(X_swiss_scaled)

print(f"\nIsomap降维后:")
print(f"  降维后特征数: {X_isomap.shape[1]}")

# 可视化
fig = plt.figure(figsize=(18, 6))

# 左图：原始3D数据
ax1 = fig.add_subplot(131, projection='3d')
ax1.scatter(X_swiss[:, 0], X_swiss[:, 1], X_swiss[:, 2], c=color, cmap='viridis', alpha=0.6, s=20)
ax1.set_xlabel('X', fontsize=12)
ax1.set_ylabel('Y', fontsize=12)
ax1.set_zlabel('Z', fontsize=12)
ax1.set_title('原始3D数据（瑞士卷）', fontsize=14)

# 中图：Isomap降维后
ax2 = fig.add_subplot(132)
scatter = ax2.scatter(X_isomap[:, 0], X_isomap[:, 1], c=color, cmap='viridis', alpha=0.6, s=20)
ax2.set_xlabel('Isomap维度1', fontsize=12)
ax2.set_ylabel('Isomap维度2', fontsize=12)
ax2.set_title('Isomap降维结果', fontsize=14)
ax2.grid(True, alpha=0.3)

# 右图：PCA降维后（对比）
from sklearn.decomposition import PCA
pca_swiss = PCA(n_components=2)
X_swiss_pca = pca_swiss.fit_transform(X_swiss_scaled)
ax3 = fig.add_subplot(133)
scatter = ax3.scatter(X_swiss_pca[:, 0], X_swiss_pca[:, 1], c=color, cmap='viridis', alpha=0.6, s=20)
ax3.set_xlabel('PCA维度1', fontsize=12)
ax3.set_ylabel('PCA维度2', fontsize=12)
ax3.set_title('PCA降维结果（对比）', fontsize=14)
ax3.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('isomap_demo.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 对比线性和非线性降维 ==========
print("\n" + "=" * 60)
print("4. 对比线性和非线性降维")
print("=" * 60)

# 使用手写数字数据集
X_digits_small = X_digits[:500]  # 使用更少的样本以加快速度
y_digits_small = y_digits[:500]
X_digits_small_scaled = scaler.fit_transform(X_digits_small)

# PCA降维
pca_digits = PCA(n_components=2)
X_digits_pca = pca_digits.fit_transform(X_digits_small_scaled)

# t-SNE降维
print("\n正在运行t-SNE...")
tsne_digits = TSNE(n_components=2, random_state=42, perplexity=30)
X_digits_tsne = tsne_digits.fit_transform(X_digits_small_scaled)

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# PCA
scatter1 = axes[0].scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], c=y_digits_small, cmap='tab10', alpha=0.6, s=30)
axes[0].set_xlabel('PCA维度1', fontsize=12)
axes[0].set_ylabel('PCA维度2', fontsize=12)
axes[0].set_title(f'PCA降维（解释方差比: {np.sum(pca_digits.explained_variance_ratio_):.3f}）', fontsize=14)
axes[0].grid(True, alpha=0.3)

# t-SNE
scatter2 = axes[1].scatter(X_digits_tsne[:, 0], X_digits_tsne[:, 1], c=y_digits_small, cmap='tab10', alpha=0.6, s=30)
axes[1].set_xlabel('t-SNE维度1', fontsize=12)
axes[1].set_ylabel('t-SNE维度2', fontsize=12)
axes[1].set_title('t-SNE降维（保留局部结构）', fontsize=14)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('linear_vs_nonlinear.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. t-SNE：保留局部结构，适合可视化，但计算慢
2. UMAP：保留拓扑结构，比t-SNE更快
3. Isomap：保留全局结构，适合流形数据
4. 非线性降维可以处理非线性数据
5. 需要根据数据特点选择合适的方法
""")

