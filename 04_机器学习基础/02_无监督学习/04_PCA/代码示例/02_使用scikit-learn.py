"""
使用scikit-learn实现PCA
本示例展示如何使用scikit-learn库快速实现PCA降维
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris, load_digits
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ========== 1. 基础PCA降维 ==========
print("=" * 60)
print("1. 基础PCA降维")
print("=" * 60)
print("""
PCA降维：
- 将高维数据降到低维
- 保留主要信息
- 去除冗余信息
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

# 创建PCA模型
# n_components: 要保留的主成分数
#   - int: 保留前n个主成分
#   - float: 保留累计方差贡献率 >= n的主成分
#   - None: 保留所有主成分
pca = PCA(n_components=2)

# 训练模型并转换数据
print("\n训练模型...")
X_pca = pca.fit_transform(X_scaled)

print(f"\n降维结果:")
print(f"  降维后的数据形状: {X_pca.shape}")
print(f"  方差贡献率: {pca.explained_variance_ratio_}")
print(f"  累计方差贡献率: {np.sum(pca.explained_variance_ratio_):.4f}")

# 可视化
plt.figure(figsize=(14, 6))

# 左图：原始数据（前两个特征）
plt.subplot(1, 2, 1)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('原始数据（前两个特征）', fontsize=14)
plt.xlabel('特征1', fontsize=12)
plt.ylabel('特征2', fontsize=12)
plt.grid(True, alpha=0.3)

# 右图：PCA降维后的数据
plt.subplot(1, 2, 2)
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
plt.title('PCA降维后的数据（2维）', fontsize=14)
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_basic.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. 累计方差贡献率 ==========
print("\n" + "=" * 60)
print("2. 累计方差贡献率")
print("=" * 60)
print("""
累计方差贡献率：
- 表示保留的信息量
- 通常保留累计方差贡献率 > 85%或90%的主成分
- 帮助选择合适的降维维度
""")

# 使用所有主成分训练PCA
pca_full = PCA(n_components=None)
pca_full.fit(X_scaled)

# 计算累计方差贡献率
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)

print(f"\n各主成分的方差贡献率:")
for i in range(len(pca_full.explained_variance_ratio_)):
    print(f"  主成分 {i+1}: {pca_full.explained_variance_ratio_[i]:.4f} "
          f"({pca_full.explained_variance_ratio_[i]*100:.2f}%)")

print(f"\n累计方差贡献率:")
for i in range(len(cumulative_variance)):
    print(f"  前{i+1}个主成分: {cumulative_variance[i]:.4f} "
          f"({cumulative_variance[i]*100:.2f}%)")

# 可视化
plt.figure(figsize=(12, 5))

# 左图：方差贡献率
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca_full.explained_variance_ratio_) + 1),
        pca_full.explained_variance_ratio_, alpha=0.7, color='skyblue')
plt.xlabel('主成分', fontsize=12)
plt.ylabel('方差贡献率', fontsize=12)
plt.title('各主成分的方差贡献率', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 右图：累计方差贡献率
plt.subplot(1, 2, 2)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'o-', linewidth=2, markersize=8)
plt.axhline(y=0.85, color='r', linestyle='--', label='85%阈值')
plt.axhline(y=0.90, color='g', linestyle='--', label='90%阈值')
plt.xlabel('主成分数', fontsize=12)
plt.ylabel('累计方差贡献率', fontsize=12)
plt.title('累计方差贡献率', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 使用累计方差贡献率选择维度 ==========
print("\n" + "=" * 60)
print("3. 使用累计方差贡献率选择维度")
print("=" * 60)

# 使用累计方差贡献率选择维度
# 保留累计方差贡献率 >= 0.95的主成分
pca_95 = PCA(n_components=0.95)
X_pca_95 = pca_95.fit_transform(X_scaled)

print(f"\n保留95%方差的主成分数: {pca_95.n_components_}")
print(f"实际累计方差贡献率: {np.sum(pca_95.explained_variance_ratio_):.4f}")

# ========== 4. 高维数据可视化 ==========
print("\n" + "=" * 60)
print("4. 高维数据可视化")
print("=" * 60)
print("""
PCA用于高维数据可视化：
- 将高维数据降到2维或3维
- 可视化数据分布
- 分析数据模式
""")

# 加载手写数字数据集（8x8图像，64维）
digits = load_digits()
X_digits = digits.data
y_digits = digits.target

print(f"\n手写数字数据:")
print(f"  样本数: {X_digits.shape[0]}")
print(f"  特征数: {X_digits.shape[1]} (8x8图像)")

# 标准化
scaler_digits = StandardScaler()
X_digits_scaled = scaler_digits.fit_transform(X_digits)

# PCA降维到2维
pca_digits = PCA(n_components=2)
X_digits_pca = pca_digits.fit_transform(X_digits_scaled)

print(f"降维后的数据形状: {X_digits_pca.shape}")
print(f"累计方差贡献率: {np.sum(pca_digits.explained_variance_ratio_):.4f}")

# 可视化
plt.figure(figsize=(12, 6))
scatter = plt.scatter(X_digits_pca[:, 0], X_digits_pca[:, 1], c=y_digits, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='数字')
plt.xlabel('第一主成分', fontsize=12)
plt.ylabel('第二主成分', fontsize=12)
plt.title('手写数字数据PCA降维可视化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('pca_digits.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 主成分分析 ==========
print("\n" + "=" * 60)
print("5. 主成分分析")
print("=" * 60)
print("""
主成分分析：
- 分析主成分的系数（权重）
- 理解主成分的含义
- 解释数据的主要变化方向
""")

# 分析鸢尾花数据的主成分
print(f"\n鸢尾花数据的主成分:")
for i in range(2):
    print(f"\n主成分 {i+1} (方差贡献率: {pca.explained_variance_ratio_[i]:.4f}):")
    print(f"  系数: {pca.components_[i]}")
    print(f"  主要特征: {iris.feature_names[np.argmax(np.abs(pca.components_[i]))]}")

# 可视化主成分
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for i in range(2):
    ax = axes[i]
    # 绘制主成分系数
    bars = ax.bar(range(len(pca.components_[i])), pca.components_[i], alpha=0.7, color='skyblue')
    ax.set_xlabel('特征', fontsize=12)
    ax.set_ylabel('系数', fontsize=12)
    ax.set_title(f'主成分 {i+1} 的系数', fontsize=14)
    ax.set_xticks(range(len(iris.feature_names)))
    ax.set_xticklabels(iris.feature_names, rotation=45, ha='right')
    ax.grid(True, axis='y', alpha=0.3)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=1)

plt.tight_layout()
plt.savefig('pca_components.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 数据还原 ==========
print("\n" + "=" * 60)
print("6. 数据还原")
print("=" * 60)
print("""
数据还原：
- 将降维后的数据还原到原始空间
- 还原是近似的，会丢失一些信息
- 可以用于数据压缩和去噪
""")

# 将降维后的数据还原
X_reconstructed = pca.inverse_transform(X_pca)

# 计算还原误差
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"\n还原误差（MSE）: {reconstruction_error:.6f}")

# 可视化原始数据和还原数据
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# 左图：原始数据
axes[0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
axes[0].set_title('原始数据', fontsize=14)
axes[0].set_xlabel('特征1', fontsize=12)
axes[0].set_ylabel('特征2', fontsize=12)
axes[0].grid(True, alpha=0.3)

# 右图：还原数据
axes[1].scatter(X_reconstructed[:, 0], X_reconstructed[:, 1], c=y, cmap='viridis', alpha=0.6, s=50)
axes[1].set_title('还原数据（近似）', fontsize=14)
axes[1].set_xlabel('特征1', fontsize=12)
axes[1].set_ylabel('特征2', fontsize=12)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_reconstruction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. PCA通过找到数据中方差最大的方向来降维
2. 累计方差贡献率表示保留的信息量
3. 可以使用累计方差贡献率自动选择降维维度
4. PCA可以用于高维数据可视化
5. 主成分的系数可以帮助理解数据的主要变化方向
6. 降维后的数据可以还原，但会丢失一些信息
""")

