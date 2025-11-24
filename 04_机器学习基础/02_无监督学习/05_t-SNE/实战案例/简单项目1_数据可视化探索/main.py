"""
简单项目1：数据可视化探索
使用t-SNE对高维数据进行可视化

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# 加载手写数字数据集（8x8图像，64维）
digits = load_digits()
X = digits.data
y = digits.target

# 使用前1000个样本（t-SNE计算慢）
X_subset = X[:1000]
y_subset = y[:1000]

print(f"数据信息:")
print(f"  样本数: {X_subset.shape[0]}")
print(f"  特征数: {X_subset.shape[1]} (8x8图像)")
print(f"  类别数: {len(np.unique(y_subset))}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_subset)

print("\n数据标准化完成！")

# ========== 2. t-SNE降维 ==========
print("\n" + "=" * 60)
print("2. t-SNE降维")
print("=" * 60)

# 使用t-SNE降维到2维
tsne = TSNE(
    n_components=2,
    perplexity=30,
    learning_rate=200,
    n_iter=1000,
    random_state=42
)

print("训练模型（这可能需要一些时间）...")
X_tsne = tsne.fit_transform(X_scaled)

print(f"降维后的数据形状: {X_tsne.shape}")

# ========== 3. 可视化结果 ==========
print("\n" + "=" * 60)
print("3. 可视化结果")
print("=" * 60)

# 可视化降维后的数据
plt.figure(figsize=(12, 8))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_subset, cmap='tab10', alpha=0.6, s=30)
plt.colorbar(scatter, label='数字')
plt.xlabel('第一维度', fontsize=12)
plt.ylabel('第二维度', fontsize=12)
plt.title('手写数字数据t-SNE降维可视化', fontsize=14)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('tsne_visualization.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 分析数据模式 ==========
print("\n" + "=" * 60)
print("4. 分析数据模式")
print("=" * 60)

print(f"\n各类别在降维后的分布:")
for digit in np.unique(y_subset):
    digit_data = X_tsne[y_subset == digit]
    print(f"  数字 {digit}:")
    print(f"    样本数: {len(digit_data)}")
    print(f"    第一维度范围: [{digit_data[:, 0].min():.2f}, {digit_data[:, 0].max():.2f}]")
    print(f"    第二维度范围: [{digit_data[:, 1].min():.2f}, {digit_data[:, 1].max():.2f}]")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. t-SNE可以将高维数据降到2维进行可视化
2. 降维后的数据保留了局部结构
3. 可以用于探索数据的结构和模式
4. 特别适合可视化高维数据
""")

