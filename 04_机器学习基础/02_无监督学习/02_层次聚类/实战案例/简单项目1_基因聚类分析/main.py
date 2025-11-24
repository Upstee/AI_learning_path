"""
简单项目1：基因聚类分析
使用层次聚类对基因表达数据进行聚类

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.datasets import make_blobs

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备基因表达数据")
print("=" * 60)

# 生成模拟基因表达数据
# 在实际应用中，你需要从数据库或文件中加载真实的基因表达数据
# 这里我们使用模拟数据来演示

# 假设有50个基因，10个样本（实验条件）
n_genes = 50
n_samples = 10

# 生成基因表达数据
# 每个基因在不同样本中的表达水平
np.random.seed(42)
gene_data = np.random.randn(n_genes, n_samples)

# 添加一些模式，使某些基因相似
# 前20个基因有相似的模式
gene_data[:20] += np.random.randn(1, n_samples) * 0.5
# 中间15个基因有相似的模式
gene_data[20:35] += np.random.randn(1, n_samples) * 0.5
# 最后15个基因有相似的模式
gene_data[35:] += np.random.randn(1, n_samples) * 0.5

print(f"基因表达数据:")
print(f"  基因数: {n_genes}")
print(f"  样本数: {n_samples}")
print(f"  数据形状: {gene_data.shape}")

# 标准化数据
# 标准化每个基因的表达水平
scaler = StandardScaler()
gene_data_scaled = scaler.fit_transform(gene_data)

print("\n数据标准化完成！")

# ========== 2. 层次聚类 ==========
print("\n" + "=" * 60)
print("2. 层次聚类")
print("=" * 60)

# 使用层次聚类
# 将基因聚类，找到表达模式相似的基因
clustering = AgglomerativeClustering(
    n_clusters=3,      # 假设分成3个基因簇
    linkage='average'  # 使用平均链接
)

# 训练模型
print("训练模型...")
labels = clustering.fit_predict(gene_data_scaled)

print(f"\n聚类结果:")
print(f"  簇数: {len(np.unique(labels))}")
print(f"  各簇的基因数:")
for cluster_id in np.unique(labels):
    count = np.sum(labels == cluster_id)
    print(f"    簇 {cluster_id}: {count} 个基因")

# ========== 3. 绘制树状图 ==========
print("\n" + "=" * 60)
print("3. 绘制树状图")
print("=" * 60)

# 计算链接矩阵
Z = linkage(gene_data_scaled, method='average', metric='euclidean')

# 绘制树状图
plt.figure(figsize=(14, 8))
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=10,
    labels=[f'基因{i+1}' for i in range(n_genes)],
    truncate_mode='level',
    p=10
)
plt.title('基因聚类树状图', fontsize=16, fontweight='bold')
plt.xlabel('基因', fontsize=14)
plt.ylabel('距离', fontsize=14)
plt.tight_layout()
plt.savefig('gene_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 可视化聚类结果 ==========
print("\n" + "=" * 60)
print("4. 可视化聚类结果")
print("=" * 60)

# 可视化每个基因簇的表达模式
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for cluster_id in range(3):
    # 获取属于该簇的基因
    cluster_genes = gene_data_scaled[labels == cluster_id]
    
    # 绘制表达模式
    ax = axes[cluster_id]
    for gene in cluster_genes[:10]:  # 只显示前10个基因
        ax.plot(gene, alpha=0.3, linewidth=1)
    
    # 绘制平均表达模式
    if len(cluster_genes) > 0:
        mean_expression = np.mean(cluster_genes, axis=0)
        ax.plot(mean_expression, 'r-', linewidth=3, label='平均表达')
    
    ax.set_title(f'簇 {cluster_id} ({len(cluster_genes)} 个基因)', fontsize=14)
    ax.set_xlabel('样本', fontsize=12)
    ax.set_ylabel('标准化表达水平', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('gene_clustering_patterns.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 分析基因簇 ==========
print("\n" + "=" * 60)
print("5. 分析基因簇")
print("=" * 60)

print("\n各基因簇的特征:")
for cluster_id in range(3):
    cluster_genes = gene_data_scaled[labels == cluster_id]
    if len(cluster_genes) > 0:
        mean_expr = np.mean(cluster_genes, axis=0)
        std_expr = np.std(cluster_genes, axis=0)
        print(f"\n簇 {cluster_id} ({len(cluster_genes)} 个基因):")
        print(f"  平均表达水平: {mean_expr}")
        print(f"  表达变异性: {std_expr}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. 层次聚类可以用于基因表达数据分析
2. 树状图可以可视化基因的相似性
3. 相似的基因可能具有相似的功能
4. 可以用于发现基因调控网络
""")

