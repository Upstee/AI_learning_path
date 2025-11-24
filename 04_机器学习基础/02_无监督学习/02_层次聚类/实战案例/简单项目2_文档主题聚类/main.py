"""
简单项目2：文档主题聚类
使用层次聚类对文档进行主题聚类

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.cluster.hierarchy import dendrogram, linkage

# ========== 1. 准备文档数据 ==========
print("=" * 60)
print("1. 准备文档数据")
print("=" * 60)

# 创建简单的文档数据
# 在实际应用中，你需要从文件或数据库中加载真实的文档
documents = [
    "机器学习是人工智能的一个分支",
    "深度学习使用神经网络",
    "自然语言处理处理文本数据",
    "计算机视觉处理图像数据",
    "强化学习通过奖励学习",
    "监督学习使用标签数据",
    "无监督学习不使用标签",
    "聚类算法将数据分组",
    "分类算法预测类别",
    "回归算法预测数值",
    "神经网络模拟大脑",
    "卷积神经网络用于图像",
    "循环神经网络用于序列",
    "Transformer用于NLP",
    "BERT是预训练模型",
]

print(f"文档数: {len(documents)}")
print(f"\n前5个文档:")
for i, doc in enumerate(documents[:5]):
    print(f"  {i+1}. {doc}")

# ========== 2. 特征提取 ==========
print("\n" + "=" * 60)
print("2. 特征提取")
print("=" * 60)

# 使用TF-IDF提取特征
# TF-IDF将文本转换为数值特征
vectorizer = TfidfVectorizer(max_features=50)
X = vectorizer.fit_transform(documents)

print(f"特征矩阵形状: {X.shape}")
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")
print(f"前10个词: {list(vectorizer.vocabulary_.keys())[:10]}")

# 转换为密集矩阵（层次聚类需要）
X_dense = X.toarray()

# ========== 3. 层次聚类 ==========
print("\n" + "=" * 60)
print("3. 层次聚类")
print("=" * 60)

# 使用层次聚类
clustering = AgglomerativeClustering(
    n_clusters=3,      # 假设分成3个主题
    linkage='average'  # 使用平均链接
)

# 训练模型
print("训练模型...")
labels = clustering.fit_predict(X_dense)

print(f"\n聚类结果:")
print(f"  主题数: {len(np.unique(labels))}")
print(f"  各主题的文档数:")
for topic_id in np.unique(labels):
    count = np.sum(labels == topic_id)
    print(f"    主题 {topic_id}: {count} 个文档")

# ========== 4. 绘制树状图 ==========
print("\n" + "=" * 60)
print("4. 绘制树状图")
print("=" * 60)

# 计算链接矩阵
Z = linkage(X_dense, method='average', metric='euclidean')

# 绘制树状图
plt.figure(figsize=(14, 8))
dendrogram(
    Z,
    leaf_rotation=90,
    leaf_font_size=10,
    labels=[f'文档{i+1}' for i in range(len(documents))],
    truncate_mode='level',
    p=5
)
plt.title('文档聚类树状图', fontsize=16, fontweight='bold')
plt.xlabel('文档', fontsize=14)
plt.ylabel('距离', fontsize=14)
plt.tight_layout()
plt.savefig('document_clustering_dendrogram.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 分析每个主题 ==========
print("\n" + "=" * 60)
print("5. 分析每个主题")
print("=" * 60)

print("\n各主题的文档:")
for topic_id in np.unique(labels):
    topic_docs = [documents[i] for i in range(len(documents)) if labels[i] == topic_id]
    print(f"\n主题 {topic_id} ({len(topic_docs)} 个文档):")
    for i, doc in enumerate(topic_docs, 1):
        print(f"  {i}. {doc}")

# ========== 6. 可视化主题分布 ==========
print("\n" + "=" * 60)
print("6. 可视化主题分布")
print("=" * 60)

# 统计每个主题的文档数
topic_counts = [np.sum(labels == i) for i in range(3)]

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(range(3), topic_counts, color=['skyblue', 'lightgreen', 'lightcoral'], alpha=0.7)
plt.xlabel('主题', fontsize=12)
plt.ylabel('文档数', fontsize=12)
plt.title('各主题的文档数', fontsize=14)
plt.xticks(range(3), [f'主题{i}' for i in range(3)])
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('document_clustering_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. 层次聚类可以用于文档主题聚类
2. 树状图可以可视化文档的相似性
3. 相似的文档可能属于同一主题
4. 可以用于文档组织和检索
""")

