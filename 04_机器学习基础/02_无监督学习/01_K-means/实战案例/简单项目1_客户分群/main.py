"""
简单项目1：客户分群
使用K-means对客户进行分群

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import seaborn as sns

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成模拟客户数据
# 在实际应用中，你需要从数据库或文件中加载真实数据
np.random.seed(42)

# 创建客户数据
# 特征：年龄、年收入、消费金额
n_customers = 200
data = {
    '年龄': np.random.randint(20, 70, n_customers),
    '年收入': np.random.randint(20000, 150000, n_customers),
    '消费金额': np.random.randint(1000, 50000, n_customers)
}

df = pd.DataFrame(data)

print(f"客户数据:")
print(f"  客户数: {len(df)}")
print(f"  特征: {list(df.columns)}")
print(f"\n数据预览:")
print(df.head())

# ========== 2. 数据预处理 ==========
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

# 提取特征
X = df.values

# 标准化特征
# K-means对特征尺度敏感，需要标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("标准化完成！")
print(f"标准化前（前5个样本）:")
print(X[:5])
print(f"\n标准化后（前5个样本）:")
print(X_scaled[:5])

# ========== 3. 选择最优K值 ==========
print("\n" + "=" * 60)
print("3. 选择最优K值")
print("=" * 60)

# 使用肘部法则和轮廓系数选择K值
k_range = range(2, 11)
inertias = []
silhouette_scores = []

print("\n测试不同的K值:")
for k in k_range:
    kmeans_k = KMeans(n_clusters=k, random_state=42)
    labels_k = kmeans_k.fit_predict(X_scaled)
    inertias.append(kmeans_k.inertia_)
    score = silhouette_score(X_scaled, labels_k)
    silhouette_scores.append(score)
    if k <= 5 or k % 2 == 0:
        print(f"  K={k}: WCSS={kmeans_k.inertia_:.4f}, 轮廓系数={score:.4f}")

# 找到最佳K值（轮廓系数最大）
best_k = k_range[np.argmax(silhouette_scores)]
best_score = max(silhouette_scores)
print(f"\n最佳K值: {best_k}, 轮廓系数: {best_score:.4f}")

# 可视化K值选择
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：肘部法则
axes[0].plot(k_range, inertias, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('K值', fontsize=12)
axes[0].set_ylabel('簇内平方和 (WCSS)', fontsize=12)
axes[0].set_title('肘部法则', fontsize=14)
axes[0].grid(True, alpha=0.3)

# 右图：轮廓系数
axes[1].plot(k_range, silhouette_scores, 'o-', linewidth=2, markersize=8, color='green')
axes[1].axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'最佳K={best_k}')
axes[1].set_xlabel('K值', fontsize=12)
axes[1].set_ylabel('轮廓系数', fontsize=12)
axes[1].set_title('轮廓系数', fontsize=14)
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('customer_clustering_k_selection.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 4. 训练模型 ==========
print("\n" + "=" * 60)
print("4. 训练K-means模型")
print("=" * 60)

# 使用最佳K值训练模型
kmeans = KMeans(n_clusters=best_k, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# 添加聚类标签到数据框
df['客户群体'] = labels

print(f"训练完成！")
print(f"  簇数: {best_k}")
print(f"  簇内平方和: {kmeans.inertia_:.4f}")

# ========== 5. 分析客户群体 ==========
print("\n" + "=" * 60)
print("5. 分析客户群体")
print("=" * 60)

# 分析每个群体的特征
print("\n各客户群体特征:")
for cluster_id in range(best_k):
    cluster_data = df[df['客户群体'] == cluster_id]
    print(f"\n群体 {cluster_id} ({len(cluster_data)} 人):")
    print(f"  平均年龄: {cluster_data['年龄'].mean():.1f} 岁")
    print(f"  平均年收入: {cluster_data['年收入'].mean():.0f} 元")
    print(f"  平均消费金额: {cluster_data['消费金额'].mean():.0f} 元")

# ========== 6. 可视化结果 ==========
print("\n" + "=" * 60)
print("6. 可视化结果")
print("=" * 60)

# 创建特征对的可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 年龄 vs 年收入
axes[0, 0].scatter(df['年龄'], df['年收入'], c=df['客户群体'], cmap='viridis', alpha=0.6, s=50)
axes[0, 0].set_xlabel('年龄', fontsize=12)
axes[0, 0].set_ylabel('年收入', fontsize=12)
axes[0, 0].set_title('年龄 vs 年收入', fontsize=14)
axes[0, 0].grid(True, alpha=0.3)

# 年龄 vs 消费金额
axes[0, 1].scatter(df['年龄'], df['消费金额'], c=df['客户群体'], cmap='viridis', alpha=0.6, s=50)
axes[0, 1].set_xlabel('年龄', fontsize=12)
axes[0, 1].set_ylabel('消费金额', fontsize=12)
axes[0, 1].set_title('年龄 vs 消费金额', fontsize=14)
axes[0, 1].grid(True, alpha=0.3)

# 年收入 vs 消费金额
axes[1, 0].scatter(df['年收入'], df['消费金额'], c=df['客户群体'], cmap='viridis', alpha=0.6, s=50)
axes[1, 0].set_xlabel('年收入', fontsize=12)
axes[1, 0].set_ylabel('消费金额', fontsize=12)
axes[1, 0].set_title('年收入 vs 消费金额', fontsize=14)
axes[1, 0].grid(True, alpha=0.3)

# 各群体人数分布
cluster_counts = df['客户群体'].value_counts().sort_index()
axes[1, 1].bar(cluster_counts.index, cluster_counts.values, color='skyblue', alpha=0.7)
axes[1, 1].set_xlabel('客户群体', fontsize=12)
axes[1, 1].set_ylabel('人数', fontsize=12)
axes[1, 1].set_title('各群体人数分布', fontsize=14)
axes[1, 1].grid(True, axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('customer_clustering_results.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. K-means可以用于客户分群
2. 需要选择合适的K值
3. 特征标准化很重要
4. 可以分析每个客户群体的特征，制定不同的营销策略
""")

