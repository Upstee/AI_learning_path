# 层次聚类常见问题FAQ

> **目的**：快速解决学习过程中的常见问题，避免卡住

---

## 📚 目录

- [概念理解问题](#概念理解问题)
- [代码实现问题](#代码实现问题)
- [参数调优问题](#参数调优问题)
- [实际应用问题](#实际应用问题)
- [错误排查](#错误排查)

---

## 概念理解问题

### Q1: 层次聚类和K-means有什么区别？

**A**: 主要区别：

| 特性 | 层次聚类 | K-means |
|------|---------|---------|
| **K值** | 不需要预设，可以从树状图选择 | 需要预设K值 |
| **计算速度** | 慢，O(n³) | 快，O(n) |
| **结果** | 树状图，可以看层次结构 | 平面结果 |
| **适用数据量** | 小到中等（<10000） | 大数据 |
| **可逆性** | 可以看完整合并过程 | 不可逆 |

**选择建议**：
- 数据量小，需要树状图 → 层次聚类
- 数据量大，需要快速聚类 → K-means
- 不知道K值 → 层次聚类（从树状图选择）

---

### Q2: 凝聚聚类和分裂聚类有什么区别？

**A**: 

**凝聚聚类（Agglomerative）**：
- **方向**：自底向上
- **开始**：每个点是一个簇
- **过程**：逐步合并最近的簇
- **结束**：所有点合并成一个簇（或达到K值）
- **常用**：✅ 更常用，实现简单

**分裂聚类（Divisive）**：
- **方向**：自顶向下
- **开始**：所有点在一个簇
- **过程**：逐步分裂成更小的簇
- **结束**：每个点是一个簇（或达到K值）
- **常用**：❌ 较少使用，实现复杂

**实际应用**：通常使用凝聚聚类。

---

### Q3: 什么是链接准则（Linkage）？

**A**: 链接准则决定**如何计算两个簇之间的距离**：

**常见方法**：

1. **Ward（最小方差）**：
   - 合并后簇内方差增加最小的两个簇
   - 通常产生紧凑的球形簇
   - ✅ 最常用

2. **Complete（完全链接）**：
   - 两个簇中**最远两点**的距离
   - 对异常值更鲁棒
   - 可能产生紧凑但分离的簇

3. **Average（平均链接）**：
   - 两个簇中**所有点对**的平均距离
   - 平衡的方法
   - 介于ward和complete之间

4. **Single（单链接）**：
   - 两个簇中**最近两点**的距离
   - 可能产生链状簇（chaining）
   - ⚠️ 对噪声敏感

**选择建议**：
- 默认使用 **ward**
- 数据有异常值 → **complete**
- 需要平衡 → **average**
- 避免使用 **single**（除非特殊需求）

---

### Q4: 如何从树状图选择K值？

**A**: 三种方法：

#### 方法1：观察距离变化（推荐）

```python
# 计算链接矩阵
linkage_matrix = linkage(X, method='ward')

# 提取距离信息
distances = linkage_matrix[:, 2]  # 第3列是距离

# 计算距离变化率
distance_changes = np.diff(distances)

# 找到变化最大的点（肘部点）
elbow_point = np.argmax(distance_changes)
k_value = len(X) - elbow_point - 1

print(f"建议K值: {k_value}")
```

#### 方法2：可视化选择

```python
# 绘制树状图
dendrogram(linkage_matrix)

# 在图上画水平线，观察切割点
plt.axhline(y=threshold, color='r', linestyle='--')
plt.show()

# 根据切割点数量确定K值
```

#### 方法3：使用轮廓系数

```python
from sklearn.metrics import silhouette_score

k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    clustering = AgglomerativeClustering(n_clusters=k, linkage='ward')
    labels = clustering.fit_predict(X)
    score = silhouette_score(X, labels)
    silhouette_scores.append(score)

best_k = k_range[np.argmax(silhouette_scores)]
print(f"最佳K值: {best_k}")
```

---

### Q5: 层次聚类的时间复杂度是多少？

**A**: 

**时间复杂度**：O(n³) 或 O(n² log n)

- **n³**：朴素实现
- **n² log n**：使用优先队列优化

**空间复杂度**：O(n²)（存储距离矩阵）

**对比**：
- K-means：O(n)（线性）
- 层次聚类：O(n³)（立方）

**影响**：
- 数据量 < 1000：层次聚类可以接受
- 数据量 1000-10000：层次聚类较慢，但可用
- 数据量 > 10000：建议使用K-means或DBSCAN

---

## 代码实现问题

### Q6: 如何从零实现层次聚类？

**A**: 核心步骤：

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform

def agglomerative_clustering(X, n_clusters, linkage='ward'):
    """从零实现凝聚层次聚类"""
    n_samples = len(X)
    
    # 1. 初始化：每个点是一个簇
    clusters = [[i] for i in range(n_samples)]
    
    # 2. 计算距离矩阵
    distances = squareform(pdist(X))
    
    # 3. 逐步合并
    while len(clusters) > n_clusters:
        # 找到最近的两个簇
        min_dist = float('inf')
        merge_i, merge_j = -1, -1
        
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                # 计算簇间距离（根据链接方法）
                dist = compute_cluster_distance(
                    clusters[i], clusters[j], distances, linkage
                )
                if dist < min_dist:
                    min_dist = dist
                    merge_i, merge_j = i, j
        
        # 合并两个簇
        clusters[merge_i].extend(clusters[merge_j])
        clusters.pop(merge_j)
    
    # 4. 生成标签
    labels = np.zeros(n_samples)
    for cluster_id, cluster in enumerate(clusters):
        for point_id in cluster:
            labels[point_id] = cluster_id
    
    return labels.astype(int)

def compute_cluster_distance(cluster1, cluster2, distances, linkage):
    """计算两个簇之间的距离"""
    if linkage == 'single':
        # 单链接：最近距离
        return min(distances[i, j] for i in cluster1 for j in cluster2)
    elif linkage == 'complete':
        # 完全链接：最远距离
        return max(distances[i, j] for i in cluster1 for j in cluster2)
    elif linkage == 'average':
        # 平均链接：平均距离
        return np.mean([distances[i, j] for i in cluster1 for j in cluster2])
    # ward方法更复杂，这里简化
```

---

### Q7: 如何绘制树状图？

**A**: 使用scipy：

```python
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt

# 1. 计算链接矩阵
linkage_matrix = linkage(X, method='ward')

# 2. 绘制树状图
plt.figure(figsize=(12, 8))
dendrogram(linkage_matrix,
           truncate_mode='level',  # 截断模式
           p=5,                    # 显示5层
           show_leaf_counts=True,  # 显示叶子数量
           leaf_font_size=10)      # 叶子字体大小
plt.title('层次聚类树状图')
plt.xlabel('样本索引或（簇大小）')
plt.ylabel('距离')
plt.show()
```

**参数说明**：
- `truncate_mode='level'`：按层级截断
- `p=5`：显示5层
- `show_leaf_counts=True`：显示每个簇的大小

---

### Q8: 如何处理大数据集？

**A**: 层次聚类对大数据集很慢，可以使用：

#### 方法1：采样

```python
from sklearn.utils import resample

# 采样到1000个点
X_sampled = resample(X, n_samples=1000, random_state=42)

# 在采样数据上聚类
clustering = AgglomerativeClustering(n_clusters=5)
labels_sampled = clustering.fit_predict(X_sampled)

# 对剩余数据分配标签（使用KNN）
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_sampled, labels_sampled)
labels_all = knn.predict(X)
```

#### 方法2：使用其他算法

```python
# 大数据集使用K-means或DBSCAN
from sklearn.cluster import KMeans, DBSCAN

# K-means（快速）
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X)

# DBSCAN（不需要K值）
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)
```

---

## 参数调优问题

### Q9: 如何选择链接方法？

**A**: 选择建议：

| 场景 | 推荐方法 | 原因 |
|------|---------|------|
| **默认情况** | ward | 通常效果最好 |
| **数据有异常值** | complete | 对异常值鲁棒 |
| **需要平衡** | average | 平衡的方法 |
| **特殊需求** | single | 可能产生链状簇 |

**实验方法**：
```python
methods = ['ward', 'complete', 'average', 'single']
for method in methods:
    clustering = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = clustering.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"{method}: {score:.4f}")
```

---

### Q10: 如何加速层次聚类？

**A**: 优化方法：

1. **减少数据量**：采样或降维
2. **使用近似方法**：FastCluster库
3. **并行计算**：使用多进程
4. **提前停止**：达到目标K值后停止

```python
# 使用FastCluster（如果可用）
try:
    import fastcluster
    linkage_matrix = fastcluster.linkage(X, method='ward')
except ImportError:
    from scipy.cluster.hierarchy import linkage
    linkage_matrix = linkage(X, method='ward')
```

---

## 实际应用问题

### Q11: 层次聚类适合处理什么类型的数据？

**A**: 

**适合**：
- ✅ **小到中等数据量**（<10000）
- ✅ **需要树状图**：生物信息学、系统发育
- ✅ **不知道K值**：可以从树状图选择
- ✅ **需要层次结构**：文档分类、基因分析

**不适合**：
- ❌ **大数据量**（>10000）：计算太慢
- ❌ **需要快速结果**：K-means更快
- ❌ **高维数据**：距离计算不准确

---

### Q12: 层次聚类在哪些实际场景中应用？

**A**: 

**常见应用**：

1. **生物信息学**：
   - 基因聚类
   - 蛋白质分类
   - 系统发育树

2. **文档分析**：
   - 文档主题聚类
   - 文本分类

3. **社交网络**：
   - 社区发现
   - 用户分群

4. **图像分析**：
   - 图像分割
   - 特征聚类

更多场景请参考：[实战场景库.md](./实战场景库.md)

---

## 错误排查

### Q13: 报错"MemoryError"

**A**: 

**原因**：距离矩阵太大，内存不足

**解决**：
```python
# 1. 减少数据量
X_sampled = X[:1000]  # 只使用1000个样本

# 2. 使用采样方法
from sklearn.utils import resample
X_sampled = resample(X, n_samples=1000)

# 3. 使用其他算法（K-means）
kmeans = KMeans(n_clusters=5)
labels = kmeans.fit_predict(X)
```

---

### Q14: 树状图太复杂，看不清

**A**: 

**解决**：
```python
# 1. 截断树状图
dendrogram(linkage_matrix, 
           truncate_mode='level',  # 按层级截断
           p=5)                    # 只显示5层

# 2. 只显示部分叶子
dendrogram(linkage_matrix,
           truncate_mode='lastp',  # 只显示最后p个合并
           p=10)                   # 显示最后10个

# 3. 增大图像尺寸
plt.figure(figsize=(20, 10))
```

---

### Q15: 聚类结果不理想

**A**: 

**可能原因和解决方案**：

1. **链接方法不当**
   - 解决：尝试不同的链接方法（ward、complete、average）

2. **K值选择不当**
   - 解决：从树状图重新选择K值，或使用轮廓系数

3. **数据不适合层次聚类**
   - 解决：检查数据量，大数据集考虑K-means

4. **数据未标准化**
   - 解决：先标准化数据

---

## 📖 更多资源

- **理论详解**：[理论笔记/层次聚类原理详解.md](./理论笔记/层次聚类原理详解.md)
- **代码示例**：[代码示例/](./代码示例/)
- **实战案例**：[实战案例/](./实战案例/)
- **学习检查点**：[学习检查点.md](./学习检查点.md)

---

**如果这里没有你遇到的问题，请查看其他资源或继续学习！** 💪
