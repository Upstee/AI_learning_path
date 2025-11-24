# 层次聚类

## 1. 课程概述

### 课程目标
1. 理解层次聚类的基本原理和两种方法（凝聚、分裂）
2. 掌握不同链接准则（单链接、完全链接、平均链接）
3. 理解树状图（Dendrogram）的解读
4. 能够从零实现层次聚类算法（简化版）
5. 能够使用scikit-learn实现层次聚类
6. 掌握层次聚类的优缺点和适用场景

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 需要理解树结构和链接准则

### 课程定位
- **前置课程**：01_K-means、02_数学基础（线性代数）
- **后续课程**：03_DBSCAN、04_PCA
- **在体系中的位置**：不需要预设K值的聚类方法，可生成树状图

### 学完能做什么
- 能够理解和使用层次聚类进行数据聚类
- 能够从零实现层次聚类算法
- 能够解读树状图并选择合适的聚类数
- 能够理解不同链接准则的影响

---

## 2. 前置知识检查

### 必备前置概念清单
- **K-means**：理解聚类的基本概念
- **树结构**：理解树状结构
- **距离计算**：理解距离度量

### 回顾链接/跳转
- 如果不熟悉K-means：`04_机器学习基础/02_无监督学习/01_K-means/`
- 如果不熟悉距离计算：`02_数学基础/01_线性代数/`

### 入门小测

**选择题**（每题2分，共10分）

1. 层次聚类的主要类型是？
   A. 凝聚和分裂  B. 单链接和完全链接  C. 平均和加权  D. 向上和向下
   **答案**：A

2. 凝聚层次聚类从什么开始？
   A. 一个簇  B. 每个样本一个簇  C. K个簇  D. 随机簇
   **答案**：B

3. 单链接准则使用什么距离？
   A. 最近距离  B. 最远距离  C. 平均距离  D. 中心距离
   **答案**：A

4. 树状图（Dendrogram）显示什么？
   A. 聚类过程  B. 最终结果  C. 距离信息  D. 以上都是
   **答案**：D

5. 层次聚类的优点不包括？
   A. 不需要预设K值  B. 可生成树状图  C. 计算快速  D. 结果可解释
   **答案**：C（层次聚类计算复杂度较高）

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 层次聚类原理

#### 概念引入与直观类比

**类比**：层次聚类就像"家族树"，逐步合并或分裂形成层次结构。

- **凝聚方法**：从下往上，逐步合并
- **分裂方法**：从上往下，逐步分裂
- **树状图**：显示整个聚类过程

例如：
- 生物分类：从物种到门类的层次结构
- 组织架构：从个人到部门的层次结构

#### 逐步理论推导

**步骤1：凝聚层次聚类（自底向上）**

1. 初始化：每个样本一个簇
2. 计算簇间距离
3. 合并距离最近的两个簇
4. 重复步骤2-3，直到只剩一个簇

**步骤2：分裂层次聚类（自顶向下）**

1. 初始化：所有样本一个簇
2. 选择要分裂的簇
3. 将簇分裂成两个子簇
4. 重复步骤2-3，直到每个样本一个簇

**步骤3：链接准则**

- **单链接（Single Linkage）**：最近距离
  d(C₁, C₂) = min{d(x, y) | x ∈ C₁, y ∈ C₂}

- **完全链接（Complete Linkage）**：最远距离
  d(C₁, C₂) = max{d(x, y) | x ∈ C₁, y ∈ C₂}

- **平均链接（Average Linkage）**：平均距离
  d(C₁, C₂) = (1/|C₁||C₂|) ∑∑ d(x, y)

- **中心链接（Centroid Linkage）**：中心距离
  d(C₁, C₂) = d(μ₁, μ₂)

#### 数学公式与必要证明

**距离更新（Lance-Williams公式）**：

合并簇Cᵢ和Cⱼ后，新簇Cₖ与簇Cₕ的距离：
d(Cₖ, Cₕ) = αᵢd(Cᵢ, Cₕ) + αⱼd(Cⱼ, Cₕ) + βd(Cᵢ, Cⱼ) + γ|d(Cᵢ, Cₕ) - d(Cⱼ, Cₕ)|

不同链接准则的系数不同。

#### 算法伪代码

```
凝聚层次聚类算法：
1. 初始化：每个样本一个簇，计算距离矩阵
2. 重复直到只剩一个簇：
   a. 找到距离最近的两个簇C_i和C_j
   b. 合并C_i和C_j为新簇C_k
   c. 更新距离矩阵（使用链接准则）
   d. 记录合并信息
3. 根据合并信息构建树状图
4. 返回树状图和聚类结果
```

#### 关键性质

**优点**：
- **不需要预设K值**：可以生成任意数量的簇
- **可生成树状图**：可视化整个聚类过程
- **结果可解释**：层次结构清晰
- **确定性**：给定链接准则，结果确定

**缺点**：
- **计算复杂度高**：O(n³)或O(n²log n)
- **对噪声敏感**：噪声可能影响整个结构
- **不可逆**：一旦合并不能撤销
- **内存占用大**：需要存储距离矩阵

**适用场景**：
- 需要层次结构
- 数据量不大（<10000样本）
- 需要可视化聚类过程
- 需要探索不同K值

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy >= 1.20.0
  - pandas >= 1.3.0
  - matplotlib >= 3.3.0
  - scipy >= 1.7.0
  - scikit-learn >= 0.24.0

### 4.2 从零开始的完整可运行示例

#### 示例1：使用scipy（推荐）

```python
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X, y_true = make_blobs(n_samples=50, centers=3, n_features=2, random_state=42)

# 计算距离矩阵
distance_matrix = pdist(X, metric='euclidean')

# 执行层次聚类（使用不同链接准则）
linkage_methods = ['single', 'complete', 'average', 'ward']
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

for idx, method in enumerate(linkage_methods):
    # 计算链接矩阵
    Z = linkage(distance_matrix, method=method)
    
    # 绘制树状图
    ax = axes[idx // 2, idx % 2]
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8)
    ax.set_title(f'{method.capitalize()} Linkage')
    ax.set_xlabel('样本索引')
    ax.set_ylabel('距离')

plt.tight_layout()
plt.show()

# 使用Ward链接（通常效果最好）
Z_ward = linkage(distance_matrix, method='ward')

# 获取不同K值的聚类结果
for k in [2, 3, 4]:
    labels = fcluster(Z_ward, k, criterion='maxclust')
    print(f"K={k}时的聚类结果: {np.unique(labels, return_counts=True)}")
```

#### 示例2：使用scikit-learn

```python
from sklearn.cluster import AgglomerativeClustering
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X, y_true = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)

# 测试不同的链接准则和K值
linkage_methods = ['ward', 'complete', 'average']
k_values = [2, 3, 4, 5]

results = {}
for method in linkage_methods:
    for k in k_values:
        clustering = AgglomerativeClustering(n_clusters=k, linkage=method)
        labels = clustering.fit_predict(X)
        score = silhouette_score(X, labels)
        results[(method, k)] = score
        print(f"{method}, K={k}: 轮廓系数={score:.4f}")

# 使用最佳参数
best_params = max(results, key=results.get)
best_method, best_k = best_params
print(f"\n最佳参数: {best_method}, K={best_k}")

# 训练最终模型
clustering = AgglomerativeClustering(n_clusters=best_k, linkage=best_method)
labels = clustering.fit_predict(X)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title(f'层次聚类结果 ({best_method}, K={best_k})')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.colorbar(label='簇标签')
plt.show()
```

#### 示例3：从零实现（简化版）

```python
import numpy as np
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

class HierarchicalClustering:
    """层次聚类（从零实现，简化版）"""
    
    def __init__(self, n_clusters=2, linkage='complete'):
        self.n_clusters = n_clusters
        self.linkage = linkage
    
    def _single_linkage(self, dist_matrix, i, j, k):
        """单链接：最近距离"""
        return min(dist_matrix[i, k], dist_matrix[j, k])
    
    def _complete_linkage(self, dist_matrix, i, j, k):
        """完全链接：最远距离"""
        return max(dist_matrix[i, k], dist_matrix[j, k])
    
    def _average_linkage(self, dist_matrix, i, j, k, sizes):
        """平均链接：平均距离"""
        return (sizes[i] * dist_matrix[i, k] + sizes[j] * dist_matrix[j, k]) / (sizes[i] + sizes[j])
    
    def fit(self, X):
        """训练模型"""
        n_samples = X.shape[0]
        
        # 计算距离矩阵
        distances = pdist(X, metric='euclidean')
        dist_matrix = squareform(distances)
        
        # 初始化：每个样本一个簇
        clusters = [[i] for i in range(n_samples)]
        sizes = [1] * n_samples
        
        # 记录合并历史
        merge_history = []
        
        # 凝聚聚类
        while len(clusters) > self.n_clusters:
            # 找到距离最近的两个簇
            min_dist = np.inf
            merge_i, merge_j = -1, -1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    # 计算簇间距离
                    if self.linkage == 'single':
                        dist = min([dist_matrix[ci, cj] for ci in clusters[i] for cj in clusters[j]])
                    elif self.linkage == 'complete':
                        dist = max([dist_matrix[ci, cj] for ci in clusters[i] for cj in clusters[j]])
                    else:  # average
                        dist = np.mean([dist_matrix[ci, cj] for ci in clusters[i] for cj in clusters[j]])
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # 合并簇
            clusters[merge_i].extend(clusters[merge_j])
            clusters.pop(merge_j)
            sizes[merge_i] += sizes[merge_j]
            sizes.pop(merge_j)
            
            merge_history.append((merge_i, merge_j, min_dist))
        
        # 生成标签
        self.labels_ = np.zeros(n_samples, dtype=int)
        for label, cluster in enumerate(clusters):
            for idx in cluster:
                self.labels_[idx] = label
        
        return self

# 生成数据
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=30, centers=3, n_features=2, random_state=42)

# 训练模型
hc = HierarchicalClustering(n_clusters=3, linkage='complete')
hc.fit(X)

# 可视化
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('真实标签')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=hc.labels_, cmap='viridis', alpha=0.6)
plt.title('层次聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.tight_layout()
plt.show()
```

### 4.3 常见错误与排查

**错误1**：数据量太大
```python
# 错误：数据量太大，计算慢
hc = AgglomerativeClustering(n_clusters=3)
hc.fit(X_large)  # X_large有100000个样本

# 正确：使用采样或Mini-batch方法
from sklearn.cluster import MiniBatchKMeans
kmeans = MiniBatchKMeans(n_clusters=3)
kmeans.fit(X_large)
```

**错误2**：链接准则选择不当
```python
# 问题：不同链接准则适合不同数据
# 解决：尝试不同链接准则，选择最佳
for method in ['ward', 'complete', 'average']:
    hc = AgglomerativeClustering(n_clusters=3, linkage=method)
    labels = hc.fit_predict(X)
    score = silhouette_score(X, labels)
    print(f"{method}: {score:.4f}")
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：使用scipy实现层次聚类**
使用scipy实现层次聚类并绘制树状图。

**练习2：对比不同链接准则**
对比单链接、完全链接、平均链接的效果。

**练习3：K值选择**
从树状图中选择合适的K值。

### 进阶练习（2-3题）

**练习1：从零实现层次聚类**
从零实现层次聚类算法（简化版）。

**练习2：大规模数据优化**
使用采样或近似方法处理大规模数据。

### 挑战练习（1-2题）

**练习1：完整的聚类系统**
实现完整的聚类系统，包括数据预处理、链接准则选择、K值选择、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：基因表达数据分析

**业务背景**：
根据基因表达数据对样本进行聚类，发现不同的样本类型。

**问题抽象**：
- 特征：基因表达值（高维）
- 目标：将样本分成不同群体
- 方法：层次聚类

**端到端实现**：
```python
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 创建模拟数据
np.random.seed(42)
n_samples = 50
n_features = 100

# 生成3个不同的样本群体
X = np.zeros((n_samples, n_features))
for i in range(3):
    start = i * 16
    end = (i + 1) * 16 if i < 2 else n_samples
    X[start:end] = np.random.normal(i * 2, 1, (end - start, n_features))

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 降维可视化（可选）
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# 计算距离矩阵
distance_matrix = pdist(X_scaled, metric='euclidean')

# 层次聚类
Z = linkage(distance_matrix, method='ward')

# 绘制树状图
plt.figure(figsize=(15, 8))
dendrogram(Z, leaf_rotation=90, leaf_font_size=8)
plt.title('层次聚类树状图')
plt.xlabel('样本索引')
plt.ylabel('距离')
plt.show()

# 选择K=3
labels = fcluster(Z, 3, criterion='maxclust')

# 可视化（2D投影）
plt.figure(figsize=(10, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.6)
plt.title('层次聚类结果（PCA投影）')
plt.xlabel('第一主成分')
plt.ylabel('第二主成分')
plt.colorbar(label='簇标签')
plt.show()

# 分析每个簇的特征
print("\n各簇特征:")
for i in range(1, 4):
    cluster_data = X_scaled[labels == i]
    print(f"\n簇{i} (共{len(cluster_data)}个样本):")
    print(f"  平均表达值: {cluster_data.mean(axis=0)[:5]}...")  # 只显示前5个特征
```

**结果解读**：
- 层次聚类成功识别出不同的样本群体
- 树状图显示了聚类的层次结构

**改进方向**：
- 使用更多特征
- 尝试其他聚类算法
- 结合领域知识

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 层次聚类的主要类型是？
   A. 凝聚和分裂  B. 单链接和完全链接  C. 平均和加权  D. 向上和向下
   **答案**：A

2. 凝聚层次聚类从什么开始？
   A. 一个簇  B. 每个样本一个簇  C. K个簇  D. 随机簇
   **答案**：B

3. 单链接准则使用什么距离？
   A. 最近距离  B. 最远距离  C. 平均距离  D. 中心距离
   **答案**：A

4. 树状图（Dendrogram）显示什么？
   A. 聚类过程  B. 最终结果  C. 距离信息  D. 以上都是
   **答案**：D

5. 层次聚类的优点不包括？
   A. 不需要预设K值  B. 可生成树状图  C. 计算快速  D. 结果可解释
   **答案**：C

**简答题**（每题10分，共40分）

1. 解释凝聚和分裂层次聚类的区别。
   **参考答案**：凝聚从每个样本一个簇开始，逐步合并；分裂从所有样本一个簇开始，逐步分裂。

2. 说明不同链接准则的特点。
   **参考答案**：单链接用最近距离，适合链状簇；完全链接用最远距离，适合紧凑簇；平均链接用平均距离，平衡两者。

3. 解释如何从树状图选择K值。
   **参考答案**：在树状图中找到距离变化大的位置（"切割点"），对应的K值通常是合适的聚类数。

4. 说明层次聚类的优缺点。
   **参考答案**：优点：不需要预设K值、可生成树状图、结果可解释；缺点：计算复杂度高、对噪声敏感、内存占用大。

### 编程实践题（20分）

使用scipy实现层次聚类，包括不同链接准则和树状图绘制。

### 综合应用题（20分）

使用层次聚类解决真实问题，包括数据预处理、链接准则选择、K值选择、模型训练、评估、可视化。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第9章）
- 《数据挖掘：概念与技术》- Han等

**在线资源**：
- scipy官方文档
- scikit-learn官方文档

### 相关工具与库

- **scipy**：linkage, dendrogram, fcluster
- **scikit-learn**：AgglomerativeClustering
- **matplotlib**：可视化

### 进阶话题指引

完成本课程后，可以学习：
- **分裂层次聚类**：自顶向下的方法
- **大规模优化**：处理大规模数据的近似方法
- **其他聚类算法**：DBSCAN、谱聚类

### 下节课预告

下一课将学习：
- **03_DBSCAN**：基于密度的聚类算法
- DBSCAN可以发现任意形状的簇，不需要预设K值

### 学习建议

1. **理解树结构**：理解层次聚类的树状结构
2. **多实践**：尝试不同链接准则和K值
3. **可视化**：绘制树状图，理解聚类过程
4. **持续学习**：层次聚类是重要的聚类方法

---

**恭喜完成第二课！你已经掌握了层次聚类，准备好学习DBSCAN了！**

