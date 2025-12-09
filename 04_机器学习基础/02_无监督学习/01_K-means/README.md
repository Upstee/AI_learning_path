# K-means聚类

> **🎯 快速开始**：如果你是第一次学习K-means，建议先完成[快速上手](./00_快速上手.md)（30分钟），快速体验效果，建立学习信心！

---

## 📚 学习资源导航

- **[快速上手](./00_快速上手.md)** - 30分钟快速体验，建立学习信心
- **[学习检查点](./学习检查点.md)** - 自我评估，确保真正掌握
- **[常见问题FAQ](./常见问题FAQ.md)** - 快速解决学习中的问题
- **[实战场景库](./实战场景库.md)** - 真实业务场景应用案例

---

## 1. 课程概述

### 课程目标
1. 理解K-means聚类的基本原理和算法流程
2. 掌握K值选择的方法（肘部法则、轮廓系数）
3. 理解K-means的优缺点和局限性
4. 能够从零实现K-means算法
5. 能够使用scikit-learn实现K-means
6. 掌握K-means的改进方法（K-means++、Mini-batch K-means）

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 算法简单但需要理解优化过程

### 课程定位
- **前置课程**：02_数学基础（线性代数、优化理论）、03_数据处理基础
- **后续课程**：02_层次聚类、03_DBSCAN
- **在体系中的位置**：最常用的聚类算法，简单高效

### 学完能做什么
- 能够理解和使用K-means进行数据聚类
- 能够从零实现K-means算法
- 能够选择合适的K值
- 能够处理聚类结果和可视化

---

## 2. 前置知识检查

### 必备前置概念清单
- **线性代数**：向量、距离
- **优化理论**：迭代优化
- **NumPy**：数组操作

### 回顾链接/跳转
- 如果不熟悉距离计算：`02_数学基础/01_线性代数/`
- 如果不熟悉NumPy：`03_数据处理基础/01_NumPy/`

### 入门小测

**选择题**（每题2分，共10分）

1. K-means是什么类型的算法？
   A. 监督学习  B. 无监督学习  C. 强化学习  D. 深度学习
   **答案**：B

2. K-means的目标是？
   A. 最小化类内距离  B. 最大化类间距离  C. 最小化误差  D. A和C
   **答案**：D

3. K-means算法的停止条件是？
   A. 达到最大迭代次数  B. 质心不再变化  C. 误差不再减小  D. 以上都可以
   **答案**：D

4. K值的选择方法不包括？
   A. 肘部法则  B. 轮廓系数  C. 交叉验证  D. 网格搜索
   **答案**：C（无监督学习没有标签，无法交叉验证）

5. K-means的缺点不包括？
   A. 需要预设K值  B. 对初始值敏感  C. 只能处理球形簇  D. 计算复杂度高
   **答案**：D（K-means计算复杂度相对较低）

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 K-means原理

#### 概念引入与直观类比

**类比**：K-means就像"找K个中心点"，把所有数据点分配到最近的中心点。

- **中心点（质心）**：每个簇的中心
- **分配**：每个点分配到最近的质心
- **更新**：重新计算质心位置
- **重复**：直到质心不再变化

例如：
- 客户分群：根据消费行为将客户分成K类
- 图像压缩：用K种颜色代表所有颜色

#### 逐步理论推导

**步骤1：初始化**

随机选择K个初始质心：
μ₁, μ₂, ..., μₖ

**步骤2：分配**

将每个数据点分配到最近的质心：
c⁽ⁱ⁾ = argminⱼ ||x⁽ⁱ⁾ - μⱼ||²

**步骤3：更新**

重新计算每个簇的质心：
μⱼ = (1/|Cⱼ|) ∑x⁽ⁱ⁾∈Cⱼ x⁽ⁱ⁾

**步骤4：重复**

重复步骤2和3，直到：
- 质心不再变化，或
- 达到最大迭代次数，或
- 误差不再减小

#### 数学公式与必要证明

**目标函数（误差平方和）**：

J = ∑ᵢ₌₁ⁿ ∑ⱼ₌₁ᵏ wᵢⱼ ||x⁽ⁱ⁾ - μⱼ||²

其中wᵢⱼ = 1如果x⁽ⁱ⁾属于簇j，否则为0。

**K-means是最小化J的迭代算法**：

1. 固定μⱼ，最小化J关于wᵢⱼ → 分配步骤
2. 固定wᵢⱼ，最小化J关于μⱼ → 更新步骤

#### 算法伪代码

```
K-means算法：
1. 初始化K个质心（随机选择或K-means++）
2. 重复直到收敛：
   a. 分配：将每个数据点分配到最近的质心
      c^(i) = argmin_j ||x^(i) - μ_j||^2
   b. 更新：重新计算每个簇的质心
      μ_j = (1/|C_j|) Σ x^(i) for x^(i) in C_j
   c. 计算误差：J = Σ ||x^(i) - μ_{c^(i)}||^2
   d. 如果误差不再减小或达到最大迭代次数，停止
3. 返回簇分配和质心
```

#### 关键性质

**优点**：
- **简单高效**：算法简单，计算快速
- **可扩展**：适合大规模数据
- **广泛应用**：应用广泛，易于理解

**缺点**：
- **需要预设K值**：不知道K值需要尝试
- **对初始值敏感**：不同初始值可能得到不同结果
- **只能处理球形簇**：假设簇是球形的
- **对异常值敏感**：异常值影响质心计算

**适用场景**：
- 数据有明显的簇结构
- 簇是球形的
- 需要快速聚类
- 数据量大

---

### 3.2 K值选择

#### 肘部法则（Elbow Method）

**思想**：随着K增加，误差会减小，但减小速度会变慢。找到"肘部"点。

**方法**：
1. 计算不同K值的误差（SSE）
2. 绘制K-SSE曲线
3. 找到"肘部"点（误差下降速度变慢的点）

#### 轮廓系数（Silhouette Score）

**思想**：衡量样本与其所在簇的相似度。

**公式**：
s(i) = (b(i) - a(i)) / max(a(i), b(i))

其中：
- a(i)：样本i到同簇其他样本的平均距离
- b(i)：样本i到最近其他簇的平均距离

**范围**：[-1, 1]，越接近1越好。

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy >= 1.20.0
  - pandas >= 1.3.0
  - matplotlib >= 3.3.0
  - scikit-learn >= 0.24.0

### 4.2 从零开始的完整可运行示例

#### 示例1：从零实现K-means

```python
import numpy as np
import matplotlib.pyplot as plt

class KMeans:
    """K-means聚类（从零实现）"""
    
    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.random_state = random_state
        self.centroids = None
        self.labels = None
    
    def _initialize_centroids(self, X):
        """初始化质心（随机选择）"""
        np.random.seed(self.random_state)
        n_samples, n_features = X.shape
        indices = np.random.choice(n_samples, self.k, replace=False)
        return X[indices]
    
    def _assign_clusters(self, X, centroids):
        """分配数据点到最近的质心"""
        distances = np.sqrt(((X - centroids[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    
    def _update_centroids(self, X, labels):
        """更新质心"""
        centroids = np.zeros((self.k, X.shape[1]))
        for i in range(self.k):
            centroids[i] = X[labels == i].mean(axis=0)
        return centroids
    
    def _compute_sse(self, X, labels, centroids):
        """计算误差平方和"""
        sse = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            sse += np.sum((cluster_points - centroids[i])**2)
        return sse
    
    def fit(self, X):
        """训练模型"""
        # 初始化质心
        self.centroids = self._initialize_centroids(X)
        
        for iteration in range(self.max_iters):
            # 分配
            old_labels = self.labels
            self.labels = self._assign_clusters(X, self.centroids)
            
            # 检查收敛
            if old_labels is not None and np.all(old_labels == self.labels):
                break
            
            # 更新质心
            new_centroids = self._update_centroids(X, self.labels)
            
            # 检查质心变化
            if np.allclose(self.centroids, new_centroids, atol=self.tol):
                break
            
            self.centroids = new_centroids
        
        return self
    
    def predict(self, X):
        """预测新数据点的簇"""
        return self._assign_clusters(X, self.centroids)

# 生成数据
from sklearn.datasets import make_blobs
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# 训练模型
kmeans = KMeans(k=3, max_iters=100, random_state=42)
kmeans.fit(X)

# 预测
y_pred = kmeans.labels

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[:, 0], X[:, 1], c=y_true, cmap='viridis', alpha=0.6)
plt.title('真实标签')
plt.xlabel('特征1')
plt.ylabel('特征2')

plt.subplot(1, 2, 2)
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.centroids[:, 0], kmeans.centroids[:, 1], 
           c='red', marker='x', s=200, linewidths=3, label='质心')
plt.title('K-means聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()

plt.tight_layout()
plt.show()

print(f"质心位置:\n{kmeans.centroids}")
```

#### 示例2：使用scikit-learn

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# 训练模型
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X)

# 预测
y_pred = kmeans.labels_

# 评估
silhouette = silhouette_score(X, y_pred)
print(f"轮廓系数: {silhouette:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='质心')
plt.title('K-means聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.show()
```

#### 示例3：K值选择（肘部法则）

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
X, _ = make_blobs(n_samples=300, centers=4, n_features=2, random_state=42)

# 测试不同的K值
k_range = range(1, 11)
sse = []
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X)
    sse.append(kmeans.inertia_)
    if k > 1:  # 轮廓系数需要至少2个簇
        silhouette_scores.append(silhouette_score(X, kmeans.labels_))
    else:
        silhouette_scores.append(0)

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 肘部法则
axes[0].plot(k_range, sse, marker='o')
axes[0].set_xlabel('K值')
axes[0].set_ylabel('误差平方和 (SSE)')
axes[0].set_title('肘部法则')
axes[0].grid(True)

# 轮廓系数
axes[1].plot(k_range, silhouette_scores, marker='o')
axes[1].set_xlabel('K值')
axes[1].set_ylabel('轮廓系数')
axes[1].set_title('轮廓系数')
axes[1].grid(True)

plt.tight_layout()
plt.show()

# 找出最佳K值
best_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"根据轮廓系数，最佳K值: {best_k_silhouette}")
```

#### 示例4：K-means++初始化

```python
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

# 生成数据
X, y_true = make_blobs(n_samples=300, centers=3, n_features=2, random_state=42)

# 使用K-means++初始化（默认）
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42, n_init=10)
kmeans.fit(X)

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X[:, 0], X[:, 1], c=kmeans.labels_, cmap='viridis', alpha=0.6)
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
           c='red', marker='x', s=200, linewidths=3, label='质心')
plt.title('K-means++聚类结果')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.legend()
plt.show()
```

### 4.3 常见错误与排查

**错误1**：K值选择不当
```python
# 错误：K值太大或太小
kmeans = KMeans(n_clusters=100)  # K太大，过拟合
kmeans = KMeans(n_clusters=1)    # K太小，欠拟合

# 正确：使用肘部法则或轮廓系数选择K值
```

**错误2**：未标准化特征
```python
# 错误：特征量纲不同，距离计算不准确
kmeans.fit(X)  # X未标准化

# 正确：先标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
kmeans.fit(X_scaled)
```

**错误3**：初始值敏感
```python
# 问题：不同初始值可能得到不同结果
# 解决：使用K-means++或多次运行取最佳
kmeans = KMeans(n_clusters=3, init='k-means++', n_init=10)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现K-means**
不使用库，从零实现K-means算法。

**练习2：K值选择**
使用肘部法则和轮廓系数选择最佳K值。

**练习3：可视化聚类结果**
可视化不同K值的聚类结果。

### 进阶练习（2-3题）

**练习1：K-means++初始化**
实现K-means++初始化方法。

**练习2：Mini-batch K-means**
使用Mini-batch K-means处理大规模数据。

### 挑战练习（1-2题）

**练习1：完整的聚类系统**
实现完整的聚类系统，包括数据预处理、K值选择、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：客户分群系统

**业务背景**：
根据客户消费行为将客户分成不同群体。

**问题抽象**：
- 特征：消费金额、消费频率、最近消费时间等
- 目标：将客户分成K个群体
- 方法：K-means聚类

**端到端实现**：
```python
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# 创建模拟数据
np.random.seed(42)
n_customers = 500
data = {
    'total_spent': np.random.normal(1000, 300, n_customers),
    'frequency': np.random.poisson(5, n_customers),
    'recency': np.random.randint(0, 365, n_customers)
}

df = pd.DataFrame(data)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

# K值选择
k_range = range(2, 11)
silhouette_scores = []

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_scaled)
    score = silhouette_score(X_scaled, kmeans.labels_)
    silhouette_scores.append(score)

best_k = k_range[np.argmax(silhouette_scores)]
print(f"最佳K值: {best_k}")

# 使用最佳K值训练
kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
kmeans.fit(X_scaled)

# 添加聚类标签
df['cluster'] = kmeans.labels_

# 分析每个簇的特征
print("\n各簇特征:")
for i in range(best_k):
    cluster_data = df[df['cluster'] == i]
    print(f"\n簇{i} (共{len(cluster_data)}个客户):")
    print(f"  平均消费金额: {cluster_data['total_spent'].mean():.2f}")
    print(f"  平均消费频率: {cluster_data['frequency'].mean():.2f}")
    print(f"  平均最近消费: {cluster_data['recency'].mean():.2f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# K值选择
axes[0].plot(k_range, silhouette_scores, marker='o')
axes[0].axvline(x=best_k, color='r', linestyle='--', label=f'最佳K={best_k}')
axes[0].set_xlabel('K值')
axes[0].set_ylabel('轮廓系数')
axes[0].set_title('K值选择')
axes[0].legend()
axes[0].grid(True)

# 聚类结果（2D投影）
axes[1].scatter(df['total_spent'], df['frequency'], c=df['cluster'], 
               cmap='viridis', alpha=0.6)
axes[1].set_xlabel('总消费金额')
axes[1].set_ylabel('消费频率')
axes[1].set_title('客户分群结果')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

**结果解读**：
- K-means成功将客户分成不同群体
- 每个群体有不同的消费特征

**改进方向**：
- 使用更多特征
- 尝试其他聚类算法
- 处理异常值

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. K-means是什么类型的算法？
   A. 监督学习  B. 无监督学习  C. 强化学习  D. 深度学习
   **答案**：B

2. K-means的目标是？
   A. 最小化类内距离  B. 最大化类间距离  C. 最小化误差  D. A和C
   **答案**：D

3. K-means算法的停止条件是？
   A. 达到最大迭代次数  B. 质心不再变化  C. 误差不再减小  D. 以上都可以
   **答案**：D

4. K值的选择方法不包括？
   A. 肘部法则  B. 轮廓系数  C. 交叉验证  D. 网格搜索
   **答案**：C

5. K-means的缺点不包括？
   A. 需要预设K值  B. 对初始值敏感  C. 只能处理球形簇  D. 计算复杂度高
   **答案**：D

**简答题**（每题10分，共40分）

1. 解释K-means的工作原理。
   **参考答案**：随机初始化K个质心，将每个数据点分配到最近的质心，重新计算质心，重复直到收敛。

2. 说明K值选择的方法。
   **参考答案**：肘部法则（找误差下降速度变慢的点）、轮廓系数（衡量聚类质量，越接近1越好）。

3. 解释K-means的优缺点。
   **参考答案**：优点：简单高效、可扩展；缺点：需要预设K值、对初始值敏感、只能处理球形簇。

4. 说明K-means++初始化的优势。
   **参考答案**：K-means++选择初始质心时使它们相互远离，减少对初始值的敏感性，提高聚类质量。

### 编程实践题（20分）

从零实现K-means算法，包括初始化、分配、更新、收敛判断。

### 综合应用题（20分）

使用K-means解决真实聚类问题，包括K值选择、数据预处理、模型训练、评估、可视化。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第9章）
- 《数据挖掘：概念与技术》- Han等

**在线资源**：
- scikit-learn官方文档
- K-means原始论文

### 相关工具与库

- **scikit-learn**：KMeans, MiniBatchKMeans
- **numpy**：数组操作
- **matplotlib**：可视化

### 进阶话题指引

完成本课程后，可以学习：
- **K-means++**：改进的初始化方法
- **Mini-batch K-means**：处理大规模数据
- **其他聚类算法**：层次聚类、DBSCAN

### 下节课预告

下一课将学习：
- **02_层次聚类**：自底向上或自顶向下的聚类方法
- 层次聚类不需要预设K值，可以生成树状图

### 学习建议

1. **理解优化过程**：理解K-means的迭代优化过程
2. **多实践**：从零实现算法，加深理解
3. **K值选择**：掌握K值选择的方法
4. **持续学习**：K-means是聚类的基础算法

---

**恭喜完成第一课！你已经掌握了K-means，准备好学习层次聚类了！**

