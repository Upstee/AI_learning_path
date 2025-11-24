# t-SNE流形学习

## 1. 课程概述

### 课程目标
1. 理解t-SNE的基本原理和数学基础（高维/低维概率分布、KL散度）
2. 掌握t-SNE的优化过程和参数选择
3. 理解t-SNE相比PCA的优势（非线性、保留局部结构）
4. 能够使用scikit-learn实现t-SNE
5. 理解t-SNE的局限性和适用场景
6. 掌握其他流形学习方法（UMAP、Isomap）

### 预计学习时间
- **理论学习**：8-10小时
- **代码实践**：6-8小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约2-3周）

### 难度等级
- **中等偏上** - 需要理解概率分布和优化理论

### 课程定位
- **前置课程**：02_数学基础（概率统计、优化理论）、04_PCA
- **后续课程**：07_降维技术
- **在体系中的位置**：非线性降维方法，主要用于数据可视化

### 学完能做什么
- 能够理解和使用t-SNE进行非线性降维
- 能够选择合适的参数（perplexity、learning_rate）
- 能够应用t-SNE进行数据可视化
- 能够理解t-SNE的局限性和改进方法

---

## 2. 前置知识检查

### 必备前置概念清单
- **概率统计**：概率分布、条件概率、KL散度
- **优化理论**：梯度下降、学习率
- **PCA**：理解线性降维的局限性
- **NumPy**：数组操作

### 回顾链接/跳转
- 如果不熟悉KL散度：`02_数学基础/02_概率统计/`
- 如果不熟悉PCA：`04_机器学习基础/02_无监督学习/04_PCA/`
- 如果不熟悉优化：`02_数学基础/04_优化理论/`

### 入门小测

**选择题**（每题2分，共10分）

1. t-SNE的主要目的是？
   A. 分类  B. 回归  C. 非线性降维和可视化  D. 聚类
   **答案**：C

2. t-SNE相比PCA的主要优势是？
   A. 计算更快  B. 能处理非线性关系  C. 不需要参数  D. 能处理缺失值
   **答案**：B

3. t-SNE的核心思想是？
   A. 最大化方差  B. 最小化KL散度  C. 最大化相关性  D. 最小化距离
   **答案**：B

4. t-SNE的perplexity参数大致表示？
   A. 学习率  B. 局部邻居数量  C. 降维维度  D. 迭代次数
   **答案**：B

5. t-SNE的局限性不包括？
   A. 计算复杂度高  B. 结果不稳定  C. 只能处理线性关系  D. 难以解释
   **答案**：C（t-SNE能处理非线性关系）

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 t-SNE原理

#### 概念引入与直观类比

**类比**：t-SNE就像"在低维空间重建高维数据的邻居关系"，保持局部结构。

- **高维空间**：原始数据空间
- **低维空间**：降维后的可视化空间
- **邻居关系**：相似的点在低维空间也相近

#### 逐步理论推导

**步骤1：高维空间概率分布**

对于高维空间中的点i和j，定义相似度：
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i}\exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

对称化：
$$p_{ij} = \frac{p_{j|i} + p_{i|j}}{2n}$$

**步骤2：低维空间概率分布**

使用t分布（自由度=1）：
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l}(1 + ||y_k - y_l||^2)^{-1}}$$

**步骤3：最小化KL散度**

目标函数：
$$C = \sum_{i,j} p_{ij} \log \frac{p_{ij}}{q_{ij}}$$

使用梯度下降优化。

#### 关键参数

- **perplexity**：控制局部邻居数量，通常5-50
- **learning_rate**：学习率，通常10-1000
- **n_iter**：迭代次数，通常1000+

---

## 4. Python代码实践

### 4.1 使用scikit-learn实现t-SNE

```python
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
import matplotlib.pyplot as plt

# 加载数据
digits = load_digits()
X = digits.data
y = digits.target

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6)
plt.colorbar()
plt.title('t-SNE可视化结果')
plt.show()
```

---

## 5. 动手练习（分层次）

### 基础练习（3-5题）⚠️【必须至少3题，难度递增】

#### 练习1：使用t-SNE进行数据可视化
**目标**：使用t-SNE将高维数据降到2D可视化

**要求**：
1. 加载高维数据集
2. 使用t-SNE降维到2D
3. 可视化结果
4. 分析不同类别的分布

**难度**：⭐⭐

---

#### 练习2：t-SNE参数调优
**目标**：学习如何为t-SNE选择合适的参数

**要求**：
1. 测试不同的perplexity值
2. 测试不同的learning_rate
3. 比较不同参数的效果
4. 选择最优参数

**难度**：⭐⭐⭐

---

#### 练习3：t-SNE与PCA对比
**目标**：比较t-SNE和PCA的降维效果

**要求**：
1. 使用PCA和t-SNE分别降维
2. 可视化对比结果
3. 分析两种方法的差异
4. 总结适用场景

**难度**：⭐⭐⭐

---

### 进阶练习（2-3题）⚠️【必须至少2题，难度递增】

#### 练习1：大规模数据t-SNE
**目标**：优化t-SNE以处理大规模数据

**要求**：
1. 使用PCA预降维
2. 使用Barnes-Hut近似
3. 在10万+数据点上测试
4. 比较优化前后的性能

**难度**：⭐⭐⭐⭐

---

#### 练习2：t-SNE在图像数据中的应用
**目标**：使用t-SNE可视化图像数据的分布

**要求**：
1. 加载图像数据集
2. 提取特征
3. 使用t-SNE降维
4. 可视化图像分布
5. 分析相似图像的聚集情况

**难度**：⭐⭐⭐⭐

---

### 挑战练习（1-2题）⚠️【必须至少1题】

#### 练习1：实现简化版t-SNE
**目标**：从零实现t-SNE算法的核心部分

**要求**：
1. 实现高维/低维概率分布计算
2. 实现KL散度计算
3. 实现梯度下降优化
4. 在小型数据集上测试
5. 与scikit-learn结果对比

**难度**：⭐⭐⭐⭐⭐

---

## 6. 实际案例

### 案例1：MNIST数据可视化（简单项目）

**业务背景**：将MNIST手写数字图像降到2D可视化，观察不同数字的分布。

**端到端实现**：
```python
from sklearn.manifold import TSNE
from sklearn.datasets import fetch_openml
import matplotlib.pyplot as plt
import numpy as np

# 加载MNIST数据（采样）
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X = mnist.data[:5000]  # 采样5000个样本
y = mnist.target[:5000].astype(int)

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(12, 10))
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.6, s=10)
plt.colorbar(scatter)
plt.title('MNIST t-SNE可视化')
plt.show()
```

---

### 案例2：高维特征可视化（中等项目）

**业务背景**：将高维特征降到2D，分析特征分布和聚类情况。

**端到端实现**：
```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.datasets import load_wine
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
wine = load_wine()
X = wine.data
y = wine.target

# 方法1：直接t-SNE
tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# 方法2：PCA预降维 + t-SNE
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X)
tsne_pca = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne_pca = tsne_pca.fit_transform(X_pca)

# 可视化对比
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

axes[0].scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[0].set_title('直接t-SNE')
axes[0].set_xlabel('t-SNE 1')
axes[0].set_ylabel('t-SNE 2')

axes[1].scatter(X_tsne_pca[:, 0], X_tsne_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
axes[1].set_title('PCA预降维 + t-SNE')
axes[1].set_xlabel('t-SNE 1')
axes[1].set_ylabel('t-SNE 2')

plt.tight_layout()
plt.show()
```

---

### 案例3：文本数据可视化（进阶项目）

**业务背景**：将文本数据的词向量降到2D，观察文本的语义分布。

**端到端实现**：
```python
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import numpy as np

# 示例文本数据
texts = [
    "machine learning deep learning neural network",
    "natural language processing text analysis",
    "computer vision image recognition",
    # ... 更多文本
]

# 特征提取
vectorizer = TfidfVectorizer(max_features=100)
X = vectorizer.fit_transform(texts).toarray()

# t-SNE降维
tsne = TSNE(n_components=2, perplexity=min(30, len(texts)-1), random_state=42)
X_tsne = tsne.fit_transform(X)

# 可视化
plt.figure(figsize=(10, 8))
plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6)
for i, text in enumerate(texts[:20]):  # 只标注前20个
    plt.annotate(text[:20], (X_tsne[i, 0], X_tsne[i, 1]), fontsize=8)
plt.title('文本数据t-SNE可视化')
plt.show()
```

---

## 7. 自我评估

### 概念题

#### 选择题（10-15道）

1. t-SNE的主要目的是？
   A. 分类  B. 回归  C. 非线性降维和可视化  D. 聚类
   **答案**：C

2. t-SNE的核心思想是？
   A. 最大化方差  B. 最小化KL散度  C. 最大化相关性  D. 最小化距离
   **答案**：B

3. t-SNE相比PCA的主要优势是？
   A. 计算更快  B. 能处理非线性关系  C. 不需要参数  D. 能处理缺失值
   **答案**：B

4. perplexity参数大致表示？
   A. 学习率  B. 局部邻居数量  C. 降维维度  D. 迭代次数
   **答案**：B

5. t-SNE的局限性不包括？
   A. 计算复杂度高  B. 结果不稳定  C. 只能处理线性关系  D. 难以解释
   **答案**：C

#### 简答题（5-8道）

1. 解释t-SNE的基本原理。
   **参考答案**：t-SNE通过最小化高维和低维空间概率分布的KL散度，在低维空间保持高维数据的局部结构。

2. 说明t-SNE与PCA的主要区别。
   **参考答案**：
   - PCA是线性降维，t-SNE是非线性降维
   - PCA保留全局结构，t-SNE保留局部结构
   - PCA计算快，t-SNE计算慢

3. 如何选择t-SNE的参数？
   **参考答案**：
   - perplexity：通常5-50，根据数据规模调整
   - learning_rate：通常10-1000，默认200
   - n_iter：通常1000+，确保收敛

---

### 编程实践题（2-3道）

#### 题目1：使用t-SNE进行数据可视化
**要求**：
1. 加载高维数据集
2. 使用t-SNE降维
3. 可视化结果
4. 分析不同参数的影响

**评分标准**：
- 正确使用t-SNE（40分）
- 可视化清晰（20分）
- 参数分析深入（20分）
- 代码质量（20分）

---

### 综合应用题（1-2道）

#### 题目1：t-SNE在图像分类中的应用
**要求**：
1. 加载图像数据集
2. 提取特征
3. 使用t-SNE降维可视化
4. 分析不同类别的分布
5. 提出改进建议

**评分标准**：
- 数据处理正确（25分）
- t-SNE应用正确（25分）
- 分析深入（25分）
- 建议有价值（25分）

---

## 8. 拓展学习

### 论文推荐

1. **van der Maaten, L., & Hinton, G. (2008). "Visualizing data using t-SNE."** Journal of Machine Learning Research
   - t-SNE的原始论文
   - 理解算法的理论基础

2. **McInnes, L., et al. (2018). "UMAP: Uniform Manifold Approximation and Projection."** arXiv
   - UMAP算法论文
   - 学习t-SNE的改进方法

### 书籍推荐

1. **《机器学习》- 周志华**
   - 第10章：降维与度量学习
   - 包含流形学习的讲解

### 相关工具与库

1. **scikit-learn**
   - t-SNE实现
   - 文档：https://scikit-learn.org/stable/modules/manifold.html#t-sne

2. **umap-learn**
   - UMAP实现
   - GitHub: https://github.com/lmcinnes/umap

### 进阶话题指引

1. **流形学习变体**
   - UMAP
   - Isomap
   - LLE

2. **t-SNE优化**
   - Barnes-Hut近似
   - 大规模数据t-SNE
   - 增量t-SNE

3. **t-SNE在特定领域的应用**
   - 图像数据可视化
   - 文本数据可视化
   - 生物信息学

### 下节课预告与学习建议

**下节课**：`06_异常检测`

**学习建议**：
1. 完成所有练习题
2. 理解非线性降维的必要性
3. 掌握参数选择方法
4. 了解t-SNE的局限性

**前置准备**：
- 了解异常检测的基本概念
- 复习概率统计
- 准备数据集进行实践

---

**完成本课程后，你将能够：**
- ✅ 理解t-SNE的原理和应用
- ✅ 使用t-SNE进行数据可视化
- ✅ 选择合适的参数
- ✅ 理解t-SNE的局限性和改进方法

**继续学习，成为AI大师！** 🚀

