# K近邻（KNN）

## 1. 课程概述

### 课程目标
1. 理解KNN的基本原理和"懒惰学习"思想
2. 掌握距离度量的计算方法（欧氏距离、曼哈顿距离等）
3. 理解K值选择的影响
4. 能够从零实现KNN算法
5. 能够使用scikit-learn实现KNN
6. 掌握KNN的优化方法（KD树、球树）

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 算法简单但需要理解距离和优化

### 课程定位
- **前置课程**：02_数学基础（线性代数）、03_数据处理基础
- **后续课程**：02_无监督学习、05_深度学习基础
- **在体系中的位置**：简单的非参数分类器，基于实例的学习

### 学完能做什么
- 能够理解和使用KNN解决分类和回归问题
- 能够从零实现KNN算法
- 能够选择合适的距离度量和K值
- 能够理解KNN的优化方法

---

## 2. 前置知识检查

### 必备前置概念清单
- **线性代数**：向量、距离
- **数据结构**：最近邻搜索
- **概率统计**：投票、平均

### 回顾链接/跳转
- 如果不熟悉距离计算：`02_数学基础/01_线性代数/`
- 如果不熟悉数据结构：`01_Python进阶/`

### 入门小测

**选择题**（每题2分，共10分）

1. KNN是什么类型的学习方法？
   A. 参数学习  B. 非参数学习  C. 深度学习  D. 强化学习
   **答案**：B

2. KNN在训练时做什么？
   A. 训练模型  B. 存储数据  C. 优化参数  D. 计算梯度
   **答案**：B

3. K值的选择对KNN的影响是？
   A. K越大越容易过拟合  B. K越小越容易过拟合  C. K不影响  D. K越大越复杂
   **答案**：B

4. 欧氏距离的公式是？
   A. √(∑(xᵢ-yᵢ)²)  B. ∑|xᵢ-yᵢ|  C. max|xᵢ-yᵢ|  D. ∑(xᵢ-yᵢ)²
   **答案**：A

5. KNN的缺点不包括？
   A. 计算成本高  B. 对K值敏感  C. 需要大量内存  D. 训练时间长
   **答案**：D（KNN训练时间短，但预测时间长）

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 KNN原理

#### 概念引入与直观类比

**类比**：KNN就像"物以类聚"，根据最近的K个邻居判断类别。

- **邻居**：最近的K个训练样本
- **投票**：K个邻居投票决定类别
- **距离**：用距离衡量"近"

例如：
- 分类：根据最近的K个样本的类别投票
- 回归：根据最近的K个样本的值平均

#### 逐步理论推导

**步骤1：距离计算**

对于两个样本x和y：
- **欧氏距离**：d(x,y) = √(∑(xᵢ-yᵢ)²)
- **曼哈顿距离**：d(x,y) = ∑|xᵢ-yᵢ|
- **切比雪夫距离**：d(x,y) = max|xᵢ-yᵢ|
- **余弦相似度**：cos(θ) = (x·y)/(||x|| ||y||)

**步骤2：找K个最近邻**

对于新样本x，找到训练集中距离最近的K个样本。

**步骤3：分类决策**

- **多数投票**：K个邻居中多数类
- **加权投票**：根据距离加权投票

**步骤4：回归预测**

- **简单平均**：K个邻居值的平均
- **加权平均**：根据距离加权平均

#### 数学公式与必要证明

**欧氏距离**：

d(x,y) = √(∑ᵢ₌₁ⁿ (xᵢ - yᵢ)²)

**加权投票**：

对于分类，权重可以是距离的倒数：
wᵢ = 1 / (d(x, xᵢ) + ε)

其中ε是小值防止除零。

**加权平均**：

对于回归：
ŷ = (∑ᵢ₌₁ᴷ wᵢyᵢ) / (∑ᵢ₌₁ᴷ wᵢ)

#### 算法伪代码

```
KNN算法（分类）：
1. 训练阶段：
   a. 存储训练数据X和标签y
2. 预测阶段：
   a. 对于新样本x：
      i. 计算x到所有训练样本的距离
      ii. 找到距离最近的K个样本
      iii. 这K个样本的标签投票
      iv. 返回多数类
3. 返回预测结果

KNN算法（回归）：
1. 训练阶段：
   a. 存储训练数据X和标签y
2. 预测阶段：
   a. 对于新样本x：
      i. 计算x到所有训练样本的距离
      ii. 找到距离最近的K个样本
      iii. 这K个样本的值平均
      iv. 返回平均值
3. 返回预测结果
```

#### 关键性质

**优点**：
- **简单直观**：算法简单，易于理解
- **无需训练**：训练时只存储数据
- **适合非线性**：可以处理复杂的非线性关系
- **多分类**：天然支持多分类

**缺点**：
- **计算成本高**：预测时需要计算所有距离
- **内存占用大**：需要存储所有训练数据
- **对K值敏感**：K值选择影响性能
- **对特征缩放敏感**：需要标准化特征
- **维度灾难**：高维数据效果差

**适用场景**：
- 小到中等规模数据
- 非线性关系
- 需要简单模型
- 低维数据

---

### 3.2 优化方法

#### KD树

**思想**：将数据组织成树结构，加速最近邻搜索。

**构建**：
1. 选择方差最大的维度
2. 选择中位数作为分割点
3. 递归构建左右子树

**搜索**：
1. 从根节点开始
2. 根据分割维度决定搜索方向
3. 回溯检查是否需要搜索另一侧

#### 球树

**思想**：将数据组织成球结构，用球包含样本。

**构建**：
1. 选择距离最远的两个点作为球心
2. 将其他点分配到最近的球
3. 递归构建子树

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

#### 示例1：从零实现KNN

```python
import numpy as np
from collections import Counter

class KNN:
    """KNN类（从零实现）"""
    
    def __init__(self, k=3, distance='euclidean', weights='uniform'):
        self.k = k
        self.distance = distance
        self.weights = weights
    
    def _euclidean_distance(self, x1, x2):
        """欧氏距离"""
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
    def _manhattan_distance(self, x1, x2):
        """曼哈顿距离"""
        return np.sum(np.abs(x1 - x2))
    
    def _compute_distance(self, x1, x2):
        """计算距离"""
        if self.distance == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            return self._euclidean_distance(x1, x2)
    
    def fit(self, X, y):
        """训练模型（只存储数据）"""
        self.X_train = X
        self.y_train = y
        return self
    
    def predict(self, X):
        """预测"""
        predictions = []
        
        for x in X:
            # 计算到所有训练样本的距离
            distances = [self._compute_distance(x, x_train) 
                        for x_train in self.X_train]
            
            # 找到K个最近邻的索引
            k_indices = np.argsort(distances)[:self.k]
            
            # 获取K个最近邻的标签
            k_nearest_labels = [self.y_train[i] for i in k_indices]
            
            if self.weights == 'uniform':
                # 多数投票
                most_common = Counter(k_nearest_labels).most_common(1)[0][0]
                predictions.append(most_common)
            else:
                # 加权投票
                k_distances = [distances[i] for i in k_indices]
                weights = [1 / (d + 1e-10) for d in k_distances]
                weighted_votes = {}
                for label, weight in zip(k_nearest_labels, weights):
                    weighted_votes[label] = weighted_votes.get(label, 0) + weight
                predictions.append(max(weighted_votes, key=weighted_votes.get))
        
        return np.array(predictions)

# 生成数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                          random_state=42)

# 训练模型
model = KNN(k=5, distance='euclidean', weights='uniform')
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"准确率: {accuracy:.4f}")
```

#### 示例2：使用scikit-learn

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X, y = make_classification(n_samples=200, n_features=2, n_classes=3, 
                          random_state=42)

# 标准化（重要！）
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', metric='euclidean')
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策边界
plt.figure(figsize=(10, 8))
xx, yy = np.meshgrid(np.linspace(X_scaled[:, 0].min()-1, X_scaled[:, 0].max()+1, 100),
                     np.linspace(X_scaled[:, 1].min()-1, X_scaled[:, 1].max()+1, 100))
Z = knn.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=y, cmap=plt.cm.RdYlBu, edgecolors='black')
plt.xlabel('特征1')
plt.ylabel('特征2')
plt.title('KNN决策边界')
plt.colorbar()
plt.show()
```

#### 示例3：K值选择

```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 测试不同的K值
k_range = range(1, 31)
k_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv=5, scoring='accuracy')
    k_scores.append(scores.mean())

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range, k_scores, marker='o')
plt.xlabel('K值')
plt.ylabel('交叉验证准确率')
plt.title('K值选择')
plt.grid(True)
plt.show()

# 找出最佳K值
best_k = k_range[np.argmax(k_scores)]
best_score = max(k_scores)
print(f"最佳K值: {best_k}")
print(f"最佳准确率: {best_score:.4f}")
```

#### 示例4：KNN回归

```python
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(200, 2)
y = X[:, 0] * 2 + X[:, 1] * 3 + np.random.randn(200) * 0.5

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练模型
knn_reg = KNeighborsRegressor(n_neighbors=5, weights='distance')
knn_reg.fit(X_train, y_train)

# 预测
y_pred = knn_reg.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
```

### 4.3 常见错误与排查

**错误1**：未标准化特征
```python
# 错误：特征量纲不同，距离计算不准确
knn.fit(X)  # X未标准化

# 正确：先标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
knn.fit(X_scaled)
```

**错误2**：K值选择不当
```python
# 错误：K值太小或太大
knn = KNeighborsClassifier(n_neighbors=1)  # 可能过拟合
knn = KNeighborsClassifier(n_neighbors=100)  # 可能欠拟合

# 正确：使用交叉验证选择K值
from sklearn.model_selection import GridSearchCV
param_grid = {'n_neighbors': range(1, 21)}
grid_search = GridSearchCV(knn, param_grid, cv=5)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现KNN**
不使用库，从零实现KNN算法。

**练习2：K值选择**
使用交叉验证选择最佳K值。

**练习3：距离度量对比**
对比不同距离度量的效果。

### 进阶练习（2-3题）

**练习1：加权KNN**
实现加权KNN（根据距离加权）。

**练习2：KNN回归**
使用KNN解决回归问题。

### 挑战练习（1-2题）

**练习1：完整的分类系统**
实现完整的分类系统，包括数据预处理、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：手写数字识别（KNN）

**业务背景**：
使用KNN识别手写数字。

**问题抽象**：
- 特征：图像像素值
- 目标：数字类别（0-9）
- 方法：KNN

**端到端实现**：
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# 加载数据
digits = load_digits()
X, y = digits.data, digits.target

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 训练模型
knn = KNeighborsClassifier(n_neighbors=5, weights='distance', metric='euclidean')
knn.fit(X_train, y_train)

# 预测
y_pred = knn.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title('混淆矩阵')
plt.colorbar()
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.tight_layout()
plt.show()

# 可视化一些预测结果
fig, axes = plt.subplots(2, 5, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    idx = np.random.randint(0, len(X_test))
    ax.imshow(X_test[idx].reshape(8, 8), cmap='gray')
    ax.set_title(f'真实: {y_test[idx]}, 预测: {y_pred[idx]}')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

**结果解读**：
- KNN能够很好地识别手写数字
- 准确率高，但计算成本较高

**改进方向**：
- 使用KD树加速
- 特征降维
- 使用更复杂的模型

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. KNN是什么类型的学习方法？
   A. 参数学习  B. 非参数学习  C. 深度学习  D. 强化学习
   **答案**：B

2. KNN在训练时做什么？
   A. 训练模型  B. 存储数据  C. 优化参数  D. 计算梯度
   **答案**：B

3. K值的选择对KNN的影响是？
   A. K越大越容易过拟合  B. K越小越容易过拟合  C. K不影响  D. K越大越复杂
   **答案**：B

4. 欧氏距离的公式是？
   A. √(∑(xᵢ-yᵢ)²)  B. ∑|xᵢ-yᵢ|  C. max|xᵢ-yᵢ|  D. ∑(xᵢ-yᵢ)²
   **答案**：A

5. KNN的缺点不包括？
   A. 计算成本高  B. 对K值敏感  C. 需要大量内存  D. 训练时间长
   **答案**：D

**简答题**（每题10分，共40分）

1. 解释KNN的工作原理。
   **参考答案**：对于新样本，找到训练集中距离最近的K个样本，根据这K个样本的标签投票（分类）或值平均（回归）。

2. 说明K值选择的影响。
   **参考答案**：K值小容易过拟合，对噪声敏感；K值大容易欠拟合，边界平滑。需要交叉验证选择。

3. 解释为什么KNN需要标准化特征。
   **参考答案**：不同特征的量纲不同，距离计算会被大值特征主导。标准化后所有特征同等重要。

4. 说明KNN的优缺点。
   **参考答案**：优点：简单直观、无需训练、适合非线性；缺点：计算成本高、内存占用大、对K值敏感、维度灾难。

### 编程实践题（20分）

从零实现KNN算法，包括距离计算和K近邻搜索。

### 综合应用题（20分）

使用KNN解决真实问题，包括数据预处理、K值选择、模型训练、评估。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第10章）
- 《统计学习方法》- 李航（第3章）

**在线资源**：
- scikit-learn官方文档
- KD树和球树算法

### 相关工具与库

- **scikit-learn**：KNeighborsClassifier, KNeighborsRegressor
- **numpy**：距离计算
- **pandas**：数据处理

### 进阶话题指引

完成本课程后，可以学习：
- **KD树和球树**：加速最近邻搜索
- **局部敏感哈希**：高维数据的近似最近邻
- **流形学习**：处理高维数据

### 下节课预告

完成01_监督学习后，将学习：
- **02_无监督学习**：聚类和降维
- 无监督学习不需要标签，发现数据中的模式

### 学习建议

1. **理解距离**：理解不同距离度量的含义
2. **多实践**：从零实现算法，加深理解
3. **K值选择**：使用交叉验证选择最佳K值
4. **持续学习**：KNN是简单但重要的算法

---

**恭喜完成第八课！你已经完成了01_监督学习模块的所有算法！准备好学习无监督学习了！**

