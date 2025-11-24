# 支持向量机（SVM）

## 1. 课程概述

### 课程目标
1. 理解SVM的基本原理和最大间隔思想
2. 掌握硬间隔和软间隔SVM
3. 理解核函数的作用和常见核函数
4. 能够从零实现SVM算法（简化版）
5. 能够使用scikit-learn实现SVM
6. 理解SVM在非线性问题中的应用

### 预计学习时间
- **理论学习**：10-12小时
- **代码实践**：10-12小时
- **练习巩固**：8-10小时
- **总计**：28-34小时（约2-3周）

### 难度等级
- **中高** - 需要理解优化理论和核技巧

### 课程定位
- **前置课程**：02_数学基础（线性代数、优化理论）、01_逻辑回归
- **后续课程**：06_集成学习、05_深度学习基础
- **在体系中的位置**：强大的分类器，可处理线性和非线性问题

### 学完能做什么
- 能够理解和使用SVM解决分类和回归问题
- 能够理解和使用核函数处理非线性问题
- 能够进行超参数调优
- 能够理解支持向量的概念

---

## 2. 前置知识检查

### 必备前置概念清单
- **线性代数**：向量、内积、超平面
- **优化理论**：拉格朗日乘数法、对偶问题
- **微积分**：导数、梯度

### 回顾链接/跳转
- 如果不熟悉优化理论：`02_数学基础/04_优化理论/`
- 如果不熟悉线性代数：`02_数学基础/01_线性代数/`

### 入门小测

**选择题**（每题2分，共10分）

1. SVM的目标是？
   A. 最小化误差  B. 最大化间隔  C. 最小化参数  D. 最大化准确率
   **答案**：B

2. 支持向量是？
   A. 所有训练样本  B. 离超平面最近的样本  C. 分类错误的样本  D. 随机选择的样本
   **答案**：B

3. 核函数的作用是？
   A. 加速计算  B. 处理非线性问题  C. 减少参数  D. 提高准确率
   **答案**：B

4. 软间隔SVM允许？
   A. 线性不可分  B. 分类错误  C. 间隔为负  D. 以上都可以
   **答案**：D

5. 常见的核函数不包括？
   A. 线性核  B. 多项式核  C. RBF核  D. Sigmoid核
   **答案**：D（Sigmoid核不常用）

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 SVM原理

#### 概念引入与直观类比

**类比**：SVM就像"找最宽的路"，在两类数据之间找最宽的间隔。

- **超平面**：分隔两类数据的平面
- **间隔**：两类数据到超平面的距离
- **支持向量**：离超平面最近的样本点
- **目标**：最大化间隔

例如：
- 分类问题：在两类数据之间找最佳分隔线
- 非线性问题：通过核函数映射到高维空间

#### 逐步理论推导

**步骤1：硬间隔SVM（线性可分）**

优化问题：
min ||w||²/2
s.t. yᵢ(wᵀxᵢ + b) ≥ 1, ∀i

**步骤2：软间隔SVM（线性不可分）**

引入松弛变量ξᵢ：
min ||w||²/2 + C∑ξᵢ
s.t. yᵢ(wᵀxᵢ + b) ≥ 1 - ξᵢ, ξᵢ ≥ 0

**步骤3：对偶问题**

使用拉格朗日乘数法，转化为对偶问题：
max ∑αᵢ - (1/2)∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼxᵢᵀxⱼ
s.t. ∑αᵢyᵢ = 0, 0 ≤ αᵢ ≤ C

**步骤4：核函数**

将内积xᵢᵀxⱼ替换为K(xᵢ, xⱼ)：
max ∑αᵢ - (1/2)∑ᵢ∑ⱼ αᵢαⱼyᵢyⱼK(xᵢ, xⱼ)

#### 数学公式与必要证明

**间隔的推导**：

点到超平面的距离：
d = |wᵀx + b| / ||w||

对于支持向量，y(wᵀx + b) = 1，所以间隔为：
margin = 2 / ||w||

最大化间隔等价于最小化||w||²。

**核函数的作用**：

通过核函数K(x, z) = φ(x)ᵀφ(z)，将数据映射到高维空间，在高维空间中线性可分。

#### 算法伪代码

```
SVM算法（SMO简化版）：
1. 初始化α = 0
2. 重复直到收敛：
   a. 选择两个变量αᵢ, αⱼ
   b. 优化这两个变量
   c. 更新α
3. 计算w和b
4. 返回模型
```

#### 关键性质

**优点**：
- **高维有效**：在高维空间中表现好
- **内存高效**：只使用支持向量
- **通用性强**：通过核函数处理非线性问题
- **理论基础强**：有坚实的数学基础

**缺点**：
- **训练时间长**：大规模数据训练慢
- **对参数敏感**：需要仔细调参
- **可解释性差**：不如决策树可解释
- **不直接提供概率**：需要额外计算

**适用场景**：
- 高维数据
- 非线性问题（使用核函数）
- 小到中等规模数据
- 需要高准确率

---

### 3.2 核函数

#### 常见核函数

**线性核**：
K(x, z) = xᵀz

**多项式核**：
K(x, z) = (γxᵀz + r)ᵈ

**RBF核（高斯核）**：
K(x, z) = exp(-γ||x - z||²)

**Sigmoid核**：
K(x, z) = tanh(γxᵀz + r)

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy >= 1.20.0
  - pandas >= 1.3.0
  - matplotlib >= 3.3.0
  - scikit-learn >= 0.24.0
  - cvxopt >= 1.2.0（可选，用于QP求解）

### 4.2 从零开始的完整可运行示例

#### 示例1：使用scikit-learn（推荐）

```python
from sklearn.svm import SVC, SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import numpy as np
import matplotlib.pyplot as plt

# 生成数据
from sklearn.datasets import make_classification, make_circles

# 线性可分数据
X_linear, y_linear = make_classification(n_samples=100, n_features=2, 
                                        n_redundant=0, n_clusters_per_class=1, 
                                        random_state=42)

# 非线性数据（同心圆）
X_nonlinear, y_nonlinear = make_circles(n_samples=100, noise=0.1, factor=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_linear_scaled = scaler.fit_transform(X_linear)
X_nonlinear_scaled = scaler.fit_transform(X_nonlinear)

# 线性SVM
svm_linear = SVC(kernel='linear', C=1.0, random_state=42)
svm_linear.fit(X_linear_scaled, y_linear)
accuracy_linear = svm_linear.score(X_linear_scaled, y_linear)
print(f"线性SVM准确率: {accuracy_linear:.4f}")

# 非线性SVM（RBF核）
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_rbf.fit(X_nonlinear_scaled, y_nonlinear)
accuracy_rbf = svm_rbf.score(X_nonlinear_scaled, y_nonlinear)
print(f"RBF核SVM准确率: {accuracy_rbf:.4f}")

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 线性SVM
ax = axes[0]
ax.scatter(X_linear_scaled[y_linear==0, 0], X_linear_scaled[y_linear==0, 1], 
          c='red', label='Class 0', alpha=0.6)
ax.scatter(X_linear_scaled[y_linear==1, 0], X_linear_scaled[y_linear==1, 1], 
          c='blue', label='Class 1', alpha=0.6)

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_linear_scaled[:, 0].min()-1, 
                                 X_linear_scaled[:, 0].max()+1, 100),
                     np.linspace(X_linear_scaled[:, 1].min()-1, 
                                X_linear_scaled[:, 1].max()+1, 100))
Z = svm_linear.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.set_title('线性SVM')
ax.legend()
ax.grid(True, alpha=0.3)

# RBF核SVM
ax = axes[1]
ax.scatter(X_nonlinear_scaled[y_nonlinear==0, 0], X_nonlinear_scaled[y_nonlinear==0, 1], 
          c='red', label='Class 0', alpha=0.6)
ax.scatter(X_nonlinear_scaled[y_nonlinear==1, 0], X_nonlinear_scaled[y_nonlinear==1, 1], 
          c='blue', label='Class 1', alpha=0.6)

# 绘制决策边界
xx, yy = np.meshgrid(np.linspace(X_nonlinear_scaled[:, 0].min()-1, 
                                 X_nonlinear_scaled[:, 0].max()+1, 100),
                     np.linspace(X_nonlinear_scaled[:, 1].min()-1, 
                                X_nonlinear_scaled[:, 1].max()+1, 100))
Z = svm_rbf.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
ax.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
ax.set_title('RBF核SVM（非线性）')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# 支持向量
print(f"\n线性SVM支持向量数量: {len(svm_linear.support_vectors_)}")
print(f"RBF核SVM支持向量数量: {len(svm_rbf.support_vectors_)}")
```

#### 示例2：不同核函数对比

```python
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.datasets import make_classification
import numpy as np

# 生成数据
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# 不同核函数
kernels = ['linear', 'poly', 'rbf', 'sigmoid']
results = {}

for kernel in kernels:
    svm = SVC(kernel=kernel, random_state=42)
    scores = cross_val_score(svm, X, y, cv=5)
    results[kernel] = {
        'mean': scores.mean(),
        'std': scores.std()
    }
    print(f"{kernel}核: 准确率={scores.mean():.4f} ± {scores.std():.4f}")

# 找出最佳核函数
best_kernel = max(results, key=lambda k: results[k]['mean'])
print(f"\n最佳核函数: {best_kernel}")
```

#### 示例3：超参数调优

```python
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 生成数据
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 网格搜索
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
    'kernel': ['rbf', 'poly', 'sigmoid']
}

svm = SVC(random_state=42)
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print(f"最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 测试集评估
y_pred = grid_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"测试集准确率: {accuracy:.4f}")
```

### 4.3 常见错误与排查

**错误1**：未标准化数据
```python
# 错误：SVM对特征缩放敏感
svm.fit(X)  # X未标准化

# 正确：先标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
svm.fit(X_scaled)
```

**错误2**：参数选择不当
```python
# 错误：C和gamma选择不当
svm = SVC(C=1000, gamma=1000)  # 可能过拟合

# 正确：使用网格搜索
grid_search = GridSearchCV(svm, param_grid, cv=5)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：线性SVM**
使用线性SVM解决二分类问题。

**练习2：核函数对比**
对比不同核函数的效果。

**练习3：超参数调优**
使用网格搜索调优SVM参数。

### 进阶练习（2-3题）

**练习1：多分类SVM**
使用SVM解决多分类问题（One-vs-One或One-vs-Rest）。

**练习2：SVM回归**
使用SVR解决回归问题。

### 挑战练习（1-2题）

**练习1：完整的分类系统**
实现完整的分类系统，包括数据预处理、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：手写数字识别

**业务背景**：
识别手写数字（0-9）。

**问题抽象**：
- 特征：图像像素值
- 目标：数字类别（0-9）
- 方法：SVM（多分类）

**端到端实现**：
```python
from sklearn.svm import SVC
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
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train, y_train)

# 预测
y_pred = svm.predict(X_test)
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
```

**结果解读**：
- SVM能够很好地识别手写数字
- 准确率高，混淆矩阵显示分类效果

**改进方向**：
- 使用更复杂的特征
- 调整超参数
- 使用深度学习模型

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. SVM的目标是？
   A. 最小化误差  B. 最大化间隔  C. 最小化参数  D. 最大化准确率
   **答案**：B

2. 支持向量是？
   A. 所有训练样本  B. 离超平面最近的样本  C. 分类错误的样本  D. 随机选择的样本
   **答案**：B

3. 核函数的作用是？
   A. 加速计算  B. 处理非线性问题  C. 减少参数  D. 提高准确率
   **答案**：B

4. 软间隔SVM允许？
   A. 线性不可分  B. 分类错误  C. 间隔为负  D. 以上都可以
   **答案**：D

5. 常见的核函数不包括？
   A. 线性核  B. 多项式核  C. RBF核  D. Sigmoid核
   **答案**：D（Sigmoid核不常用）

**简答题**（每题10分，共40分）

1. 解释SVM的最大间隔原理。
   **参考答案**：SVM寻找使两类数据间隔最大的超平面，间隔越大，泛化能力越强。

2. 说明核函数的作用和常见类型。
   **参考答案**：核函数将数据映射到高维空间，使非线性问题线性可分。常见类型：线性、多项式、RBF。

3. 解释硬间隔和软间隔SVM的区别。
   **参考答案**：硬间隔要求所有样本正确分类，软间隔允许部分样本分类错误，通过松弛变量控制。

4. 说明SVM的优缺点。
   **参考答案**：优点：高维有效、内存高效、通用性强；缺点：训练时间长、对参数敏感、可解释性差。

### 编程实践题（20分）

使用SVM解决分类问题，包括数据预处理、模型训练、评估。

### 综合应用题（20分）

使用SVM解决真实问题，包括核函数选择、超参数调优、结果分析。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第6章）
- 《统计学习方法》- 李航（第7章）
- 《支持向量机导论》- Nello Cristianini

**在线资源**：
- scikit-learn官方文档
- SVM原始论文

### 相关工具与库

- **scikit-learn**：SVC, SVR
- **libsvm**：C++实现的SVM库
- **cvxopt**：凸优化库

### 进阶话题指引

完成本课程后，可以学习：
- **序列最小优化（SMO）**：SVM的优化算法
- **多分类SVM**：One-vs-One, One-vs-Rest
- **SVM回归**：支持向量回归

### 下节课预告

下一课将学习：
- **06_集成学习**：综合多种学习方法
- 集成学习通过组合多个模型提升性能

### 学习建议

1. **理解优化理论**：理解拉格朗日乘数法和对偶问题
2. **多实践**：尝试不同核函数和参数
3. **可视化**：可视化决策边界，理解SVM的工作原理
4. **持续学习**：SVM是强大的分类器，需要扎实掌握

---

**恭喜完成第五课！你已经掌握了SVM，准备好学习集成学习了！**

