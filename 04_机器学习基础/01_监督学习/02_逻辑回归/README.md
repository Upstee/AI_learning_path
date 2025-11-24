# 逻辑回归

## 1. 课程概述

### 课程目标
1. 理解逻辑回归的基本原理和假设
2. 掌握Sigmoid函数的作用和性质
3. 理解最大似然估计的原理
4. 能够从零实现逻辑回归算法
5. 能够使用scikit-learn实现逻辑回归
6. 理解多分类逻辑回归（Softmax）

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 需要理解概率和优化

### 课程定位
- **前置课程**：01_线性回归、02_数学基础（概率统计、优化理论）
- **后续课程**：03_决策树、04_随机森林
- **在体系中的位置**：分类问题的基础算法，线性分类器

### 学完能做什么
- 能够理解和使用逻辑回归解决分类问题
- 能够从零实现逻辑回归算法
- 能够处理二分类和多分类问题
- 能够理解概率输出和决策边界

---

## 2. 前置知识检查

### 必备前置概念清单
- **线性回归**：理解线性模型
- **概率统计**：概率、似然函数
- **微积分**：导数、梯度
- **优化理论**：梯度下降

### 回顾链接/跳转
- 如果不熟悉线性回归：`04_机器学习基础/01_监督学习/01_线性回归/`
- 如果不熟悉概率：`02_数学基础/02_概率统计/`

### 入门小测

**选择题**（每题2分，共10分）

1. 逻辑回归用于什么任务？
   A. 回归  B. 分类  C. 聚类  D. 降维
   **答案**：B

2. Sigmoid函数的值域是？
   A. [0, 1]  B. (-∞, ∞)  C. [0, ∞)  D. (-1, 1)
   **答案**：A

3. 逻辑回归使用什么优化方法？
   A. 最小二乘法  B. 最大似然估计  C. 梯度下降  D. B和C
   **答案**：D

4. 逻辑回归的输出是什么？
   A. 类别  B. 概率  C. 数值  D. 以上都可以
   **答案**：D

5. 多分类逻辑回归使用什么函数？
   A. Sigmoid  B. Softmax  C. ReLU  D. Tanh
   **答案**：B

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 逻辑回归原理

#### 概念引入与直观类比

**类比**：逻辑回归就像"概率判断器"，根据特征判断属于某个类别的概率。

- **线性回归**：输出连续值
- **逻辑回归**：输出概率（0到1之间）

例如：
- 邮件分类：判断是垃圾邮件的概率
- 疾病诊断：判断患病的概率

#### 逐步理论推导

**步骤1：从线性回归到逻辑回归**
线性回归：h(x) = wᵀx + b

问题：输出可能是任意值，不适合概率。

**步骤2：引入Sigmoid函数**
σ(z) = 1 / (1 + e^(-z))

性质：
- 值域：[0, 1]
- 单调递增
- S形曲线

**步骤3：逻辑回归假设**
h(x) = σ(wᵀx + b) = 1 / (1 + e^(-(wᵀx + b)))

**步骤4：概率解释**
P(y=1|x) = h(x)
P(y=0|x) = 1 - h(x)

#### 数学公式与必要证明

**Sigmoid函数的性质**：

导数：
σ'(z) = σ(z)(1 - σ(z))

这个性质使得梯度计算简单。

**最大似然估计**：

似然函数：
L(w) = ∏ᵢ h(x⁽ⁱ⁾)^(y⁽ⁱ⁾) (1 - h(x⁽ⁱ⁾))^(1-y⁽ⁱ⁾)

对数似然：
l(w) = ∑ᵢ [y⁽ⁱ⁾log(h(x⁽ⁱ⁾)) + (1-y⁽ⁱ⁾)log(1-h(x⁽ⁱ⁾))]

梯度：
∂l/∂w = Xᵀ(h(X) - y)

#### 图解/可视化

```
Sigmoid函数：
     y
  1.0|     ╱──────
     |   ╱
 0.5 | ╱
     |╱
  0.0 └───────→ x
    S形曲线
```

#### 算法伪代码

```
逻辑回归算法（梯度上升）：
1. 初始化参数 w, b
2. 重复直到收敛：
   a. 计算预测概率：h = σ(Xw + b)
   b. 计算梯度：
      dw = (1/m)X^T * (h - y)
      db = (1/m)sum(h - y)
   c. 更新参数：
      w = w + α * dw  （梯度上升）
      b = b + α * db
3. 返回 w, b
```

#### 关键性质

**优点**：
- **概率输出**：输出是概率，可解释
- **不需要特征缩放**：对特征缩放不敏感
- **不易过拟合**：有正则化时
- **计算快速**：训练和预测都很快

**缺点**：
- **线性决策边界**：只能处理线性可分问题
- **需要大量样本**：小样本效果差
- **对异常值敏感**：异常值影响大

**适用场景**：
- 二分类问题
- 需要概率输出
- 特征与目标呈线性关系

---

### 3.2 多分类逻辑回归

#### 概念引入与直观类比

**类比**：多分类就像"多选一"，从多个类别中选择一个。

- **二分类**：是/否
- **多分类**：类别1/类别2/类别3/...

#### 逐步理论推导

**步骤1：One-vs-Rest（OvR）**
训练K个二分类器，每个对应一个类别。

**步骤2：Softmax回归**
对于K个类别：
P(y=k|x) = e^(wₖᵀx) / ∑ⱼ e^(wⱼᵀx)

**步骤3：损失函数（交叉熵）**
J(w) = -∑ᵢ ∑ₖ yₖ⁽ⁱ⁾log(P(y=k|x⁽ⁱ⁾))

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

#### 示例1：从零实现逻辑回归

```python
import numpy as np
import matplotlib.pyplot as plt

class LogisticRegression:
    """逻辑回归类（从零实现）"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """Sigmoid函数"""
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))  # 防止溢出
    
    def fit(self, X, y):
        """训练模型"""
        m, n = X.shape
        
        # 初始化参数
        self.w = np.zeros(n)
        self.b = 0
        
        # 梯度上升（最大化似然）
        for i in range(self.max_iter):
            # 预测概率
            z = X @ self.w + self.b
            h = self.sigmoid(z)
            
            # 计算损失（对数似然的负值）
            cost = -(1/m) * np.sum(y * np.log(h + 1e-10) + (1-y) * np.log(1-h + 1e-10))
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/m) * X.T @ (h - y)
            db = (1/m) * np.sum(h - y)
            
            # 更新参数
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # 检查收敛
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                break
        
        return self
    
    def predict_proba(self, X):
        """预测概率"""
        z = X @ self.w + self.b
        return self.sigmoid(z)
    
    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 2)
# 创建线性可分的二分类数据
y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

# 训练模型
model = LogisticRegression(learning_rate=0.1, max_iter=1000)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)

print(f"准确率: {accuracy:.4f}")
print(f"学习到的参数: w={model.w}, b={model.b:.4f}")

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X[y==0, 0], X[y==0, 1], c='red', label='Class 0', alpha=0.6)
plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', label='Class 1', alpha=0.6)

# 绘制决策边界
x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx1, xx2 = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                       np.linspace(x2_min, x2_max, 100))
X_grid = np.c_[xx1.ravel(), xx2.ravel()]
Z = model.predict(X_grid).reshape(xx1.shape)
plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)
plt.xlabel('X1')
plt.ylabel('X2')
plt.title('决策边界')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(model.cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('损失函数收敛过程')
plt.grid(True)

plt.tight_layout()
plt.show()
```

#### 示例2：使用scikit-learn

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(200, 2)
y = ((X[:, 0] + X[:, 1]) > 0).astype(int)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))
```

#### 示例3：多分类逻辑回归

```python
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# 生成多分类数据
X, y = make_classification(n_samples=300, n_features=4, n_classes=3, 
                          n_informative=4, n_redundant=0, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型（自动使用多分类）
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)

# 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
print("\n预测概率（前5个样本）:")
print(y_proba[:5])
```

### 4.3 常见错误与排查

**错误1**：Sigmoid溢出
```python
# 错误：z太大导致溢出
h = 1 / (1 + np.exp(-z))  # z很大时会溢出

# 正确：限制z的范围
h = 1 / (1 + np.exp(-np.clip(z, -250, 250)))
```

**错误2**：对数计算出现0
```python
# 错误：h可能为0，log(0)未定义
cost = -np.sum(y * np.log(h))

# 正确：添加小值防止log(0)
cost = -np.sum(y * np.log(h + 1e-10))
```

**错误3**：混淆梯度上升和梯度下降
```python
# 逻辑回归使用梯度上升（最大化似然）
# 但通常写成梯度下降（最小化负对数似然）
w = w - learning_rate * dw  # 最小化损失
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现逻辑回归**
不使用库，从零实现逻辑回归算法。

**练习2：使用scikit-learn**
使用scikit-learn实现逻辑回归。

**练习3：多分类问题**
实现多分类逻辑回归。

### 进阶练习（2-3题）

**练习1：正则化逻辑回归**
实现L1和L2正则化的逻辑回归。

**练习2：特征工程**
对数据进行特征工程，提升模型性能。

### 挑战练习（1-2题）

**练习1：完整的分类系统**
实现完整的分类系统，包括数据预处理、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：垃圾邮件分类系统

**业务背景**：
根据邮件内容判断是否为垃圾邮件。

**问题抽象**：
- 特征：邮件内容特征（词频等）
- 目标：垃圾邮件（1）或正常邮件（0）
- 方法：逻辑回归

**端到端实现**：
```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import pandas as pd

# 创建模拟数据
emails = [
    "buy now special offer free money",
    "meeting tomorrow at 3pm",
    "click here win prize",
    "project update attached",
    "limited time discount",
    "team meeting notes",
    "urgent action required",
    "quarterly report review"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0]  # 1=垃圾邮件，0=正常邮件

# 特征提取（词频）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(emails).toarray()
y = np.array(labels)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

# 评估
accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '垃圾']))

# 查看特征重要性
feature_names = vectorizer.get_feature_names_out()
coef = model.coef_[0]
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'coefficient': coef
}).sort_values('coefficient', ascending=False)

print("\n特征重要性（前10）:")
print(feature_importance.head(10))
```

**结果解读**：
- 模型能够区分垃圾邮件和正常邮件
- 特征重要性显示哪些词对分类重要

**改进方向**：
- 使用TF-IDF特征
- 添加更多特征
- 处理类别不平衡

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 逻辑回归用于什么任务？
   A. 回归  B. 分类  C. 聚类  D. 降维
   **答案**：B

2. Sigmoid函数的值域是？
   A. [0, 1]  B. (-∞, ∞)  C. [0, ∞)  D. (-1, 1)
   **答案**：A

3. 逻辑回归使用什么优化方法？
   A. 最小二乘法  B. 最大似然估计  C. 梯度下降  D. B和C
   **答案**：D

4. 逻辑回归的输出是什么？
   A. 类别  B. 概率  C. 数值  D. 以上都可以
   **答案**：D

5. 多分类逻辑回归使用什么函数？
   A. Sigmoid  B. Softmax  C. ReLU  D. Tanh
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释逻辑回归和线性回归的区别。
   **参考答案**：线性回归输出连续值，用于回归；逻辑回归输出概率，用于分类。

2. 说明Sigmoid函数的作用。
   **参考答案**：将线性组合映射到[0,1]区间，表示概率。

3. 解释最大似然估计的原理。
   **参考答案**：选择使观测数据出现概率最大的参数值。

4. 说明逻辑回归的优缺点。
   **参考答案**：优点：概率输出、快速、可解释；缺点：线性边界、需要大量样本。

### 编程实践题（20分）

从零实现逻辑回归算法，包括Sigmoid函数和梯度下降。

### 综合应用题（20分）

使用逻辑回归解决真实分类问题，包括数据预处理、模型训练、评估。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第3章）
- 《统计学习方法》- 李航（第6章）

**在线资源**：
- Andrew Ng的Machine Learning课程
- scikit-learn官方文档

### 相关工具与库

- **scikit-learn**：LogisticRegression
- **statsmodels**：统计建模
- **pandas**：数据处理

### 进阶话题指引

完成本课程后，可以学习：
- **正则化逻辑回归**：L1、L2正则化
- **多分类扩展**：Softmax回归
- **非线性扩展**：多项式特征

### 下节课预告

下一课将学习：
- **03_决策树**：树形分类器
- 决策树是非线性分类器，可解释性强

### 学习建议

1. **理解概率**：理解概率输出和决策边界
2. **多实践**：从零实现算法，加深理解
3. **对比算法**：对比逻辑回归和其他分类算法
4. **持续学习**：逻辑回归是分类的基础，需要扎实掌握

---

**恭喜完成第二课！你已经掌握了逻辑回归，准备好学习决策树了！**

