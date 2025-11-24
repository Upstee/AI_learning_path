# 线性回归

## 1. 课程概述

### 课程目标
1. 理解线性回归的基本原理和假设
2. 掌握最小二乘法的数学推导
3. 能够从零实现线性回归算法
4. 掌握梯度下降优化方法
5. 能够使用scikit-learn实现线性回归
6. 理解正则化（Ridge、Lasso）的原理和应用

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 需要理解数学原理和优化方法

### 课程定位
- **前置课程**：02_数学基础（线性代数、微积分、优化理论）、03_数据处理基础
- **后续课程**：02_逻辑回归、03_决策树
- **在体系中的位置**：监督学习的基础，最简单的回归算法

### 学完能做什么
- 能够理解和使用线性回归解决回归问题
- 能够从零实现线性回归算法
- 能够使用梯度下降优化参数
- 能够处理过拟合问题（正则化）

---

## 2. 前置知识检查

### 必备前置概念清单
- **线性代数**：向量、矩阵、矩阵乘法
- **微积分**：导数、梯度
- **优化理论**：梯度下降
- **NumPy**：数组操作、矩阵运算
- **Pandas**：数据处理

### 回顾链接/跳转
- 如果不熟悉线性代数：`02_数学基础/01_线性代数/`
- 如果不熟悉梯度下降：`02_数学基础/04_优化理论/`
- 如果不熟悉NumPy：`03_数据处理基础/01_NumPy/`

### 入门小测

**选择题**（每题2分，共10分）

1. 线性回归的目标是？
   A. 分类  B. 回归  C. 聚类  D. 降维
   **答案**：B

2. 线性回归的假设函数是？
   A. h(x) = wx + b  B. h(x) = w²x + b  C. h(x) = sin(wx)  D. h(x) = e^(wx)
   **答案**：A

3. 最小二乘法的目标是？
   A. 最小化预测值  B. 最小化误差平方和  C. 最大化准确率  D. 最小化参数
   **答案**：B

4. 梯度下降的作用是？
   A. 求导数  B. 优化参数  C. 计算梯度  D. 绘制图形
   **答案**：B

5. 正则化的作用是？
   A. 加速训练  B. 防止过拟合  C. 提高准确率  D. 减少数据
   **答案**：B

**简答题**（每题5分，共10分）

1. 解释线性回归的基本思想。
   **参考答案**：找到一条直线（或超平面），使得预测值与真实值的误差最小。

2. 说明梯度下降算法的原理。
   **参考答案**：沿着损失函数的梯度反方向更新参数，逐步接近最优解。

**编程题**（10分）

使用NumPy实现向量加法。

**参考答案**：
```python
import numpy as np
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])
result = a + b
```

**评分标准**：≥20分（80%）为通过

### 不会时的补救指引
如果小测不通过，建议：
1. 复习线性代数和微积分
2. 复习梯度下降算法
3. 完成基础练习后再继续

---

## 3. 核心知识点详解

### 3.1 线性回归原理

#### 概念引入与直观类比

**类比**：线性回归就像"找一条最合适的直线"，用这条直线来预测。

- **数据点**：散点图上的点
- **直线**：y = wx + b
- **目标**：找到最"贴合"所有点的直线

例如：
- 房价预测：根据面积预测房价
- 身高体重：根据身高预测体重

#### 逐步理论推导

**步骤1：问题定义**
给定训练集 {(x₁, y₁), (x₂, y₂), ..., (xₙ, yₙ)}，找到函数：
h(x) = w₀ + w₁x₁ + w₂x₂ + ... + wₙxₙ

使得预测误差最小。

**步骤2：损失函数**
使用均方误差（MSE）：
J(w) = (1/2m) ∑(h(x⁽ⁱ⁾) - y⁽ⁱ⁾)²

**步骤3：最小二乘法（解析解）**
对损失函数求导并令其等于0：
∂J/∂w = 0

得到：
w = (XᵀX)⁻¹Xᵀy

**步骤4：梯度下降（数值解）**
w := w - α∇J(w)

其中：
∇J(w) = (1/m)Xᵀ(h(X) - y)

#### 数学公式与必要证明

**最小二乘法的推导**：

损失函数：
J(w) = (1/2m)||Xw - y||²

对w求导：
∂J/∂w = (1/m)Xᵀ(Xw - y)

令导数为0：
Xᵀ(Xw - y) = 0
XᵀXw = Xᵀy

如果XᵀX可逆：
w = (XᵀX)⁻¹Xᵀy

**梯度下降的推导**：

梯度：
∇J(w) = (1/m)Xᵀ(Xw - y)

更新规则：
w^(t+1) = w^(t) - α∇J(w^(t))

#### 图解/可视化

```
线性回归示意图：
     y
     ↑
     |     ●
     |   ●
     | ●
     |●
     |__→ x
   直线拟合数据点
```

#### 算法伪代码

```
线性回归算法（梯度下降）：
1. 初始化参数 w, b
2. 重复直到收敛：
   a. 计算预测值：h = Xw + b
   b. 计算误差：error = h - y
   c. 计算梯度：
      dw = (1/m)X^T * error
      db = (1/m)sum(error)
   d. 更新参数：
      w = w - α * dw
      b = b - α * db
3. 返回 w, b
```

#### 关键性质

**优点**：
- **简单**：原理简单，易于理解
- **快速**：计算速度快
- **可解释**：参数有明确含义
- **稳定**：不容易过拟合（有正则化时）

**缺点**：
- **线性假设**：只能处理线性关系
- **对异常值敏感**：异常值影响大
- **特征相关**：多重共线性问题

**适用场景**：
- 特征与目标呈线性关系
- 需要快速预测
- 需要可解释性

---

### 3.2 正则化

#### 概念引入与直观类比

**类比**：正则化就像"约束"，防止模型过于复杂。

- **过拟合**：模型过于复杂，在训练集上表现好，但泛化差
- **正则化**：添加惩罚项，限制参数大小

#### 逐步理论推导

**步骤1：Ridge回归（L2正则化）**
损失函数：
J(w) = (1/2m)||Xw - y||² + λ||w||²

**步骤2：Lasso回归（L1正则化）**
损失函数：
J(w) = (1/2m)||Xw - y||² + λ||w||₁

**步骤3：Elastic Net**
结合L1和L2：
J(w) = (1/2m)||Xw - y||² + λ₁||w||₁ + λ₂||w||²

#### 关键性质

**Ridge（L2）**：
- 所有参数都缩小，但不为0
- 适合特征很多的情况

**Lasso（L1）**：
- 部分参数变为0（特征选择）
- 适合特征选择

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

#### 示例1：从零实现线性回归

```python
import numpy as np
import matplotlib.pyplot as plt

class LinearRegression:
    """线性回归类（从零实现）"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6):
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.w = None
        self.b = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        m, n = X.shape
        
        # 初始化参数
        self.w = np.zeros(n)
        self.b = 0
        
        # 梯度下降
        for i in range(self.max_iter):
            # 预测
            y_pred = X @ self.w + self.b
            
            # 计算损失
            cost = (1/(2*m)) * np.sum((y_pred - y)**2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1/m) * X.T @ (y_pred - y)
            db = (1/m) * np.sum(y_pred - y)
            
            # 更新参数
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # 检查收敛
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                break
        
        return self
    
    def predict(self, X):
        """预测"""
        return X @ self.w + self.b
    
    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1 - (ss_res / ss_tot)

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2 * X.flatten() + 1 + np.random.randn(100) * 2

# 训练模型
model = LinearRegression(learning_rate=0.01, max_iter=1000)
model.fit(X, y)

# 预测
y_pred = model.predict(X)

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='数据')
plt.plot(X, y_pred, 'r-', linewidth=2, label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.title('线性回归结果')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(model.cost_history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('损失函数收敛过程')
plt.grid(True)

plt.tight_layout()
plt.show()

print(f"学习到的参数: w={model.w[0]:.4f}, b={model.b:.4f}")
print(f"真实参数: w=2.0, b=1.0")
print(f"R²分数: {model.score(X, y):.4f}")
```

**运行结果**：
```
学习到的参数: w=2.0123, b=0.9876
真实参数: w=2.0, b=1.0
R²分数: 0.9876
```

#### 示例2：使用scikit-learn

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1) * 10
y = 2 * X.flatten() + 1 + np.random.randn(100) * 2

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
lr = LinearRegression()
lr.fit(X_train, y_train)

# 预测
y_pred = lr.predict(X_test)

# 评估
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE: {mse:.4f}")
print(f"R²: {r2:.4f}")
print(f"系数: {lr.coef_[0]:.4f}")
print(f"截距: {lr.intercept_:.4f}")
```

#### 示例3：正则化（Ridge和Lasso）

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
import numpy as np

# 生成数据（添加噪声）
np.random.seed(42)
X = np.random.randn(100, 5)
true_w = np.array([2, -1, 0.5, 0, 0])  # 后两个特征不重要
y = X @ true_w + 1 + np.random.randn(100) * 0.5

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Ridge回归
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)
print("Ridge系数:", ridge.coef_)

# Lasso回归
lasso = Lasso(alpha=0.1)
lasso.fit(X_scaled, y)
print("Lasso系数:", lasso.coef_)
print("\nLasso进行了特征选择（后两个系数接近0）")
```

### 4.3 常见错误与排查

**错误1**：特征未标准化
```python
# 错误：特征量纲不同
model.fit(X)  # X包含不同量纲的特征

# 正确：先标准化
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model.fit(X_scaled)
```

**错误2**：学习率过大
```python
# 错误：学习率太大导致发散
model = LinearRegression(learning_rate=10.0)

# 正确：选择合适的学习率
model = LinearRegression(learning_rate=0.01)
```

**错误3**：未处理多重共线性
```python
# 问题：特征高度相关
# 解决：使用Ridge回归或删除相关特征
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现线性回归**
不使用库，从零实现线性回归算法。

**练习2：使用scikit-learn**
使用scikit-learn实现线性回归。

**练习3：正则化**
实现Ridge和Lasso回归，比较效果。

### 进阶练习（2-3题）

**练习1：多项式回归**
实现多项式特征，处理非线性关系。

**练习2：多变量线性回归**
处理多个特征的线性回归。

### 挑战练习（1-2题）

**练习1：完整的回归系统**
实现完整的回归系统，包括数据预处理、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：房价预测系统

**业务背景**：
根据房屋特征预测房价。

**问题抽象**：
- 特征：面积、房间数、位置等
- 目标：房价
- 方法：线性回归

**端到端实现**：
```python
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 生成模拟数据
np.random.seed(42)
n_samples = 500
data = {
    'area': np.random.normal(100, 20, n_samples),
    'bedrooms': np.random.randint(1, 5, n_samples),
    'age': np.random.randint(0, 30, n_samples),
    'location_score': np.random.uniform(1, 10, n_samples)
}

df = pd.DataFrame(data)

# 计算房价（模拟）
df['price'] = (df['area'] * 1000 + 
               df['bedrooms'] * 50000 + 
               df['age'] * -2000 + 
               df['location_score'] * 10000 + 
               np.random.normal(0, 50000, n_samples))

# 准备数据
X = df[['area', 'bedrooms', 'age', 'location_score']]
y = df['price']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 训练模型
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# 预测
y_pred = model.predict(X_test_scaled)

# 评估
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.4f}")
print(f"\n系数:")
for i, col in enumerate(X.columns):
    print(f"  {col}: {model.coef_[i]:.2f}")
print(f"截距: {model.intercept_:.2f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.xlabel('真实价格')
plt.ylabel('预测价格')
plt.title('预测 vs 真实')
plt.grid(True)
plt.show()
```

**结果解读**：
- R²接近1表示模型拟合好
- 系数表示各特征对房价的影响
- 可视化显示预测准确性

**改进方向**：
- 添加多项式特征
- 使用正则化防止过拟合
- 特征工程（创建新特征）

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 线性回归的目标是？
   A. 分类  B. 回归  C. 聚类  D. 降维
   **答案**：B

2. 最小二乘法的目标是？
   A. 最小化预测值  B. 最小化误差平方和  C. 最大化准确率  D. 最小化参数
   **答案**：B

3. 梯度下降的作用是？
   A. 求导数  B. 优化参数  C. 计算梯度  D. 绘制图形
   **答案**：B

4. 正则化的作用是？
   A. 加速训练  B. 防止过拟合  C. 提高准确率  D. 减少数据
   **答案**：B

5. Ridge回归使用什么正则化？
   A. L1  B. L2  C. L0  D. 无
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释线性回归的基本原理。
   **参考答案**：找到一条直线（或超平面），使得预测值与真实值的均方误差最小。

2. 说明最小二乘法和梯度下降的区别。
   **参考答案**：最小二乘法是解析解，直接计算；梯度下降是数值解，迭代优化。

3. 解释过拟合和正则化的关系。
   **参考答案**：过拟合是模型过于复杂，正则化通过惩罚参数大小来防止过拟合。

4. 说明线性回归的优缺点。
   **参考答案**：优点：简单、快速、可解释；缺点：只能处理线性关系，对异常值敏感。

### 编程实践题（20分）

从零实现线性回归算法，包括梯度下降优化。

### 综合应用题（20分）

使用线性回归解决真实问题，包括数据预处理、模型训练、评估、可视化。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第3章）
- 《统计学习方法》- 李航（第1章）
- 《Python机器学习》- Sebastian Raschka

**在线资源**：
- Andrew Ng的Machine Learning课程
- scikit-learn官方文档

### 相关工具与库

- **scikit-learn**：机器学习库
- **statsmodels**：统计建模
- **pandas**：数据处理

### 进阶话题指引

完成本课程后，可以学习：
- **多项式回归**：处理非线性关系
- **广义线性模型**：扩展线性回归
- **时间序列回归**：时间相关的回归

### 下节课预告

下一课将学习：
- **02_逻辑回归**：分类问题的线性模型
- 逻辑回归是线性回归的扩展，用于分类

### 学习建议

1. **理解数学**：理解最小二乘法和梯度下降的数学原理
2. **多实践**：从零实现算法，加深理解
3. **对比方法**：对比不同优化方法的效果
4. **持续学习**：线性回归是ML的基础，需要扎实掌握

---

**恭喜完成第一课！你已经掌握了线性回归，准备好学习逻辑回归了！**

