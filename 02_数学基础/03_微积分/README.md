# 微积分

## 1. 课程概述

### 课程目标
1. 理解导数和微分的概念
2. 掌握链式法则和复合函数求导
3. 理解偏导数和梯度的概念
4. 掌握梯度在优化中的应用
5. 理解积分的基本概念
6. 能够使用Python进行微积分计算

### 预计学习时间
- **理论学习**：10-12小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：24-30小时（约2-3周）

### 难度等级
- **中等** - 需要理解抽象概念，但通过可视化可以掌握

### 课程定位
- **前置课程**：高中数学（导数基础）、01_线性代数
- **后续课程**：04_优化理论、04_机器学习基础
- **在体系中的位置**：优化算法的基础，反向传播的核心

### 学完能做什么
- 能够计算函数的导数和梯度
- 能够理解梯度下降算法
- 能够理解反向传播算法
- 能够使用Python进行微积分计算

---

## 2. 前置知识检查

### 必备前置概念清单
- **高中数学**：函数、导数基础
- **线性代数**：向量、矩阵
- **Python基础**：函数、NumPy

### 回顾链接/跳转
- 如果不熟悉线性代数：`02_数学基础/01_线性代数/`
- 如果不熟悉Python：`01_Python进阶/`

### 入门小测

**选择题**（每题2分，共10分）

1. 导数表示什么？
   A. 函数值  B. 变化率  C. 积分  D. 极限
   **答案**：B

2. 链式法则用于什么？
   A. 复合函数求导  B. 函数复合  C. 函数分解  D. 函数简化
   **答案**：A

3. 梯度是什么？
   A. 标量  B. 向量  C. 矩阵  D. 张量
   **答案**：B

4. 梯度下降用于什么？
   A. 求最大值  B. 求最小值  C. 求积分  D. 求导数
   **答案**：B

5. 偏导数是什么？
   A. 对多个变量求导  B. 对单个变量求导  C. 对函数求导  D. 对向量求导
   **答案**：B

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 导数与微分

#### 概念引入与直观类比

**类比**：导数就像"速度"，表示函数值变化的快慢。

- **导数**：函数在某点的瞬时变化率
- **几何意义**：切线的斜率
- **物理意义**：速度、加速度

#### 逐步理论推导

**步骤1：导数的定义**
f'(x) = lim(h→0) [f(x+h) - f(x)] / h

**步骤2：常见函数的导数**
- (xⁿ)' = nxⁿ⁻¹
- (eˣ)' = eˣ
- (ln x)' = 1/x
- (sin x)' = cos x
- (cos x)' = -sin x

**步骤3：导数的运算法则**
- (f+g)' = f' + g'
- (fg)' = f'g + fg'
- (f/g)' = (f'g - fg') / g²

**步骤4：链式法则**
如果y = f(g(x))，则：
dy/dx = (dy/dg) × (dg/dx)

#### 数学公式与必要证明

**链式法则的证明**（直观理解）：
如果y = f(u)，u = g(x)，则：
Δy/Δx = (Δy/Δu) × (Δu/Δx)

当Δx → 0时：
dy/dx = (dy/du) × (du/dx)

---

### 3.2 梯度

#### 概念引入与直观类比

**类比**：梯度就像"最陡的上坡方向"，指向函数值增长最快的方向。

- **梯度**：多变量函数的导数向量
- **方向**：函数值增长最快的方向
- **大小**：变化率的大小

#### 逐步理论推导

**步骤1：偏导数**
对于函数f(x₁, x₂, ..., xₙ)，对xᵢ的偏导数：
∂f/∂xᵢ = lim(h→0) [f(..., xᵢ+h, ...) - f(..., xᵢ, ...)] / h

**步骤2：梯度定义**
梯度是偏导数组成的向量：
∇f = [∂f/∂x₁, ∂f/∂x₂, ..., ∂f/∂xₙ]ᵀ

**步骤3：梯度的性质**
- 梯度指向函数值增长最快的方向
- 梯度的大小表示变化率
- 在极值点，梯度为0

**步骤4：梯度下降**
x^(t+1) = x^(t) - α∇f(x^(t))

其中α是学习率。

---

### 3.3 链式法则在神经网络中的应用

#### 概念引入与直观类比

**类比**：链式法则就像"连锁反应"，误差从输出层反向传播到输入层。

- **前向传播**：数据从输入到输出
- **反向传播**：误差从输出到输入
- **链式法则**：连接前向和反向传播

#### 逐步理论推导

**步骤1：损失函数**
L = (1/2)(y - ŷ)²

**步骤2：输出层误差**
∂L/∂ŷ = ŷ - y

**步骤3：隐藏层误差（链式法则）**
∂L/∂w = (∂L/∂ŷ) × (∂ŷ/∂z) × (∂z/∂w)

**步骤4：权重更新**
w = w - α(∂L/∂w)

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy
  - scipy
  - matplotlib
  - sympy（符号计算，可选）

### 4.2 从零开始的完整可运行示例

#### 示例1：数值导数

```python
import numpy as np
import matplotlib.pyplot as plt

def numerical_derivative(f, x, h=1e-5):
    """数值计算导数"""
    return (f(x + h) - f(x)) / h

# 定义函数
def f(x):
    return x**2

# 计算导数
x = 2.0
derivative = numerical_derivative(f, x)
print(f"f(x) = x² 在 x={x} 处的导数: {derivative:.4f}")
print(f"理论值: {2*x}")

# 可视化
x_vals = np.linspace(-3, 3, 100)
y_vals = f(x_vals)
dy_vals = 2 * x_vals  # 理论导数

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, label='f(x) = x²')
plt.plot(x_vals, dy_vals, label="f'(x) = 2x")
plt.xlabel('x')
plt.ylabel('y')
plt.title('函数及其导数')
plt.legend()
plt.grid(True)
plt.show()
```

#### 示例2：梯度计算

```python
import numpy as np

def gradient(f, x, h=1e-5):
    """计算梯度（数值方法）"""
    grad = np.zeros_like(x)
    for i in range(len(x)):
        x_plus = x.copy()
        x_plus[i] += h
        grad[i] = (f(x_plus) - f(x)) / h
    return grad

# 定义函数 f(x, y) = x² + y²
def f(x):
    return x[0]**2 + x[1]**2

# 计算梯度
x = np.array([1.0, 2.0])
grad = gradient(f, x)
print(f"函数 f(x,y) = x² + y²")
print(f"在点 ({x[0]}, {x[1]}) 处的梯度: {grad}")
print(f"理论梯度: [{2*x[0]}, {2*x[1]}] = [2.0, 4.0]")
```

#### 示例3：梯度下降

```python
import numpy as np
import matplotlib.pyplot as plt

def gradient_descent(f, grad_f, x0, learning_rate=0.1, max_iter=100, tol=1e-6):
    """梯度下降算法"""
    x = x0.copy()
    history = [x.copy()]
    
    for i in range(max_iter):
        grad = grad_f(x)
        x_new = x - learning_rate * grad
        
        # 检查收敛
        if np.linalg.norm(x_new - x) < tol:
            break
        
        x = x_new
        history.append(x.copy())
    
    return x, history

# 定义函数 f(x, y) = (x-1)² + (y-2)²
def f(x):
    return (x[0] - 1)**2 + (x[1] - 2)**2

def grad_f(x):
    return np.array([2*(x[0] - 1), 2*(x[1] - 2)])

# 初始点
x0 = np.array([5.0, 5.0])

# 运行梯度下降
x_opt, history = gradient_descent(f, grad_f, x0, learning_rate=0.1)

print(f"初始点: ({x0[0]}, {x0[1]})")
print(f"最优解: ({x_opt[0]:.4f}, {x_opt[1]:.4f})")
print(f"最优值: {f(x_opt):.4f}")
print(f"迭代次数: {len(history)}")

# 可视化
history = np.array(history)
plt.figure(figsize=(10, 6))
plt.plot(history[:, 0], history[:, 1], 'o-', label='梯度下降路径')
plt.plot(1, 2, 'r*', markersize=15, label='最优点')
plt.plot(x0[0], x0[1], 'go', markersize=10, label='起始点')
plt.xlabel('x')
plt.ylabel('y')
plt.title('梯度下降优化过程')
plt.legend()
plt.grid(True)
plt.show()
```

#### 示例4：链式法则

```python
import numpy as np

# 链式法则示例：y = f(g(x))
# 其中 g(x) = x², f(u) = sin(u)

def g(x):
    return x**2

def f(u):
    return np.sin(u)

def composite_function(x):
    return f(g(x))

# 计算导数（链式法则）
def derivative_chain_rule(x):
    # dy/dx = (df/du) × (du/dx)
    # df/du = cos(u) = cos(x²)
    # du/dx = 2x
    return np.cos(x**2) * 2*x

# 数值验证
x = 1.0
h = 1e-5
numerical_derivative = (composite_function(x + h) - composite_function(x)) / h
theoretical_derivative = derivative_chain_rule(x)

print(f"x = {x}")
print(f"数值导数: {numerical_derivative:.6f}")
print(f"理论导数: {theoretical_derivative:.6f}")
print(f"误差: {abs(numerical_derivative - theoretical_derivative):.2e}")
```

### 4.3 常见错误与排查

**错误1**：学习率过大
```python
# 错误：学习率太大导致发散
x = x - 10.0 * grad  # 可能发散

# 正确：选择合适的学习率
x = x - 0.01 * grad  # 通常0.001-0.1
```

**错误2**：梯度计算错误
```python
# 错误：忘记链式法则
# 正确：使用链式法则计算复合函数导数
```

**错误3**：数值精度问题
```python
# 使用合适的h值
h = 1e-5  # 太小：数值误差大；太大：近似误差大
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：导数计算**
计算各种函数的导数。

**练习2：梯度计算**
计算多变量函数的梯度。

**练习3：梯度下降**
实现简单的梯度下降算法。

### 进阶练习（2-3题）

**练习1：反向传播**
实现简单的反向传播算法。

**练习2：优化算法**
实现不同的优化算法（SGD、Adam等）。

---

## 6. 实际案例

### 案例：使用梯度下降优化线性回归

**业务背景**：
使用梯度下降找到线性回归的最优参数。

**问题抽象**：
- 损失函数：L = (1/2m)∑(y - ŷ)²
- 梯度：∂L/∂w
- 优化：使用梯度下降最小化损失

**端到端实现**：
```python
import numpy as np
import matplotlib.pyplot as plt

def linear_regression_gd(X, y, learning_rate=0.01, max_iter=1000):
    """使用梯度下降的线性回归"""
    m, n = X.shape
    w = np.zeros(n)
    b = 0
    history = []
    
    for i in range(max_iter):
        # 预测
        y_pred = X @ w + b
        
        # 计算梯度
        dw = (1/m) * X.T @ (y_pred - y)
        db = (1/m) * np.sum(y_pred - y)
        
        # 更新参数
        w = w - learning_rate * dw
        b = b - learning_rate * db
        
        # 记录损失
        loss = (1/(2*m)) * np.sum((y_pred - y)**2)
        history.append(loss)
    
    return w, b, history

# 生成数据
np.random.seed(42)
X = np.random.randn(100, 1)
y = 2 * X.flatten() + 1 + 0.1 * np.random.randn(100)

# 添加偏置项
X_with_bias = np.column_stack([np.ones(100), X])

# 训练
w, b, history = linear_regression_gd(X_with_bias, y)

print(f"学习到的参数: w={w[1]:.4f}, b={w[0]:.4f}")
print(f"真实参数: w=2.0, b=1.0")

# 可视化
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(X, y, alpha=0.5, label='数据')
x_line = np.linspace(X.min(), X.max(), 100)
y_line = w[1] * x_line + w[0]
plt.plot(x_line, y_line, 'r-', label='拟合直线')
plt.xlabel('X')
plt.ylabel('y')
plt.legend()
plt.title('线性回归结果')

plt.subplot(1, 2, 2)
plt.plot(history)
plt.xlabel('迭代次数')
plt.ylabel('损失')
plt.title('损失函数收敛过程')
plt.grid(True)

plt.tight_layout()
plt.show()
```

**结果解读**：
- 梯度下降成功找到最优参数
- 损失函数逐渐收敛

**改进方向**：
- 添加学习率衰减
- 使用批量梯度下降
- 添加正则化

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 导数表示什么？
   A. 函数值  B. 变化率  C. 积分  D. 极限
   **答案**：B

2. 梯度是什么？
   A. 标量  B. 向量  C. 矩阵  D. 函数
   **答案**：B

3. 梯度下降用于什么？
   A. 求最大值  B. 求最小值  C. 求积分  D. 求导数
   **答案**：B

4. 链式法则用于什么？
   A. 复合函数求导  B. 函数复合  C. 函数分解  D. 函数简化
   **答案**：A

5. 偏导数是什么？
   A. 对多个变量求导  B. 对单个变量求导  C. 对函数求导  D. 对向量求导
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释导数的几何意义。
   **参考答案**：导数表示函数图像上某点处切线的斜率。

2. 说明梯度的意义。
   **参考答案**：梯度是多变量函数的导数向量，指向函数值增长最快的方向。

3. 解释梯度下降算法。
   **参考答案**：沿着梯度反方向更新参数，逐步接近函数的最小值。

4. 说明链式法则在反向传播中的应用。
   **参考答案**：通过链式法则，误差可以从输出层反向传播到输入层，计算各层参数的梯度。

### 编程实践题（20分）

实现梯度下降算法优化一个函数。

### 综合应用题（20分）

使用梯度下降实现线性回归，并可视化结果。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《微积分》- 各种版本
- 《深度学习》- Ian Goodfellow（第4章）

**在线资源**：
- Khan Academy微积分课程
- 3Blue1Brown微积分系列

### 相关工具与库

- **sympy**：符号计算
- **scipy.optimize**：优化算法
- **autograd**：自动微分

### 进阶话题指引

完成本课程后，可以学习：
- **自动微分**：自动计算梯度
- **优化算法**：SGD、Adam、RMSprop等
- **二阶优化**：牛顿法、拟牛顿法

### 下节课预告

下一课将学习：
- **04_优化理论**：凸优化、约束优化
- 优化理论是机器学习的核心

### 学习建议

1. **理解几何意义**：通过图形理解导数
2. **多练习**：通过计算加深理解
3. **结合应用**：理解在AI中的实际应用
4. **持续学习**：微积分是优化的基础，需要扎实掌握

---

**恭喜完成第三课！你已经掌握了微积分的基础，准备好学习优化理论了！**

