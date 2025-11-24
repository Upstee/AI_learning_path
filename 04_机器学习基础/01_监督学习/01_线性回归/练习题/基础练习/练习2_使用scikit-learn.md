# 基础练习2：使用scikit-learn实现线性回归

## 练习目标

使用scikit-learn库实现线性回归，包括：
1. 基本线性回归
2. Ridge和Lasso回归
3. 模型评估和对比

## 练习要求

### 1. 数据准备

使用以下代码生成数据：
```python
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
```

### 2. 实现基本线性回归

使用`LinearRegression`训练模型，并：
- 计算MSE和R²分数
- 可视化数据点和拟合直线
- 打印学习到的参数

### 3. 实现Ridge回归

使用`Ridge`训练模型，尝试不同的alpha值（0.1, 1.0, 10.0），并：
- 对比不同alpha值的效果
- 观察参数的变化

### 4. 实现Lasso回归

使用`Lasso`训练模型，尝试不同的alpha值，并：
- 观察哪些特征的系数变为0
- 理解Lasso的特征选择能力

### 5. 模型对比

对比三种模型：
- 训练集和测试集的MSE
- R²分数
- 参数值

## 预期结果

- 能够使用scikit-learn实现线性回归
- 理解Ridge和Lasso的区别
- 能够评估和对比模型

## 提示

1. 使用`train_test_split`分割数据
2. 使用`StandardScaler`标准化特征（对正则化很重要）
3. 使用`mean_squared_error`和`r2_score`评估模型

---

**完成后，请查看答案文件进行对比！**

