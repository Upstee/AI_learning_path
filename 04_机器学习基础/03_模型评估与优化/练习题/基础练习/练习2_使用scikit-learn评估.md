# 练习2：使用scikit-learn进行模型评估

## 练习目标

掌握使用scikit-learn进行完整的模型评估流程，包括分类和回归任务的评估。

## 练习要求

### 任务1：分类模型评估

使用scikit-learn完成以下任务：

1. **加载数据集**
   - 使用`sklearn.datasets`加载一个分类数据集（如`load_breast_cancer`或`load_iris`）
   - 划分训练集和测试集（80%训练，20%测试）

2. **训练多个分类模型**
   - 逻辑回归（LogisticRegression）
   - 决策树（DecisionTreeClassifier）
   - 随机森林（RandomForestClassifier）
   - SVM（SVC）

3. **评估每个模型**
   对每个模型计算并输出：
   - 准确率（Accuracy）
   - 精确率（Precision）
   - 召回率（Recall）
   - F1分数（F1-Score）
   - 混淆矩阵（Confusion Matrix）
   - ROC曲线和AUC值（如果是二分类）

4. **可视化结果**
   - 绘制混淆矩阵热力图
   - 绘制ROC曲线（如果是二分类）
   - 创建模型性能对比表

### 任务2：回归模型评估

1. **加载数据集**
   - 使用`sklearn.datasets`加载一个回归数据集（如`load_boston`或`load_diabetes`）
   - 划分训练集和测试集

2. **训练多个回归模型**
   - 线性回归（LinearRegression）
   - 岭回归（Ridge）
   - Lasso回归（Lasso）
   - 随机森林回归（RandomForestRegressor）

3. **评估每个模型**
   对每个模型计算并输出：
   - 均方误差（MSE）
   - 均方根误差（RMSE）
   - 平均绝对误差（MAE）
   - R²分数（R-squared）

4. **可视化结果**
   - 绘制预测值vs真实值散点图
   - 绘制残差图
   - 创建模型性能对比表

### 任务3：模型选择

根据评估结果：
1. 选择分类任务中表现最好的模型
2. 选择回归任务中表现最好的模型
3. 分析为什么这些模型表现最好

## 代码框架

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, r2_score,
    mean_squared_error, mean_absolute_error
)

# TODO: 完成分类任务评估
# 1. 加载数据
# 2. 划分数据集
# 3. 训练模型
# 4. 评估模型
# 5. 可视化结果

# TODO: 完成回归任务评估
# 1. 加载数据
# 2. 划分数据集
# 3. 训练模型
# 4. 评估模型
# 5. 可视化结果

# TODO: 模型选择和分析
```

## 输出要求

1. **分类任务输出**：
   - 每个模型的评估指标表格
   - 混淆矩阵热力图（每个模型一张图）
   - ROC曲线图（如果是二分类）
   - 模型性能对比柱状图

2. **回归任务输出**：
   - 每个模型的评估指标表格
   - 预测值vs真实值散点图（每个模型一张图）
   - 残差图（每个模型一张图）
   - 模型性能对比柱状图

3. **分析报告**：
   - 最佳分类模型及其原因
   - 最佳回归模型及其原因
   - 模型性能差异分析

## 提示

1. 使用`train_test_split`划分数据集时，设置`random_state`以确保结果可复现
2. 对于多分类问题，ROC曲线需要使用`roc_auc_score`的`multi_class`参数
3. 使用`seaborn.heatmap`绘制混淆矩阵更美观
4. 使用`pandas.DataFrame`创建性能对比表更清晰

## 思考题

1. 为什么在分类任务中，准确率可能不是最好的评估指标？
2. 在什么情况下，精确率比召回率更重要？
3. R²分数为0.8意味着什么？
4. 如何根据ROC曲线选择最佳的分类阈值？

## 难度等级

**基础** - 需要熟悉scikit-learn的评估函数和可视化

## 预计时间

3-4小时

