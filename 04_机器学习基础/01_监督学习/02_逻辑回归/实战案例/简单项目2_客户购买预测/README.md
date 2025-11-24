# 简单项目2：客户购买预测

## 项目描述

使用逻辑回归预测客户是否会购买产品，基于客户的年龄、收入、浏览历史等特征。

## 项目要求

### 难度等级
- **简单** - 使用现成工具/库，完成基础任务
- **代码量**：< 200行
- **时间**：2-4小时

### 功能要求

1. **数据准备**
   - 生成或加载客户数据
   - 数据预处理（缺失值、异常值）

2. **特征工程**
   - 特征选择
   - 特征标准化

3. **模型训练**
   - 使用逻辑回归训练模型
   - 评估模型性能

4. **结果分析**
   - 计算评估指标
   - 分析特征重要性
   - 可视化结果

## 实现步骤

### 步骤1：数据生成/加载

```python
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# 生成模拟数据
X, y = make_classification(n_samples=1000, n_features=5, 
                           n_informative=3, n_redundant=1,
                           random_state=42)

# 创建DataFrame
df = pd.DataFrame(X, columns=['age', 'income', 'browse_time', 
                               'click_count', 'page_views'])
df['purchase'] = y
```

### 步骤2：数据预处理

```python
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# 划分特征和标签
X = df.drop('purchase', axis=1)
y = df['purchase']

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

### 步骤3：模型训练

```python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_scaled, y_train)
```

### 步骤4：评估和分析

```python
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"准确率: {accuracy:.4f}")
print(f"AUC: {auc:.4f}")
```

## 预期结果

- 模型能够预测客户购买行为
- 准确率 > 0.75
- 能够识别重要特征

## 改进方向

1. 使用真实数据集
2. 更多特征工程
3. 处理类别不平衡
4. 尝试不同的模型

## 项目文件

- `main.py` - 主程序
- `README.md` - 项目说明（本文件）

