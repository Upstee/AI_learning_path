# 简单项目1：垃圾邮件分类

## 项目描述

使用逻辑回归构建一个简单的垃圾邮件分类器，能够区分垃圾邮件和正常邮件。

## 项目要求

### 难度等级
- **简单** - 使用现成工具/库，完成基础任务
- **代码量**：< 200行
- **时间**：2-4小时

### 功能要求

1. **数据准备**
   - 加载邮件数据集（可以使用sklearn的示例数据或公开数据集）
   - 数据预处理（文本向量化）

2. **特征提取**
   - 使用TF-IDF向量化文本
   - 选择重要特征

3. **模型训练**
   - 使用逻辑回归训练分类器
   - 评估模型性能

4. **结果评估**
   - 计算准确率、精确率、召回率、F1分数
   - 绘制混淆矩阵
   - 分析分类结果

## 实现步骤

### 步骤1：数据加载

```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

# 加载数据（示例：使用20newsgroups的部分类别）
categories = ['alt.atheism', 'soc.religion.christian']
newsgroups_train = fetch_20newsgroups(subset='train', categories=categories)
newsgroups_test = fetch_20newsgroups(subset='test', categories=categories)

X_train = newsgroups_train.data
y_train = newsgroups_train.target
X_test = newsgroups_test.data
y_test = newsgroups_test.target
```

### 步骤2：特征提取

```python
# TF-IDF向量化
vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)
```

### 步骤3：模型训练

```python
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 训练模型
model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_train_tfidf, y_train)

# 预测
y_pred = model.predict(X_test_tfidf)
```

### 步骤4：评估

```python
# 评估性能
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))
```

## 预期结果

- 模型能够正确分类大部分邮件
- 准确率 > 0.80
- 清晰的评估报告

## 改进方向

1. 尝试不同的特征提取方法
2. 调整TF-IDF参数
3. 尝试不同的分类器
4. 处理类别不平衡（如果有）

## 项目文件

- `main.py` - 主程序
- `README.md` - 项目说明（本文件）

