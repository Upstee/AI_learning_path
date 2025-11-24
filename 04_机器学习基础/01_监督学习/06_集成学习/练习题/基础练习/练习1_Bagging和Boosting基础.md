# 基础练习1：Bagging和Boosting基础

## 练习目标

掌握Bagging和Boosting的基本使用方法。

## 练习要求

### 1. Bagging实现（30分）

1. **使用BaggingClassifier**
   - 使用Bagging进行分类
   - 调整n_estimators参数
   - 对比单棵决策树和Bagging

2. **分析Bagging效果**
   - 观察准确率提升
   - 分析过拟合程度
   - 理解Bagging的优势

### 2. AdaBoost实现（30分）

1. **使用AdaBoostClassifier**
   - 使用AdaBoost进行分类
   - 调整n_estimators和learning_rate
   - 观察学习器权重

2. **分析AdaBoost效果**
   - 观察准确率提升
   - 分析样本权重变化
   - 理解Boosting的优势

### 3. 梯度提升实现（30分）

1. **使用GradientBoostingClassifier**
   - 使用GBDT进行分类
   - 调整主要参数
   - 观察学习曲线

2. **分析GBDT效果**
   - 观察准确率提升
   - 分析特征重要性
   - 理解梯度提升的优势

### 4. 方法对比（10分）

1. **对比不同方法**
   - 对比Bagging、AdaBoost、GBDT
   - 分析各自的优缺点
   - 总结适用场景

## 代码要求

- 使用scikit-learn
- 代码清晰，有注释
- 包含对比分析

## 评估标准

- **正确性**（60分）：所有操作正确
- **理解深度**（30分）：理解不同方法
- **代码质量**（10分）：代码清晰

---

**完成后，请将代码保存为 `练习1_答案.py`，并放在 `答案/` 文件夹中。**

