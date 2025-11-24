# 模型评估与优化

## 1. 课程概述

### 课程目标
1. 理解模型评估的重要性和评估指标（准确率、精确率、召回率、F1、ROC、AUC）
2. 掌握交叉验证方法（K折交叉验证、留一法、分层交叉验证）
3. 理解过拟合和欠拟合问题
4. 掌握模型优化方法（超参数调优、特征选择、正则化）
5. 能够使用scikit-learn进行模型评估和优化
6. 能够应用评估和优化技术提升模型性能

### 预计学习时间
- **理论学习**：10-12小时
- **代码实践**：12-14小时
- **练习巩固**：10-12小时
- **总计**：32-38小时（约4-5周）

### 难度等级
- **中等偏上** - 需要理解多种评估指标和优化方法

### 课程定位
- **前置课程**：01_监督学习、02_无监督学习
- **后续课程**：04_实战项目、05_深度学习基础
- **在体系中的位置**：模型开发的关键环节，确保模型质量

### 学完能做什么
- 能够正确评估模型性能
- 能够选择合适的评估指标
- 能够使用交叉验证
- 能够进行超参数调优
- 能够优化模型性能

---

## 2. 前置知识检查

### 必备前置概念清单
- **监督学习**：分类、回归
- **概率统计**：混淆矩阵、ROC曲线
- **NumPy、Pandas**：数据处理
- **scikit-learn**：基本使用

### 回顾链接/跳转
- 如果不熟悉分类：`04_机器学习基础/01_监督学习/`
- 如果不熟悉概率统计：`02_数学基础/02_概率统计/`
- 如果不熟悉scikit-learn：`03_数据处理基础/`

### 入门小测

**选择题**（每题2分，共10分）

1. 准确率适用于哪种情况？
   A. 类别平衡  B. 类别不平衡  C. 回归问题  D. 聚类问题
   **答案**：A

2. 精确率衡量的是？
   A. 预测为正例中真正例的比例  B. 真正例中预测为正例的比例  C. 准确率  D. 召回率
   **答案**：A

3. ROC曲线的横轴是？
   A. 真正例率  B. 假正例率  C. 精确率  D. 召回率
   **答案**：B

4. K折交叉验证中，K通常取？
   A. 2-3  B. 5-10  C. 20-30  D. 50+
   **答案**：B

5. 过拟合的表现是？
   A. 训练集和测试集误差都高  B. 训练集误差低，测试集误差高  C. 训练集和测试集误差都低  D. 训练集误差高
   **答案**：B

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 分类评估指标

#### 混淆矩阵

|  | 预测正例 | 预测负例 |
|--|---------|---------|
| 实际正例 | TP | FN |
| 实际负例 | FP | TN |

#### 评估指标

**准确率（Accuracy）**：
$$Accuracy = \frac{TP + TN}{TP + TN + FP + FN}$$

**精确率（Precision）**：
$$Precision = \frac{TP}{TP + FP}$$

**召回率（Recall）**：
$$Recall = \frac{TP}{TP + FN}$$

**F1分数**：
$$F1 = 2 \times \frac{Precision \times Recall}{Precision + Recall}$$

**ROC曲线和AUC**：
- ROC曲线：以假正例率为横轴，真正例率为纵轴
- AUC：ROC曲线下面积，值越大越好

---

### 3.2 回归评估指标

**均方误差（MSE）**：
$$MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2$$

**均方根误差（RMSE）**：
$$RMSE = \sqrt{MSE}$$

**平均绝对误差（MAE）**：
$$MAE = \frac{1}{n}\sum_{i=1}^{n}|y_i - \hat{y}_i|$$

**R²分数**：
$$R^2 = 1 - \frac{\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{n}(y_i - \bar{y})^2}$$

---

### 3.3 交叉验证

#### K折交叉验证

将数据分成K份，每次用K-1份训练，1份测试，重复K次。

**优点**：
- 充分利用数据
- 减少随机性影响
- 更可靠的性能估计

#### 分层交叉验证

保持每折中类别比例与原始数据一致。

---

### 3.4 模型优化

#### 超参数调优

**网格搜索（Grid Search）**：遍历所有参数组合

**随机搜索（Random Search）**：随机采样参数组合

**贝叶斯优化**：使用贝叶斯方法选择参数

#### 特征选择

**过滤法**：基于统计特征选择

**包装法**：基于模型性能选择

**嵌入法**：模型训练过程中选择

---

## 4. Python代码实践

### 4.1 分类评估

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

# 评估分类模型
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]
y_pred = [0, 1, 1, 0, 0, 0, 1, 1, 0, 1]

# 基本指标
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"准确率: {accuracy:.4f}")
print(f"精确率: {precision:.4f}")
print(f"召回率: {recall:.4f}")
print(f"F1分数: {f1:.4f}")

# 混淆矩阵
cm = confusion_matrix(y_true, y_pred)
print("\n混淆矩阵:")
print(cm)

# 详细报告
print("\n分类报告:")
print(classification_report(y_true, y_pred))

# ROC曲线
y_scores = [0.1, 0.9, 0.8, 0.2, 0.3, 0.1, 0.9, 0.8, 0.2, 0.9]
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 4.2 交叉验证

```python
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 创建模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)

# K折交叉验证
cv_scores = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
print(f"交叉验证准确率: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

# 分层交叉验证
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores_stratified = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
print(f"分层交叉验证准确率: {cv_scores_stratified.mean():.4f} ± {cv_scores_stratified.std():.4f}")
```

---

### 4.3 超参数调优

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import load_iris
from scipy.stats import randint

# 加载数据
iris = load_iris()
X, y = iris.data, iris.target

# 定义参数网格
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

# 网格搜索
clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X, y)

print("最佳参数:", grid_search.best_params_)
print("最佳得分:", grid_search.best_score_)

# 随机搜索
param_dist = {
    'n_estimators': randint(50, 200),
    'max_depth': [3, 5, 7, None],
    'min_samples_split': randint(2, 10)
}

random_search = RandomizedSearchCV(clf, param_dist, n_iter=20, cv=5, 
                                   scoring='accuracy', random_state=42, n_jobs=-1)
random_search.fit(X, y)

print("\n随机搜索最佳参数:", random_search.best_params_)
print("随机搜索最佳得分:", random_search.best_score_)
```

---

## 5. 动手练习（分层次）

### 基础练习（3-5题）⚠️【必须至少3题，难度递增】

#### 练习1：实现分类评估指标
**目标**：从零实现分类评估指标

**要求**：
1. 实现准确率、精确率、召回率、F1
2. 实现混淆矩阵
3. 在模拟数据上测试
4. 与scikit-learn结果对比

**难度**：⭐⭐

---

#### 练习2：实现交叉验证
**目标**：从零实现K折交叉验证

**要求**：
1. 实现K折交叉验证
2. 实现分层交叉验证
3. 在真实数据集上测试
4. 与scikit-learn结果对比

**难度**：⭐⭐⭐

---

#### 练习3：超参数调优实践
**目标**：使用网格搜索和随机搜索进行超参数调优

**要求**：
1. 使用网格搜索
2. 使用随机搜索
3. 比较两种方法
4. 分析最优参数

**难度**：⭐⭐⭐

---

### 进阶练习（2-3题）⚠️【必须至少2题，难度递增】

#### 练习1：完整的模型评估流程
**目标**：构建完整的模型评估系统

**要求**：
1. 实现多种评估指标
2. 实现交叉验证
3. 可视化评估结果
4. 生成评估报告

**难度**：⭐⭐⭐⭐

---

#### 练习2：模型优化系统
**目标**：构建自动化的模型优化系统

**要求**：
1. 实现超参数调优
2. 实现特征选择
3. 实现模型集成
4. 自动化优化流程

**难度**：⭐⭐⭐⭐

---

### 挑战练习（1-2题）⚠️【必须至少1题】

#### 练习1：大规模模型评估与优化
**目标**：处理大规模数据的模型评估和优化

**要求**：
1. 实现增量交叉验证
2. 优化超参数搜索效率
3. 处理大规模数据
4. 并行化评估过程
5. 实现分布式优化

**难度**：⭐⭐⭐⭐⭐

---

## 6. 实际案例

### 案例1：分类模型评估（简单项目）

**业务背景**：评估一个二分类模型的性能。

**端到端实现**：
```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt

# 生成数据
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, 
                           n_redundant=10, random_state=42)

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)
y_proba = clf.predict_proba(X_test)[:, 1]

# 评估
print("分类报告:")
print(classification_report(y_test, y_pred))

print("\n混淆矩阵:")
print(confusion_matrix(y_test, y_pred))

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('假正例率')
plt.ylabel('真正例率')
plt.title('ROC曲线')
plt.legend()
plt.grid(True)
plt.show()
```

---

### 案例2：回归模型评估（中等项目）

**业务背景**：评估一个回归模型的性能。

**端到端实现**：
```python
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.datasets import load_boston
import numpy as np

# 加载数据
boston = load_boston()
X, y = boston.data, boston.target

# 创建模型
reg = RandomForestRegressor(n_estimators=100, random_state=42)

# 交叉验证
cv_scores = cross_val_score(reg, X, y, cv=5, scoring='neg_mean_squared_error')
rmse_scores = np.sqrt(-cv_scores)

print(f"交叉验证RMSE: {rmse_scores.mean():.2f} ± {rmse_scores.std():.2f}")

# 训练和评估
reg.fit(X, y)
y_pred = reg.predict(X)

mse = mean_squared_error(y, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)

print(f"\nMSE: {mse:.2f}")
print(f"RMSE: {rmse:.2f}")
print(f"MAE: {mae:.2f}")
print(f"R²: {r2:.4f}")
```

---

### 案例3：完整的模型优化流程（进阶项目）

**业务背景**：构建一个完整的模型优化系统，包括特征选择、超参数调优、模型评估。

**端到端实现**：
```python
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.datasets import load_breast_cancer
import numpy as np

# 加载数据
cancer = load_breast_cancer()
X, y = cancer.data, cancer.target

# 划分数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 步骤1：特征选择
selector = SelectKBest(f_classif, k=10)
X_train_selected = selector.fit_transform(X_train, y_train)
X_test_selected = selector.transform(X_test)

print(f"原始特征数: {X_train.shape[1]}")
print(f"选择特征数: {X_train_selected.shape[1]}")

# 步骤2：超参数调优
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10]
}

clf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(clf, param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid_search.fit(X_train_selected, y_train)

print(f"\n最佳参数: {grid_search.best_params_}")
print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")

# 步骤3：最终评估
best_clf = grid_search.best_estimator_
y_pred = best_clf.predict(X_test_selected)
y_proba = best_clf.predict_proba(X_test_selected)[:, 1]

print("\n测试集性能:")
print(classification_report(y_test, y_pred))
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba):.4f}")
```

---

## 7. 自我评估

### 概念题

#### 选择题（10-15道）

1. 准确率适用于哪种情况？
   A. 类别平衡  B. 类别不平衡  C. 回归问题  D. 聚类问题
   **答案**：A

2. 精确率衡量的是？
   A. 预测为正例中真正例的比例  B. 真正例中预测为正例的比例  C. 准确率  D. 召回率
   **答案**：A

3. ROC曲线的横轴是？
   A. 真正例率  B. 假正例率  C. 精确率  D. 召回率
   **答案**：B

4. K折交叉验证中，K通常取？
   A. 2-3  B. 5-10  C. 20-30  D. 50+
   **答案**：B

5. 过拟合的表现是？
   A. 训练集和测试集误差都高  B. 训练集误差低，测试集误差高  C. 训练集和测试集误差都低  D. 训练集误差高
   **答案**：B

#### 简答题（5-8道）

1. 解释准确率、精确率、召回率、F1分数的含义和适用场景。
   **参考答案**：
   - 准确率：适用于类别平衡的情况
   - 精确率：关注预测为正例的准确性
   - 召回率：关注真正例的识别率
   - F1分数：精确率和召回率的调和平均

2. 说明交叉验证的原理和优势。
   **参考答案**：将数据分成K份，每次用K-1份训练，1份测试，重复K次。优势：充分利用数据，减少随机性，更可靠的性能估计。

---

### 编程实践题（2-3道）

#### 题目1：实现分类评估系统
**要求**：
1. 实现多种评估指标
2. 实现混淆矩阵
3. 实现ROC曲线
4. 生成评估报告

**评分标准**：
- 正确实现指标（40分）
- 可视化清晰（20分）
- 报告完整（20分）
- 代码质量（20分）

---

### 综合应用题（1-2道）

#### 题目1：构建完整的模型优化系统
**要求**：
1. 实现特征选择
2. 实现超参数调优
3. 实现交叉验证
4. 生成优化报告
5. 分析优化效果

**评分标准**：
- 功能实现正确（30分）
- 优化效果明显（30分）
- 分析深入（20分）
- 代码质量（20分）

---

## 8. 拓展学习

### 论文推荐

1. **Kohavi, R. (1995). "A study of cross-validation and bootstrap for accuracy estimation and model selection."** IJCAI
   - 交叉验证经典论文

### 书籍推荐

1. **《机器学习》- 周志华**
   - 第2章：模型评估与选择

2. **《统计学习方法》- 李航**
   - 模型评估相关章节

### 相关工具与库

1. **scikit-learn**
   - 模型评估和优化工具
   - 文档：https://scikit-learn.org/stable/modules/model_evaluation.html

2. **optuna**
   - 超参数优化框架
   - GitHub: https://github.com/optuna/optuna

### 进阶话题指引

1. **高级评估方法**
   - 时间序列交叉验证
   - 嵌套交叉验证
   - 自助法（Bootstrap）

2. **高级优化方法**
   - 贝叶斯优化
   - 进化算法
   - 强化学习优化

3. **模型解释性**
   - SHAP值
   - LIME
   - 特征重要性

### 下节课预告与学习建议

**下节课**：`04_实战项目`

**学习建议**：
1. 完成所有练习题
2. 理解不同评估指标的适用场景
3. 掌握交叉验证方法
4. 了解模型优化的策略

**前置准备**：
- 复习监督学习和无监督学习
- 准备真实数据集
- 了解项目开发流程

---

**完成本课程后，你将能够：**
- ✅ 正确评估模型性能
- ✅ 选择合适的评估指标
- ✅ 使用交叉验证
- ✅ 进行超参数调优
- ✅ 优化模型性能

**继续学习，成为AI大师！** 🚀

