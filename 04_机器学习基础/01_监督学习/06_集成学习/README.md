# 集成学习

## 1. 课程概述

### 课程目标
1. 理解集成学习的基本思想和原理
2. 掌握Bagging、Boosting、Stacking三种集成方法
3. 理解为什么集成能提升性能
4. 能够实现和使用各种集成方法
5. 掌握梯度提升树（GBDT、XGBoost、LightGBM）
6. 能够进行模型融合和超参数调优

### 预计学习时间
- **理论学习**：10-12小时
- **代码实践**：12-15小时
- **练习巩固**：10-12小时
- **总计**：32-39小时（约3-4周）

### 难度等级
- **中高** - 需要理解多种集成方法和优化算法

### 课程定位
- **前置课程**：03_决策树、04_随机森林、02_数学基础（优化理论）
- **后续课程**：05_深度学习基础
- **在体系中的位置**：提升模型性能的重要方法，实际应用广泛

### 学完能做什么
- 能够理解和使用各种集成学习方法
- 能够实现和使用梯度提升树
- 能够进行模型融合和超参数调优
- 能够理解集成学习的优势和局限性

---

## 2. 前置知识检查

### 必备前置概念清单
- **决策树**：理解决策树算法
- **随机森林**：理解Bagging方法
- **优化理论**：理解梯度下降和优化
- **概率统计**：理解偏差和方差

### 回顾链接/跳转
- 如果不熟悉随机森林：`04_机器学习基础/01_监督学习/04_随机森林/`
- 如果不熟悉优化理论：`02_数学基础/04_优化理论/`

### 入门小测

**选择题**（每题2分，共10分）

1. 集成学习的三种主要方法是？
   A. Bagging, Boosting, Stacking  B. 决策树, 随机森林, SVM  C. 线性, 非线性, 核方法  D. 分类, 回归, 聚类
   **答案**：A

2. Bagging的主要思想是？
   A. 顺序训练模型  B. 并行训练模型  C. 堆叠模型  D. 融合模型
   **答案**：B

3. Boosting的主要思想是？
   A. 并行训练  B. 顺序训练，关注错误样本  C. 随机采样  D. 投票
   **答案**：B

4. 梯度提升树使用什么优化方法？
   A. 最小二乘法  B. 梯度下降  C. 最大似然  D. 信息增益
   **答案**：B

5. XGBoost相比GBDT的主要改进不包括？
   A. 正则化  B. 并行计算  C. 缺失值处理  D. 更简单的算法
   **答案**：D

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 集成学习原理

#### 概念引入与直观类比

**类比**：集成学习就像"三个臭皮匠顶个诸葛亮"，多个弱模型组合成强模型。

- **弱模型**：单个模型可能不够好
- **集成**：多个模型组合
- **强模型**：组合后性能提升

例如：
- 投票：多个专家投票做决策
- 平均：多个预测取平均
- 加权：根据表现加权组合

#### 逐步理论推导

**步骤1：偏差-方差分解**

泛化误差 = 偏差² + 方差 + 噪声

- **偏差**：模型的拟合能力
- **方差**：模型对数据变化的敏感度
- **集成**：减少方差，可能略微增加偏差

**步骤2：Bagging（并行）**

1. 从训练集中有放回地随机采样
2. 每个子集训练一个模型
3. 所有模型投票或平均

**步骤3：Boosting（顺序）**

1. 训练第一个模型
2. 关注第一个模型的错误样本
3. 训练第二个模型修正错误
4. 重复，逐步提升

**步骤4：Stacking（堆叠）**

1. 训练多个基学习器
2. 用基学习器的预测作为新特征
3. 训练元学习器（第二层模型）

#### 数学公式与必要证明

**Bagging的方差减少**：

单模型方差：Var(f)

Bagging方差：Var(1/n ∑ᵢ fᵢ) = (1/n)Var(f)（当模型不相关时）

**Boosting的误差减少**：

第t轮的误差：εₜ

最终误差：ε ≤ ∏ₜ (2√(εₜ(1-εₜ)))

当每个弱分类器略好于随机猜测时，误差指数下降。

#### 算法伪代码

```
Bagging算法：
1. 对于 i = 1 到 n_models：
   a. 从训练集中有放回地随机采样，得到子集D_i
   b. 使用D_i训练模型M_i
2. 对于新样本x：
   a. 每模型预测：y_i = M_i(x)
   b. 最终预测：投票或平均
3. 返回预测结果

Boosting算法（AdaBoost）：
1. 初始化样本权重w_i = 1/n
2. 对于 t = 1 到 T：
   a. 训练弱分类器h_t（使用权重w）
   b. 计算错误率ε_t
   c. 计算权重α_t = (1/2)ln((1-ε_t)/ε_t)
   d. 更新样本权重：w_i = w_i * exp(-α_t * y_i * h_t(x_i))
   e. 归一化权重
3. 最终分类器：H(x) = sign(∑α_t * h_t(x))
```

#### 关键性质

**优点**：
- **高准确率**：集成多个模型，准确率高
- **稳定性好**：不容易受数据小变化影响
- **泛化能力强**：减少过拟合风险
- **灵活性强**：可以组合不同类型的模型

**缺点**：
- **计算成本高**：需要训练多个模型
- **可解释性差**：不如单模型可解释
- **内存占用大**：需要存储多个模型
- **可能过拟合**：如果基模型过拟合，集成也可能过拟合

**适用场景**：
- 需要高准确率
- 有足够的计算资源
- 不需要强可解释性
- 数据量大

---

### 3.2 梯度提升树（GBDT）

#### 概念引入与直观类比

**类比**：梯度提升就像"逐步改进"，每一步都修正前一步的错误。

- **第一步**：训练一个模型
- **第二步**：训练新模型修正第一步的残差
- **重复**：逐步减少误差

#### 逐步理论推导

**步骤1：前向分步算法**

F₀(x) = 0
Fₘ(x) = Fₘ₋₁(x) + αₘhₘ(x)

**步骤2：梯度提升**

使用负梯度作为残差的近似：
rᵢₘ = -[∂L(yᵢ, F(xᵢ))/∂F(xᵢ)]|F=Fₘ₋₁

**步骤3：训练决策树**

使用残差rᵢₘ训练决策树hₘ(x)

**步骤4：更新模型**

Fₘ(x) = Fₘ₋₁(x) + αₘhₘ(x)

#### 关键性质

**GBDT**：
- 使用CART回归树
- 使用负梯度作为残差
- 逐步减少损失

**XGBoost**：
- 添加L1和L2正则化
- 使用二阶梯度信息
- 并行计算和缺失值处理

**LightGBM**：
- 使用直方图算法加速
- 使用Leaf-wise生长策略
- 支持类别特征

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy >= 1.20.0
  - pandas >= 1.3.0
  - matplotlib >= 3.3.0
  - scikit-learn >= 0.24.0
  - xgboost >= 1.5.0
  - lightgbm >= 3.3.0

### 4.2 从零开始的完整可运行示例

#### 示例1：Bagging（使用scikit-learn）

```python
from sklearn.ensemble import BaggingClassifier, BaggingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification
import numpy as np

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 单棵决策树
single_tree = DecisionTreeClassifier(random_state=42)
single_tree.fit(X_train, y_train)
single_score = single_tree.score(X_test, y_test)
print(f"单棵决策树准确率: {single_score:.4f}")

# Bagging
bagging = BaggingClassifier(
    base_estimator=DecisionTreeClassifier(),
    n_estimators=10,
    random_state=42
)
bagging.fit(X_train, y_train)
bagging_score = bagging.score(X_test, y_test)
print(f"Bagging准确率: {bagging_score:.4f}")
print(f"提升: {bagging_score - single_score:.4f}")
```

#### 示例2：AdaBoost

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# AdaBoost
adaboost = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=50,
    learning_rate=1.0,
    random_state=42
)
adaboost.fit(X_train, y_train)
adaboost_score = adaboost.score(X_test, y_test)
print(f"AdaBoost准确率: {adaboost_score:.4f}")

# 查看特征重要性
print("\n特征重要性:")
for i, importance in enumerate(adaboost.feature_importances_):
    print(f"特征{i}: {importance:.4f}")
```

#### 示例3：XGBoost

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# XGBoost
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBoost准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 特征重要性
print("\n特征重要性（前5）:")
feature_importance = model.feature_importances_
top5 = sorted(range(len(feature_importance)), 
              key=lambda i: feature_importance[i], reverse=True)[:5]
for i in top5:
    print(f"特征{i}: {feature_importance[i]:.4f}")
```

#### 示例4：LightGBM

```python
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LightGBM
model = lgb.LGBMClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42
)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"LightGBM准确率: {accuracy:.4f}")
```

#### 示例5：Stacking

```python
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.datasets import make_classification

# 生成数据
X, y = make_classification(n_samples=500, n_features=10, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义基学习器
base_learners = [
    ('dt', DecisionTreeClassifier(random_state=42)),
    ('svm', SVC(probability=True, random_state=42)),
    ('lr', LogisticRegression(random_state=42))
]

# 元学习器
meta_learner = LogisticRegression()

# Stacking
stacking = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5
)
stacking.fit(X_train, y_train)

# 预测
y_pred = stacking.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Stacking准确率: {accuracy:.4f}")

# 对比单个模型
for name, model in base_learners:
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name}准确率: {score:.4f}")
```

### 4.3 常见错误与排查

**错误1**：过拟合
```python
# 错误：树太深或学习率太大
model = xgb.XGBClassifier(max_depth=20, learning_rate=1.0)

# 正确：限制深度和学习率
model = xgb.XGBClassifier(max_depth=5, learning_rate=0.1)
```

**错误2**：未使用交叉验证
```python
# 错误：直接使用全部数据
model.fit(X, y)

# 正确：使用交叉验证（Stacking中）
stacking = StackingClassifier(estimators=..., cv=5)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：实现Bagging**
从零实现Bagging算法。

**练习2：实现AdaBoost**
从零实现AdaBoost算法。

**练习3：使用XGBoost**
使用XGBoost解决分类和回归问题。

### 进阶练习（2-3题）

**练习1：实现Stacking**
实现Stacking集成方法。

**练习2：超参数调优**
使用网格搜索调优XGBoost参数。

### 挑战练习（1-2题）

**练习1：完整的集成系统**
实现完整的集成系统，包括多种集成方法、模型融合、评估。

---

## 6. 实际案例

### 案例：房价预测系统（集成学习）

**业务背景**：
使用集成学习预测房价。

**问题抽象**：
- 特征：面积、位置、房龄等
- 目标：房价
- 方法：XGBoost + LightGBM + Stacking

**端到端实现**：
```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import lightgbm as lgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.ensemble import StackingRegressor

# 创建模拟数据
np.random.seed(42)
n_samples = 1000
data = {
    'area': np.random.normal(100, 20, n_samples),
    'location_score': np.random.uniform(1, 10, n_samples),
    'age': np.random.randint(0, 30, n_samples),
    'bedrooms': np.random.randint(1, 5, n_samples)
}

df = pd.DataFrame(data)
df['price'] = (df['area'] * 1000 + 
               df['location_score'] * 10000 + 
               df['age'] * -2000 + 
               np.random.normal(0, 50000, n_samples))

# 准备数据
X = df[['area', 'location_score', 'age', 'bedrooms']]
y = df['price']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 单个模型
models = {
    'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, random_state=42),
    'LightGBM': lgb.LGBMRegressor(n_estimators=100, max_depth=5, random_state=42),
    'RandomForest': RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
}

results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    results[name] = {'RMSE': rmse, 'R²': r2}
    print(f"{name}: RMSE={rmse:.2f}, R²={r2:.4f}")

# Stacking
base_learners = [
    ('xgb', xgb.XGBRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ('lgb', lgb.LGBMRegressor(n_estimators=50, max_depth=5, random_state=42)),
    ('rf', RandomForestRegressor(n_estimators=50, max_depth=5, random_state=42))
]

stacking = StackingRegressor(
    estimators=base_learners,
    final_estimator=Ridge(),
    cv=5
)
stacking.fit(X_train_scaled, y_train)
y_pred_stacking = stacking.predict(X_test_scaled)
rmse_stacking = np.sqrt(mean_squared_error(y_test, y_pred_stacking))
r2_stacking = r2_score(y_test, y_pred_stacking)

print(f"\nStacking: RMSE={rmse_stacking:.2f}, R²={r2_stacking:.4f}")
print(f"相比最佳单模型提升: {min([r['RMSE'] for r in results.values()]) - rmse_stacking:.2f}")
```

**结果解读**：
- Stacking通常能提升性能
- 不同模型组合效果更好

**改进方向**：
- 添加更多基学习器
- 使用更复杂的元学习器
- 特征工程

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 集成学习的三种主要方法是？
   A. Bagging, Boosting, Stacking  B. 决策树, 随机森林, SVM  C. 线性, 非线性, 核方法  D. 分类, 回归, 聚类
   **答案**：A

2. Bagging的主要思想是？
   A. 顺序训练模型  B. 并行训练模型  C. 堆叠模型  D. 融合模型
   **答案**：B

3. Boosting的主要思想是？
   A. 并行训练  B. 顺序训练，关注错误样本  C. 随机采样  D. 投票
   **答案**：B

4. 梯度提升树使用什么优化方法？
   A. 最小二乘法  B. 梯度下降  C. 最大似然  D. 信息增益
   **答案**：B

5. XGBoost相比GBDT的主要改进不包括？
   A. 正则化  B. 并行计算  C. 缺失值处理  D. 更简单的算法
   **答案**：D

**简答题**（每题10分，共40分）

1. 解释集成学习为什么能提升性能。
   **参考答案**：通过组合多个模型，减少方差，提高泛化能力。偏差-方差分解说明集成可以减少方差。

2. 说明Bagging和Boosting的区别。
   **参考答案**：Bagging并行训练多个模型，通过投票或平均；Boosting顺序训练，关注错误样本，逐步提升。

3. 解释梯度提升树的工作原理。
   **参考答案**：使用前向分步算法，每一步训练新模型修正前一步的残差（用负梯度近似），逐步减少损失。

4. 说明Stacking的原理。
   **参考答案**：训练多个基学习器，用它们的预测作为新特征，训练元学习器（第二层模型）进行最终预测。

### 编程实践题（20分）

实现Bagging或AdaBoost算法，包括训练和预测。

### 综合应用题（20分）

使用集成学习解决真实问题，包括多种集成方法、模型融合、评估。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第8章）
- 《统计学习方法》- 李航（第8章）
- 《XGBoost: A Scalable Tree Boosting System》论文

**在线资源**：
- XGBoost官方文档
- LightGBM官方文档
- scikit-learn官方文档

### 相关工具与库

- **scikit-learn**：BaggingClassifier, AdaBoostClassifier, StackingClassifier
- **xgboost**：XGBClassifier, XGBRegressor
- **lightgbm**：LGBMClassifier, LGBMRegressor

### 进阶话题指引

完成本课程后，可以学习：
- **更高级的集成方法**：Blending, Voting
- **神经网络集成**：深度学习的集成方法
- **AutoML**：自动机器学习

### 下节课预告

下一课将学习：
- **07_朴素贝叶斯**：基于概率的分类器
- 朴素贝叶斯简单高效，适合文本分类

### 学习建议

1. **理解原理**：理解为什么集成能提升性能
2. **多实践**：尝试不同的集成方法和参数
3. **对比方法**：对比不同集成方法的效果
4. **持续学习**：集成学习是提升性能的重要方法

---

**恭喜完成第六课！你已经掌握了集成学习，准备好学习朴素贝叶斯了！**

