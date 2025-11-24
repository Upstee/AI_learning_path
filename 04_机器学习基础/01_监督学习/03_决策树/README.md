# 决策树

## 1. 课程概述

### 课程目标
1. 理解决策树的基本原理和构建过程
2. 掌握信息熵、信息增益、基尼不纯度等概念
3. 理解ID3、C4.5、CART算法
4. 能够从零实现决策树算法
5. 掌握剪枝技术防止过拟合
6. 能够使用scikit-learn实现决策树

### 预计学习时间
- **理论学习**：8-10小时
- **代码实践**：10-12小时
- **练习巩固**：8-10小时
- **总计**：26-32小时（约2-3周）

### 难度等级
- **中等** - 需要理解信息论和树结构

### 课程定位
- **前置课程**：02_逻辑回归、02_数学基础（概率统计）
- **后续课程**：04_随机森林、06_集成学习
- **在体系中的位置**：非线性分类器，可解释性强

### 学完能做什么
- 能够理解和使用决策树解决分类和回归问题
- 能够从零实现决策树算法
- 能够进行特征重要性分析
- 能够理解和使用剪枝技术

---

## 2. 前置知识检查

### 必备前置概念清单
- **概率统计**：概率、熵
- **信息论**：信息熵、信息增益
- **树结构**：二叉树、递归
- **Python**：递归、数据结构

### 回顾链接/跳转
- 如果不熟悉信息论：`02_数学基础/02_概率统计/`
- 如果不熟悉树结构：`01_Python进阶/`

### 入门小测

**选择题**（每题2分，共10分）

1. 决策树用于什么任务？
   A. 只用于分类  B. 只用于回归  C. 分类和回归  D. 聚类
   **答案**：C

2. 信息熵越大表示什么？
   A. 信息量越大  B. 不确定性越大  C. 确定性越大  D. 信息量越小
   **答案**：B

3. CART算法使用什么指标？
   A. 信息增益  B. 信息增益比  C. 基尼不纯度  D. 以上都可以
   **答案**：C

4. 剪枝的目的是？
   A. 加速训练  B. 防止过拟合  C. 提高准确率  D. 减少数据
   **答案**：B

5. 决策树的优点不包括？
   A. 可解释性强  B. 不需要特征缩放  C. 处理非线性关系  D. 不容易过拟合
   **答案**：D

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 决策树原理

#### 概念引入与直观类比

**类比**：决策树就像"问答游戏"，通过一系列问题逐步缩小范围。

- **根节点**：第一个问题
- **内部节点**：中间问题
- **叶节点**：最终答案

例如：
- 天气预测：先问"温度>25°C？"，再问"湿度>70%？"
- 疾病诊断：先问"发烧？"，再问"咳嗽？"

#### 逐步理论推导

**步骤1：信息熵**
H(S) = -∑ᵢ pᵢlog₂(pᵢ)

其中pᵢ是类别i的概率。

**步骤2：信息增益**
IG(S, A) = H(S) - ∑ᵥ (|Sᵥ|/|S|)H(Sᵥ)

其中A是特征，Sᵥ是特征A取值为v的样本子集。

**步骤3：基尼不纯度**
Gini(S) = 1 - ∑ᵢ pᵢ²

**步骤4：构建决策树**
1. 选择最佳特征（信息增益最大或基尼不纯度最小）
2. 根据特征值分割数据
3. 递归构建子树
4. 直到满足停止条件

#### 数学公式与必要证明

**信息增益的推导**：

信息增益 = 父节点熵 - 加权子节点熵

IG(S, A) = H(S) - ∑ᵥ (|Sᵥ|/|S|)H(Sᵥ)

选择信息增益最大的特征。

**基尼不纯度的推导**：

基尼不纯度 = 1 - ∑ᵢ pᵢ²

对于二分类：
Gini = 1 - p² - (1-p)² = 2p(1-p)

#### 算法伪代码

```
决策树算法（ID3）：
1. 如果所有样本属于同一类别，返回叶节点
2. 如果特征集为空，返回多数类
3. 选择信息增益最大的特征A
4. 对于特征A的每个取值v：
   a. 创建分支
   b. 递归构建子树
5. 返回树
```

#### 关键性质

**优点**：
- **可解释性强**：决策过程清晰
- **不需要特征缩放**：对特征缩放不敏感
- **处理非线性关系**：可以处理复杂关系
- **特征选择**：自动选择重要特征

**缺点**：
- **容易过拟合**：树太深容易过拟合
- **不稳定**：数据小变化可能导致树大变化
- **偏向多值特征**：信息增益偏向多值特征

**适用场景**：
- 需要可解释性
- 特征混合类型（数值、类别）
- 非线性关系

---

### 3.2 剪枝技术

#### 概念引入与直观类比

**类比**：剪枝就像"修剪树枝"，去掉不必要的部分。

- **过拟合**：树太复杂，在训练集上表现好，但泛化差
- **剪枝**：去掉不必要的分支，简化树

#### 逐步理论推导

**步骤1：预剪枝**
在构建树时提前停止：
- 最大深度
- 最小样本数
- 最小信息增益

**步骤2：后剪枝**
构建完整树后剪枝：
- 自底向上剪枝
- 使用验证集评估
- 如果剪枝后性能不降，则剪枝

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

#### 示例1：从零实现决策树（简化版）

```python
import numpy as np
from collections import Counter

class DecisionTree:
    """决策树类（从零实现，简化版）"""
    
    def __init__(self, max_depth=10, min_samples_split=2):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None
    
    def entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probs = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) for p in probs if p > 0)
    
    def information_gain(self, y, y_left, y_right):
        """计算信息增益"""
        parent_entropy = self.entropy(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n == 0:
            return 0
        
        child_entropy = (n_left / n) * self.entropy(y_left) + (n_right / n) * self.entropy(y_right)
        return parent_entropy - child_entropy
    
    def best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            thresholds = np.unique(X[:, feature_idx])
            
            for threshold in thresholds:
                left_indices = X[:, feature_idx] <= threshold
                right_indices = ~left_indices
                
                if np.sum(left_indices) == 0 or np.sum(right_indices) == 0:
                    continue
                
                y_left = y[left_indices]
                y_right = y[right_indices]
                
                gain = self.information_gain(y, y_left, y_right)
                
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def build_tree(self, X, y, depth=0):
        """递归构建树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or 
            n_classes == 1 or 
            n_samples < self.min_samples_split):
            return Counter(y).most_common(1)[0][0]
        
        # 找到最佳分割
        feature, threshold = self.best_split(X, y)
        
        if feature is None:
            return Counter(y).most_common(1)[0][0]
        
        # 分割数据
        left_indices = X[:, feature] <= threshold
        right_indices = ~left_indices
        
        # 递归构建子树
        left_tree = self.build_tree(X[left_indices], y[left_indices], depth + 1)
        right_tree = self.build_tree(X[right_indices], y[right_indices], depth + 1)
        
        return {
            'feature': feature,
            'threshold': threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """训练模型"""
        self.tree = self.build_tree(X, y)
        return self
    
    def predict_one(self, x, tree):
        """预测单个样本"""
        if not isinstance(tree, dict):
            return tree
        
        if x[tree['feature']] <= tree['threshold']:
            return self.predict_one(x, tree['left'])
        else:
            return self.predict_one(x, tree['right'])
    
    def predict(self, X):
        """预测"""
        return np.array([self.predict_one(x, self.tree) for x in X])

# 生成数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                          n_redundant=0, random_state=42)

# 训练模型
model = DecisionTree(max_depth=5)
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)

print(f"准确率: {accuracy:.4f}")
```

#### 示例2：使用scikit-learn

```python
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import numpy as np

# 生成数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=4, n_classes=3, 
                          random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = DecisionTreeClassifier(max_depth=5, min_samples_split=5, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=[f'特征{i}' for i in range(4)], 
          class_names=[f'类别{i}' for i in range(3)])
plt.title('决策树可视化')
plt.show()

# 特征重要性
print("\n特征重要性:")
for i, importance in enumerate(model.feature_importances_):
    print(f"特征{i}: {importance:.4f}")
```

### 4.3 常见错误与排查

**错误1**：树过深导致过拟合
```python
# 错误：不限制深度
model = DecisionTreeClassifier()

# 正确：限制深度和最小样本数
model = DecisionTreeClassifier(max_depth=5, min_samples_split=5)
```

**错误2**：信息增益偏向多值特征
```python
# 问题：多值特征的信息增益通常更大
# 解决：使用信息增益比（C4.5）或基尼不纯度（CART）
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现决策树**
不使用库，从零实现决策树算法。

**练习2：使用scikit-learn**
使用scikit-learn实现决策树。

**练习3：可视化决策树**
可视化决策树的决策过程。

### 进阶练习（2-3题）

**练习1：剪枝技术**
实现预剪枝和后剪枝。

**练习2：回归树**
实现决策树回归。

### 挑战练习（1-2题）

**练习1：完整的分类系统**
实现完整的分类系统，包括数据预处理、模型训练、评估、可视化。

---

## 6. 实际案例

### 案例：贷款审批决策系统

**业务背景**：
根据客户信息判断是否批准贷款。

**问题抽象**：
- 特征：收入、信用评分、工作年限等
- 目标：批准（1）或拒绝（0）
- 方法：决策树

**端到端实现**：
```python
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt

# 创建模拟数据
np.random.seed(42)
n_samples = 500
data = {
    'income': np.random.normal(50000, 15000, n_samples),
    'credit_score': np.random.randint(300, 850, n_samples),
    'work_years': np.random.randint(0, 30, n_samples),
    'debt_ratio': np.random.uniform(0, 0.5, n_samples)
}

df = pd.DataFrame(data)

# 计算是否批准（模拟规则）
df['approved'] = ((df['income'] > 45000) & 
                  (df['credit_score'] > 600) & 
                  (df['debt_ratio'] < 0.4)).astype(int)

# 准备数据
X = df[['income', 'credit_score', 'work_years', 'debt_ratio']]
y = df['approved']

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = DecisionTreeClassifier(max_depth=4, min_samples_split=20, random_state=42)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策树
plt.figure(figsize=(20, 10))
plot_tree(model, filled=True, feature_names=X.columns, 
          class_names=['拒绝', '批准'])
plt.title('贷款审批决策树')
plt.show()

# 特征重要性
print("\n特征重要性:")
for i, col in enumerate(X.columns):
    print(f"{col}: {model.feature_importances_[i]:.4f}")
```

**结果解读**：
- 决策树清晰显示决策规则
- 特征重要性显示哪些因素最重要

**改进方向**：
- 使用随机森林提升性能
- 添加更多特征
- 处理类别不平衡

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 决策树用于什么任务？
   A. 只用于分类  B. 只用于回归  C. 分类和回归  D. 聚类
   **答案**：C

2. 信息熵越大表示什么？
   A. 信息量越大  B. 不确定性越大  C. 确定性越大  D. 信息量越小
   **答案**：B

3. CART算法使用什么指标？
   A. 信息增益  B. 信息增益比  C. 基尼不纯度  D. 以上都可以
   **答案**：C

4. 剪枝的目的是？
   A. 加速训练  B. 防止过拟合  C. 提高准确率  D. 减少数据
   **答案**：B

5. 决策树的优点不包括？
   A. 可解释性强  B. 不需要特征缩放  C. 处理非线性关系  D. 不容易过拟合
   **答案**：D

**简答题**（每题10分，共40分）

1. 解释信息熵和信息增益的含义。
   **参考答案**：信息熵衡量不确定性，信息增益衡量特征对分类的贡献。

2. 说明决策树容易过拟合的原因。
   **参考答案**：树太深会记住训练数据的细节，导致泛化能力差。

3. 解释预剪枝和后剪枝的区别。
   **参考答案**：预剪枝在构建时停止，后剪枝在构建后剪掉分支。

4. 说明决策树的优缺点。
   **参考答案**：优点：可解释、不需要特征缩放；缺点：容易过拟合、不稳定。

### 编程实践题（20分）

从零实现决策树算法，包括信息增益计算和树构建。

### 综合应用题（20分）

使用决策树解决真实问题，包括数据预处理、模型训练、评估、可视化。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第4章）
- 《统计学习方法》- 李航（第5章）

**在线资源**：
- scikit-learn官方文档
- 决策树可视化工具

### 相关工具与库

- **scikit-learn**：DecisionTreeClassifier
- **graphviz**：树可视化
- **pandas**：数据处理

### 进阶话题指引

完成本课程后，可以学习：
- **随机森林**：集成多个决策树
- **梯度提升树**：GBDT、XGBoost
- **剪枝技术**：更高级的剪枝方法

### 下节课预告

下一课将学习：
- **04_随机森林**：集成多个决策树
- 随机森林通过集成提升性能和稳定性

### 学习建议

1. **理解信息论**：理解信息熵和信息增益
2. **多实践**：从零实现算法，加深理解
3. **可视化**：可视化决策树，理解决策过程
4. **持续学习**：决策树是集成学习的基础

---

**恭喜完成第三课！你已经掌握了决策树，准备好学习随机森林了！**

