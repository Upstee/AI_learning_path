# 朴素贝叶斯

## 1. 课程概述

### 课程目标
1. 理解贝叶斯定理和朴素贝叶斯原理
2. 掌握三种常见的朴素贝叶斯模型（高斯、多项式、伯努利）
3. 理解"朴素"假设的含义和影响
4. 能够从零实现朴素贝叶斯算法
5. 能够使用scikit-learn实现朴素贝叶斯
6. 掌握朴素贝叶斯在文本分类中的应用

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：8-10小时
- **练习巩固**：6-8小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **中等** - 需要理解概率和贝叶斯定理

### 课程定位
- **前置课程**：02_数学基础（概率统计）、01_逻辑回归
- **后续课程**：08_KNN、05_深度学习基础
- **在体系中的位置**：基于概率的分类器，简单高效，适合文本分类

### 学完能做什么
- 能够理解和使用朴素贝叶斯解决分类问题
- 能够从零实现朴素贝叶斯算法
- 能够处理文本分类问题
- 能够理解概率输出和特征重要性

---

## 2. 前置知识检查

### 必备前置概念清单
- **概率统计**：条件概率、贝叶斯定理
- **概率分布**：高斯分布、多项式分布、伯努利分布
- **文本处理**：词频、TF-IDF

### 回顾链接/跳转
- 如果不熟悉贝叶斯定理：`02_数学基础/02_概率统计/`
- 如果不熟悉文本处理：`03_数据处理基础/`

### 入门小测

**选择题**（每题2分，共10分）

1. 朴素贝叶斯的"朴素"假设是？
   A. 特征独立  B. 特征相关  C. 特征相同  D. 特征连续
   **答案**：A

2. 贝叶斯定理的公式是？
   A. P(A|B) = P(B|A)P(A)/P(B)  B. P(A|B) = P(A)P(B)  C. P(A|B) = P(A)/P(B)  D. P(A|B) = P(B|A)
   **答案**：A

3. 文本分类通常使用什么朴素贝叶斯？
   A. 高斯  B. 多项式  C. 伯努利  D. B和C
   **答案**：D

4. 朴素贝叶斯的优点不包括？
   A. 简单快速  B. 需要大量数据  C. 适合文本分类  D. 概率输出
   **答案**：B

5. 拉普拉斯平滑的作用是？
   A. 加速计算  B. 处理零概率问题  C. 减少参数  D. 提高准确率
   **答案**：B

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 朴素贝叶斯原理

#### 概念引入与直观类比

**类比**：朴素贝叶斯就像"根据症状诊断疾病"，根据多个症状的概率判断最可能的疾病。

- **症状**：特征
- **疾病**：类别
- **诊断**：根据症状概率判断疾病

例如：
- 垃圾邮件分类：根据词的出现概率判断是否为垃圾邮件
- 疾病诊断：根据症状概率判断疾病

#### 逐步理论推导

**步骤1：贝叶斯定理**

P(y|x) = P(x|y)P(y) / P(x)

其中：
- P(y|x)：后验概率（给定特征x，类别y的概率）
- P(x|y)：似然（给定类别y，特征x的概率）
- P(y)：先验概率（类别y的概率）
- P(x)：证据（特征x的概率）

**步骤2：朴素假设**

假设特征之间相互独立：
P(x₁, x₂, ..., xₙ|y) = P(x₁|y)P(x₂|y)...P(xₙ|y)

**步骤3：分类决策**

选择后验概率最大的类别：
y = argmax P(y|x) = argmax P(x|y)P(y)

**步骤4：三种模型**

- **高斯朴素贝叶斯**：特征服从高斯分布
- **多项式朴素贝叶斯**：特征为计数（如词频）
- **伯努利朴素贝叶斯**：特征为二值（出现/不出现）

#### 数学公式与必要证明

**贝叶斯定理的推导**：

从条件概率定义：
P(y|x) = P(x, y) / P(x)
P(x|y) = P(x, y) / P(y)

因此：
P(x, y) = P(y|x)P(x) = P(x|y)P(y)

所以：
P(y|x) = P(x|y)P(y) / P(x)

**朴素假设下的分类**：

y = argmax P(y|x)
  = argmax P(x|y)P(y) / P(x)
  = argmax P(x|y)P(y)  （P(x)对所有y相同，可忽略）
  = argmax P(y) ∏ᵢ P(xᵢ|y)

#### 算法伪代码

```
朴素贝叶斯算法（训练）：
1. 计算每个类别的先验概率：P(y)
2. 对于每个特征x_i和每个类别y：
   a. 计算条件概率：P(x_i|y)
3. 返回先验概率和条件概率

朴素贝叶斯算法（预测）：
1. 对于每个类别y：
   a. 计算后验概率：P(y|x) = P(y) * ∏P(x_i|y)
2. 返回概率最大的类别
```

#### 关键性质

**优点**：
- **简单快速**：算法简单，训练和预测都快
- **适合小样本**：即使数据少也能工作
- **适合文本分类**：在文本分类中表现好
- **概率输出**：输出是概率，可解释
- **对噪声鲁棒**：对噪声不敏感

**缺点**：
- **特征独立假设**：现实中特征往往相关
- **可能欠拟合**：如果假设不成立，性能可能差
- **需要特征分布假设**：需要假设特征分布

**适用场景**：
- 文本分类（垃圾邮件、情感分析）
- 小样本问题
- 需要快速预测
- 特征相对独立

---

### 3.2 三种模型

#### 高斯朴素贝叶斯

假设特征服从高斯分布：
P(xᵢ|y) = (1/√(2πσ²)) exp(-(xᵢ-μ)²/(2σ²))

适用于连续特征。

#### 多项式朴素贝叶斯

假设特征为计数（如词频）：
P(xᵢ|y) = (Nᵢᵧ + α) / (Nᵧ + αn)

其中α是拉普拉斯平滑参数。

适用于文本分类（词频）。

#### 伯努利朴素贝叶斯

假设特征为二值（出现/不出现）：
P(xᵢ|y) = P(xᵢ=1|y) if xᵢ=1 else 1-P(xᵢ=1|y)

适用于文本分类（词出现/不出现）。

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

#### 示例1：从零实现朴素贝叶斯（高斯）

```python
import numpy as np
from collections import Counter

class GaussianNB:
    """高斯朴素贝叶斯（从零实现）"""
    
    def fit(self, X, y):
        """训练模型"""
        self.classes = np.unique(y)
        self.n_classes = len(self.classes)
        self.n_features = X.shape[1]
        
        # 计算先验概率
        self.priors = {}
        for c in self.classes:
            self.priors[c] = np.sum(y == c) / len(y)
        
        # 计算每个类别的均值和方差
        self.means = {}
        self.vars = {}
        
        for c in self.classes:
            X_c = X[y == c]
            self.means[c] = np.mean(X_c, axis=0)
            self.vars[c] = np.var(X_c, axis=0)
        
        return self
    
    def _gaussian_pdf(self, x, mean, var):
        """计算高斯概率密度"""
        eps = 1e-10  # 防止除零
        return (1 / np.sqrt(2 * np.pi * (var + eps))) * \
               np.exp(-0.5 * ((x - mean) ** 2) / (var + eps))
    
    def predict_proba(self, X):
        """预测概率"""
        n_samples = X.shape[0]
        probas = np.zeros((n_samples, self.n_classes))
        
        for i, c in enumerate(self.classes):
            # 计算似然
            likelihood = np.prod(self._gaussian_pdf(X, self.means[c], self.vars[c]), axis=1)
            # 后验概率 = 先验 * 似然
            probas[:, i] = self.priors[c] * likelihood
        
        # 归一化
        probas = probas / np.sum(probas, axis=1, keepdims=True)
        return probas
    
    def predict(self, X):
        """预测类别"""
        probas = self.predict_proba(X)
        return self.classes[np.argmax(probas, axis=1)]

# 生成数据
from sklearn.datasets import make_classification
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# 训练模型
model = GaussianNB()
model.fit(X, y)

# 预测
y_pred = model.predict(X)
accuracy = np.mean(y_pred == y)
print(f"准确率: {accuracy:.4f}")
```

#### 示例2：使用scikit-learn

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_classification
import numpy as np

# 生成数据
X, y = make_classification(n_samples=200, n_features=4, n_classes=2, random_state=42)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_train, y_train)
y_pred = gnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"高斯朴素贝叶斯准确率: {accuracy:.4f}")

# 预测概率
y_proba = gnb.predict_proba(X_test)
print(f"\n前5个样本的预测概率:")
print(y_proba[:5])
```

#### 示例3：文本分类（多项式朴素贝叶斯）

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 示例文本数据
texts = [
    "buy now special offer free money",
    "meeting tomorrow at 3pm",
    "click here win prize",
    "project update attached",
    "limited time discount",
    "team meeting notes",
    "urgent action required",
    "quarterly report review",
    "get rich quick scheme",
    "client presentation slides"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1=垃圾邮件，0=正常邮件

# 特征提取（词频）
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 多项式朴素贝叶斯
mnb = MultinomialNB(alpha=1.0)  # alpha是拉普拉斯平滑参数
mnb.fit(X_train, y_train)
y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"多项式朴素贝叶斯准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '垃圾']))

# 查看特征重要性（词的重要性）
feature_names = vectorizer.get_feature_names_out()
for i, c in enumerate(mnb.classes_):
    print(f"\n类别{c}的重要词（前5）:")
    coef = mnb.feature_log_prob_[i]
    top5 = sorted(range(len(coef)), key=lambda j: coef[j], reverse=True)[:5]
    for j in top5:
        print(f"  {feature_names[j]}: {coef[j]:.4f}")
```

#### 示例4：伯努利朴素贝叶斯

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

# 示例文本数据
texts = [
    "buy now special offer",
    "meeting tomorrow",
    "click here win",
    "project update",
    "limited time discount"
]
labels = [1, 0, 1, 0, 1]

# 特征提取（二值：出现/不出现）
vectorizer = CountVectorizer(binary=True)
X = vectorizer.fit_transform(texts).toarray()
y = np.array(labels)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 伯努利朴素贝叶斯
bnb = BernoulliNB(alpha=1.0)
bnb.fit(X_train, y_train)
y_pred = bnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"伯努利朴素贝叶斯准确率: {accuracy:.4f}")
```

### 4.3 常见错误与排查

**错误1**：未处理零概率
```python
# 错误：可能出现零概率
likelihood = np.prod(prob, axis=1)  # 如果prob为0，乘积为0

# 正确：使用拉普拉斯平滑或添加小值
prob = prob + 1e-10
likelihood = np.prod(prob, axis=1)
```

**错误2**：特征类型不匹配
```python
# 错误：连续特征使用多项式朴素贝叶斯
mnb = MultinomialNB()
mnb.fit(X_continuous, y)  # X_continuous是连续值

# 正确：连续特征使用高斯朴素贝叶斯
gnb = GaussianNB()
gnb.fit(X_continuous, y)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：从零实现高斯朴素贝叶斯**
不使用库，从零实现高斯朴素贝叶斯算法。

**练习2：文本分类**
使用多项式朴素贝叶斯进行文本分类。

**练习3：对比三种模型**
对比高斯、多项式、伯努利三种模型的效果。

### 进阶练习（2-3题）

**练习1：特征工程**
对文本进行特征工程，提升模型性能。

**练习2：处理类别不平衡**
处理类别不平衡问题。

### 挑战练习（1-2题）

**练习1：完整的文本分类系统**
实现完整的文本分类系统，包括数据预处理、特征提取、模型训练、评估。

---

## 6. 实际案例

### 案例：垃圾邮件分类系统

**业务背景**：
根据邮件内容判断是否为垃圾邮件。

**问题抽象**：
- 特征：邮件中的词（词频或二值）
- 目标：垃圾邮件（1）或正常邮件（0）
- 方法：多项式或伯努利朴素贝叶斯

**端到端实现**：
```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

# 创建模拟数据
emails = [
    "buy now special offer free money click here",
    "meeting tomorrow at 3pm conference room",
    "click here win prize limited time",
    "project update attached please review",
    "limited time discount buy now",
    "team meeting notes action items",
    "urgent action required immediate response",
    "quarterly report review financial data",
    "get rich quick scheme investment opportunity",
    "client presentation slides business proposal",
    "free money no risk guaranteed profit",
    "schedule meeting discuss project timeline"
]
labels = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

# 特征提取（TF-IDF）
vectorizer = TfidfVectorizer(max_features=100, stop_words='english')
X = vectorizer.fit_transform(emails).toarray()
y = np.array(labels)

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 训练模型
model = MultinomialNB(alpha=1.0)
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"准确率: {accuracy:.4f}")
print("\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '垃圾']))

# 混淆矩阵
cm = confusion_matrix(y_test, y_pred)
print("\n混淆矩阵:")
print(cm)

# 预测新邮件
new_emails = [
    "special offer buy now limited time",
    "meeting scheduled for next week"
]
X_new = vectorizer.transform(new_emails)
y_new_pred = model.predict(X_new)
y_new_proba = model.predict_proba(X_new)

print("\n新邮件预测:")
for email, pred, proba in zip(new_emails, y_new_pred, y_new_proba):
    print(f"邮件: {email}")
    print(f"预测: {'垃圾邮件' if pred == 1 else '正常邮件'}")
    print(f"概率: 正常={proba[0]:.4f}, 垃圾={proba[1]:.4f}")
    print()
```

**结果解读**：
- 朴素贝叶斯能够很好地分类垃圾邮件
- 概率输出提供可解释性

**改进方向**：
- 使用更多特征
- 特征工程（n-gram）
- 处理类别不平衡

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 朴素贝叶斯的"朴素"假设是？
   A. 特征独立  B. 特征相关  C. 特征相同  D. 特征连续
   **答案**：A

2. 贝叶斯定理的公式是？
   A. P(A|B) = P(B|A)P(A)/P(B)  B. P(A|B) = P(A)P(B)  C. P(A|B) = P(A)/P(B)  D. P(A|B) = P(B|A)
   **答案**：A

3. 文本分类通常使用什么朴素贝叶斯？
   A. 高斯  B. 多项式  C. 伯努利  D. B和C
   **答案**：D

4. 朴素贝叶斯的优点不包括？
   A. 简单快速  B. 需要大量数据  C. 适合文本分类  D. 概率输出
   **答案**：B

5. 拉普拉斯平滑的作用是？
   A. 加速计算  B. 处理零概率问题  C. 减少参数  D. 提高准确率
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释贝叶斯定理和朴素贝叶斯的原理。
   **参考答案**：贝叶斯定理：P(y|x) = P(x|y)P(y)/P(x)。朴素贝叶斯假设特征独立，简化计算。

2. 说明三种朴素贝叶斯模型的区别。
   **参考答案**：高斯用于连续特征，多项式用于计数特征（词频），伯努利用于二值特征（词出现/不出现）。

3. 解释"朴素"假设的含义和影响。
   **参考答案**：假设特征独立，简化计算但可能不符合现实。如果特征相关，性能可能下降。

4. 说明朴素贝叶斯的优缺点。
   **参考答案**：优点：简单快速、适合小样本、适合文本分类；缺点：特征独立假设可能不成立。

### 编程实践题（20分）

从零实现朴素贝叶斯算法，包括训练和预测。

### 综合应用题（20分）

使用朴素贝叶斯解决文本分类问题，包括数据预处理、特征提取、模型训练、评估。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《机器学习》- 周志华（第7章）
- 《统计学习方法》- 李航（第4章）

**在线资源**：
- scikit-learn官方文档
- 文本分类教程

### 相关工具与库

- **scikit-learn**：GaussianNB, MultinomialNB, BernoulliNB
- **nltk**：自然语言处理
- **pandas**：数据处理

### 进阶话题指引

完成本课程后，可以学习：
- **半朴素贝叶斯**：放松独立性假设
- **贝叶斯网络**：考虑特征依赖关系
- **文本分类进阶**：更复杂的文本分类方法

### 下节课预告

下一课将学习：
- **08_KNN**：K近邻算法
- KNN是简单的非参数分类器

### 学习建议

1. **理解概率**：理解贝叶斯定理和条件概率
2. **多实践**：从零实现算法，加深理解
3. **文本分类**：尝试文本分类问题
4. **持续学习**：朴素贝叶斯是文本分类的基础

---

**恭喜完成第七课！你已经掌握了朴素贝叶斯，准备好学习KNN了！**

