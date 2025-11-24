# 概率统计

## 1. 课程概述

### 课程目标
1. 理解概率的基本概念和公理
2. 掌握条件概率和贝叶斯定理
3. 理解随机变量和概率分布
4. 掌握常见的概率分布（正态、二项、泊松等）
5. 理解统计推断的基本方法
6. 能够使用Python进行概率统计计算

### 预计学习时间
- **理论学习**：12-15小时
- **代码实践**：10-12小时
- **练习巩固**：8-10小时
- **总计**：30-37小时（约3-4周）

### 难度等级
- **中等** - 需要理解抽象概念，但通过实例可以掌握

### 课程定位
- **前置课程**：高中数学（概率基础）、01_线性代数
- **后续课程**：03_微积分、04_机器学习基础
- **在体系中的位置**：机器学习的理论基础，几乎所有ML算法都基于概率

### 学完能做什么
- 能够理解和使用概率模型
- 能够进行统计推断和假设检验
- 能够理解机器学习中的概率方法
- 能够使用Python进行概率统计计算

---

## 2. 前置知识检查

### 必备前置概念清单
- **高中数学**：概率、统计基础
- **线性代数**：向量、矩阵
- **Python基础**：函数、列表、NumPy

### 回顾链接/跳转
- 如果不熟悉线性代数：`02_数学基础/01_线性代数/`
- 如果不熟悉Python：`01_Python进阶/`

### 入门小测

**选择题**（每题2分，共10分）

1. 概率的取值范围是？
   A. [0, 1]  B. (0, 1)  C. [0, ∞)  D. (-∞, ∞)
   **答案**：A

2. 条件概率P(A|B)表示什么？
   A. A和B同时发生  B. 在B发生的条件下A发生的概率  C. A或B发生  D. A不发生
   **答案**：B

3. 正态分布的特点是什么？
   A. 对称  B. 钟形  C. 均值决定位置  D. 以上都是
   **答案**：D

4. 样本均值是总体均值的？
   A. 估计量  B. 参数  C. 统计量  D. 变量
   **答案**：A

5. 贝叶斯定理用于什么？
   A. 计算条件概率  B. 更新先验概率  C. 统计推断  D. 以上都是
   **答案**：D

**评分标准**：≥8分（80%）为通过

### 不会时的补救指引
如果小测不通过，建议：
1. 复习高中数学（概率、统计）
2. 学习线性代数基础
3. 完成基础练习后再继续

---

## 3. 核心知识点详解

### 3.1 概率基础

#### 概念引入与直观类比

**类比**：概率就像"可能性"，表示某件事发生的可能性大小。

- **0**：不可能发生
- **0.5**：一半可能性
- **1**：必然发生

例如：
- 抛硬币：正面朝上的概率是0.5
- 掷骰子：得到6的概率是1/6

#### 逐步理论推导

**步骤1：概率的定义**
对于事件A，概率P(A)满足：
- 0 ≤ P(A) ≤ 1
- P(必然事件) = 1
- P(不可能事件) = 0

**步骤2：概率的加法规则**
对于互斥事件A和B：
P(A ∪ B) = P(A) + P(B)

**步骤3：概率的乘法规则**
对于独立事件A和B：
P(A ∩ B) = P(A) × P(B)

**步骤4：条件概率**
在事件B发生的条件下，事件A发生的概率：
P(A|B) = P(A ∩ B) / P(B)

**步骤5：贝叶斯定理**
P(A|B) = P(B|A) × P(A) / P(B)

#### 数学公式与必要证明

**贝叶斯定理的推导**：
从条件概率定义：
P(A|B) = P(A ∩ B) / P(B)
P(B|A) = P(A ∩ B) / P(A)

因此：
P(A ∩ B) = P(A|B) × P(B) = P(B|A) × P(A)

所以：
P(A|B) = P(B|A) × P(A) / P(B)

**全概率公式**：
如果B₁, B₂, ..., Bₙ是互斥且完备的事件，则：
P(A) = ∑ᵢ P(A|Bᵢ) × P(Bᵢ)

---

### 3.2 随机变量

#### 概念引入与直观类比

**类比**：随机变量就像"不确定的数值"，它的值取决于随机事件的结果。

- **离散随机变量**：取值可数（如掷骰子的结果）
- **连续随机变量**：取值连续（如身高、体重）

#### 逐步理论推导

**步骤1：随机变量的定义**
随机变量X是样本空间到实数的映射：
X: Ω → ℝ

**步骤2：概率质量函数（PMF）**
对于离散随机变量X：
p(x) = P(X = x)

**步骤3：概率密度函数（PDF）**
对于连续随机变量X：
P(a ≤ X ≤ b) = ∫ₐᵇ f(x)dx

**步骤4：累积分布函数（CDF）**
F(x) = P(X ≤ x)

**步骤5：期望和方差**
期望：E[X] = ∑ x·p(x)（离散）或 ∫ x·f(x)dx（连续）
方差：Var(X) = E[(X - E[X])²] = E[X²] - (E[X])²

---

### 3.3 常见概率分布

#### 离散分布

**二项分布**：
- **参数**：n（试验次数），p（成功概率）
- **PMF**：P(X=k) = C(n,k) × pᵏ × (1-p)ⁿ⁻ᵏ
- **应用**：n次独立试验中成功的次数

**泊松分布**：
- **参数**：λ（平均发生率）
- **PMF**：P(X=k) = (λᵏ × e⁻λ) / k!
- **应用**：单位时间内事件发生的次数

#### 连续分布

**正态分布（高斯分布）**：
- **参数**：μ（均值），σ²（方差）
- **PDF**：f(x) = (1/√(2πσ²)) × e^(-(x-μ)²/(2σ²))
- **应用**：大量随机变量的和近似正态分布（中心极限定理）

**均匀分布**：
- **参数**：a（下界），b（上界）
- **PDF**：f(x) = 1/(b-a)，x ∈ [a,b]
- **应用**：等概率的连续随机变量

---

### 3.4 统计推断

#### 概念引入与直观类比

**类比**：统计推断就像"从样本推断总体"，通过观察部分来了解整体。

- **样本**：我们观察到的数据
- **总体**：我们想了解的完整数据
- **推断**：从样本推断总体的特征

#### 逐步理论推导

**步骤1：点估计**
使用样本统计量估计总体参数：
- 样本均值 x̄ 估计总体均值 μ
- 样本方差 s² 估计总体方差 σ²

**步骤2：区间估计**
构造置信区间，包含总体参数的概率为1-α：
P(θ ∈ [L, U]) = 1 - α

**步骤3：假设检验**
- **原假设H₀**：要检验的假设
- **备择假设H₁**：与原假设对立的假设
- **显著性水平α**：犯第一类错误的概率
- **p值**：在原假设下，观察到当前或更极端结果的概率

**步骤4：检验步骤**
1. 提出假设H₀和H₁
2. 选择检验统计量
3. 确定拒绝域
4. 计算p值
5. 做出决策

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy
  - scipy
  - matplotlib

### 4.2 从零开始的完整可运行示例

#### 示例1：概率计算

```python
import numpy as np
from scipy import stats

# 抛硬币：正面概率0.5
p_heads = 0.5
print(f"正面概率: {p_heads}")

# 掷骰子：每个面概率1/6
p_dice = 1/6
print(f"掷出6的概率: {p_dice}")

# 条件概率：P(A|B) = P(A∩B) / P(B)
# 例子：从一副牌中抽一张，已知是红色，求是红心的概率
p_red = 26/52  # 红色牌的概率
p_heart_given_red = 13/26  # 在红色条件下是红心
print(f"已知是红色，是红心的概率: {p_heart_given_red}")
```

#### 示例2：随机变量和分布

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# 二项分布：n=10, p=0.5
n, p = 10, 0.5
binom_dist = stats.binom(n, p)

# 计算概率
k = 5
prob = binom_dist.pmf(k)
print(f"10次试验中成功5次的概率: {prob:.4f}")

# 生成随机样本
samples = binom_dist.rvs(size=1000)
print(f"样本均值: {np.mean(samples):.2f}")
print(f"理论均值: {n * p}")

# 正态分布：μ=0, σ=1
mu, sigma = 0, 1
normal_dist = stats.norm(mu, sigma)

# 计算概率密度
x = np.linspace(-4, 4, 100)
pdf = normal_dist.pdf(x)

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(x, pdf, label='正态分布PDF')
plt.xlabel('x')
plt.ylabel('概率密度')
plt.title('标准正态分布')
plt.legend()
plt.grid(True)
plt.show()
```

#### 示例3：统计推断

```python
import numpy as np
from scipy import stats

# 生成样本数据（假设来自正态分布）
np.random.seed(42)
true_mean = 100
true_std = 15
sample_size = 30
sample = np.random.normal(true_mean, true_std, sample_size)

# 点估计
sample_mean = np.mean(sample)
sample_std = np.std(sample, ddof=1)  # 样本标准差（无偏估计）
print(f"样本均值: {sample_mean:.2f}")
print(f"样本标准差: {sample_std:.2f}")

# 置信区间（95%）
confidence_level = 0.95
alpha = 1 - confidence_level
t_critical = stats.t.ppf(1 - alpha/2, df=sample_size-1)
margin_error = t_critical * (sample_std / np.sqrt(sample_size))
ci_lower = sample_mean - margin_error
ci_upper = sample_mean + margin_error
print(f"95%置信区间: [{ci_lower:.2f}, {ci_upper:.2f}]")

# 假设检验：H0: μ = 100
hypothesized_mean = 100
t_stat, p_value = stats.ttest_1samp(sample, hypothesized_mean)
print(f"\n假设检验:")
print(f"原假设: μ = {hypothesized_mean}")
print(f"t统计量: {t_stat:.4f}")
print(f"p值: {p_value:.4f}")
if p_value < 0.05:
    print("拒绝原假设（显著性水平0.05）")
else:
    print("不能拒绝原假设（显著性水平0.05）")
```

#### 示例4：贝叶斯定理应用

```python
import numpy as np

def bayes_theorem(prior, likelihood, evidence):
    """计算后验概率"""
    posterior = (likelihood * prior) / evidence
    return posterior

# 例子：疾病检测
# 假设某种疾病的患病率是1%
p_disease = 0.01
p_no_disease = 0.99

# 检测的准确性：患病时检测阳性95%，未患病时检测阴性90%
p_positive_given_disease = 0.95
p_positive_given_no_disease = 0.10

# 计算检测阳性的概率（全概率公式）
p_positive = (p_positive_given_disease * p_disease + 
              p_positive_given_no_disease * p_no_disease)

# 贝叶斯定理：已知检测阳性，求患病的概率
p_disease_given_positive = bayes_theorem(
    p_disease, 
    p_positive_given_disease, 
    p_positive
)

print(f"疾病患病率: {p_disease*100}%")
print(f"检测阳性概率: {p_positive*100:.2f}%")
print(f"已知检测阳性，患病的概率: {p_disease_given_positive*100:.2f}%")
```

### 4.3 常见错误与排查

**错误1**：混淆概率和条件概率
```python
# 错误理解
p_a_and_b = p_a * p_b  # 这只适用于独立事件

# 正确
p_a_and_b = p_a_given_b * p_b  # 一般情况
```

**错误2**：样本标准差计算错误
```python
# 错误：使用总体标准差公式
std_wrong = np.std(sample)

# 正确：使用样本标准差（无偏估计）
std_correct = np.std(sample, ddof=1)
```

**错误3**：假设检验的p值理解错误
```python
# 错误：p值小表示原假设正确
# 正确：p值小表示拒绝原假设
```

### 4.4 性能/工程化小技巧

1. **使用向量化计算**
```python
# 慢：循环
probs = []
for k in range(n+1):
    probs.append(stats.binom.pmf(k, n, p))

# 快：向量化
k_values = np.arange(n+1)
probs = stats.binom.pmf(k_values, n, p)
```

2. **使用scipy.stats模块**
```python
# 使用scipy.stats而不是手动计算
from scipy import stats
prob = stats.norm.cdf(x, mu, sigma)  # 累积分布函数
```

3. **批量计算**
```python
# 批量计算多个分布的统计量
distributions = [stats.norm(0, 1), stats.norm(1, 2)]
means = [dist.mean() for dist in distributions]
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：概率计算**
计算各种概率：抛硬币、掷骰子、抽牌等。

**练习2：条件概率和贝叶斯定理**
解决条件概率问题，应用贝叶斯定理。

**练习3：随机变量**
生成不同分布的随机变量，计算期望和方差。

### 进阶练习（2-3题）

**练习1：统计推断**
从样本数据估计总体参数，构造置信区间。

**练习2：假设检验**
进行假设检验，解释p值的意义。

### 挑战练习（1-2题）

**练习1：完整的统计分析**
对数据集进行完整的统计分析：描述统计、推断统计、假设检验。

---

## 6. 实际案例

### 案例：A/B测试分析

**业务背景**：
网站进行A/B测试，比较两种页面设计的效果。

**问题抽象**：
- 原假设H₀：两种设计效果相同
- 备择假设H₁：新设计效果更好
- 需要统计检验

**端到端实现**：
```python
import numpy as np
from scipy import stats

# 模拟A/B测试数据
np.random.seed(42)
# 设计A：转化率10%
n_a = 1000
conversions_a = np.random.binomial(1, 0.10, n_a)
rate_a = np.mean(conversions_a)

# 设计B：转化率12%
n_b = 1000
conversions_b = np.random.binomial(1, 0.12, n_b)
rate_b = np.mean(conversions_b)

print(f"设计A转化率: {rate_a*100:.2f}%")
print(f"设计B转化率: {rate_b*100:.2f}%")

# 双样本比例检验
# H0: p_a = p_b
# H1: p_a ≠ p_b
from statsmodels.stats.proportion import proportions_ztest

count = np.array([np.sum(conversions_a), np.sum(conversions_b)])
nobs = np.array([n_a, n_b])
z_stat, p_value = proportions_ztest(count, nobs)

print(f"\n假设检验结果:")
print(f"z统计量: {z_stat:.4f}")
print(f"p值: {p_value:.4f}")

if p_value < 0.05:
    print("拒绝原假设：两种设计效果有显著差异")
    if rate_b > rate_a:
        print("设计B效果更好")
else:
    print("不能拒绝原假设：两种设计效果无显著差异")
```

**结果解读**：
- 可以判断两种设计是否有显著差异
- 可以确定哪种设计效果更好

**改进方向**：
- 考虑样本量计算
- 考虑多重比较问题
- 添加效应量分析

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 概率的取值范围是？
   A. [0, 1]  B. (0, 1)  C. [0, ∞)  D. (-∞, ∞)
   **答案**：A

2. 贝叶斯定理用于什么？
   A. 计算条件概率  B. 更新先验概率  C. 统计推断  D. 以上都是
   **答案**：D

3. 正态分布的特点？
   A. 对称  B. 钟形  C. 均值决定位置  D. 以上都是
   **答案**：D

4. p值小于0.05表示什么？
   A. 原假设正确  B. 拒绝原假设  C. 接受原假设  D. 无法判断
   **答案**：B

5. 置信区间的含义是？
   A. 参数一定在区间内  B. 参数在区间内的概率  C. 样本在区间内  D. 以上都不对
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释条件概率和贝叶斯定理。
   **参考答案**：条件概率是在已知条件下事件发生的概率；贝叶斯定理用于更新先验概率得到后验概率。

2. 说明正态分布的重要性。
   **参考答案**：中心极限定理表明大量随机变量的和近似正态分布，使其在统计中非常重要。

3. 解释假设检验的步骤。
   **参考答案**：提出假设、选择检验统计量、确定拒绝域、计算p值、做出决策。

4. 说明概率统计在AI中的应用。
   **参考答案**：机器学习基于概率模型，贝叶斯方法用于推理，统计方法用于模型评估。

### 编程实践题（20分）

使用Python实现贝叶斯分类器。

### 综合应用题（20分）

对数据集进行完整的统计分析：描述统计、推断统计、假设检验。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《概率论与数理统计》- 各种版本
- 《统计学习方法》- 李航
- 《模式识别与机器学习》- Bishop

**在线资源**：
- Khan Academy概率统计课程
- 3Blue1Brown概率系列

### 相关工具与库

- **scipy.stats**：统计分布和检验
- **statsmodels**：统计建模
- **pandas**：数据处理和统计

### 进阶话题指引

完成本课程后，可以学习：
- **贝叶斯统计**：更深入的贝叶斯方法
- **时间序列分析**：时间相关的统计
- **多元统计**：多维数据分析

### 下节课预告

下一课将学习：
- **03_微积分**：导数、梯度、链式法则
- 微积分是优化算法的基础

### 学习建议

1. **理解概念**：不要只记公式，要理解含义
2. **多练习**：通过计算加深理解
3. **结合应用**：理解在AI中的实际应用
4. **持续学习**：概率统计是AI的基础，需要扎实掌握

---

**恭喜完成第二课！你已经掌握了概率统计的基础，准备好学习微积分了！**

