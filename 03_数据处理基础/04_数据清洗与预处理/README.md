# 数据清洗与预处理

## 1. 课程概述

### 课程目标
1. 理解数据清洗的重要性和步骤
2. 掌握缺失值处理的多种方法
3. 掌握异常值检测和处理方法
4. 掌握数据标准化和归一化
5. 掌握特征工程的基本方法
6. 能够为机器学习准备高质量数据

### 预计学习时间
- **理论学习**：8-10小时
- **代码实践**：12-15小时
- **练习巩固**：8-10小时
- **总计**：28-35小时（约2-3周）

### 难度等级
- **中等** - 需要理解各种处理方法的原理和适用场景

### 课程定位
- **前置课程**：01_NumPy、02_Pandas、03_Matplotlib_Seaborn
- **后续课程**：04_机器学习基础
- **在体系中的位置**：机器学习的前置步骤，数据质量决定模型效果

### 学完能做什么
- 能够识别和处理数据质量问题
- 能够进行数据清洗和预处理
- 能够进行特征工程
- 能够为机器学习准备高质量数据

---

## 2. 前置知识检查

### 必备前置概念清单
- **NumPy**：数组操作
- **Pandas**：DataFrame操作
- **Matplotlib/Seaborn**：数据可视化
- **统计学基础**：均值、标准差、分位数

### 回顾链接/跳转
- 如果不熟悉Pandas：`03_数据处理基础/02_Pandas/`
- 如果不熟悉统计：`02_数学基础/02_概率统计/`

### 入门小测

**选择题**（每题2分，共10分）

1. 数据清洗的主要目的？
   A. 删除数据  B. 提高数据质量  C. 增加数据  D. 修改数据
   **答案**：B

2. 缺失值处理的常见方法？
   A. 删除  B. 填充  C. 插值  D. 以上都是
   **答案**：D

3. 异常值检测的方法？
   A. 3σ原则  B. IQR方法  C. 箱线图  D. 以上都是
   **答案**：D

4. 标准化的作用？
   A. 统一量纲  B. 加速收敛  C. 提高精度  D. 以上都是
   **答案**：D

5. 特征工程的作用？
   A. 创建新特征  B. 选择特征  C. 转换特征  D. 以上都是
   **答案**：D

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 数据质量检查

#### 概念引入与直观类比

**类比**：数据质量检查就像"体检"，发现数据中的问题。

- **缺失值**：数据不完整
- **异常值**：数据异常
- **重复值**：数据重复
- **不一致**：数据格式不一致

#### 逐步理论推导

**步骤1：检查缺失值**
```python
df.isnull().sum()  # 统计缺失值
df.isnull().any()  # 检查是否有缺失值
```

**步骤2：检查重复值**
```python
df.duplicated().sum()  # 统计重复值
df.drop_duplicates()   # 删除重复值
```

**步骤3：检查数据类型**
```python
df.dtypes  # 查看数据类型
df.info()  # 查看数据信息
```

**步骤4：检查异常值**
```python
df.describe()  # 描述性统计
# 使用箱线图、散点图等可视化
```

---

### 3.2 缺失值处理

#### 概念引入与直观类比

**类比**：缺失值就像"空白"，需要填补或删除。

- **删除**：如果缺失值很少，可以删除
- **填充**：用均值、中位数、众数等填充
- **插值**：用前后值插值

#### 逐步理论推导

**步骤1：删除缺失值**
```python
df.dropna()                    # 删除包含缺失值的行
df.dropna(axis=1)              # 删除包含缺失值的列
df.dropna(subset=['col'])      # 删除指定列的缺失值
```

**步骤2：填充缺失值**
```python
df.fillna(0)                   # 用0填充
df.fillna(df.mean())           # 用均值填充
df.fillna(method='ffill')      # 前向填充
df.fillna(method='bfill')      # 后向填充
```

**步骤3：插值**
```python
df.interpolate()                # 线性插值
df.interpolate(method='polynomial', order=2)  # 多项式插值
```

#### 关键性质

**删除策略**：
- **优点**：简单直接
- **缺点**：可能丢失信息
- **适用**：缺失值很少（<5%）

**填充策略**：
- **优点**：保留数据
- **缺点**：可能引入偏差
- **适用**：缺失值较多

---

### 3.3 异常值处理

#### 概念引入与直观类比

**类比**：异常值就像"离群点"，需要识别和处理。

- **3σ原则**：超出3倍标准差
- **IQR方法**：超出1.5倍IQR
- **箱线图**：可视化识别

#### 逐步理论推导

**步骤1：3σ原则**
```python
mean = df['col'].mean()
std = df['col'].std()
lower = mean - 3 * std
upper = mean + 3 * std
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
```

**步骤2：IQR方法**
```python
Q1 = df['col'].quantile(0.25)
Q3 = df['col'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR
outliers = df[(df['col'] < lower) | (df['col'] > upper)]
```

**步骤3：处理异常值**
```python
# 删除
df_clean = df[~outliers_mask]

# 截断
df['col'] = df['col'].clip(lower, upper)

# 替换
df['col'] = df['col'].replace(outliers, df['col'].median())
```

---

### 3.4 数据标准化和归一化

#### 概念引入与直观类比

**类比**：标准化就像"统一度量衡"，让不同特征在同一尺度。

- **标准化（Z-score）**：均值0，标准差1
- **归一化（Min-Max）**：缩放到[0, 1]
- **作用**：加速收敛，提高精度

#### 逐步理论推导

**步骤1：标准化（Z-score）**
```
z = (x - μ) / σ
```

**步骤2：归一化（Min-Max）**
```
x_norm = (x - min) / (max - min)
```

**步骤3：实现**
```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 归一化
minmax_scaler = MinMaxScaler()
X_normalized = minmax_scaler.fit_transform(X)
```

---

### 3.5 特征工程

#### 概念引入与直观类比

**类比**：特征工程就像"加工原材料"，创造更有价值的特征。

- **特征创建**：从现有特征创建新特征
- **特征选择**：选择重要特征
- **特征转换**：转换特征形式

#### 逐步理论推导

**步骤1：特征创建**
```python
# 数值特征
df['feature_new'] = df['feature1'] * df['feature2']
df['feature_log'] = np.log(df['feature'])

# 分类特征
df['feature_cat'] = pd.cut(df['feature'], bins=5)
```

**步骤2：特征编码**
```python
# 独热编码
pd.get_dummies(df['category'])

# 标签编码
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df['category_encoded'] = le.fit_transform(df['category'])
```

**步骤3：特征选择**
```python
from sklearn.feature_selection import SelectKBest, f_classif

selector = SelectKBest(f_classif, k=10)
X_selected = selector.fit_transform(X, y)
```

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - pandas >= 1.3.0
  - numpy >= 1.20.0
  - scikit-learn >= 0.24.0
  - matplotlib
  - seaborn

### 4.2 从零开始的完整可运行示例

#### 示例1：数据质量检查

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 创建包含问题的数据
np.random.seed(42)
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve', 'Frank', 'Grace', 'Henry'],
    'age': [20, 21, np.nan, 22, 20, 150, 19, 21],  # 包含缺失值和异常值
    'score': [85, 90, 88, np.nan, 87, 200, 89, 91],  # 包含缺失值和异常值
    'city': ['Beijing', 'Shanghai', None, 'Beijing', 'Shanghai', 'Guangzhou', 'Beijing', 'Shanghai']
}
df = pd.DataFrame(data)

print("原始数据:")
print(df)
print("\n数据信息:")
print(df.info())

# 检查缺失值
print("\n缺失值统计:")
print(df.isnull().sum())

# 检查异常值
print("\n描述性统计:")
print(df.describe())

# 可视化异常值
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.boxplot(y=df['age'], ax=axes[0])
axes[0].set_title('Age Distribution')
sns.boxplot(y=df['score'], ax=axes[1])
axes[1].set_title('Score Distribution')
plt.tight_layout()
plt.show()
```

#### 示例2：缺失值处理

```python
import pandas as pd
import numpy as np

# 创建数据
np.random.seed(42)
df = pd.DataFrame({
    'A': [1, 2, np.nan, 4, 5],
    'B': [np.nan, 2, 3, 4, np.nan],
    'C': [1, 2, 3, np.nan, 5]
})

print("原始数据:")
print(df)

# 方法1：删除缺失值
print("\n删除包含缺失值的行:")
df_dropna = df.dropna()
print(df_dropna)

# 方法2：填充缺失值
print("\n用均值填充:")
df_filled_mean = df.fillna(df.mean())
print(df_filled_mean)

print("\n用中位数填充:")
df_filled_median = df.fillna(df.median())
print(df_filled_median)

# 方法3：前向填充
print("\n前向填充:")
df_ffill = df.fillna(method='ffill')
print(df_ffill)

# 方法4：插值
print("\n线性插值:")
df_interpolated = df.interpolate()
print(df_interpolated)
```

#### 示例3：异常值处理

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 创建包含异常值的数据
np.random.seed(42)
normal_data = np.random.normal(100, 15, 100)
outliers = np.array([200, 250, 30, 35])  # 异常值
data = np.concatenate([normal_data, outliers])
df = pd.DataFrame({'value': data})

print("原始数据统计:")
print(df.describe())

# 方法1：3σ原则
mean = df['value'].mean()
std = df['value'].std()
lower = mean - 3 * std
upper = mean + 3 * std
outliers_3sigma = df[(df['value'] < lower) | (df['value'] > upper)]
print(f"\n3σ原则检测到的异常值数量: {len(outliers_3sigma)}")

# 方法2：IQR方法
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_iqr = Q1 - 1.5 * IQR
upper_iqr = Q3 + 1.5 * IQR
outliers_iqr = df[(df['value'] < lower_iqr) | (df['value'] > upper_iqr)]
print(f"IQR方法检测到的异常值数量: {len(outliers_iqr)}")

# 处理异常值：截断
df_cleaned = df.copy()
df_cleaned['value'] = df_cleaned['value'].clip(lower_iqr, upper_iqr)

print("\n处理后的数据统计:")
print(df_cleaned.describe())

# 可视化
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].boxplot(df['value'])
axes[0].set_title('原始数据（含异常值）')
axes[1].boxplot(df_cleaned['value'])
axes[1].set_title('处理后数据')
plt.tight_layout()
plt.show()
```

#### 示例4：数据标准化和归一化

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 创建数据
np.random.seed(42)
df = pd.DataFrame({
    'feature1': np.random.normal(100, 15, 100),
    'feature2': np.random.normal(50, 5, 100),
    'feature3': np.random.normal(1000, 100, 100)
})

print("原始数据统计:")
print(df.describe())

# 标准化（Z-score）
scaler = StandardScaler()
df_standardized = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)
print("\n标准化后统计（均值≈0，标准差≈1）:")
print(df_standardized.describe())

# 归一化（Min-Max）
minmax_scaler = MinMaxScaler()
df_normalized = pd.DataFrame(
    minmax_scaler.fit_transform(df),
    columns=df.columns
)
print("\n归一化后统计（范围[0, 1]）:")
print(df_normalized.describe())
```

#### 示例5：特征工程

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# 创建数据
df = pd.DataFrame({
    'age': [20, 25, 30, 35, 40],
    'income': [50000, 60000, 70000, 80000, 90000],
    'city': ['Beijing', 'Shanghai', 'Beijing', 'Guangzhou', 'Shanghai'],
    'category': ['A', 'B', 'A', 'C', 'B']
})

print("原始数据:")
print(df)

# 特征创建
df['age_income_ratio'] = df['age'] / df['income']
df['income_log'] = np.log(df['income'])

# 标签编码
le = LabelEncoder()
df['city_encoded'] = le.fit_transform(df['city'])

# 独热编码
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')

print("\n特征工程后:")
print(df_encoded.head())
```

### 4.3 常见错误与排查

**错误1**：删除缺失值导致数据过少
```python
# 错误：删除太多数据
df_clean = df.dropna()  # 可能删除太多

# 正确：检查缺失比例
missing_ratio = df.isnull().sum() / len(df)
# 只删除缺失比例小的列
```

**错误2**：标准化前没有分割数据
```python
# 错误：在分割前标准化
scaler.fit_transform(X)  # 包含测试集信息

# 正确：先分割，再分别标准化
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
scaler.fit(X_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**错误3**：异常值处理不当
```python
# 错误：盲目删除所有异常值
# 正确：分析异常值原因，决定处理方式
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：缺失值处理**
对包含缺失值的数据进行各种处理。

**练习2：异常值检测**
使用不同方法检测异常值。

**练习3：数据标准化**
对数据进行标准化和归一化。

### 进阶练习（2-3题）

**练习1：完整数据清洗流程**
对真实数据集进行完整的数据清洗。

**练习2：特征工程**
创建新特征、编码分类特征。

### 挑战练习（1-2题）

**练习1：端到端数据预处理**
对真实数据集进行完整的预处理，为机器学习准备数据。

---

## 6. 实际案例

### 案例：房价预测数据预处理

**业务背景**：
为房价预测模型准备数据，需要进行数据清洗和预处理。

**问题抽象**：
- 数据：房价数据集
- 问题：缺失值、异常值、特征工程
- 目标：准备高质量数据

**端到端实现**：
```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# 创建模拟数据
np.random.seed(42)
n_samples = 1000
data = {
    'area': np.random.normal(100, 20, n_samples),
    'bedrooms': np.random.randint(1, 5, n_samples),
    'age': np.random.randint(0, 50, n_samples),
    'location': np.random.choice(['A', 'B', 'C'], n_samples),
    'price': None  # 待计算
}

# 添加一些缺失值和异常值
missing_indices = np.random.choice(n_samples, size=50, replace=False)
data['area'][missing_indices[:25]] = np.nan
data['age'][missing_indices[25:]] = np.nan

# 添加异常值
data['area'][0] = 500  # 异常值
data['age'][1] = 200  # 异常值

df = pd.DataFrame(data)

# 计算价格（模拟）
df['price'] = (df['area'] * 1000 + 
               df['bedrooms'] * 50000 + 
               df['age'] * -1000 + 
               np.random.normal(0, 10000, n_samples))

print("原始数据信息:")
print(df.info())
print("\n缺失值统计:")
print(df.isnull().sum())

# 步骤1：处理缺失值
# 用中位数填充数值特征
df['area'].fillna(df['area'].median(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)

# 步骤2：处理异常值（IQR方法）
def remove_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return df[(df[column] >= lower) & (df[column] <= upper)]

df_clean = df.copy()
for col in ['area', 'age']:
    df_clean = remove_outliers_iqr(df_clean, col)

# 步骤3：特征编码
le = LabelEncoder()
df_clean['location_encoded'] = le.fit_transform(df_clean['location'])

# 步骤4：特征工程
df_clean['area_per_bedroom'] = df_clean['area'] / df_clean['bedrooms']
df_clean['age_squared'] = df_clean['age'] ** 2

# 步骤5：标准化数值特征
scaler = StandardScaler()
numeric_cols = ['area', 'age', 'bedrooms', 'area_per_bedroom', 'age_squared']
df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

print("\n处理后数据信息:")
print(df_clean.info())
print("\n处理后统计:")
print(df_clean.describe())

# 可视化对比
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

axes[0, 0].hist(df['area'].dropna(), bins=30, alpha=0.5, label='原始')
axes[0, 0].hist(df_clean['area'], bins=30, alpha=0.5, label='处理后')
axes[0, 0].set_title('Area Distribution')
axes[0, 0].legend()

axes[0, 1].hist(df['age'].dropna(), bins=30, alpha=0.5, label='原始')
axes[0, 1].hist(df_clean['age'], bins=30, alpha=0.5, label='处理后')
axes[0, 1].set_title('Age Distribution')
axes[0, 1].legend()

sns.boxplot(y=df['area'].dropna(), ax=axes[1, 0])
axes[1, 0].set_title('原始Area（含异常值）')

sns.boxplot(y=df_clean['area'], ax=axes[1, 1])
axes[1, 1].set_title('处理后Area')

plt.tight_layout()
plt.show()

print(f"\n原始数据: {len(df)} 条")
print(f"处理后数据: {len(df_clean)} 条")
print(f"删除: {len(df) - len(df_clean)} 条异常值")
```

**结果解读**：
- 数据质量显著提升
- 异常值被识别和处理
- 特征工程创建了新特征

**改进方向**：
- 使用更高级的缺失值填充方法
- 使用更智能的异常值检测
- 添加更多特征工程方法

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. 数据清洗的主要目的？
   A. 删除数据  B. 提高数据质量  C. 增加数据  D. 修改数据
   **答案**：B

2. 缺失值处理的常见方法？
   A. 删除  B. 填充  C. 插值  D. 以上都是
   **答案**：D

3. 异常值检测的方法？
   A. 3σ原则  B. IQR方法  C. 箱线图  D. 以上都是
   **答案**：D

4. 标准化的作用？
   A. 统一量纲  B. 加速收敛  C. 提高精度  D. 以上都是
   **答案**：D

5. 特征工程的作用？
   A. 创建新特征  B. 选择特征  C. 转换特征  D. 以上都是
   **答案**：D

**简答题**（每题10分，共40分）

1. 解释缺失值处理的策略。
   **参考答案**：删除（缺失少）、填充（均值/中位数/众数）、插值（时间序列）、模型预测（高级方法）。

2. 说明异常值检测和处理的方法。
   **参考答案**：检测方法有3σ原则、IQR方法、箱线图；处理方法有删除、截断、替换。

3. 解释标准化和归一化的区别。
   **参考答案**：标准化将数据转换为均值0、标准差1；归一化将数据缩放到[0,1]范围。

4. 说明特征工程的重要性。
   **参考答案**：好的特征工程可以显著提升模型性能，是机器学习成功的关键。

### 编程实践题（20分）

对真实数据集进行完整的数据清洗和预处理。

### 综合应用题（20分）

为机器学习模型准备数据，包括数据清洗、特征工程、标准化等完整流程。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python数据科学手册》- Jake VanderPlas
- 《特征工程入门与实践》- 各种版本

**在线资源**：
- scikit-learn预处理文档
- 数据清洗最佳实践

### 相关工具与库

- **scikit-learn**：预处理工具
- **pandas**：数据清洗
- **missingno**：缺失值可视化

### 进阶话题指引

完成本课程后，可以学习：
- **高级特征工程**：特征选择、降维
- **自动化特征工程**：AutoML工具
- **数据管道**：构建数据预处理管道

### 下节课预告

完成数据处理基础模块后，可以进入：
- **04_机器学习基础**：开始学习机器学习算法
- 数据准备好了，可以开始训练模型了

### 学习建议

1. **多实践**：处理真实数据集
2. **理解原理**：理解每种方法的原理和适用场景
3. **对比方法**：对比不同方法的效果
4. **持续学习**：数据预处理是ML的基础，需要扎实掌握

---

**恭喜完成数据处理基础模块！你已经掌握了数据处理的核心技能，准备好开始学习机器学习了！**

