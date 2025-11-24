# Pandas

## 1. 课程概述

### 课程目标
1. 理解Series和DataFrame的概念
2. 掌握数据的读取、写入和基本操作
3. 掌握数据筛选、分组、聚合操作
4. 掌握数据清洗和预处理方法
5. 能够使用Pandas进行数据分析

### 预计学习时间
- **理论学习**：8-10小时
- **代码实践**：12-15小时
- **练习巩固**：8-10小时
- **总计**：28-35小时（约2-3周）

### 难度等级
- **中等** - 需要理解数据结构和操作

### 课程定位
- **前置课程**：01_NumPy、Python基础
- **后续课程**：04_数据清洗与预处理、04_机器学习基础
- **在体系中的位置**：数据分析的核心工具，AI数据准备必备

### 学完能做什么
- 能够读取和处理各种格式的数据
- 能够进行数据筛选、分组、聚合
- 能够进行数据清洗和预处理
- 能够为机器学习准备数据

---

## 2. 前置知识检查

### 必备前置概念清单
- **NumPy基础**：数组操作、索引
- **Python基础**：字典、列表、函数
- **文件操作**：读取CSV、Excel等文件

### 回顾链接/跳转
- 如果不熟悉NumPy：`03_数据处理基础/01_NumPy/`
- 如果不熟悉Python：`01_Python进阶/`

### 入门小测

**选择题**（每题2分，共10分）

1. Pandas的主要数据结构？
   A. Array  B. Series和DataFrame  C. List  D. Dict
   **答案**：B

2. 如何读取CSV文件？
   A. pd.read_csv()  B. pd.read_excel()  C. pd.read_json()  D. pd.read_txt()
   **答案**：A

3. DataFrame的列如何访问？
   A. df['column']  B. df.column  C. 以上都可以  D. 以上都不可以
   **答案**：C

4. 如何筛选数据？
   A. df[df['col'] > value]  B. df.filter()  C. df.select()  D. 以上都可以
   **答案**：A

5. 分组聚合的函数？
   A. groupby()  B. aggregate()  C. 以上都可以  D. 以上都不可以
   **答案**：C

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 Series和DataFrame

#### 概念引入与直观类比

**类比**：
- **Series**：像"带标签的列表"，一维数据
- **DataFrame**：像"Excel表格"，二维数据

#### 逐步理论推导

**步骤1：创建Series**
```python
import pandas as pd
s = pd.Series([1, 2, 3, 4, 5], index=['a', 'b', 'c', 'd', 'e'])
```

**步骤2：创建DataFrame**
```python
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [20, 21, 19],
    'score': [85, 90, 88]
})
```

**步骤3：数据访问**
```python
df['name']        # 访问列
df.loc[0]         # 访问行
df.iloc[0, 0]     # 访问元素
```

---

### 3.2 数据操作

#### 数据筛选

```python
# 条件筛选
df[df['age'] > 20]
df[(df['age'] > 20) & (df['score'] > 85)]

# 多条件筛选
df.query('age > 20 and score > 85')
```

#### 数据分组

```python
# 按列分组
grouped = df.groupby('category')

# 聚合操作
grouped['value'].mean()
grouped.agg({'value': ['mean', 'std'], 'count': 'sum'})
```

#### 数据合并

```python
# 合并DataFrame
pd.merge(df1, df2, on='key')
pd.concat([df1, df2], axis=0)  # 纵向合并
pd.concat([df1, df2], axis=1)  # 横向合并
```

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - pandas >= 1.3.0
  - numpy

### 4.2 从零开始的完整可运行示例

#### 示例1：创建和操作DataFrame

```python
import pandas as pd
import numpy as np

# 创建DataFrame
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [20, 21, 19, 22, 20],
    'score': [85, 90, 88, 92, 87],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Beijing', 'Shanghai']
}
df = pd.DataFrame(data)

print("DataFrame:")
print(df)
print("\n基本信息:")
print(df.info())
print("\n统计信息:")
print(df.describe())

# 访问数据
print("\n访问列:")
print(df['name'])
print("\n访问行:")
print(df.loc[0])
print("\n访问元素:")
print(df.loc[0, 'name'])
```

#### 示例2：数据筛选和分组

```python
import pandas as pd

# 创建数据
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [20, 21, 19, 22, 20],
    'score': [85, 90, 88, 92, 87],
    'city': ['Beijing', 'Shanghai', 'Guangzhou', 'Beijing', 'Shanghai']
})

# 条件筛选
print("年龄大于20的学生:")
print(df[df['age'] > 20])

print("\n分数大于85的学生:")
print(df[df['score'] > 85])

# 分组聚合
print("\n按城市分组，计算平均分数:")
city_avg = df.groupby('city')['score'].mean()
print(city_avg)

print("\n按城市分组，计算多个统计量:")
city_stats = df.groupby('city').agg({
    'score': ['mean', 'std', 'count'],
    'age': 'mean'
})
print(city_stats)
```

#### 示例3：数据清洗

```python
import pandas as pd
import numpy as np

# 创建包含缺失值的数据
df = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'age': [20, 21, np.nan, 22, 20],
    'score': [85, 90, 88, np.nan, 87],
    'city': ['Beijing', 'Shanghai', None, 'Beijing', 'Shanghai']
})

print("原始数据:")
print(df)
print("\n缺失值统计:")
print(df.isnull().sum())

# 处理缺失值
print("\n删除包含缺失值的行:")
df_dropna = df.dropna()
print(df_dropna)

print("\n填充缺失值:")
df_filled = df.fillna({'age': df['age'].mean(), 'score': df['score'].mean(), 'city': 'Unknown'})
print(df_filled)
```

### 4.3 常见错误与排查

**错误1**：使用链式赋值
```python
# 错误
df[df['age'] > 20]['score'] = 100

# 正确
df.loc[df['age'] > 20, 'score'] = 100
```

**错误2**：混淆loc和iloc
```python
# loc：使用标签索引
df.loc[0, 'name']

# iloc：使用位置索引
df.iloc[0, 0]
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：数据创建和访问**
创建DataFrame，练习各种访问方式。

**练习2：数据筛选**
使用条件筛选数据。

**练习3：数据分组**
按不同列分组并聚合。

### 进阶练习（2-3题）

**练习1：数据清洗**
处理缺失值、重复值、异常值。

**练习2：数据合并**
合并多个DataFrame。

### 挑战练习（1-2题）

**练习1：完整数据分析**
对真实数据集进行完整的数据分析。

---

## 6. 实际案例

### 案例：学生成绩分析系统

**业务背景**：
分析学生成绩数据，生成统计报告。

**问题抽象**：
- 读取成绩数据
- 计算统计信息
- 生成报告

**端到端实现**：
```python
import pandas as pd
import numpy as np

# 创建数据
np.random.seed(42)
data = {
    'student_id': range(1, 101),
    'name': [f'Student_{i}' for i in range(1, 101)],
    'math': np.random.randint(60, 100, 100),
    'english': np.random.randint(60, 100, 100),
    'science': np.random.randint(60, 100, 100),
    'class': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)

# 计算总分和平均分
df['total'] = df[['math', 'english', 'science']].sum(axis=1)
df['average'] = df[['math', 'english', 'science']].mean(axis=1)

# 按班级统计
class_stats = df.groupby('class').agg({
    'math': 'mean',
    'english': 'mean',
    'science': 'mean',
    'average': 'mean'
})

print("按班级统计:")
print(class_stats)

# 找出优秀学生（平均分>85）
excellent = df[df['average'] > 85]
print(f"\n优秀学生数量: {len(excellent)}")
print(excellent[['name', 'average']].head())
```

**结果解读**：
- 可以快速进行数据分析
- 可以生成各种统计报告

**改进方向**：
- 添加可视化
- 添加排名功能
- 导出报告

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. Pandas的主要数据结构？
   A. Array  B. Series和DataFrame  C. List  D. Dict
   **答案**：B

2. 如何读取CSV文件？
   A. pd.read_csv()  B. pd.read_excel()  C. pd.read_json()  D. pd.read_txt()
   **答案**：A

3. DataFrame的列如何访问？
   A. df['column']  B. df.column  C. 以上都可以  D. 以上都不可以
   **答案**：C

4. 如何筛选数据？
   A. df[df['col'] > value]  B. df.filter()  C. df.select()  D. 以上都可以
   **答案**：A

5. 分组聚合的函数？
   A. groupby()  B. aggregate()  C. 以上都可以  D. 以上都不可以
   **答案**：C

**简答题**（每题10分，共40分）

1. 解释Series和DataFrame的区别。
   **参考答案**：Series是一维带标签数组，DataFrame是二维表格结构。

2. 说明数据筛选的方法。
   **参考答案**：使用布尔索引、query方法、loc/iloc等方法筛选数据。

3. 解释groupby的工作原理。
   **参考答案**：按指定列分组，然后对每组应用聚合函数。

4. 说明Pandas在数据分析中的优势。
   **参考答案**：提供丰富的数据操作功能，支持多种数据格式，易于使用。

### 编程实践题（20分）

使用Pandas处理和分析数据集。

### 综合应用题（20分）

对真实数据集进行完整的数据分析。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python数据科学手册》- Jake VanderPlas（第3章）
- 《利用Python进行数据分析》- Wes McKinney

**在线资源**：
- Pandas官方文档
- Pandas教程

### 相关工具与库

- **NumPy**：Pandas的基础
- **Matplotlib/Seaborn**：数据可视化
- **scikit-learn**：机器学习（使用Pandas数据）

### 进阶话题指引

完成本课程后，可以学习：
- **时间序列分析**：Pandas的时间序列功能
- **数据透视表**：pivot_table
- **性能优化**：向量化、并行处理

### 下节课预告

下一课将学习：
- **03_Matplotlib_Seaborn**：数据可视化
- 可视化是数据分析的重要工具

### 学习建议

1. **多实践**：每学一个功能，立即写代码
2. **处理真实数据**：使用真实数据集练习
3. **理解数据结构**：理解Series和DataFrame的本质
4. **持续学习**：Pandas是数据分析的核心，需要熟练掌握

---

**恭喜完成第二课！你已经掌握了Pandas的基础，准备好学习数据可视化了！**

