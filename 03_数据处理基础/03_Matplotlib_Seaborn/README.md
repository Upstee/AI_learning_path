# Matplotlib与Seaborn

## 1. 课程概述

### 课程目标
1. 掌握Matplotlib的基础绘图功能
2. 能够创建各种类型的图表（折线图、散点图、柱状图等）
3. 掌握Seaborn的统计图表功能
4. 能够进行数据可视化分析
5. 能够创建美观、专业的图表

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：10-12小时
- **练习巩固**：4-6小时
- **总计**：20-26小时（约1-2周）

### 难度等级
- **简单到中等** - 主要是API使用，需要多练习

### 课程定位
- **前置课程**：01_NumPy、02_Pandas
- **后续课程**：04_数据清洗与预处理、04_机器学习基础
- **在体系中的位置**：数据可视化工具，数据分析必备

### 学完能做什么
- 能够创建各种类型的图表
- 能够进行数据可视化分析
- 能够创建美观、专业的可视化报告
- 能够为机器学习结果可视化

---

## 2. 前置知识检查

### 必备前置概念清单
- **NumPy基础**：数组操作
- **Pandas基础**：DataFrame操作
- **Python基础**：函数、列表

### 回顾链接/跳转
- 如果不熟悉NumPy：`03_数据处理基础/01_NumPy/`
- 如果不熟悉Pandas：`03_数据处理基础/02_Pandas/`

### 入门小测

**选择题**（每题2分，共10分）

1. Matplotlib的主要用途？
   A. 数据处理  B. 数据可视化  C. 数据分析  D. 数据存储
   **答案**：B

2. 如何创建折线图？
   A. plt.plot()  B. plt.scatter()  C. plt.bar()  D. plt.hist()
   **答案**：A

3. Seaborn的优势？
   A. 更美观  B. 统计图表  C. 更简单  D. 以上都是
   **答案**：D

4. 如何显示图表？
   A. plt.show()  B. plt.display()  C. plt.render()  D. plt.draw()
   **答案**：A

5. 如何保存图表？
   A. plt.savefig()  B. plt.save()  C. plt.export()  D. plt.write()
   **答案**：A

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 Matplotlib基础

#### 概念引入与直观类比

**类比**：Matplotlib就像"画布和画笔"，可以画出各种图表。

- **Figure**：画布
- **Axes**：子图
- **Plot**：绘图函数

#### 逐步理论推导

**步骤1：基础绘图**
```python
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 100)
y = np.sin(x)
plt.plot(x, y)
plt.show()
```

**步骤2：图表定制**
```python
plt.plot(x, y, label='sin(x)', color='blue', linewidth=2)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Sine Wave')
plt.legend()
plt.grid(True)
plt.show()
```

**步骤3：多子图**
```python
fig, axes = plt.subplots(2, 2, figsize=(10, 8))
axes[0, 0].plot(x, y)
axes[0, 1].scatter(x, y)
# ...
plt.tight_layout()
plt.show()
```

---

### 3.2 常用图表类型

#### 折线图
```python
plt.plot(x, y)
```

#### 散点图
```python
plt.scatter(x, y, s=50, alpha=0.5)
```

#### 柱状图
```python
plt.bar(categories, values)
```

#### 直方图
```python
plt.hist(data, bins=30)
```

#### 箱线图
```python
plt.boxplot(data)
```

---

### 3.3 Seaborn

#### 概念引入与直观类比

**类比**：Seaborn就像"高级画笔"，可以轻松画出统计图表。

- **更美观**：默认样式更好
- **统计图表**：内置统计功能
- **更简单**：API更简洁

#### 逐步理论推导

**步骤1：基础使用**
```python
import seaborn as sns
sns.set_style("whitegrid")
sns.lineplot(x='x', y='y', data=df)
```

**步骤2：统计图表**
```python
sns.boxplot(x='category', y='value', data=df)
sns.violinplot(x='category', y='value', data=df)
sns.heatmap(correlation_matrix, annot=True)
```

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - matplotlib >= 3.3.0
  - seaborn >= 0.11.0
  - numpy
  - pandas

### 4.2 从零开始的完整可运行示例

#### 示例1：基础绘图

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
x = np.linspace(0, 2*np.pi, 100)
y1 = np.sin(x)
y2 = np.cos(x)

# 创建图表
plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2, linestyle='--')
plt.xlabel('x', fontsize=12)
plt.ylabel('y', fontsize=12)
plt.title('Trigonometric Functions', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, alpha=0.3)
plt.show()
```

#### 示例2：多种图表类型

```python
import matplotlib.pyplot as plt
import numpy as np

# 创建数据
np.random.seed(42)
categories = ['A', 'B', 'C', 'D', 'E']
values = np.random.randint(50, 100, 5)
x = np.random.randn(100)
y = np.random.randn(100)

# 创建多子图
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 折线图
axes[0, 0].plot(categories, values, marker='o')
axes[0, 0].set_title('Line Plot')
axes[0, 0].set_xlabel('Category')
axes[0, 0].set_ylabel('Value')

# 柱状图
axes[0, 1].bar(categories, values, color=['red', 'blue', 'green', 'orange', 'purple'])
axes[0, 1].set_title('Bar Chart')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Value')

# 散点图
axes[1, 0].scatter(x, y, alpha=0.5, s=50)
axes[1, 0].set_title('Scatter Plot')
axes[1, 0].set_xlabel('X')
axes[1, 0].set_ylabel('Y')

# 直方图
axes[1, 1].hist(x, bins=20, edgecolor='black')
axes[1, 1].set_title('Histogram')
axes[1, 1].set_xlabel('Value')
axes[1, 1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()
```

#### 示例3：Seaborn统计图表

```python
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设置样式
sns.set_style("whitegrid")

# 创建数据
np.random.seed(42)
data = pd.DataFrame({
    'category': np.random.choice(['A', 'B', 'C'], 100),
    'value': np.random.randn(100),
    'score': np.random.randint(60, 100, 100)
})

# 箱线图
plt.figure(figsize=(10, 6))
sns.boxplot(x='category', y='value', data=data)
plt.title('Box Plot by Category')
plt.show()

# 小提琴图
plt.figure(figsize=(10, 6))
sns.violinplot(x='category', y='value', data=data)
plt.title('Violin Plot by Category')
plt.show()

# 相关性热力图
correlation = data[['value', 'score']].corr()
plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Heatmap')
plt.show()
```

### 4.3 常见错误与排查

**错误1**：忘记plt.show()
```python
# 错误：图表不显示
plt.plot(x, y)

# 正确
plt.plot(x, y)
plt.show()
```

**错误2**：中文乱码
```python
# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
```

**错误3**：图表重叠
```python
# 解决：使用plt.tight_layout()
plt.tight_layout()
plt.show()
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：基础图表**
创建折线图、散点图、柱状图。

**练习2：图表定制**
定制图表的颜色、标签、标题等。

**练习3：多子图**
创建包含多个子图的图表。

### 进阶练习（2-3题）

**练习1：数据可视化分析**
对数据集进行可视化分析。

**练习2：统计图表**
使用Seaborn创建统计图表。

### 挑战练习（1-2题）

**练习1：完整的可视化报告**
创建包含多种图表的完整可视化报告。

---

## 6. 实际案例

### 案例：销售数据可视化分析

**业务背景**：
分析销售数据，生成可视化报告。

**问题抽象**：
- 数据：销售记录
- 可视化：多种图表
- 输出：可视化报告

**端到端实现**：
```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# 创建模拟数据
np.random.seed(42)
dates = pd.date_range('2024-01-01', periods=30, freq='D')
sales_data = pd.DataFrame({
    'date': dates,
    'sales': np.random.randint(1000, 5000, 30),
    'category': np.random.choice(['A', 'B', 'C'], 30),
    'region': np.random.choice(['North', 'South', 'East', 'West'], 30)
})

# 创建可视化报告
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# 1. 销售趋势
axes[0, 0].plot(sales_data['date'], sales_data['sales'], marker='o')
axes[0, 0].set_title('Sales Trend')
axes[0, 0].set_xlabel('Date')
axes[0, 0].set_ylabel('Sales')
axes[0, 0].tick_params(axis='x', rotation=45)

# 2. 按类别统计
category_sales = sales_data.groupby('category')['sales'].sum()
axes[0, 1].bar(category_sales.index, category_sales.values, color=['red', 'blue', 'green'])
axes[0, 1].set_title('Sales by Category')
axes[0, 1].set_xlabel('Category')
axes[0, 1].set_ylabel('Total Sales')

# 3. 按地区箱线图
sns.boxplot(x='region', y='sales', data=sales_data, ax=axes[1, 0])
axes[1, 0].set_title('Sales Distribution by Region')

# 4. 相关性分析
correlation = sales_data[['sales']].corrwith(pd.get_dummies(sales_data[['category', 'region']]).sum(axis=1))
axes[1, 1].bar(range(len(correlation)), correlation.values)
axes[1, 1].set_title('Correlation Analysis')
axes[1, 1].set_xlabel('Feature')
axes[1, 1].set_ylabel('Correlation')

plt.tight_layout()
plt.savefig('sales_analysis.png', dpi=300, bbox_inches='tight')
plt.show()
```

**结果解读**：
- 可以直观看到销售趋势和分布
- 可以发现数据中的模式和异常

**改进方向**：
- 添加交互式图表
- 添加更多统计信息
- 优化图表美观度

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. Matplotlib的主要用途？
   A. 数据处理  B. 数据可视化  C. 数据分析  D. 数据存储
   **答案**：B

2. 如何创建折线图？
   A. plt.plot()  B. plt.scatter()  C. plt.bar()  D. plt.hist()
   **答案**：A

3. Seaborn的优势？
   A. 更美观  B. 统计图表  C. 更简单  D. 以上都是
   **答案**：D

4. 如何显示图表？
   A. plt.show()  B. plt.display()  C. plt.render()  D. plt.draw()
   **答案**：A

5. 如何保存图表？
   A. plt.savefig()  B. plt.save()  C. plt.export()  D. plt.write()
   **答案**：A

**简答题**（每题10分，共40分）

1. 解释Matplotlib和Seaborn的区别。
   **参考答案**：Matplotlib是基础绘图库，灵活但需要更多代码；Seaborn基于Matplotlib，更美观且提供统计图表。

2. 说明如何创建多子图。
   **参考答案**：使用plt.subplots()创建子图网格，然后在每个子图上绘图。

3. 解释数据可视化的重要性。
   **参考答案**：可视化帮助理解数据、发现模式、传达信息，是数据分析的重要工具。

4. 说明如何创建美观的图表。
   **参考答案**：选择合适的图表类型、使用合适的颜色、添加标签和标题、调整布局。

### 编程实践题（20分）

创建包含多种图表的可视化报告。

### 综合应用题（20分）

对真实数据集进行完整的可视化分析。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python数据科学手册》- Jake VanderPlas（第4章）
- 《数据可视化实战》- 各种版本

**在线资源**：
- Matplotlib官方文档
- Seaborn官方文档

### 相关工具与库

- **Plotly**：交互式可视化
- **Bokeh**：Web可视化
- **Altair**：声明式可视化

### 进阶话题指引

完成本课程后，可以学习：
- **交互式可视化**：Plotly、Bokeh
- **3D可视化**：mplot3d
- **动画**：matplotlib.animation

### 下节课预告

下一课将学习：
- **04_数据清洗与预处理**：缺失值处理、异常值处理、特征工程
- 数据清洗是机器学习的重要步骤

### 学习建议

1. **多实践**：每学一个图表类型，立即画图
2. **参考示例**：多看官方示例
3. **美化图表**：学习如何创建美观的图表
4. **持续学习**：可视化是数据分析的重要技能

---

**恭喜完成第三课！你已经掌握了数据可视化的基础，准备好学习数据清洗了！**

