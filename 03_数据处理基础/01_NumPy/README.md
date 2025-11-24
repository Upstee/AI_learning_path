# NumPy

## 1. 课程概述

### 课程目标
1. 理解NumPy数组的概念和优势
2. 掌握数组的创建、索引、切片操作
3. 掌握数组的数学运算和广播机制
4. 掌握数组的形状变换和拼接操作
5. 能够使用NumPy进行高效的数值计算

### 预计学习时间
- **理论学习**：6-8小时
- **代码实践**：10-12小时
- **练习巩固**：6-8小时
- **总计**：22-28小时（约2-3周）

### 难度等级
- **中等** - 需要理解数组操作和广播机制

### 课程定位
- **前置课程**：Python基础、01_线性代数
- **后续课程**：02_Pandas、04_机器学习基础
- **在体系中的位置**：数据处理的基础，几乎所有AI库都依赖NumPy

### 学完能做什么
- 能够使用NumPy进行高效的数值计算
- 能够处理多维数组数据
- 能够理解和使用广播机制
- 能够为机器学习准备数据

---

## 2. 前置知识检查

### 必备前置概念清单
- **Python基础**：列表、循环、函数
- **线性代数**：向量、矩阵的基本概念
- **数学运算**：加减乘除、指数运算

### 回顾链接/跳转
- 如果不熟悉Python：`01_Python进阶/`
- 如果不熟悉线性代数：`02_数学基础/01_线性代数/`

### 入门小测

**选择题**（每题2分，共10分）

1. Python列表和NumPy数组的主要区别？
   A. 列表更快  B. 数组支持向量化运算  C. 列表更灵活  D. 数组更简单
   **答案**：B

2. NumPy数组的维度如何表示？
   A. shape  B. size  C. dtype  D. ndim
   **答案**：A

3. 广播机制的作用？
   A. 扩展数组  B. 对不同形状数组进行运算  C. 复制数组  D. 删除数组
   **答案**：B

4. 如何创建全零数组？
   A. np.array([0])  B. np.zeros()  C. np.empty()  D. np.ones()
   **答案**：B

5. 数组转置的方法？
   A. array.T  B. array.transpose()  C. 以上都可以  D. 以上都不可以
   **答案**：C

**评分标准**：≥8分（80%）为通过

---

## 3. 核心知识点详解

### 3.1 NumPy数组基础

#### 概念引入与直观类比

**类比**：NumPy数组就像"高效的表格"，比Python列表快得多。

- **Python列表**：灵活但慢，每个元素是独立对象
- **NumPy数组**：固定类型，连续内存，向量化运算

#### 逐步理论推导

**步骤1：创建数组**
```python
import numpy as np
arr = np.array([1, 2, 3, 4, 5])
```

**步骤2：数组属性**
- **shape**：形状（维度）
- **dtype**：数据类型
- **size**：元素总数
- **ndim**：维度数

**步骤3：数组运算**
```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = arr1 + arr2  # [5, 7, 9]（向量化）
```

**步骤4：广播机制**
```python
arr = np.array([1, 2, 3])
result = arr * 2  # [2, 4, 6]（广播）
```

#### 关键性质

**优点**：
- **速度快**：C语言实现，比Python快
- **内存效率**：连续内存，节省空间
- **向量化**：批量运算，无需循环

**适用场景**：
- 数值计算
- 矩阵运算
- 数据处理
- 机器学习

---

### 3.2 数组操作

#### 索引和切片

```python
arr = np.array([0, 1, 2, 3, 4, 5])
print(arr[0])      # 0
print(arr[1:4])    # [1, 2, 3]
print(arr[::2])    # [0, 2, 4]（步长2）
```

#### 形状变换

```python
arr = np.arange(12)
arr_2d = arr.reshape(3, 4)  # 3行4列
arr_flat = arr_2d.flatten()  # 展平
```

#### 数组拼接

```python
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
result = np.concatenate([arr1, arr2])  # [1, 2, 3, 4, 5, 6]
```

---

### 3.3 数学运算

#### 基本运算

```python
arr = np.array([1, 2, 3, 4, 5])
print(arr.sum())      # 15
print(arr.mean())     # 3.0
print(arr.std())      # 标准差
print(arr.max())      # 5
print(arr.min())      # 1
```

#### 矩阵运算

```python
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
C = np.dot(A, B)  # 矩阵乘法
```

#### 广播机制

```python
# 形状 (3, 1) 和 (1, 3) 可以广播到 (3, 3)
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([1, 2, 3])        # (3,)
result = a + b  # 广播
```

---

## 4. Python代码实践

### 4.1 环境与依赖版本

- **Python版本**：3.8+
- **依赖**：
  - numpy >= 1.20.0

### 4.2 从零开始的完整可运行示例

#### 示例1：数组创建和基本操作

```python
import numpy as np

# 创建数组
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([[1, 2, 3], [4, 5, 6]])

print("一维数组:", arr1)
print("形状:", arr1.shape)
print("数据类型:", arr1.dtype)

print("\n二维数组:")
print(arr2)
print("形状:", arr2.shape)

# 创建特殊数组
zeros = np.zeros((3, 4))
ones = np.ones((2, 3))
full = np.full((2, 2), 7)
arange = np.arange(0, 10, 2)
linspace = np.linspace(0, 1, 5)

print("\n全零数组:")
print(zeros)
print("\n全1数组:")
print(ones)
print("\narange:", arange)
print("linspace:", linspace)
```

**运行结果**：
```
一维数组: [1 2 3 4 5]
形状: (5,)
数据类型: int64

二维数组:
[[1 2 3]
 [4 5 6]]
形状: (2, 3)
...
```

#### 示例2：数组运算

```python
import numpy as np

# 数组运算
a = np.array([1, 2, 3, 4])
b = np.array([5, 6, 7, 8])

print("加法:", a + b)
print("减法:", a - b)
print("乘法:", a * b)  # 元素乘积
print("除法:", a / b)
print("幂运算:", a ** 2)

# 统计函数
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print("\n统计信息:")
print(f"和: {arr.sum()}")
print(f"均值: {arr.mean():.2f}")
print(f"标准差: {arr.std():.2f}")
print(f"最大值: {arr.max()}")
print(f"最小值: {arr.min()}")
print(f"中位数: {np.median(arr)}")
```

#### 示例3：索引和切片

```python
import numpy as np

# 创建数组
arr = np.arange(12).reshape(3, 4)
print("原数组:")
print(arr)

# 索引
print("\narr[0, 0]:", arr[0, 0])
print("arr[1, 2]:", arr[1, 2])

# 切片
print("\n第一行:", arr[0, :])
print("第一列:", arr[:, 0])
print("前两行:", arr[:2, :])
print("前两列:", arr[:, :2])

# 布尔索引
mask = arr > 5
print("\n大于5的元素:")
print(arr[mask])
```

#### 示例4：广播机制

```python
import numpy as np

# 广播示例1：标量与数组
arr = np.array([1, 2, 3, 4])
result = arr * 2
print("数组 * 2:", result)

# 广播示例2：不同形状的数组
a = np.array([[1], [2], [3]])  # (3, 1)
b = np.array([1, 2, 3])        # (3,)
print("\na的形状:", a.shape)
print("b的形状:", b.shape)
print("a + b (广播):")
print(a + b)

# 广播示例3：矩阵运算
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
B = np.array([1, 2, 3])               # (3,)
print("\nA + B (广播):")
print(A + B)
```

### 4.3 常见错误与排查

**错误1**：混淆元素乘积和矩阵乘法
```python
# 元素乘积
A * B

# 矩阵乘法
np.dot(A, B)  # 或 A @ B
```

**错误2**：修改数组时影响原数组
```python
# 错误：视图共享内存
arr2 = arr1
arr2[0] = 999  # arr1也会改变

# 正确：创建副本
arr2 = arr1.copy()
arr2[0] = 999  # arr1不变
```

**错误3**：广播形状不兼容
```python
# 错误：无法广播
a = np.array([1, 2, 3])      # (3,)
b = np.array([[1], [2]])     # (2, 1)
# a + b  # 错误

# 正确：兼容的形状
a = np.array([[1, 2, 3]])    # (1, 3)
b = np.array([[1], [2]])     # (2, 1)
# a + b  # 可以广播到 (2, 3)
```

---

## 5. 动手练习

### 基础练习（3-5题）

**练习1：数组创建**
创建各种形状的数组：一维、二维、三维。

**练习2：数组运算**
对数组进行各种数学运算和统计计算。

**练习3：索引和切片**
练习各种索引和切片操作。

### 进阶练习（2-3题）

**练习1：矩阵运算**
实现矩阵的乘法、转置、求逆等操作。

**练习2：广播应用**
使用广播机制实现高效的数组运算。

### 挑战练习（1-2题）

**练习1：图像处理基础**
使用NumPy处理图像数据（灰度转换、裁剪等）。

---

## 6. 实际案例

### 案例：使用NumPy进行数据分析

**业务背景**：
分析学生成绩数据，计算统计信息。

**问题抽象**：
- 数据：学生成绩矩阵
- 操作：计算平均分、最高分、最低分等
- 输出：统计报告

**端到端实现**：
```python
import numpy as np

# 生成模拟数据（5个学生，3门课程）
np.random.seed(42)
scores = np.random.randint(60, 100, size=(5, 3))
students = ['Alice', 'Bob', 'Charlie', 'David', 'Eve']
courses = ['Math', 'English', 'Science']

print("成绩矩阵:")
print(scores)

# 每个学生的平均分
student_avg = scores.mean(axis=1)
print("\n每个学生的平均分:")
for i, name in enumerate(students):
    print(f"{name}: {student_avg[i]:.2f}")

# 每门课程的平均分
course_avg = scores.mean(axis=0)
print("\n每门课程的平均分:")
for i, course in enumerate(courses):
    print(f"{course}: {course_avg[i]:.2f}")

# 总体统计
print("\n总体统计:")
print(f"最高分: {scores.max()}")
print(f"最低分: {scores.min()}")
print(f"平均分: {scores.mean():.2f}")
print(f"标准差: {scores.std():.2f}")
```

**结果解读**：
- 可以快速计算各种统计信息
- NumPy的向量化运算非常高效

**改进方向**：
- 添加可视化
- 添加异常值检测
- 添加排名功能

---

## 7. 自我评估

### 概念题

**选择题**（每题2分，共20分）

1. NumPy数组的主要优势？
   A. 灵活  B. 快速  C. 简单  D. 美观
   **答案**：B

2. 数组的shape属性表示？
   A. 大小  B. 形状  C. 类型  D. 维度数
   **答案**：B

3. 如何创建全零数组？
   A. np.array([0])  B. np.zeros()  C. np.empty()  D. np.ones()
   **答案**：B

4. 广播机制的作用？
   A. 扩展数组  B. 对不同形状数组运算  C. 复制数组  D. 删除数组
   **答案**：B

5. 矩阵乘法的函数？
   A. *  B. np.dot()  C. np.multiply()  D. 以上都可以
   **答案**：B

**简答题**（每题10分，共40分）

1. 解释NumPy数组和Python列表的区别。
   **参考答案**：NumPy数组固定类型、连续内存、支持向量化运算，速度快；Python列表灵活但慢。

2. 说明广播机制的原理。
   **参考答案**：广播允许对不同形状的数组进行运算，通过扩展维度使形状兼容。

3. 解释数组的索引和切片。
   **参考答案**：索引访问单个元素，切片访问子数组，支持多维索引和布尔索引。

4. 说明NumPy在AI中的重要性。
   **参考答案**：NumPy是几乎所有AI库的基础，提供高效的数值计算能力。

### 编程实践题（20分）

使用NumPy实现矩阵运算和统计计算。

### 综合应用题（20分）

使用NumPy处理和分析数据集。

**总分**：100分，≥80分为通过

---

## 8. 拓展学习

### 论文/书籍/优质课程

**书籍推荐**：
- 《Python数据科学手册》- Jake VanderPlas（第2章）
- NumPy官方文档

**在线资源**：
- NumPy官方教程
- NumPy用户指南

### 相关工具与库

- **SciPy**：科学计算（基于NumPy）
- **Pandas**：数据分析（基于NumPy）
- **Matplotlib**：可视化（使用NumPy数组）

### 进阶话题指引

完成本课程后，可以学习：
- **高级索引**：花式索引、布尔索引
- **结构化数组**：处理结构化数据
- **性能优化**：向量化、并行计算

### 下节课预告

下一课将学习：
- **02_Pandas**：更强大的数据处理工具
- Pandas基于NumPy，提供更高级的数据操作

### 学习建议

1. **多实践**：每学一个功能，立即写代码
2. **理解广播**：广播是NumPy的核心特性
3. **性能对比**：对比NumPy和Python列表的性能
4. **持续学习**：NumPy是AI的基础，需要熟练掌握

---

**恭喜完成第一课！你已经掌握了NumPy的基础，准备好学习Pandas了！**

