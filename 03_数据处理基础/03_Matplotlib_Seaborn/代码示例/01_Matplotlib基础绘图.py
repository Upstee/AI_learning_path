"""
Matplotlib基础绘图示例
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# ========== 1. 基础线图 ==========
print("=" * 50)
print("1. 基础线图")
print("=" * 50)

x = np.linspace(0, 10, 100)
y = np.sin(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y, label='sin(x)')
plt.xlabel('X轴')
plt.ylabel('Y轴')
plt.title('正弦函数')
plt.legend()
plt.grid(True)
plt.savefig('line_plot.png', dpi=300)
plt.show()

# ========== 2. 散点图 ==========
print("\n2. 散点图")

x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5

plt.figure(figsize=(10, 6))
plt.scatter(x, y, alpha=0.6)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('散点图')
plt.grid(True)
plt.savefig('scatter_plot.png', dpi=300)
plt.show()

# ========== 3. 柱状图 ==========
print("\n3. 柱状图")

categories = ['A', 'B', 'C', 'D', 'E']
values = [23, 45, 56, 78, 32]

plt.figure(figsize=(10, 6))
plt.bar(categories, values, color='skyblue')
plt.xlabel('类别')
plt.ylabel('值')
plt.title('柱状图')
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('bar_plot.png', dpi=300)
plt.show()

# ========== 4. 直方图 ==========
print("\n4. 直方图")

data = np.random.normal(100, 15, 1000)

plt.figure(figsize=(10, 6))
plt.hist(data, bins=30, edgecolor='black', alpha=0.7)
plt.xlabel('值')
plt.ylabel('频率')
plt.title('直方图')
plt.grid(True, axis='y', alpha=0.3)
plt.savefig('histogram.png', dpi=300)
plt.show()

# ========== 5. 子图 ==========
print("\n5. 子图")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# 子图1：线图
x = np.linspace(0, 10, 100)
axes[0, 0].plot(x, np.sin(x))
axes[0, 0].set_title('正弦函数')
axes[0, 0].grid(True)

# 子图2：散点图
x = np.random.randn(100)
y = np.random.randn(100)
axes[0, 1].scatter(x, y, alpha=0.6)
axes[0, 1].set_title('散点图')
axes[0, 1].grid(True)

# 子图3：柱状图
categories = ['A', 'B', 'C', 'D']
values = [10, 20, 30, 40]
axes[1, 0].bar(categories, values)
axes[1, 0].set_title('柱状图')
axes[1, 0].grid(True, axis='y')

# 子图4：直方图
data = np.random.normal(0, 1, 1000)
axes[1, 1].hist(data, bins=30, edgecolor='black')
axes[1, 1].set_title('直方图')
axes[1, 1].grid(True, axis='y')

plt.tight_layout()
plt.savefig('subplots.png', dpi=300)
plt.show()

# ========== 6. 多线图 ==========
print("\n6. 多线图")

x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)
y3 = np.sin(x) * np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)', linewidth=2)
plt.plot(x, y2, label='cos(x)', linewidth=2)
plt.plot(x, y3, label='sin(x)*cos(x)', linewidth=2)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('多线图')
plt.legend()
plt.grid(True)
plt.savefig('multi_line.png', dpi=300)
plt.show()

print("\n" + "=" * 50)
print("Matplotlib基础绘图示例完成！")
print("=" * 50)

