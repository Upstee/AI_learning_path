"""
Seaborn统计图表示例
"""

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 设置Seaborn样式
sns.set_style("whitegrid")
sns.set_palette("husl")

# ========== 1. 分布图 ==========
print("=" * 50)
print("1. 分布图")
print("=" * 50)

# 生成示例数据
data = np.random.normal(100, 15, 1000)
df = pd.DataFrame({'value': data})

# 单变量分布
plt.figure(figsize=(10, 6))
sns.histplot(data=df, x='value', kde=True)
plt.title('分布图（带密度曲线）')
plt.savefig('seaborn_dist.png', dpi=300)
plt.show()

# ========== 2. 散点图 ==========
print("\n2. 散点图")

# 生成示例数据
np.random.seed(42)
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': 2 * np.random.randn(100) + 1,
    'category': np.random.choice(['A', 'B', 'C'], 100)
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='x', y='y', hue='category', style='category')
plt.title('散点图（按类别着色）')
plt.savefig('seaborn_scatter.png', dpi=300)
plt.show()

# ========== 3. 箱线图 ==========
print("\n3. 箱线图")

# 生成示例数据
df = pd.DataFrame({
    'category': np.repeat(['A', 'B', 'C'], 100),
    'value': np.concatenate([
        np.random.normal(10, 2, 100),
        np.random.normal(15, 3, 100),
        np.random.normal(12, 2.5, 100)
    ])
})

plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='category', y='value')
plt.title('箱线图')
plt.savefig('seaborn_box.png', dpi=300)
plt.show()

# ========== 4. 小提琴图 ==========
print("\n4. 小提琴图")

plt.figure(figsize=(10, 6))
sns.violinplot(data=df, x='category', y='value')
plt.title('小提琴图')
plt.savefig('seaborn_violin.png', dpi=300)
plt.show()

# ========== 5. 热力图 ==========
print("\n5. 热力图")

# 生成相关性矩阵
data = np.random.randn(100, 5)
df_corr = pd.DataFrame(data).corr()

plt.figure(figsize=(10, 8))
sns.heatmap(df_corr, annot=True, cmap='coolwarm', center=0)
plt.title('相关性热力图')
plt.savefig('seaborn_heatmap.png', dpi=300)
plt.show()

# ========== 6. 成对关系图 ==========
print("\n6. 成对关系图")

# 生成示例数据
iris = sns.load_dataset('iris')

sns.pairplot(iris, hue='species')
plt.savefig('seaborn_pairplot.png', dpi=300)
plt.show()

# ========== 7. 回归图 ==========
print("\n7. 回归图")

# 生成示例数据
np.random.seed(42)
x = np.random.randn(100)
y = 2 * x + np.random.randn(100) * 0.5
df = pd.DataFrame({'x': x, 'y': y})

plt.figure(figsize=(10, 6))
sns.regplot(data=df, x='x', y='y')
plt.title('回归图')
plt.savefig('seaborn_regplot.png', dpi=300)
plt.show()

print("\n" + "=" * 50)
print("Seaborn统计图表示例完成！")
print("=" * 50)

