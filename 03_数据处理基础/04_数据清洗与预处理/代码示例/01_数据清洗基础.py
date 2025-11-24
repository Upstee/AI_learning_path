"""
数据清洗基础示例
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# ========== 1. 缺失值处理 ==========
print("=" * 50)
print("1. 缺失值处理")
print("=" * 50)

# 创建包含缺失值的数据
data = {
    'A': [1, 2, np.nan, 4, 5],
    'B': [1, np.nan, 3, np.nan, 5],
    'C': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
print("原始数据:")
print(df)
print(f"\n缺失值统计:\n{df.isna().sum()}")

# 删除缺失值
df_dropped = df.dropna()
print(f"\n删除缺失值后:\n{df_dropped}")

# 填充缺失值
df_filled_mean = df.fillna(df.mean())
print(f"\n用均值填充:\n{df_filled_mean}")

df_filled_median = df.fillna(df.median())
print(f"\n用中位数填充:\n{df_filled_median}")

# 前向填充
df_ffill = df.fillna(method='ffill')
print(f"\n前向填充:\n{df_ffill}")

# ========== 2. 异常值处理 ==========
print("\n" + "=" * 50)
print("2. 异常值处理")
print("=" * 50)

# 创建包含异常值的数据
np.random.seed(42)
data = np.random.normal(100, 15, 100)
data = np.append(data, [200, 250, 30, 20])  # 添加异常值
df = pd.DataFrame({'value': data})
print(f"原始数据统计:\n{df.describe()}")

# Z-score方法检测异常值
mean = df['value'].mean()
std = df['value'].std()
z_scores = np.abs((df['value'] - mean) / std)
outliers_z = df[z_scores > 3]
print(f"\nZ-score方法检测到的异常值数量: {len(outliers_z)}")

# IQR方法检测异常值
Q1 = df['value'].quantile(0.25)
Q3 = df['value'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
outliers_iqr = df[(df['value'] < lower_bound) | (df['value'] > upper_bound)]
print(f"IQR方法检测到的异常值数量: {len(outliers_iqr)}")

# 处理异常值：截断
df_clipped = df.copy()
df_clipped['value'] = df_clipped['value'].clip(lower=lower_bound, upper=upper_bound)
print(f"\n截断后的数据范围: [{df_clipped['value'].min():.2f}, {df_clipped['value'].max():.2f}]")

# ========== 3. 重复值处理 ==========
print("\n" + "=" * 50)
print("3. 重复值处理")
print("=" * 50)

# 创建包含重复值的数据
data = {
    'A': [1, 2, 2, 3, 4],
    'B': [1, 2, 2, 3, 4],
    'C': [1, 2, 3, 4, 5]
}
df = pd.DataFrame(data)
print("原始数据:")
print(df)
print(f"\n重复行数: {df.duplicated().sum()}")

# 删除重复值
df_no_dup = df.drop_duplicates()
print(f"\n删除重复值后:\n{df_no_dup}")

# 基于特定列删除重复值
df_no_dup_col = df.drop_duplicates(subset=['A', 'B'])
print(f"\n基于A和B列删除重复值后:\n{df_no_dup_col}")

# ========== 4. 数据标准化 ==========
print("\n" + "=" * 50)
print("4. 数据标准化")
print("=" * 50)

# 创建数据
data = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12]])
df = pd.DataFrame(data, columns=['A', 'B', 'C'])
print("原始数据:")
print(df)
print(f"\n原始数据统计:\n{df.describe()}")

# Z-score标准化
scaler_std = StandardScaler()
df_std = pd.DataFrame(scaler_std.fit_transform(df), columns=df.columns)
print(f"\nZ-score标准化后:\n{df_std}")
print(f"\n标准化后统计:\n{df_std.describe()}")

# Min-Max归一化
scaler_minmax = MinMaxScaler()
df_minmax = pd.DataFrame(scaler_minmax.fit_transform(df), columns=df.columns)
print(f"\nMin-Max归一化后:\n{df_minmax}")
print(f"\n归一化后统计:\n{df_minmax.describe()}")

# Robust标准化（对异常值稳健）
scaler_robust = RobustScaler()
df_robust = pd.DataFrame(scaler_robust.fit_transform(df), columns=df.columns)
print(f"\nRobust标准化后:\n{df_robust}")

# ========== 5. 特征工程 ==========
print("\n" + "=" * 50)
print("5. 特征工程")
print("=" * 50)

# 创建示例数据
df = pd.DataFrame({
    'age': [25, 30, 35, 40, 45],
    'income': [50000, 60000, 70000, 80000, 90000],
    'category': ['A', 'B', 'A', 'B', 'A']
})
print("原始数据:")
print(df)

# 特征组合
df['age_income'] = df['age'] * df['income']
print(f"\n特征组合后:\n{df[['age', 'income', 'age_income']]}")

# 特征变换
df['age_log'] = np.log(df['age'])
df['income_sqrt'] = np.sqrt(df['income'])
print(f"\n特征变换后:\n{df[['age', 'age_log', 'income', 'income_sqrt']]}")

# 独热编码
df_encoded = pd.get_dummies(df, columns=['category'], prefix='cat')
print(f"\n独热编码后:\n{df_encoded}")

# ========== 6. 完整数据清洗流程 ==========
print("\n" + "=" * 50)
print("6. 完整数据清洗流程")
print("=" * 50)

# 创建包含多种问题的数据
np.random.seed(42)
data = {
    'A': np.random.normal(100, 15, 100),
    'B': np.random.normal(50, 10, 100),
    'C': np.random.choice(['X', 'Y', 'Z'], 100)
}
df = pd.DataFrame(data)

# 添加问题
df.loc[10:15, 'A'] = np.nan  # 添加缺失值
df.loc[20:25, 'B'] = 200  # 添加异常值
df = pd.concat([df, df.iloc[0:5]], ignore_index=True)  # 添加重复值

print("问题数据:")
print(f"形状: {df.shape}")
print(f"缺失值: {df.isna().sum().sum()}")
print(f"重复值: {df.duplicated().sum()}")

# 清洗流程
# 1. 处理缺失值
df_clean = df.fillna(df.mean())

# 2. 处理异常值
Q1 = df_clean['B'].quantile(0.25)
Q3 = df_clean['B'].quantile(0.75)
IQR = Q3 - Q1
df_clean['B'] = df_clean['B'].clip(lower=Q1-1.5*IQR, upper=Q3+1.5*IQR)

# 3. 删除重复值
df_clean = df_clean.drop_duplicates()

# 4. 标准化
scaler = StandardScaler()
df_clean[['A', 'B']] = scaler.fit_transform(df_clean[['A', 'B']])

print(f"\n清洗后数据:")
print(f"形状: {df_clean.shape}")
print(f"缺失值: {df_clean.isna().sum().sum()}")
print(f"重复值: {df_clean.duplicated().sum()}")
print(f"\n清洗后数据预览:\n{df_clean.head()}")

print("\n" + "=" * 50)
print("数据清洗基础示例完成！")
print("=" * 50)

