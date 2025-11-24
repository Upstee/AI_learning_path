"""
Pandas Series和DataFrame基础操作示例
"""

import pandas as pd
import numpy as np

# ========== 1. Series创建和操作 ==========
print("=" * 50)
print("1. Series创建和操作")
print("=" * 50)

# 从列表创建
s1 = pd.Series([1, 2, 3, 4, 5])
print(f"从列表创建:\n{s1}")

# 从字典创建
s2 = pd.Series({'a': 1, 'b': 2, 'c': 3})
print(f"\n从字典创建:\n{s2}")

# 指定索引
s3 = pd.Series([1, 2, 3, 4], index=['A', 'B', 'C', 'D'])
print(f"\n指定索引:\n{s3}")

# Series属性
print(f"\nSeries属性:")
print(f"  值: {s3.values}")
print(f"  索引: {s3.index}")
print(f"  数据类型: {s3.dtype}")
print(f"  大小: {s3.size}")

# Series索引
print(f"\nSeries索引:")
print(f"  s3['A']: {s3['A']}")
print(f"  s3[0:2]:\n{s3[0:2]}")

# ========== 2. DataFrame创建 ==========
print("\n" + "=" * 50)
print("2. DataFrame创建")
print("=" * 50)

# 从字典创建
data = {
    'name': ['Alice', 'Bob', 'Charlie', 'David'],
    'age': [25, 30, 35, 40],
    'city': ['New York', 'London', 'Tokyo', 'Paris']
}
df = pd.DataFrame(data)
print(f"从字典创建:\n{df}")

# 从列表创建
data_list = [['Alice', 25], ['Bob', 30], ['Charlie', 35]]
df2 = pd.DataFrame(data_list, columns=['name', 'age'])
print(f"\n从列表创建:\n{df2}")

# 从NumPy数组创建
arr = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df3 = pd.DataFrame(arr, columns=['A', 'B', 'C'])
print(f"\n从NumPy数组创建:\n{df3}")

# ========== 3. DataFrame基本操作 ==========
print("\n" + "=" * 50)
print("3. DataFrame基本操作")
print("=" * 50)

# 查看基本信息
print(f"DataFrame形状: {df.shape}")
print(f"DataFrame列名: {df.columns.tolist()}")
print(f"DataFrame索引: {df.index.tolist()}")
print(f"\nDataFrame信息:")
print(df.info())
print(f"\nDataFrame描述性统计:")
print(df.describe())

# 访问列
print(f"\n访问列:")
print(f"df['name']:\n{df['name']}")
print(f"df.name:\n{df.name}")

# 访问行
print(f"\n访问行:")
print(f"df.iloc[0]:\n{df.iloc[0]}")
print(f"df.loc[0]:\n{df.loc[0]}")

# ========== 4. 数据筛选 ==========
print("\n" + "=" * 50)
print("4. 数据筛选")
print("=" * 50)

# 条件筛选
print(f"年龄大于30的人:\n{df[df['age'] > 30]}")
print(f"\n多个条件:\n{df[(df['age'] > 30) & (df['city'] == 'Tokyo')]}")

# 选择列
print(f"\n选择特定列:\n{df[['name', 'age']]}")

# 使用query方法
print(f"\n使用query:\n{df.query('age > 30')}")

# ========== 5. 数据修改 ==========
print("\n" + "=" * 50)
print("5. 数据修改")
print("=" * 50)

# 添加列
df['salary'] = [50000, 60000, 70000, 80000]
print(f"添加salary列:\n{df}")

# 修改值
df.loc[0, 'age'] = 26
print(f"\n修改值后:\n{df}")

# 删除列
df_dropped = df.drop('salary', axis=1)
print(f"\n删除列后:\n{df_dropped}")

# ========== 6. 缺失值处理 ==========
print("\n" + "=" * 50)
print("6. 缺失值处理")
print("=" * 50)

# 创建包含缺失值的数据
df_missing = pd.DataFrame({
    'A': [1, 2, np.nan, 4],
    'B': [5, np.nan, 7, 8],
    'C': [9, 10, 11, 12]
})
print(f"包含缺失值的数据:\n{df_missing}")
print(f"\n缺失值检测:\n{df_missing.isna()}")
print(f"\n缺失值统计:\n{df_missing.isna().sum()}")

# 删除缺失值
df_dropped_na = df_missing.dropna()
print(f"\n删除缺失值后:\n{df_dropped_na}")

# 填充缺失值
df_filled = df_missing.fillna(0)
print(f"\n用0填充后:\n{df_filled}")

# ========== 7. 分组聚合 ==========
print("\n" + "=" * 50)
print("7. 分组聚合")
print("=" * 50)

# 创建示例数据
df_group = pd.DataFrame({
    'category': ['A', 'A', 'B', 'B', 'A', 'B'],
    'value': [10, 20, 30, 40, 50, 60],
    'count': [1, 2, 3, 4, 5, 6]
})
print(f"原始数据:\n{df_group}")

# 分组求和
grouped_sum = df_group.groupby('category').sum()
print(f"\n按category分组求和:\n{grouped_sum}")

# 分组求平均
grouped_mean = df_group.groupby('category').mean()
print(f"\n按category分组求平均:\n{grouped_mean}")

# 多个聚合函数
grouped_agg = df_group.groupby('category').agg(['sum', 'mean', 'std'])
print(f"\n多个聚合函数:\n{grouped_agg}")

# ========== 8. 数据合并 ==========
print("\n" + "=" * 50)
print("8. 数据合并")
print("=" * 50)

# 创建两个DataFrame
df1 = pd.DataFrame({
    'key': ['A', 'B', 'C'],
    'value1': [1, 2, 3]
})
df2 = pd.DataFrame({
    'key': ['B', 'C', 'D'],
    'value2': [4, 5, 6]
})
print(f"df1:\n{df1}")
print(f"\ndf2:\n{df2}")

# 内连接
merged_inner = pd.merge(df1, df2, on='key', how='inner')
print(f"\n内连接:\n{merged_inner}")

# 外连接
merged_outer = pd.merge(df1, df2, on='key', how='outer')
print(f"\n外连接:\n{merged_outer}")

# 左连接
merged_left = pd.merge(df1, df2, on='key', how='left')
print(f"\n左连接:\n{merged_left}")

# ========== 9. 数据排序 ==========
print("\n" + "=" * 50)
print("9. 数据排序")
print("=" * 50)

df_sort = pd.DataFrame({
    'name': ['Alice', 'Bob', 'Charlie'],
    'age': [30, 25, 35],
    'score': [85, 90, 80]
})
print(f"原始数据:\n{df_sort}")

# 按年龄排序
sorted_by_age = df_sort.sort_values('age')
print(f"\n按年龄排序:\n{sorted_by_age}")

# 多列排序
sorted_multi = df_sort.sort_values(['score', 'age'], ascending=[False, True])
print(f"\n按分数降序、年龄升序排序:\n{sorted_multi}")

# ========== 10. 数据读写 ==========
print("\n" + "=" * 50)
print("10. 数据读写")
print("=" * 50)

# 保存为CSV
df.to_csv('example.csv', index=False)
print("已保存为CSV文件: example.csv")

# 读取CSV
df_read = pd.read_csv('example.csv')
print(f"\n从CSV读取:\n{df_read}")

# 保存为Excel（需要openpyxl）
# df.to_excel('example.xlsx', index=False)

print("\n" + "=" * 50)
print("示例完成！")
print("=" * 50)

