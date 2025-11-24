"""
NumPy数组基础操作示例
"""

import numpy as np

# ========== 1. 数组创建 ==========
print("=" * 50)
print("1. 数组创建")
print("=" * 50)

# 从列表创建
arr1 = np.array([1, 2, 3, 4, 5])
print(f"从列表创建: {arr1}")

# 预分配数组
zeros = np.zeros((3, 4))
print(f"\n全零数组 (3x4):\n{zeros}")

ones = np.ones((2, 3))
print(f"\n全1数组 (2x3):\n{ones}")

# 使用arange
arr_range = np.arange(0, 10, 2)
print(f"\narange(0, 10, 2): {arr_range}")

# 使用linspace
arr_linspace = np.linspace(0, 1, 5)
print(f"linspace(0, 1, 5): {arr_linspace}")

# 随机数组
random_arr = np.random.rand(3, 3)
print(f"\n随机数组 (3x3):\n{random_arr}")

# ========== 2. 数组属性 ==========
print("\n" + "=" * 50)
print("2. 数组属性")
print("=" * 50)

arr = np.array([[1, 2, 3], [4, 5, 6]])
print(f"数组:\n{arr}")
print(f"形状 (shape): {arr.shape}")
print(f"维度数 (ndim): {arr.ndim}")
print(f"元素总数 (size): {arr.size}")
print(f"数据类型 (dtype): {arr.dtype}")
print(f"元素大小 (itemsize): {arr.itemsize} 字节")
print(f"总内存 (nbytes): {arr.nbytes} 字节")

# ========== 3. 数组索引和切片 ==========
print("\n" + "=" * 50)
print("3. 数组索引和切片")
print("=" * 50)

arr = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
print(f"原数组: {arr}")
print(f"arr[0]: {arr[0]}")
print(f"arr[-1]: {arr[-1]}")
print(f"arr[2:5]: {arr[2:5]}")
print(f"arr[::2]: {arr[::2]}")  # 步长为2

# 多维数组索引
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D数组:\n{arr2d}")
print(f"arr2d[0, 1]: {arr2d[0, 1]}")
print(f"arr2d[1:, :2]:\n{arr2d[1:, :2]}")

# 布尔索引
arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
mask = arr > 5
print(f"\n布尔索引 (arr > 5): {arr[mask]}")

# ========== 4. 数组运算 ==========
print("\n" + "=" * 50)
print("4. 数组运算")
print("=" * 50)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"arr1 + arr2: {arr1 + arr2}")
print(f"arr1 * arr2: {arr1 * arr2}")
print(f"arr1 ** 2: {arr1 ** 2}")
print(f"np.sin(arr1): {np.sin(arr1)}")

# 矩阵运算
A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])
print(f"\n矩阵A:\n{A}")
print(f"矩阵B:\n{B}")
print(f"矩阵乘法 A @ B:\n{A @ B}")
print(f"元素级乘法 A * B:\n{A * B}")

# ========== 5. 广播机制 ==========
print("\n" + "=" * 50)
print("5. 广播机制")
print("=" * 50)

arr = np.array([[1, 2, 3], [4, 5, 6]])
scalar = 10
print(f"数组:\n{arr}")
print(f"标量: {scalar}")
print(f"数组 + 标量:\n{arr + scalar}")

arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([10, 20, 30])
print(f"\n数组1:\n{arr1}")
print(f"数组2: {arr2}")
print(f"数组1 + 数组2 (广播):\n{arr1 + arr2}")

# ========== 6. 形状变换 ==========
print("\n" + "=" * 50)
print("6. 形状变换")
print("=" * 50)

arr = np.arange(12)
print(f"原数组: {arr}")
print(f"reshape(3, 4):\n{arr.reshape(3, 4)}")
print(f"reshape(-1, 4):\n{arr.reshape(-1, 4)}")

arr2d = np.array([[1, 2, 3], [4, 5, 6]])
print(f"\n2D数组:\n{arr2d}")
print(f"转置:\n{arr2d.T}")
print(f"展平 (flatten): {arr2d.flatten()}")
print(f"展平 (ravel): {arr2d.ravel()}")

# ========== 7. 数组拼接 ==========
print("\n" + "=" * 50)
print("7. 数组拼接")
print("=" * 50)

arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])
print(f"arr1: {arr1}")
print(f"arr2: {arr2}")
print(f"concatenate: {np.concatenate([arr1, arr2])}")

arr1_2d = np.array([[1, 2], [3, 4]])
arr2_2d = np.array([[5, 6], [7, 8]])
print(f"\n垂直堆叠:\n{np.vstack([arr1_2d, arr2_2d])}")
print(f"水平堆叠:\n{np.hstack([arr1_2d, arr2_2d])}")

# ========== 8. 数组统计 ==========
print("\n" + "=" * 50)
print("8. 数组统计")
print("=" * 50)

arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
print(f"数组: {arr}")
print(f"平均值: {np.mean(arr)}")
print(f"中位数: {np.median(arr)}")
print(f"标准差: {np.std(arr)}")
print(f"方差: {np.var(arr)}")
print(f"最大值: {np.max(arr)}")
print(f"最小值: {np.min(arr)}")
print(f"求和: {np.sum(arr)}")
print(f"累积和: {np.cumsum(arr)}")

# 多维数组统计
arr2d = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\n2D数组:\n{arr2d}")
print(f"按列求和: {np.sum(arr2d, axis=0)}")
print(f"按行求和: {np.sum(arr2d, axis=1)}")
print(f"总体平均值: {np.mean(arr2d)}")

# ========== 9. 数组搜索和排序 ==========
print("\n" + "=" * 50)
print("9. 数组搜索和排序")
print("=" * 50)

arr = np.array([3, 1, 4, 1, 5, 9, 2, 6])
print(f"原数组: {arr}")
print(f"排序后: {np.sort(arr)}")
print(f"排序索引: {np.argsort(arr)}")

# 搜索
print(f"\n最大值索引: {np.argmax(arr)}")
print(f"最小值索引: {np.argmin(arr)}")
print(f"值4的索引: {np.where(arr == 4)}")

# ========== 10. 性能对比 ==========
print("\n" + "=" * 50)
print("10. 性能对比示例")
print("=" * 50)

import time

# Python列表
python_list = list(range(1000000))
start = time.time()
result_python = [x * 2 for x in python_list]
time_python = time.time() - start

# NumPy数组
numpy_arr = np.array(range(1000000))
start = time.time()
result_numpy = numpy_arr * 2
time_numpy = time.time() - start

print(f"Python列表时间: {time_python:.4f}秒")
print(f"NumPy数组时间: {time_numpy:.4f}秒")
print(f"加速比: {time_python/time_numpy:.2f}倍")

