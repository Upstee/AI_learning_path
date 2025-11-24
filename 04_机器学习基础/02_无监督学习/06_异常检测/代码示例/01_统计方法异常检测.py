"""
统计方法异常检测
本示例展示如何使用统计方法进行异常检测
适合小白学习，包含大量详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from scipy import stats

# ========== 1. Z-score方法 ==========
print("=" * 60)
print("1. Z-score方法")
print("=" * 60)
print("""
Z-score方法：
- 计算样本与均值的距离（标准差倍数）
- Z-score > 3 或 < -3 的样本视为异常
- 假设数据遵循正态分布
""")

# 生成正常数据
np.random.seed(42)
normal_data = np.random.normal(0, 1, 1000)

# 添加一些异常数据
anomaly_data = np.random.normal(0, 1, 50) * 3 + 5  # 远离正常数据
data = np.concatenate([normal_data, anomaly_data])

print(f"数据信息:")
print(f"  正常样本数: {len(normal_data)}")
print(f"  异常样本数: {len(anomaly_data)}")
print(f"  总样本数: {len(data)}")

# 计算Z-score
mean = np.mean(data)
std = np.std(data)
z_scores = np.abs((data - mean) / std)

# 识别异常（Z-score > 3）
threshold = 3
anomalies = z_scores > threshold

print(f"\nZ-score统计:")
print(f"  均值: {mean:.4f}")
print(f"  标准差: {std:.4f}")
print(f"  阈值: {threshold}")
print(f"  识别出的异常数: {np.sum(anomalies)}")

# 可视化
plt.figure(figsize=(12, 6))

# 左图：数据分布
plt.subplot(1, 2, 1)
plt.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean, color='r', linestyle='--', label=f'均值={mean:.2f}')
plt.axvline(mean + threshold * std, color='g', linestyle='--', label=f'阈值={mean + threshold * std:.2f}')
plt.axvline(mean - threshold * std, color='g', linestyle='--')
plt.xlabel('值', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('数据分布', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：Z-score
plt.subplot(1, 2, 2)
plt.scatter(range(len(data)), z_scores, c=anomalies, cmap='RdYlGn', alpha=0.6, s=20)
plt.axhline(threshold, color='r', linestyle='--', label=f'阈值={threshold}')
plt.xlabel('样本索引', fontsize=12)
plt.ylabel('Z-score', fontsize=12)
plt.title('Z-score异常检测', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('zscore_anomaly.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 2. IQR方法 ==========
print("\n" + "=" * 60)
print("2. IQR方法（四分位距）")
print("=" * 60)
print("""
IQR方法：
- 使用四分位距识别异常
- 异常值：< Q1 - 1.5*IQR 或 > Q3 + 1.5*IQR
- 不假设数据分布
""")

# 计算四分位数
q1 = np.percentile(data, 25)
q3 = np.percentile(data, 75)
iqr = q3 - q1

# 计算异常边界
lower_bound = q1 - 1.5 * iqr
upper_bound = q3 + 1.5 * iqr

# 识别异常
anomalies_iqr = (data < lower_bound) | (data > upper_bound)

print(f"\nIQR统计:")
print(f"  Q1 (25%分位数): {q1:.4f}")
print(f"  Q3 (75%分位数): {q3:.4f}")
print(f"  IQR: {iqr:.4f}")
print(f"  下界: {lower_bound:.4f}")
print(f"  上界: {upper_bound:.4f}")
print(f"  识别出的异常数: {np.sum(anomalies_iqr)}")

# 可视化
plt.figure(figsize=(12, 6))

# 左图：箱线图
plt.subplot(1, 2, 1)
plt.boxplot(data, vert=True)
plt.ylabel('值', fontsize=12)
plt.title('箱线图（显示异常值）', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 右图：数据分布和边界
plt.subplot(1, 2, 2)
plt.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(lower_bound, color='r', linestyle='--', label=f'下界={lower_bound:.2f}')
plt.axvline(upper_bound, color='r', linestyle='--', label=f'上界={upper_bound:.2f}')
plt.xlabel('值', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('IQR异常检测', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iqr_anomaly.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 3σ原则 ==========
print("\n" + "=" * 60)
print("3. 3σ原则")
print("=" * 60)
print("""
3σ原则：
- 超过3个标准差的数据视为异常
- 假设数据遵循正态分布
- 约99.7%的数据在3σ范围内
""")

# 计算3σ边界
mean_3sigma = np.mean(data)
std_3sigma = np.std(data)
lower_3sigma = mean_3sigma - 3 * std_3sigma
upper_3sigma = mean_3sigma + 3 * std_3sigma

# 识别异常
anomalies_3sigma = (data < lower_3sigma) | (data > upper_3sigma)

print(f"\n3σ统计:")
print(f"  均值: {mean_3sigma:.4f}")
print(f"  标准差: {std_3sigma:.4f}")
print(f"  下界: {lower_3sigma:.4f}")
print(f"  上界: {upper_3sigma:.4f}")
print(f"  识别出的异常数: {np.sum(anomalies_3sigma)}")

# 可视化
plt.figure(figsize=(12, 6))

# 左图：数据分布和3σ边界
plt.subplot(1, 2, 1)
plt.hist(data, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
plt.axvline(mean_3sigma, color='g', linestyle='-', label=f'均值={mean_3sigma:.2f}')
plt.axvline(lower_3sigma, color='r', linestyle='--', label=f'下界={lower_3sigma:.2f}')
plt.axvline(upper_3sigma, color='r', linestyle='--', label=f'上界={upper_3sigma:.2f}')
plt.xlabel('值', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('3σ原则异常检测', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：对比三种方法
plt.subplot(1, 2, 2)
methods = ['Z-score', 'IQR', '3σ']
anomaly_counts = [np.sum(anomalies), np.sum(anomalies_iqr), np.sum(anomalies_3sigma)]
colors = ['skyblue', 'lightgreen', 'lightcoral']

bars = plt.bar(methods, anomaly_counts, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('异常数', fontsize=12)
plt.title('三种方法对比', fontsize=14)
plt.grid(True, axis='y', alpha=0.3)

# 添加数值标签
for bar, count in zip(bars, anomaly_counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
             f'{count}', ha='center', va='bottom', fontsize=12)

plt.tight_layout()
plt.savefig('3sigma_anomaly.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. Z-score方法：假设正态分布，计算简单
2. IQR方法：不假设分布，更鲁棒
3. 3σ原则：假设正态分布，约99.7%的数据在范围内
4. 不同方法适用于不同的数据分布
5. 需要根据数据特点选择合适的方法
""")


