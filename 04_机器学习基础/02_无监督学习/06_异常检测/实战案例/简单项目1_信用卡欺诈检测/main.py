"""
简单项目1：信用卡欺诈检测
使用异常检测方法检测信用卡交易中的欺诈行为

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成模拟信用卡交易数据
# 在实际应用中，你需要从数据库或文件中加载真实的交易数据
np.random.seed(42)

# 正常交易特征：金额、时间、商户类型等
n_normal = 1000
normal_amount = np.random.normal(100, 30, n_normal)  # 正常交易金额
normal_time = np.random.uniform(0, 24, n_normal)  # 交易时间（小时）
normal_merchant = np.random.randint(0, 10, n_normal)  # 商户类型

# 异常交易特征：金额异常大、时间异常、商户类型异常
n_anomaly = 50
anomaly_amount = np.random.normal(500, 100, n_anomaly)  # 异常交易金额（更大）
anomaly_time = np.random.uniform(0, 24, n_anomaly)
anomaly_merchant = np.random.randint(0, 10, n_anomaly)

# 合并数据
X_normal = np.column_stack([normal_amount, normal_time, normal_merchant])
X_anomaly = np.column_stack([anomaly_amount, anomaly_time, anomaly_merchant])
X = np.vstack([X_normal, X_anomaly])
y_true = np.hstack([np.zeros(n_normal), np.ones(n_anomaly)])

print(f"数据信息:")
print(f"  正常交易数: {n_normal}")
print(f"  异常交易数: {n_anomaly}")
print(f"  总交易数: {len(X)}")
print(f"  特征数: {X.shape[1]}")

# 标准化数据
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\n数据标准化完成！")

# ========== 2. 异常检测 ==========
print("\n" + "=" * 60)
print("2. 异常检测")
print("=" * 60)

# 使用Isolation Forest检测异常
# contamination: 异常样本的比例（估计值）
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred = iso_forest.fit_predict(X_scaled)

# Isolation Forest返回-1表示异常，1表示正常
y_pred = (y_pred == -1).astype(int)

print(f"识别出的异常交易数: {np.sum(y_pred)}")

# ========== 3. 性能评估 ==========
print("\n" + "=" * 60)
print("3. 性能评估")
print("=" * 60)

print("\n分类报告:")
print(classification_report(y_true, y_pred, target_names=['正常', '异常']))

print("\n混淆矩阵:")
cm = confusion_matrix(y_true, y_pred)
print(cm)

# ========== 4. 可视化结果 ==========
print("\n" + "=" * 60)
print("4. 可视化结果")
print("=" * 60)

# 可视化交易金额分布
plt.figure(figsize=(14, 6))

# 左图：交易金额分布
plt.subplot(1, 2, 1)
normal_amounts = X[y_true == 0, 0]
anomaly_amounts = X[y_true == 1, 0]
plt.hist(normal_amounts, bins=30, alpha=0.7, color='blue', label='正常交易', edgecolor='black')
plt.hist(anomaly_amounts, bins=30, alpha=0.7, color='red', label='异常交易', edgecolor='black')
plt.xlabel('交易金额', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('交易金额分布', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：检测结果
plt.subplot(1, 2, 2)
normal_detected = X[y_pred == 0, 0]
anomaly_detected = X[y_pred == 1, 0]
plt.hist(normal_detected, bins=30, alpha=0.7, color='blue', label='检测为正常', edgecolor='black')
plt.hist(anomaly_detected, bins=30, alpha=0.7, color='red', label='检测为异常', edgecolor='black')
plt.xlabel('交易金额', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('检测结果', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('fraud_detection.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 分析异常特征 ==========
print("\n" + "=" * 60)
print("5. 分析异常特征")
print("=" * 60)

if np.sum(y_pred) > 0:
    detected_anomalies = X[y_pred == 1]
    print(f"\n检测出的异常交易特征:")
    print(f"  异常交易数: {len(detected_anomalies)}")
    print(f"  平均金额: {detected_anomalies[:, 0].mean():.2f}")
    print(f"  金额范围: [{detected_anomalies[:, 0].min():.2f}, {detected_anomalies[:, 0].max():.2f}]")
    print(f"  平均时间: {detected_anomalies[:, 1].mean():.2f}小时")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. 异常检测可以用于信用卡欺诈检测
2. Isolation Forest是有效的异常检测方法
3. 需要根据业务需求调整参数
4. 可以分析异常交易的特征，理解欺诈模式
""")


