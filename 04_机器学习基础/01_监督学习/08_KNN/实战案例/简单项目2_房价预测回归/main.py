"""
简单项目2：房价预测回归
使用KNN回归器预测房价

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.datasets import make_regression

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 生成回归数据（模拟房价数据）
# 在实际应用中，你需要从文件或数据库中加载真实数据
# 这里我们使用模拟数据来演示

# make_regression 生成回归数据集
# n_samples: 样本数
# n_features: 特征数（房屋特征，如面积、房间数等）
# noise: 噪声水平（模拟数据中的随机性）
X, y = make_regression(
    n_samples=200,
    n_features=3,  # 3个特征：面积、房间数、位置
    noise=10,      # 噪声水平
    random_state=42
)

# 将目标值转换为正数（房价应该是正数）
y = y - y.min() + 100  # 调整到合理范围

print(f"数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  房价范围: {y.min():.2f} - {y.max():.2f}")

# ========== 2. 数据预处理 ==========
print("\n" + "=" * 60)
print("2. 数据预处理")
print("=" * 60)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 特征标准化
# KNN对特征尺度敏感，需要标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n标准化完成！")

# ========== 3. 训练模型 ==========
print("\n" + "=" * 60)
print("3. 训练模型")
print("=" * 60)

# 创建KNN回归器
# n_neighbors=5: 选择最近的5个邻居
# weights='distance': 距离越近，权重越大
knn_regressor = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance',
    metric='euclidean'
)

# 训练模型
print("训练模型...")
knn_regressor.fit(X_train_scaled, y_train)
print("训练完成！")

# ========== 4. 预测和评估 ==========
print("\n" + "=" * 60)
print("4. 预测和评估")
print("=" * 60)

# 预测测试集
y_pred = knn_regressor.predict(X_test_scaled)

# 计算评估指标
# R²分数：衡量模型解释的方差比例，越接近1越好
r2 = r2_score(y_test, y_pred)

# MSE（均方误差）：预测误差的平方的平均值
mse = mean_squared_error(y_test, y_pred)

# MAE（平均绝对误差）：预测误差的绝对值的平均值
mae = mean_absolute_error(y_test, y_pred)

# RMSE（均方根误差）：MSE的平方根，与目标值同单位
rmse = np.sqrt(mse)

print(f"R²分数: {r2:.4f}")
print(f"MSE: {mse:.4f}")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")

# ========== 5. 可视化结果 ==========
print("\n" + "=" * 60)
print("5. 可视化结果")
print("=" * 60)

# 可视化预测结果
plt.figure(figsize=(12, 5))

# 左图：真实值vs预测值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.6, s=50)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 
         'r--', linewidth=2, label='完美预测线')
plt.xlabel('真实房价', fontsize=12)
plt.ylabel('预测房价', fontsize=12)
plt.title('真实值 vs 预测值', fontsize=14)
plt.legend()
plt.grid(True, alpha=0.3)

# 右图：误差分布
plt.subplot(1, 2, 2)
errors = y_test - y_pred
plt.hist(errors, bins=20, alpha=0.7, edgecolor='black')
plt.xlabel('预测误差', fontsize=12)
plt.ylabel('频数', fontsize=12)
plt.title('误差分布', fontsize=14)
plt.axvline(x=0, color='r', linestyle='--', linewidth=2, label='零误差线')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('knn_regression_results.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 测试不同K值 ==========
print("\n" + "=" * 60)
print("6. 测试不同K值")
print("=" * 60)

# 测试不同的K值
k_values = range(1, 21)
r2_scores = []
rmse_scores = []

print("\n测试不同K值:")
for k in k_values:
    knn_k = KNeighborsRegressor(n_neighbors=k, weights='distance')
    knn_k.fit(X_train_scaled, y_train)
    y_pred_k = knn_k.predict(X_test_scaled)
    r2_k = r2_score(y_test, y_pred_k)
    rmse_k = np.sqrt(mean_squared_error(y_test, y_pred_k))
    r2_scores.append(r2_k)
    rmse_scores.append(rmse_k)
    if k <= 5 or k % 5 == 0:
        print(f"  K={k}: R²={r2_k:.4f}, RMSE={rmse_k:.4f}")

# 找到最佳K值
best_k = k_values[np.argmax(r2_scores)]
best_r2 = max(r2_scores)
print(f"\n最佳K值: {best_k}, R²: {best_r2:.4f}")

# 可视化K值对性能的影响
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 左图：R²分数
axes[0].plot(k_values, r2_scores, 'o-', linewidth=2, markersize=8)
axes[0].set_xlabel('K值', fontsize=12)
axes[0].set_ylabel('R²分数', fontsize=12)
axes[0].set_title('K值对R²分数的影响', fontsize=14)
axes[0].grid(True, alpha=0.3)
axes[0].axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'最佳K={best_k}')
axes[0].legend()

# 右图：RMSE
axes[1].plot(k_values, rmse_scores, 'o-', linewidth=2, markersize=8, color='green')
axes[1].set_xlabel('K值', fontsize=12)
axes[1].set_ylabel('RMSE', fontsize=12)
axes[1].set_title('K值对RMSE的影响', fontsize=14)
axes[1].grid(True, alpha=0.3)
axes[1].axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'最佳K={best_k}')
axes[1].legend()

plt.tight_layout()
plt.savefig('knn_regression_k_values.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 7. 测试新样本 ==========
print("\n" + "=" * 60)
print("7. 测试新样本")
print("=" * 60)

# 使用最佳K值重新训练
knn_best = KNeighborsRegressor(n_neighbors=best_k, weights='distance')
knn_best.fit(X_train_scaled, y_train)

# 新样本
new_samples = np.array([
    [0.5, 0.3, 0.2],  # 新房屋1
    [1.2, 0.8, 0.5],  # 新房屋2
    [0.8, 0.6, 0.4],  # 新房屋3
])

# 标准化新样本
new_samples_scaled = scaler.transform(new_samples)

# 预测
predictions = knn_best.predict(new_samples_scaled)

print("\n预测结果:")
for i, sample in enumerate(new_samples):
    print(f"  房屋 {i+1}: 特征={sample}, 预测房价={predictions[i]:.2f}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. KNN也可以用于回归任务
2. 回归时使用K个邻居的平均值（或加权平均）
3. 特征标准化对KNN回归同样重要
4. 需要选择合适的K值以获得最佳性能
""")

