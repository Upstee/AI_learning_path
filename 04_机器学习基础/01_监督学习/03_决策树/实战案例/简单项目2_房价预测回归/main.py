"""
简单项目2：房价预测回归
使用决策树进行房价预测
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print("=" * 50)
print("房价预测 - 决策树回归")
print("=" * 50)

# 数据探索
print(f"\n数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")
print(f"目标变量均值: {y.mean():.2f}")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 创建决策树回归器
dt = DecisionTreeRegressor(
    max_depth=10,
    min_samples_split=20,
    min_samples_leaf=10,
    random_state=42
)

# 训练模型
print("\n训练决策树回归器...")
dt.fit(X_train, y_train)

# 预测
y_pred_train = dt.predict(X_train)
y_pred_test = dt.predict(X_test)

# 评估性能
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)
test_mae = mean_absolute_error(y_test, y_pred_test)

print(f"\n模型性能:")
print(f"训练集 RMSE: {train_rmse:.4f}")
print(f"测试集 RMSE: {test_rmse:.4f}")
print(f"训练集 R²: {train_r2:.4f}")
print(f"测试集 R²: {test_r2:.4f}")
print(f"测试集 MAE: {test_mae:.4f}")

# 特征重要性
print(f"\n特征重要性:")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': dt.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# 可视化预测结果
plt.figure(figsize=(12, 5))

# 预测vs真实值
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title('预测 vs 真实值')
plt.grid(True, alpha=0.3)

# 残差图
plt.subplot(1, 2, 2)
residuals = y_test - y_pred_test
plt.scatter(y_pred_test, residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--', lw=2)
plt.xlabel('预测值')
plt.ylabel('残差')
plt.title('残差图')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('housing_prediction.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

