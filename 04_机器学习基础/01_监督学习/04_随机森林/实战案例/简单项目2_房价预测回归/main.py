"""
简单项目2：房价预测回归
使用随机森林进行房价预测
"""

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt

# 加载数据
housing = fetch_california_housing()
X, y = housing.data, housing.target
feature_names = housing.feature_names

print("=" * 50)
print("房价预测 - 随机森林 vs 回归树")
print("=" * 50)

# 数据探索
print(f"\n数据形状: {X.shape}")
print(f"特征名称: {feature_names}")
print(f"目标变量范围: [{y.min():.2f}, {y.max():.2f}]")

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 训练单棵回归树
print("\n训练单棵回归树...")
dt = DecisionTreeRegressor(max_depth=10, random_state=42)
dt.fit(X_train, y_train)
dt_pred = dt.predict(X_test)
dt_rmse = np.sqrt(mean_squared_error(y_test, dt_pred))
dt_r2 = r2_score(y_test, dt_pred)
print(f"回归树 RMSE: {dt_rmse:.4f}, R²: {dt_r2:.4f}")

# 训练随机森林
print("\n训练随机森林回归器...")
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_rmse = np.sqrt(mean_squared_error(y_test, rf_pred))
rf_r2 = r2_score(y_test, rf_pred)
print(f"随机森林 RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")

# 特征重要性
print(f"\n特征重要性（随机森林）:")
feature_importance = pd.DataFrame({
    'feature': feature_names,
    'importance': rf.feature_importances_
}).sort_values('importance', ascending=False)
print(feature_importance)

# 可视化预测结果
plt.figure(figsize=(15, 5))

# 预测vs真实值 - 回归树
plt.subplot(1, 3, 1)
plt.scatter(y_test, dt_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'回归树 (R²={dt_r2:.3f})')
plt.grid(True, alpha=0.3)

# 预测vs真实值 - 随机森林
plt.subplot(1, 3, 2)
plt.scatter(y_test, rf_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel('真实值')
plt.ylabel('预测值')
plt.title(f'随机森林 (R²={rf_r2:.3f})')
plt.grid(True, alpha=0.3)

# 特征重要性
plt.subplot(1, 3, 3)
plt.barh(feature_importance['feature'], feature_importance['importance'])
plt.xlabel('特征重要性')
plt.title('随机森林特征重要性')
plt.grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig('housing_prediction_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 性能对比
print(f"\n性能对比:")
print(f"  回归树 RMSE: {dt_rmse:.4f}, R²: {dt_r2:.4f}")
print(f"  随机森林 RMSE: {rf_rmse:.4f}, R²: {rf_r2:.4f}")
print(f"  RMSE改善: {dt_rmse - rf_rmse:.4f}")
print(f"  R²提升: {rf_r2 - dt_r2:.4f}")

print("\n" + "=" * 50)
print("项目完成！")
print("=" * 50)

