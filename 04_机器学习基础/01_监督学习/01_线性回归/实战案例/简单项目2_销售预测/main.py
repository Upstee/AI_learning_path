"""
简单项目2：销售预测
使用多特征线性回归预测销售额
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n_samples=150, random_state=42):
    """生成模拟销售数据"""
    np.random.seed(random_state)
    
    # 特征
    ad_spend = np.random.normal(10, 3, n_samples)  # 广告投入（万元）
    promotion = np.random.choice([0, 1], n_samples)  # 促销活动（0/1）
    season = np.random.choice([1, 2, 3, 4], n_samples)  # 季节（1-4）
    
    # 目标：销售额 = 广告*2 + 促销*5 + 季节因子 + 噪声
    sales = (ad_spend * 2 + 
             promotion * 5 + 
             season * 1.5 + 
             np.random.normal(0, 3, n_samples))
    
    # 确保销售额为正值
    sales = np.maximum(sales, 0)
    
    # 组合特征
    X = np.column_stack([ad_spend, promotion, season])
    
    return X, sales


def main():
    """主函数"""
    print("=" * 60)
    print("简单项目2：销售预测")
    print("=" * 60)
    
    # 1. 生成数据
    X, y = generate_data(n_samples=150)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征: 广告投入、促销活动、季节")
    
    # 2. 训练模型
    print("\n训练模型...")
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)
    
    # 3. 预测
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # 4. 评估
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n模型评估:")
    print(f"  训练集MSE: {train_mse:.2f}")
    print(f"  测试集MSE: {test_mse:.2f}")
    print(f"  训练集R²: {train_r2:.4f}")
    print(f"  测试集R²: {test_r2:.4f}")
    
    print(f"\n特征重要性（系数）:")
    feature_names = ['广告投入', '促销活动', '季节']
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.4f}")
    print(f"  截距: {model.intercept_:.4f}")
    
    # 5. 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 预测 vs 真实
    axes[0].scatter(y_test, y_pred_test, alpha=0.6)
    min_val = min(y_test.min(), y_pred_test.min())
    max_val = max(y_test.max(), y_pred_test.max())
    axes[0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0].set_xlabel('真实销售额')
    axes[0].set_ylabel('预测销售额')
    axes[0].set_title('预测 vs 真实')
    axes[0].grid(True, alpha=0.3)
    
    # 特征重要性
    axes[1].barh(feature_names, model.coef_)
    axes[1].set_xlabel('系数值')
    axes[1].set_title('特征重要性')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('销售预测结果.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 项目完成！")


if __name__ == "__main__":
    main()

