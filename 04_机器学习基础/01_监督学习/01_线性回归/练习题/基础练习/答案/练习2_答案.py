"""
基础练习2答案：使用scikit-learn实现线性回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score


def main():
    """主函数"""
    print("=" * 60)
    print("基础练习2：使用scikit-learn实现线性回归")
    print("=" * 60)
    
    # 1. 生成数据
    X, y = make_regression(n_samples=100, n_features=1, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化（对正则化很重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. 基本线性回归
    print("\n" + "-" * 60)
    print("2. 基本线性回归")
    print("-" * 60)
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    
    y_pred_train = lr.predict(X_train_scaled)
    y_pred_test = lr.predict(X_test_scaled)
    
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"训练集MSE: {train_mse:.4f}")
    print(f"测试集MSE: {test_mse:.4f}")
    print(f"训练集R²: {train_r2:.4f}")
    print(f"测试集R²: {test_r2:.4f}")
    print(f"系数: {lr.coef_[0]:.4f}")
    print(f"截距: {lr.intercept_:.4f}")
    
    # 3. Ridge回归
    print("\n" + "-" * 60)
    print("3. Ridge回归（不同alpha值）")
    print("-" * 60)
    alphas = [0.1, 1.0, 10.0]
    
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        y_pred_test = ridge.predict(X_test_scaled)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        print(f"Alpha={alpha}: 测试MSE={test_mse:.4f}, R²={test_r2:.4f}, 系数={ridge.coef_[0]:.4f}")
    
    # 4. Lasso回归（多特征示例）
    print("\n" + "-" * 60)
    print("4. Lasso回归（特征选择）")
    print("-" * 60)
    # 生成多特征数据
    X_multi, y_multi = make_regression(
        n_samples=100, n_features=5, n_informative=3, noise=10, random_state=42
    )
    X_train_multi, X_test_multi, y_train_multi, y_test_multi = train_test_split(
        X_multi, y_multi, test_size=0.2, random_state=42
    )
    
    scaler_multi = StandardScaler()
    X_train_multi_scaled = scaler_multi.fit_transform(X_train_multi)
    X_test_multi_scaled = scaler_multi.transform(X_test_multi)
    
    lasso = Lasso(alpha=0.1)
    lasso.fit(X_train_multi_scaled, y_train_multi)
    
    print("Lasso系数:")
    for i, coef in enumerate(lasso.coef_):
        status = "非零" if abs(coef) > 1e-5 else "为零（被选择掉）"
        print(f"  特征{i}: {coef:.4f} ({status})")
    
    # 5. 模型对比
    print("\n" + "-" * 60)
    print("5. 模型对比")
    print("-" * 60)
    
    models = {
        '线性回归': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1)
    }
    
    print(f"{'模型':<20} {'测试MSE':<15} {'测试R²':<15}")
    print("-" * 50)
    
    for name, model in models.items():
        if name == 'Lasso (α=0.1)':
            model.fit(X_train_multi_scaled, y_train_multi)
            y_pred = model.predict(X_test_multi_scaled)
            mse = mean_squared_error(y_test_multi, y_pred)
            r2 = r2_score(y_test_multi, y_pred)
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            mse = mean_squared_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
        
        print(f"{name:<20} {mse:<15.4f} {r2:<15.4f}")
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.5, label='训练数据')
    plt.scatter(X_test, y_test, alpha=0.5, label='测试数据', marker='x')
    
    X_line = np.linspace(X_train.min(), X_train.max(), 100).reshape(-1, 1)
    X_line_scaled = scaler.transform(X_line)
    y_line = lr.predict(X_line_scaled)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label='拟合直线')
    
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()
    
    print("\n✅ 练习完成！")


if __name__ == "__main__":
    main()

