"""
基础练习3答案：正则化对比
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def main():
    """主函数"""
    print("=" * 60)
    print("基础练习3：正则化对比")
    print("=" * 60)
    
    # 1. 生成数据（只有5个特征真正有用）
    X, y = make_regression(
        n_samples=200, n_features=10, n_informative=5, noise=10, random_state=42
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n数据信息:")
    print(f"  特征数量: {X.shape[1]}")
    print(f"  真正有用的特征: 5个")
    
    # 2. 对比无正则化、Ridge、Lasso
    print("\n" + "-" * 60)
    print("2. 模型对比")
    print("-" * 60)
    
    from sklearn.linear_model import LinearRegression
    
    models = {
        '无正则化': LinearRegression(),
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1)
    }
    
    print(f"{'模型':<20} {'测试MSE':<15} {'测试R²':<15} {'非零系数':<15}")
    print("-" * 65)
    
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        print(f"{name:<20} {mse:<15.4f} {r2:<15.4f} {n_nonzero:<15}")
    
    # 3. 正则化路径
    print("\n" + "-" * 60)
    print("3. 正则化路径分析")
    print("-" * 60)
    
    alphas = np.logspace(-3, 2, 50)
    
    ridge_coefs = []
    lasso_coefs = []
    ridge_test_mse = []
    lasso_test_mse = []
    
    for alpha in alphas:
        # Ridge
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train_scaled, y_train)
        ridge_coefs.append(ridge.coef_)
        y_pred = ridge.predict(X_test_scaled)
        ridge_test_mse.append(mean_squared_error(y_test, y_pred))
        
        # Lasso
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        lasso_coefs.append(lasso.coef_)
        y_pred = lasso.predict(X_test_scaled)
        lasso_test_mse.append(mean_squared_error(y_test, y_pred))
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # 可视化正则化路径
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Ridge系数路径
    ax = axes[0, 0]
    for i in range(ridge_coefs.shape[1]):
        ax.plot(alphas, ridge_coefs[:, i], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('系数值')
    ax.set_title('Ridge回归 - 系数路径（所有系数都缩小但不为0）')
    ax.grid(True, alpha=0.3)
    
    # Lasso系数路径
    ax = axes[0, 1]
    for i in range(lasso_coefs.shape[1]):
        ax.plot(alphas, lasso_coefs[:, i], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('系数值')
    ax.set_title('Lasso回归 - 系数路径（部分系数变为0）')
    ax.grid(True, alpha=0.3)
    
    # 测试MSE vs Alpha
    ax = axes[1, 0]
    ax.plot(alphas, ridge_test_mse, label='Ridge', linewidth=2)
    ax.plot(alphas, lasso_test_mse, label='Lasso', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('测试集MSE')
    ax.set_title('测试集MSE vs Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 非零系数数量 vs Alpha (Lasso)
    ax = axes[1, 1]
    n_nonzero = [np.sum(np.abs(coef) > 1e-5) for coef in lasso_coefs]
    ax.plot(alphas, n_nonzero, linewidth=2, color='orange')
    ax.set_xscale('log')
    ax.set_xlabel('Alpha')
    ax.set_ylabel('非零系数数量')
    ax.set_title('Lasso - 非零系数数量 vs Alpha')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('正则化路径.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 找出最佳alpha
    best_ridge_alpha = alphas[np.argmin(ridge_test_mse)]
    best_lasso_alpha = alphas[np.argmin(lasso_test_mse)]
    print(f"\n最佳Ridge alpha: {best_ridge_alpha:.4f}")
    print(f"最佳Lasso alpha: {best_lasso_alpha:.4f}")
    
    # 4. 超参数调优
    print("\n" + "-" * 60)
    print("4. 超参数调优（GridSearchCV）")
    print("-" * 60)
    
    # Ridge调优
    ridge_param_grid = {'alpha': np.logspace(-2, 2, 20)}
    ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
    ridge_grid.fit(X_train_scaled, y_train)
    
    print(f"最佳Ridge参数: alpha={ridge_grid.best_params_['alpha']:.4f}")
    print(f"交叉验证R²: {ridge_grid.best_score_:.4f}")
    
    # Lasso调优
    lasso_param_grid = {'alpha': np.logspace(-3, 1, 20)}
    lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_param_grid, cv=5, scoring='r2')
    lasso_grid.fit(X_train_scaled, y_train)
    
    print(f"最佳Lasso参数: alpha={lasso_grid.best_params_['alpha']:.4f}")
    print(f"交叉验证R²: {lasso_grid.best_score_:.4f}")
    
    # 使用最佳参数评估
    best_ridge = ridge_grid.best_estimator_
    best_lasso = lasso_grid.best_estimator_
    
    y_pred_ridge = best_ridge.predict(X_test_scaled)
    y_pred_lasso = best_lasso.predict(X_test_scaled)
    
    print(f"\n最佳模型测试集性能:")
    print(f"  Ridge: MSE={mean_squared_error(y_test, y_pred_ridge):.4f}, "
          f"R²={r2_score(y_test, y_pred_ridge):.4f}")
    print(f"  Lasso: MSE={mean_squared_error(y_test, y_pred_lasso):.4f}, "
          f"R²={r2_score(y_test, y_pred_lasso):.4f}")
    
    print("\n✅ 练习完成！")


if __name__ == "__main__":
    main()

