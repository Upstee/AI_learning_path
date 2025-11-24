"""
多项式回归示例
展示如何使用多项式特征处理非线性关系
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def generate_nonlinear_data(n_samples=100, noise=0.1, random_state=42):
    """生成非线性数据"""
    np.random.seed(random_state)
    X = np.linspace(-3, 3, n_samples).reshape(-1, 1)
    # 二次关系：y = x^2 + noise
    y = X.flatten() ** 2 + np.random.randn(n_samples) * noise
    return X, y


def compare_polynomial_degrees(X_train, X_test, y_train, y_test):
    """对比不同多项式次数"""
    print("=" * 70)
    print("多项式次数对比")
    print("=" * 70)
    
    degrees = [1, 2, 3, 5, 10, 20]
    results = {}
    
    print(f"\n{'次数':<10} {'训练MSE':<15} {'测试MSE':<15} {'R²':<15}")
    print("-" * 55)
    
    for degree in degrees:
        # 创建多项式特征
        poly_features = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_poly = poly_features.fit_transform(X_train)
        X_test_poly = poly_features.transform(X_test)
        
        # 训练模型
        model = LinearRegression()
        model.fit(X_train_poly, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train_poly)
        y_pred_test = model.predict(X_test_poly)
        
        # 评估
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        results[degree] = {
            'model': model,
            'poly_features': poly_features,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'y_pred_test': y_pred_test
        }
        
        print(f"{degree:<10} {train_mse:<15.4f} {test_mse:<15.4f} {test_r2:<15.4f}")
    
    return results


def visualize_polynomial_fits(X_train, y_train, X_test, y_test, results):
    """可视化不同次数的拟合结果"""
    n_degrees = len(results)
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    
    for idx, (degree, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # 数据点
        ax.scatter(X_train, y_train, alpha=0.5, label='训练数据', s=20)
        ax.scatter(X_test, y_test, alpha=0.5, label='测试数据', s=20, marker='x')
        
        # 拟合曲线
        X_plot_poly = result['poly_features'].transform(X_plot)
        y_plot = result['model'].predict(X_plot_poly)
        ax.plot(X_plot, y_plot, 'r-', linewidth=2, label=f'拟合曲线 (degree={degree})')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(f'次数={degree}, R²={result["test_r2"]:.4f}, 测试MSE={result["test_mse"]:.4f}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('多项式回归对比.png', dpi=150, bbox_inches='tight')
    plt.show()


def overfitting_demonstration(X_train, y_train, X_test, y_test):
    """展示过拟合现象"""
    print("\n" + "=" * 70)
    print("过拟合演示")
    print("=" * 70)
    
    # 使用高次多项式（容易过拟合）
    degree = 15
    poly_features = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly_features.fit_transform(X_train)
    X_test_poly = poly_features.transform(X_test)
    
    # 普通线性回归（可能过拟合）
    model_no_reg = LinearRegression()
    model_no_reg.fit(X_train_poly, y_train)
    
    # Ridge回归（防止过拟合）
    model_ridge = Ridge(alpha=0.1)
    model_ridge.fit(X_train_poly, y_train)
    
    # 评估
    y_pred_train_no_reg = model_no_reg.predict(X_train_poly)
    y_pred_test_no_reg = model_no_reg.predict(X_test_poly)
    y_pred_train_ridge = model_ridge.predict(X_train_poly)
    y_pred_test_ridge = model_ridge.predict(X_test_poly)
    
    train_mse_no_reg = mean_squared_error(y_train, y_pred_train_no_reg)
    test_mse_no_reg = mean_squared_error(y_test, y_pred_test_no_reg)
    train_mse_ridge = mean_squared_error(y_train, y_pred_train_ridge)
    test_mse_ridge = mean_squared_error(y_test, y_pred_test_ridge)
    
    print(f"\n高次多项式 (degree={degree}):")
    print(f"{'模型':<20} {'训练MSE':<15} {'测试MSE':<15} {'差距':<15}")
    print("-" * 65)
    print(f"{'无正则化':<20} {train_mse_no_reg:<15.4f} {test_mse_no_reg:<15.4f} "
          f"{test_mse_no_reg - train_mse_no_reg:<15.4f}")
    print(f"{'Ridge回归':<20} {train_mse_ridge:<15.4f} {test_mse_ridge:<15.4f} "
          f"{test_mse_ridge - train_mse_ridge:<15.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    X_plot = np.linspace(X_train.min(), X_train.max(), 200).reshape(-1, 1)
    X_plot_poly = poly_features.transform(X_plot)
    
    for idx, (model, title) in enumerate([(model_no_reg, '无正则化（过拟合）'),
                                          (model_ridge, 'Ridge回归（防止过拟合）')]):
        ax = axes[idx]
        
        ax.scatter(X_train, y_train, alpha=0.5, label='训练数据', s=20)
        ax.scatter(X_test, y_test, alpha=0.5, label='测试数据', s=20, marker='x')
        
        y_plot = model.predict(X_plot_poly)
        ax.plot(X_plot, y_plot, 'r-', linewidth=2, label='拟合曲线')
        
        ax.set_xlabel('X')
        ax.set_ylabel('y')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('过拟合演示.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("多项式回归示例")
    print("=" * 70)
    
    # 生成非线性数据
    X, y = generate_nonlinear_data(n_samples=100, noise=0.5)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  数据关系: y = x² + noise (非线性)")
    
    # 对比不同多项式次数
    results = compare_polynomial_degrees(X_train, X_test, y_train, y_test)
    
    # 可视化
    visualize_polynomial_fits(X_train, y_train, X_test, y_test, results)
    
    # 过拟合演示
    overfitting_demonstration(X_train, y_train, X_test, y_test)
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("1. 多项式回归可以处理非线性关系")
    print("2. 次数太低：欠拟合，无法捕捉非线性")
    print("3. 次数太高：过拟合，在训练集好但测试集差")
    print("4. 需要选择合适的次数，或使用正则化")
    print("5. 对于二次关系，degree=2通常是最佳选择")


if __name__ == "__main__":
    main()

