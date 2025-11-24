"""
使用scikit-learn实现线性回归
包含基本线性回归、Ridge回归、Lasso回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.datasets import make_regression


def generate_data(n_samples=200, n_features=1, noise=10, random_state=42):
    """生成回归数据"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        noise=noise,
        random_state=random_state
    )
    return X, y


def compare_models(X_train, X_test, y_train, y_test):
    """对比不同模型"""
    models = {
        '线性回归': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1),
        'Elastic Net': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    print("=" * 70)
    print(f"{'模型':<20} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
    print("-" * 70)
    
    for name, model in models.items():
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred = model.predict(X_test)
        
        # 评估
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': model,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'y_pred': y_pred
        }
        
        print(f"{name:<20} {mse:<15.4f} {rmse:<15.4f} {mae:<15.4f} {r2:<15.4f}")
    
    return results


def visualize_comparison(X_test, y_test, results):
    """可视化对比结果"""
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # 散点图：预测 vs 真实
        ax.scatter(y_test, result['y_pred'], alpha=0.6)
        
        # 理想线（y=x）
        min_val = min(y_test.min(), result['y_pred'].min())
        max_val = max(y_test.max(), result['y_pred'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
        
        ax.set_xlabel('真实值')
        ax.set_ylabel('预测值')
        ax.set_title(f'{name}\nR² = {result["r2"]:.4f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def cross_validation_example(X, y):
    """交叉验证示例"""
    print("\n" + "=" * 70)
    print("交叉验证（5折）")
    print("=" * 70)
    
    models = {
        '线性回归': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1)
    }
    
    print(f"{'模型':<20} {'平均R²':<15} {'标准差':<15}")
    print("-" * 50)
    
    for name, model in models.items():
        scores = cross_val_score(model, X, y, cv=5, scoring='r2')
        print(f"{name:<20} {scores.mean():<15.4f} {scores.std():<15.4f}")


def regularization_effect(X_train, X_test, y_train, y_test):
    """展示正则化效果"""
    print("\n" + "=" * 70)
    print("正则化参数的影响（Ridge回归）")
    print("=" * 70)
    
    alphas = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    print(f"{'Alpha':<15} {'MSE':<15} {'R²':<15} {'权重L2范数':<15}")
    print("-" * 60)
    
    mse_list = []
    r2_list = []
    weight_norm_list = []
    
    for alpha in alphas:
        model = Ridge(alpha=alpha)
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        weight_norm = np.linalg.norm(model.coef_)
        
        mse_list.append(mse)
        r2_list.append(r2)
        weight_norm_list.append(weight_norm)
        
        print(f"{alpha:<15.3f} {mse:<15.4f} {r2:<15.4f} {weight_norm:<15.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].semilogx(alphas, mse_list, marker='o')
    axes[0].set_xlabel('Alpha (正则化参数)')
    axes[0].set_ylabel('MSE')
    axes[0].set_title('MSE vs Alpha')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].semilogx(alphas, weight_norm_list, marker='o', color='orange')
    axes[1].set_xlabel('Alpha (正则化参数)')
    axes[1].set_ylabel('权重L2范数')
    axes[1].set_title('权重L2范数 vs Alpha')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("使用scikit-learn实现线性回归")
    print("=" * 70)
    
    # 生成数据
    X, y = generate_data(n_samples=200, n_features=5, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化（对正则化模型很重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征维度: {X_train.shape[1]}")
    
    # 对比不同模型
    results = compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 可视化对比
    visualize_comparison(y_test, y_test, results)
    
    # 交叉验证
    cross_validation_example(X_train_scaled, y_train)
    
    # 正则化效果
    regularization_effect(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 显示系数
    print("\n" + "=" * 70)
    print("模型系数对比")
    print("=" * 70)
    print(f"{'特征':<10} {'线性回归':<15} {'Ridge':<15} {'Lasso':<15}")
    print("-" * 55)
    for i in range(len(results['线性回归']['model'].coef_)):
        print(f"{f'特征{i+1}':<10} "
              f"{results['线性回归']['model'].coef_[i]:<15.4f} "
              f"{results['Ridge回归']['model'].coef_[i]:<15.4f} "
              f"{results['Lasso回归']['model'].coef_[i]:<15.4f}")


if __name__ == "__main__":
    main()

