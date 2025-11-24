"""
正则化示例：Ridge、Lasso、Elastic Net
展示正则化对模型的影响
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import make_regression


def generate_data(n_samples=100, n_features=20, noise=10, random_state=42):
    """生成回归数据（特征多于样本，模拟过拟合场景）"""
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=5,  # 只有5个特征真正有用
        noise=noise,
        random_state=random_state
    )
    return X, y


def compare_regularization(X_train, X_test, y_train, y_test):
    """对比不同正则化方法"""
    print("=" * 70)
    print("正则化方法对比")
    print("=" * 70)
    
    models = {
        'Ridge (α=1.0)': Ridge(alpha=1.0),
        'Lasso (α=0.1)': Lasso(alpha=0.1),
        'Elastic Net (α=0.1, l1_ratio=0.5)': ElasticNet(alpha=0.1, l1_ratio=0.5)
    }
    
    results = {}
    
    print(f"\n{'模型':<35} {'训练MSE':<15} {'测试MSE':<15} {'R²':<15} {'非零系数':<15}")
    print("-" * 95)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        n_nonzero = np.sum(np.abs(model.coef_) > 1e-5)
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_r2': test_r2,
            'n_nonzero': n_nonzero,
            'coef': model.coef_
        }
        
        print(f"{name:<35} {train_mse:<15.4f} {test_mse:<15.4f} {test_r2:<15.4f} {n_nonzero:<15}")
    
    return results


def regularization_path(X_train, y_train, X_test, y_test):
    """绘制正则化路径（系数随alpha的变化）"""
    print("\n" + "=" * 70)
    print("正则化路径分析")
    print("=" * 70)
    
    alphas = np.logspace(-3, 2, 50)  # 从0.001到100
    
    # Ridge路径
    ridge_coefs = []
    ridge_test_mse = []
    for alpha in alphas:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        ridge_coefs.append(ridge.coef_)
        y_pred = ridge.predict(X_test)
        ridge_test_mse.append(mean_squared_error(y_test, y_pred))
    
    # Lasso路径
    lasso_coefs = []
    lasso_test_mse = []
    for alpha in alphas:
        lasso = Lasso(alpha=alpha, max_iter=10000)
        lasso.fit(X_train, y_train)
        lasso_coefs.append(lasso.coef_)
        y_pred = lasso.predict(X_test)
        lasso_test_mse.append(mean_squared_error(y_test, y_pred))
    
    ridge_coefs = np.array(ridge_coefs)
    lasso_coefs = np.array(lasso_coefs)
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Ridge系数路径
    ax = axes[0, 0]
    for i in range(ridge_coefs.shape[1]):
        ax.plot(alphas, ridge_coefs[:, i], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (正则化参数)')
    ax.set_ylabel('系数值')
    ax.set_title('Ridge回归 - 系数路径')
    ax.grid(True, alpha=0.3)
    
    # Lasso系数路径
    ax = axes[0, 1]
    for i in range(lasso_coefs.shape[1]):
        ax.plot(alphas, lasso_coefs[:, i], alpha=0.7)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (正则化参数)')
    ax.set_ylabel('系数值')
    ax.set_title('Lasso回归 - 系数路径（注意系数变为0）')
    ax.grid(True, alpha=0.3)
    
    # 测试MSE vs Alpha
    ax = axes[1, 0]
    ax.plot(alphas, ridge_test_mse, label='Ridge', linewidth=2)
    ax.plot(alphas, lasso_test_mse, label='Lasso', linewidth=2)
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (正则化参数)')
    ax.set_ylabel('测试集MSE')
    ax.set_title('测试集MSE vs Alpha')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 非零系数数量 vs Alpha (Lasso)
    ax = axes[1, 1]
    n_nonzero = [np.sum(np.abs(coef) > 1e-5) for coef in lasso_coefs]
    ax.plot(alphas, n_nonzero, linewidth=2, color='orange')
    ax.set_xscale('log')
    ax.set_xlabel('Alpha (正则化参数)')
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


def feature_selection_comparison(X_train, y_train, X_test, y_test):
    """对比特征选择效果"""
    print("\n" + "=" * 70)
    print("特征选择对比（Lasso vs Ridge）")
    print("=" * 70)
    
    # 使用最佳alpha
    ridge = Ridge(alpha=1.0)
    lasso = Lasso(alpha=0.1)
    
    ridge.fit(X_train, y_train)
    lasso.fit(X_train, y_train)
    
    # 找出重要特征（系数绝对值大的）
    ridge_importance = np.abs(ridge.coef_)
    lasso_importance = np.abs(lasso.coef_)
    
    # 排序
    ridge_top5 = np.argsort(ridge_importance)[-5:][::-1]
    lasso_top5 = np.argsort(lasso_importance)[-5:][::-1]
    
    print("\nRidge回归 - 前5个重要特征:")
    for i, idx in enumerate(ridge_top5):
        print(f"  特征{idx}: 系数={ridge.coef_[idx]:.4f}, 重要性={ridge_importance[idx]:.4f}")
    
    print("\nLasso回归 - 前5个重要特征（非零）:")
    lasso_nonzero = np.where(np.abs(lasso.coef_) > 1e-5)[0]
    if len(lasso_nonzero) > 0:
        lasso_sorted = sorted(lasso_nonzero, key=lambda x: np.abs(lasso.coef_[x]), reverse=True)
        for i, idx in enumerate(lasso_sorted[:5]):
            print(f"  特征{idx}: 系数={lasso.coef_[idx]:.4f}, 重要性={lasso_importance[idx]:.4f}")
    else:
        print("  所有特征系数都为0（alpha太大）")
    
    # 可视化系数对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].bar(range(len(ridge.coef_)), ridge.coef_)
    axes[0].set_xlabel('特征索引')
    axes[0].set_ylabel('系数值')
    axes[0].set_title('Ridge回归 - 所有特征系数')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    axes[1].bar(range(len(lasso.coef_)), lasso.coef_)
    axes[1].set_xlabel('特征索引')
    axes[1].set_ylabel('系数值')
    axes[1].set_title('Lasso回归 - 特征选择（部分系数为0）')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('特征选择对比.png', dpi=150, bbox_inches='tight')
    plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("正则化示例：Ridge、Lasso、Elastic Net")
    print("=" * 70)
    
    # 生成数据（特征多于样本，模拟过拟合场景）
    X, y = generate_data(n_samples=100, n_features=20, noise=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化（对正则化很重要）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数量: {X_train.shape[1]}")
    print(f"  注意：特征数量接近样本数量，容易过拟合")
    
    # 对比不同正则化方法
    results = compare_regularization(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # 正则化路径
    regularization_path(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 特征选择对比
    feature_selection_comparison(X_train_scaled, y_train, X_test_scaled, y_test)
    
    # 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    print("1. Ridge回归：所有特征都保留，系数缩小但不为0")
    print("2. Lasso回归：进行特征选择，部分系数变为0")
    print("3. Elastic Net：结合两者优点，在特征相关时更稳定")
    print("4. 正则化参数alpha需要仔细调优")


if __name__ == "__main__":
    main()

