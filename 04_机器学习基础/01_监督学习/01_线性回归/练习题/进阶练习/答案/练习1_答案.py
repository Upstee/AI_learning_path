"""
进阶练习1答案：完整的回归系统
代码量：200-300行
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n_samples=300, random_state=42):
    """生成复杂数据"""
    np.random.seed(random_state)
    
    data = {
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(20, 3, n_samples),
        'feature3': np.random.normal(5, 1, n_samples),
        'feature4': np.random.randint(0, 10, n_samples),
        'feature5': np.random.uniform(0, 1, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 目标：非线性关系
    df['target'] = (
        df['feature1'] * 2 +
        df['feature2'] * 1.5 +
        df['feature3'] ** 2 * 0.5 +
        df['feature4'] * 3 +
        np.random.normal(0, 5, n_samples)
    )
    
    return df


def explore_data(df):
    """数据探索"""
    print("=" * 70)
    print("1. 数据探索")
    print("=" * 70)
    
    print(f"\n数据形状: {df.shape}")
    print(f"\n数据统计信息:")
    print(df.describe())
    
    print(f"\n缺失值:")
    print(df.isnull().sum())
    
    # 相关性分析
    print("\n特征与目标的相关性:")
    correlations = df.corr()['target'].sort_values(ascending=False)
    print(correlations)
    
    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        if col != 'target':
            axes[idx].scatter(df[col], df['target'], alpha=0.5)
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('target')
            axes[idx].set_title(f'{col} vs target')
            axes[idx].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('数据探索.png', dpi=150, bbox_inches='tight')
    plt.show()


def preprocess_data(df):
    """数据预处理"""
    print("\n" + "=" * 70)
    print("2. 数据预处理")
    print("=" * 70)
    
    X = df.drop('target', axis=1)
    y = df['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def train_and_select_models(X_train, X_test, y_train, y_test):
    """训练和选择模型"""
    print("\n" + "=" * 70)
    print("3. 模型训练和选择")
    print("=" * 70)
    
    models = {
        '线性回归': LinearRegression(),
        'Ridge': Ridge(),
        'Lasso': Lasso(),
        'Elastic Net': ElasticNet()
    }
    
    results = {}
    
    # 基础模型
    print("\n基础模型性能:")
    print(f"{'模型':<20} {'测试MSE':<15} {'测试R²':<15} {'CV R²':<15}")
    print("-" * 65)
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'mse': mse,
            'r2': r2,
            'cv_r2': cv_scores.mean(),
            'y_pred': y_pred
        }
        
        print(f"{name:<20} {mse:<15.4f} {r2:<15.4f} {cv_scores.mean():.4f}")
    
    # 超参数调优
    print("\n超参数调优:")
    ridge_param_grid = {'alpha': np.logspace(-2, 2, 20)}
    ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
    ridge_grid.fit(X_train, y_train)
    
    print(f"最佳Ridge参数: alpha={ridge_grid.best_params_['alpha']:.4f}")
    print(f"最佳Ridge CV R²: {ridge_grid.best_score_:.4f}")
    
    results['Ridge (优化)'] = {
        'model': ridge_grid.best_estimator_,
        'mse': mean_squared_error(y_test, ridge_grid.predict(X_test)),
        'r2': r2_score(y_test, ridge_grid.predict(X_test)),
        'cv_r2': ridge_grid.best_score_,
        'y_pred': ridge_grid.predict(X_test)
    }
    
    return results


def evaluate_models(y_test, results):
    """模型评估"""
    print("\n" + "=" * 70)
    print("4. 模型评估")
    print("=" * 70)
    
    print(f"{'模型':<20} {'MSE':<15} {'RMSE':<15} {'MAE':<15} {'R²':<15}")
    print("-" * 80)
    
    for name, result in results.items():
        mse = result['mse']
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, result['y_pred'])
        r2 = result['r2']
        print(f"{name:<20} {mse:<15.4f} {rmse:<15.4f} {mae:<15.4f} {r2:<15.4f}")
    
    # 可视化
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # 预测 vs 真实
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_result = results[best_model_name]
    
    axes[0, 0].scatter(y_test, best_result['y_pred'], alpha=0.6)
    min_val = min(y_test.min(), best_result['y_pred'].min())
    max_val = max(y_test.max(), best_result['y_pred'].max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    axes[0, 0].set_xlabel('真实值')
    axes[0, 0].set_ylabel('预测值')
    axes[0, 0].set_title(f'{best_model_name} - 预测 vs 真实')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 残差图
    residuals = y_test - best_result['y_pred']
    axes[0, 1].scatter(best_result['y_pred'], residuals, alpha=0.6)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 1].set_xlabel('预测值')
    axes[0, 1].set_ylabel('残差')
    axes[0, 1].set_title('残差图')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 残差分布
    axes[1, 0].hist(residuals, bins=30, alpha=0.7)
    axes[1, 0].set_xlabel('残差')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('残差分布')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # 模型对比
    model_names = list(results.keys())
    r2_scores = [results[m]['r2'] for m in model_names]
    axes[1, 1].barh(model_names, r2_scores)
    axes[1, 1].set_xlabel('R²分数')
    axes[1, 1].set_title('模型性能对比')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('模型评估.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_features(results, feature_names):
    """特征重要性分析"""
    print("\n" + "=" * 70)
    print("5. 特征重要性分析")
    print("=" * 70)
    
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    best_model = results[best_model_name]['model']
    
    if hasattr(best_model, 'coef_'):
        importance_df = pd.DataFrame({
            '特征': feature_names,
            '系数': best_model.coef_
        }).sort_values('系数', key=abs, ascending=False)
        
        print("\n特征重要性:")
        print(importance_df.to_string(index=False))
        
        # 可视化
        plt.figure(figsize=(10, 6))
        plt.barh(importance_df['特征'], importance_df['系数'])
        plt.xlabel('系数值')
        plt.title(f'{best_model_name} - 特征重要性')
        plt.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig('特征重要性.png', dpi=150, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    print("=" * 70)
    print("进阶练习1：完整的回归系统")
    print("=" * 70)
    
    # 1. 数据探索
    df = generate_data(n_samples=300)
    explore_data(df)
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 3. 模型训练和选择
    results = train_and_select_models(X_train, X_test, y_train, y_test)
    
    # 4. 模型评估
    evaluate_models(y_test, results)
    
    # 5. 特征重要性分析
    analyze_features(results, feature_names)
    
    # 6. 总结
    print("\n" + "=" * 70)
    print("总结")
    print("=" * 70)
    best_model_name = max(results.items(), key=lambda x: x[1]['r2'])[0]
    print(f"\n最佳模型: {best_model_name}")
    print(f"  测试集R²: {results[best_model_name]['r2']:.4f}")
    print(f"  交叉验证R²: {results[best_model_name]['cv_r2']:.4f}")
    
    print("\n✅ 练习完成！")


if __name__ == "__main__":
    main()

