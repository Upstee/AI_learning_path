"""
案例1：房价预测系统
使用线性回归预测房价
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def generate_data(n_samples=500, random_state=42):
    """生成模拟房价数据"""
    np.random.seed(random_state)
    
    data = {
        'area': np.random.normal(100, 20, n_samples),
        'bedrooms': np.random.randint(1, 5, n_samples),
        'bathrooms': np.random.randint(1, 4, n_samples),
        'age': np.random.randint(0, 30, n_samples),
        'location_score': np.random.uniform(1, 10, n_samples),
        'floor': np.random.randint(1, 20, n_samples),
        'parking': np.random.choice([0, 1], n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 计算房价（模拟真实关系）
    df['price'] = (
        df['area'] * 0.8 +
        df['bedrooms'] * 5 +
        df['bathrooms'] * 3 +
        df['location_score'] * 8 +
        df['floor'] * 0.5 -
        df['age'] * 0.3 +
        df['parking'] * 10 +
        np.random.normal(0, 5, n_samples)
    )
    
    # 确保价格为正值
    df['price'] = np.maximum(df['price'], 20)
    
    return df


def explore_data(df):
    """数据探索"""
    print("=" * 70)
    print("数据探索")
    print("=" * 70)
    
    print(f"\n数据形状: {df.shape}")
    print(f"\n数据前5行:")
    print(df.head())
    
    print(f"\n数据统计信息:")
    print(df.describe())
    
    print(f"\n缺失值:")
    print(df.isnull().sum())
    
    # 可视化
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    for idx, col in enumerate(df.columns):
        if col != 'price':
            axes[idx].hist(df[col], bins=30, alpha=0.7)
            axes[idx].set_title(f'{col} 分布')
            axes[idx].set_xlabel(col)
            axes[idx].set_ylabel('频数')
    
    plt.tight_layout()
    plt.savefig('数据分布.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    # 相关性分析
    print("\n特征与价格的相关性:")
    correlations = df.corr()['price'].sort_values(ascending=False)
    print(correlations)
    
    # 相关性热力图
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt='.2f', cmap='coolwarm', center=0)
    plt.title('特征相关性热力图')
    plt.tight_layout()
    plt.savefig('相关性热力图.png', dpi=150, bbox_inches='tight')
    plt.show()


def preprocess_data(df):
    """数据预处理"""
    print("\n" + "=" * 70)
    print("数据预处理")
    print("=" * 70)
    
    # 分离特征和目标
    X = df.drop('price', axis=1)
    y = df['price']
    
    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def train_models(X_train, X_test, y_train, y_test):
    """训练多个模型"""
    print("\n" + "=" * 70)
    print("模型训练")
    print("=" * 70)
    
    models = {
        '线性回归': LinearRegression(),
        'Ridge回归': Ridge(alpha=1.0),
        'Lasso回归': Lasso(alpha=0.1)
    }
    
    results = {}
    
    for name, model in models.items():
        # 训练
        model.fit(X_train, y_train)
        
        # 预测
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # 评估
        train_mse = mean_squared_error(y_train, y_pred_train)
        test_mse = mean_squared_error(y_test, y_pred_test)
        test_rmse = np.sqrt(test_mse)
        test_mae = mean_absolute_error(y_test, y_pred_test)
        test_r2 = r2_score(y_test, y_pred_test)
        
        # 交叉验证
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
        
        results[name] = {
            'model': model,
            'train_mse': train_mse,
            'test_mse': test_mse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'cv_r2_mean': cv_scores.mean(),
            'cv_r2_std': cv_scores.std(),
            'y_pred_test': y_pred_test
        }
        
        print(f"\n{name}:")
        print(f"  训练集MSE: {train_mse:.4f}")
        print(f"  测试集MSE: {test_mse:.4f}")
        print(f"  测试集RMSE: {test_rmse:.4f}")
        print(f"  测试集MAE: {test_mae:.4f}")
        print(f"  测试集R²: {test_r2:.4f}")
        print(f"  交叉验证R²: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    
    return results


def visualize_results(y_test, results):
    """可视化结果"""
    print("\n" + "=" * 70)
    print("结果可视化")
    print("=" * 70)
    
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        
        # 散点图：预测 vs 真实
        ax.scatter(y_test, result['y_pred_test'], alpha=0.6)
        
        # 理想线
        min_val = min(y_test.min(), result['y_pred_test'].min())
        max_val = max(y_test.max(), result['y_pred_test'].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='理想线')
        
        ax.set_xlabel('真实价格')
        ax.set_ylabel('预测价格')
        ax.set_title(f'{name}\nR² = {result["test_r2"]:.4f}, RMSE = {result["test_rmse"]:.2f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('预测结果对比.png', dpi=150, bbox_inches='tight')
    plt.show()


def analyze_features(results, feature_names):
    """分析特征重要性"""
    print("\n" + "=" * 70)
    print("特征重要性分析")
    print("=" * 70)
    
    # 创建特征重要性DataFrame
    importance_df = pd.DataFrame({
        '特征': feature_names,
        '线性回归': results['线性回归']['model'].coef_,
        'Ridge回归': results['Ridge回归']['model'].coef_,
        'Lasso回归': results['Lasso回归']['model'].coef_
    })
    
    print("\n特征系数:")
    print(importance_df.to_string(index=False))
    
    # 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (name, result) in enumerate(results.items()):
        ax = axes[idx]
        coef = result['model'].coef_
        ax.barh(feature_names, coef)
        ax.set_xlabel('系数值')
        ax.set_title(f'{name} - 特征系数')
        ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('特征重要性.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return importance_df


def main():
    """主函数"""
    print("=" * 70)
    print("案例1：房价预测系统")
    print("=" * 70)
    
    # 1. 生成数据
    df = generate_data(n_samples=500)
    
    # 2. 数据探索
    explore_data(df)
    
    # 3. 数据预处理
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 4. 训练模型
    results = train_models(X_train, X_test, y_train, y_test)
    
    # 5. 可视化结果
    visualize_results(y_test, results)
    
    # 6. 特征重要性分析
    importance_df = analyze_features(results, feature_names)
    
    # 7. 总结
    print("\n" + "=" * 70)
    print("项目总结")
    print("=" * 70)
    
    best_model = max(results.items(), key=lambda x: x[1]['test_r2'])
    print(f"\n最佳模型: {best_model[0]}")
    print(f"  测试集R²: {best_model[1]['test_r2']:.4f}")
    print(f"  测试集RMSE: {best_model[1]['test_rmse']:.2f} 万元")
    
    print("\n最重要的特征（按线性回归系数）:")
    top_features = importance_df.nlargest(3, '线性回归')
    for _, row in top_features.iterrows():
        print(f"  {row['特征']}: {row['线性回归']:.4f}")


if __name__ == "__main__":
    main()

