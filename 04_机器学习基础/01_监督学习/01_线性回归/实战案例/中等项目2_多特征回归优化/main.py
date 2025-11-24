"""
中等项目2：多特征回归优化
重点练习特征工程和模型优化
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n_samples=300, random_state=42):
    """生成多特征数据"""
    np.random.seed(random_state)
    
    data = {
        'feature1': np.random.normal(10, 2, n_samples),
        'feature2': np.random.normal(20, 3, n_samples),
        'feature3': np.random.normal(5, 1, n_samples),
        'feature4': np.random.randint(0, 10, n_samples),
        'feature5': np.random.choice(['A', 'B', 'C'], n_samples),
        'feature6': np.random.uniform(0, 1, n_samples),
        'feature7': np.random.poisson(3, n_samples),
        'feature8': np.random.normal(15, 2, n_samples),
        'feature9': np.random.choice([0, 1], n_samples),
        'feature10': np.random.normal(8, 1.5, n_samples)
    }
    
    df = pd.DataFrame(data)
    
    # 目标：只有部分特征真正有用
    df['target'] = (
        df['feature1'] * 2 +
        df['feature2'] * 1.5 +
        df['feature3'] * 3 +
        df['feature5'].map({'A': 5, 'B': 3, 'C': 1}) +
        df['feature9'] * 4 +
        np.random.normal(0, 5, n_samples)
    )
    
    return df


def preprocess_data(df):
    """数据预处理"""
    # 分离特征和目标
    X = df.drop('target', axis=1)
    y = df['target']
    
    # 处理类别特征
    le = LabelEncoder()
    X_processed = X.copy()
    X_processed['feature5_encoded'] = le.fit_transform(X['feature5'])
    X_processed = X_processed.drop('feature5', axis=1)
    
    return X_processed, y, le


def feature_selection(X_train, y_train, X_test, k=5):
    """特征选择"""
    selector = SelectKBest(score_func=f_regression, k=k)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)
    
    selected_features = X_train.columns[selector.get_support()]
    print(f"\n选择的特征: {list(selected_features)}")
    
    return X_train_selected, X_test_selected, selector


def main():
    """主函数"""
    print("=" * 70)
    print("中等项目2：多特征回归优化")
    print("=" * 70)
    
    # 1. 生成数据
    df = generate_data(n_samples=300)
    X, y, le = preprocess_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征数量: {X_train.shape[1]}")
    
    # 2. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 3. 特征选择
    print("\n" + "-" * 70)
    print("3. 特征选择")
    print("-" * 70)
    X_train_selected, X_test_selected, selector = feature_selection(
        pd.DataFrame(X_train_scaled, columns=X_train.columns),
        y_train,
        pd.DataFrame(X_test_scaled, columns=X_test.columns),
        k=5
    )
    
    # 4. 模型训练和优化
    print("\n" + "-" * 70)
    print("4. 模型训练和优化")
    print("-" * 70)
    
    # 4.1 基础模型
    models_before = {
        '线性回归': LinearRegression(),
        'Ridge': Ridge(alpha=1.0),
        'Lasso': Lasso(alpha=0.1)
    }
    
    print("\n优化前（使用所有特征）:")
    print(f"{'模型':<20} {'测试MSE':<15} {'测试R²':<15}")
    print("-" * 50)
    
    results_before = {}
    for name, model in models_before.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        results_before[name] = {'mse': mse, 'r2': r2}
        print(f"{name:<20} {mse:<15.4f} {r2:<15.4f}")
    
    # 4.2 优化后（特征选择+超参数调优）
    print("\n优化后（特征选择+超参数调优）:")
    
    # Ridge调优
    ridge_param_grid = {'alpha': np.logspace(-2, 2, 20)}
    ridge_grid = GridSearchCV(Ridge(), ridge_param_grid, cv=5, scoring='r2')
    ridge_grid.fit(X_train_selected, y_train)
    
    # Lasso调优
    lasso_param_grid = {'alpha': np.logspace(-3, 1, 20)}
    lasso_grid = GridSearchCV(Lasso(max_iter=10000), lasso_param_grid, cv=5, scoring='r2')
    lasso_grid.fit(X_train_selected, y_train)
    
    models_after = {
        '线性回归': LinearRegression(),
        'Ridge (优化)': ridge_grid.best_estimator_,
        'Lasso (优化)': lasso_grid.best_estimator_
    }
    
    print(f"{'模型':<20} {'测试MSE':<15} {'测试R²':<15} {'改进':<15}")
    print("-" * 65)
    
    for name, model in models_after.items():
        model.fit(X_train_selected, y_train)
        y_pred = model.predict(X_test_selected)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # 计算改进
        if name in results_before:
            improvement = r2 - results_before[name]['r2']
            print(f"{name:<20} {mse:<15.4f} {r2:<15.4f} {improvement:+.4f}")
        else:
            print(f"{name:<20} {mse:<15.4f} {r2:<15.4f} {'N/A':<15}")
    
    print(f"\n最佳Ridge参数: alpha={ridge_grid.best_params_['alpha']:.4f}")
    print(f"最佳Lasso参数: alpha={lasso_grid.best_params_['alpha']:.4f}")
    
    # 5. 可视化对比
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 性能对比
    model_names = list(results_before.keys())
    r2_before = [results_before[m]['r2'] for m in model_names]
    r2_after = [r2_score(y_test, models_after[m].predict(X_test_selected)) 
                for m in model_names]
    
    x = np.arange(len(model_names))
    width = 0.35
    axes[0].bar(x - width/2, r2_before, width, label='优化前', alpha=0.7)
    axes[0].bar(x + width/2, r2_after, width, label='优化后', alpha=0.7)
    axes[0].set_xlabel('模型')
    axes[0].set_ylabel('R²分数')
    axes[0].set_title('优化前后性能对比')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(model_names)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 特征重要性（Lasso）
    best_lasso = lasso_grid.best_estimator_
    selected_feature_names = X_train.columns[selector.get_support()]
    coef = best_lasso.coef_
    
    axes[1].barh(range(len(coef)), coef)
    axes[1].set_yticks(range(len(coef)))
    axes[1].set_yticklabels(selected_feature_names)
    axes[1].set_xlabel('系数值')
    axes[1].set_title('Lasso特征重要性（优化后）')
    axes[1].grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig('优化结果.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 项目完成！")


if __name__ == "__main__":
    main()

