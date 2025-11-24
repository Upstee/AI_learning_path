"""
XGBoost和LightGBM示例
需要安装: pip install xgboost lightgbm
"""

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# 尝试导入XGBoost和LightGBM
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，跳过XGBoost示例")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("LightGBM未安装，跳过LightGBM示例")

# ========== 1. XGBoost分类 ==========
if XGBOOST_AVAILABLE:
    print("=" * 50)
    print("1. XGBoost分类")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_classifier.fit(X_train, y_train)
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"XGBoost准确率: {accuracy:.4f}")
    
    # 特征重要性
    print(f"\n特征重要性（前10个）:")
    feature_importance = pd.DataFrame({
        'feature': [f'特征{i}' for i in range(20)],
        'importance': xgb_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    # ========== 2. XGBoost回归 ==========
    print("\n" + "=" * 50)
    print("2. XGBoost回归")
    print("=" * 50)
    
    # 生成回归数据
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # XGBoost回归器
    xgb_regressor = xgb.XGBRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    xgb_regressor.fit(X_train, y_train)
    y_pred = xgb_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"XGBoost RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")

# ========== 3. LightGBM分类 ==========
if LIGHTGBM_AVAILABLE:
    print("\n" + "=" * 50)
    print("3. LightGBM分类")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM分类器
    lgb_classifier = lgb.LGBMClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_classifier.fit(X_train, y_train)
    y_pred = lgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"LightGBM准确率: {accuracy:.4f}")
    
    # ========== 4. LightGBM回归 ==========
    print("\n" + "=" * 50)
    print("4. LightGBM回归")
    print("=" * 50)
    
    # 生成回归数据
    X, y = make_regression(n_samples=500, n_features=10, noise=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # LightGBM回归器
    lgb_regressor = lgb.LGBMRegressor(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    lgb_regressor.fit(X_train, y_train)
    y_pred = lgb_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f"LightGBM RMSE: {np.sqrt(mse):.4f}, R²: {r2:.4f}")

# ========== 5. 性能对比 ==========
if XGBOOST_AVAILABLE and LIGHTGBM_AVAILABLE:
    print("\n" + "=" * 50)
    print("5. XGBoost vs LightGBM性能对比")
    print("=" * 50)
    
    from sklearn.ensemble import GradientBoostingClassifier
    import time
    
    # 生成数据
    X, y = make_classification(n_samples=5000, n_features=50, n_informative=25,
                              n_redundant=25, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    models = {
        'GBDT': GradientBoostingClassifier(n_estimators=100, max_depth=3, random_state=42),
        'XGBoost': xgb.XGBClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1),
        'LightGBM': lgb.LGBMClassifier(n_estimators=100, max_depth=3, random_state=42, n_jobs=-1)
    }
    
    results = []
    for name, model in models.items():
        start = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start
        
        start = time.time()
        y_pred = model.predict(X_test)
        pred_time = time.time() - start
        
        accuracy = accuracy_score(y_test, y_pred)
        results.append({
            '模型': name,
            '准确率': accuracy,
            '训练时间(秒)': train_time,
            '预测时间(秒)': pred_time
        })
        print(f"{name}: 准确率={accuracy:.4f}, 训练时间={train_time:.2f}秒, 预测时间={pred_time:.4f}秒")
    
    # 可视化
    df_results = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].bar(df_results['模型'], df_results['准确率'])
    axes[0].set_ylabel('准确率')
    axes[0].set_title('准确率对比')
    axes[0].grid(True, axis='y', alpha=0.3)
    
    axes[1].bar(df_results['模型'], df_results['训练时间(秒)'])
    axes[1].set_ylabel('训练时间(秒)')
    axes[1].set_title('训练时间对比')
    axes[1].grid(True, axis='y', alpha=0.3)
    
    axes[2].bar(df_results['模型'], df_results['预测时间(秒)'])
    axes[2].set_ylabel('预测时间(秒)')
    axes[2].set_title('预测时间对比')
    axes[2].grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ensemble_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

print("\n" + "=" * 50)
print("XGBoost和LightGBM示例完成！")
print("=" * 50)
print("\n注意：如果XGBoost或LightGBM未安装，请运行:")
print("  pip install xgboost lightgbm")

