"""
简单项目2：XGBoost应用
使用XGBoost进行分类任务
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
try:
    import xgboost as xgb
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split, GridSearchCV
    from sklearn.metrics import accuracy_score, classification_report
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost未安装，请运行: pip install xgboost")

if XGBOOST_AVAILABLE:
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                              n_redundant=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("=" * 50)
    print("XGBoost分类应用")
    print("=" * 50)
    
    # 创建XGBoost分类器
    xgb_classifier = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=42,
        n_jobs=-1
    )
    
    # 训练
    print("\n训练XGBoost模型...")
    xgb_classifier.fit(X_train, y_train)
    
    # 预测
    y_pred = xgb_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n模型性能:")
    print(f"准确率: {accuracy:.4f}")
    print(f"\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # 特征重要性
    print(f"\n特征重要性（前10个）:")
    feature_importance = pd.DataFrame({
        'feature': [f'特征{i}' for i in range(20)],
        'importance': xgb_classifier.feature_importances_
    }).sort_values('importance', ascending=False)
    print(feature_importance.head(10))
    
    # 可视化特征重要性
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance.head(10)['feature'], 
             feature_importance.head(10)['importance'])
    plt.xlabel('特征重要性')
    plt.title('XGBoost特征重要性（前10个）')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('xgboost_feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # 参数调优示例
    print("\n" + "=" * 50)
    print("参数调优示例")
    print("=" * 50)
    
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2]
    }
    
    grid_search = GridSearchCV(
        xgb.XGBClassifier(random_state=42, n_jobs=-1),
        param_grid,
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )
    
    print("进行网格搜索（这可能需要一些时间）...")
    grid_search.fit(X_train, y_train)
    
    print(f"最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证得分: {grid_search.best_score_:.4f}")
    
    # 使用最佳参数
    best_xgb = grid_search.best_estimator_
    y_pred_best = best_xgb.predict(X_test)
    accuracy_best = accuracy_score(y_test, y_pred_best)
    print(f"测试集准确率: {accuracy_best:.4f}")
    print(f"提升: {accuracy_best - accuracy:.4f}")
    
    print("\n" + "=" * 50)
    print("项目完成！")
    print("=" * 50)
else:
    print("请先安装XGBoost: pip install xgboost")

