"""
简单项目2：客户购买预测
使用逻辑回归预测客户是否会购买产品
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, 
                            confusion_matrix, roc_auc_score, roc_curve)
import seaborn as sns


def generate_data(n_samples=1000):
    """生成模拟客户数据"""
    print("生成数据...")
    X, y = make_classification(
        n_samples=n_samples,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        n_clusters_per_class=1,
        random_state=42
    )
    
    # 创建DataFrame
    df = pd.DataFrame(X, columns=['age', 'income', 'browse_time', 
                                   'click_count', 'page_views'])
    df['purchase'] = y
    
    print(f"数据形状: {df.shape}")
    print(f"购买率: {df['purchase'].mean():.2%}")
    
    return df


def preprocess_data(df):
    """数据预处理"""
    print("\n数据预处理...")
    X = df.drop('purchase', axis=1)
    y = df['purchase']
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X.columns


def train_model(X_train, y_train):
    """训练模型"""
    print("\n训练模型...")
    model = LogisticRegression(random_state=42, max_iter=1000, solver='lbfgs')
    model.fit(X_train, y_train)
    print("模型训练完成")
    return model


def evaluate_model(model, X_test, y_test):
    """评估模型"""
    print("\n评估模型...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # 评估指标
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    
    print(f"准确率: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, 
                                target_names=['不购买', '购买']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    
    # 混淆矩阵
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                xticklabels=['不购买', '购买'],
                yticklabels=['不购买', '购买'])
    axes[0].set_title('混淆矩阵')
    axes[0].set_ylabel('真实标签')
    axes[0].set_xlabel('预测标签')
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    axes[1].plot(fpr, tpr, label=f'ROC曲线 (AUC = {auc:.2f})')
    axes[1].plot([0, 1], [0, 1], 'k--')
    axes[1].set_xlabel('假正例率')
    axes[1].set_ylabel('真正例率')
    axes[1].set_title('ROC曲线')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('evaluation_results.png', dpi=300)
    plt.show()
    
    return y_pred, y_proba


def analyze_features(model, feature_names):
    """分析特征重要性"""
    print("\n特征重要性分析:")
    coefficients = model.coef_[0]
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'coefficient': coefficients,
        'abs_coefficient': np.abs(coefficients)
    }).sort_values('abs_coefficient', ascending=False)
    
    print(feature_importance)
    
    # 可视化
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance['feature'], feature_importance['coefficient'])
    plt.xlabel('系数值')
    plt.title('特征重要性（逻辑回归系数）')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300)
    plt.show()
    
    return feature_importance


def main():
    """主函数"""
    print("=" * 50)
    print("客户购买预测项目")
    print("=" * 50)
    
    # 1. 生成数据
    df = generate_data(n_samples=1000)
    
    # 2. 数据预处理
    X_train, X_test, y_train, y_test, scaler, feature_names = preprocess_data(df)
    
    # 3. 训练模型
    model = train_model(X_train, y_train)
    
    # 4. 评估模型
    y_pred, y_proba = evaluate_model(model, X_test, y_test)
    
    # 5. 分析特征
    feature_importance = analyze_features(model, feature_names)
    
    print("\n" + "=" * 50)
    print("项目完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

