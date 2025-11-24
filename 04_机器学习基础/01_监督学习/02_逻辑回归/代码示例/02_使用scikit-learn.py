"""
使用scikit-learn实现逻辑回归
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.preprocessing import StandardScaler


def example1_binary_classification():
    """示例1：二分类问题"""
    print("=" * 50)
    print("示例1：二分类问题")
    print("=" * 50)
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, 
                               random_state=42)
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化（逻辑回归通常需要标准化）
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建模型
    model = LogisticRegression(random_state=42, max_iter=1000)
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred))
    
    # ROC曲线
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(fpr, tpr, label=f'ROC曲线 (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('假正例率')
    plt.ylabel('真正例率')
    plt.title('ROC曲线')
    plt.legend()
    plt.grid(True)
    
    # 决策边界
    plt.subplot(1, 2, 2)
    x_min, x_max = X_test_scaled[:, 0].min() - 1, X_test_scaled[:, 0].max() + 1
    y_min, y_max = X_test_scaled[:, 1].min() - 1, X_test_scaled[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, levels=50, alpha=0.5, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X_test_scaled[:, 0], X_test_scaled[:, 1], c=y_test, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('决策边界')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n模型参数:")
    print(f"偏置项: {model.intercept_[0]:.4f}")
    print(f"权重: {model.coef_[0]}")


def example2_multiclass():
    """示例2：多分类问题"""
    print("\n" + "=" * 50)
    print("示例2：多分类问题（Iris数据集）")
    print("=" * 50)
    
    from sklearn.datasets import load_iris
    
    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 创建模型（多分类使用softmax）
    model = LogisticRegression(multi_class='multinomial', solver='lbfgs', random_state=42, max_iter=1000)
    
    # 训练
    model.fit(X_train_scaled, y_train)
    
    # 预测
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)
    
    # 评估
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=iris.target_names))
    
    print(f"\n混淆矩阵:")
    print(confusion_matrix(y_test, y_pred))
    
    print(f"\n每个类别的概率预测示例（前5个样本）:")
    for i in range(min(5, len(y_test))):
        print(f"样本 {i+1}: 真实={y_test[i]}, 预测={y_pred[i]}, 概率={y_proba[i]}")


def example3_regularization():
    """示例3：正则化"""
    print("\n" + "=" * 50)
    print("示例3：正则化（L1和L2）")
    print("=" * 50)
    
    # 加载数据
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # 划分数据
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 测试不同的正则化
    regularizations = [
        ('L2', LogisticRegression(penalty='l2', C=1.0, random_state=42, max_iter=1000)),
        ('L2 (强)', LogisticRegression(penalty='l2', C=0.1, random_state=42, max_iter=1000)),
        ('L1', LogisticRegression(penalty='l1', C=1.0, solver='liblinear', random_state=42, max_iter=1000)),
        ('L1 (强)', LogisticRegression(penalty='l1', C=0.1, solver='liblinear', random_state=42, max_iter=1000)),
    ]
    
    results = []
    for name, model in regularizations:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        accuracy = accuracy_score(y_test, y_pred)
        
        # 统计非零参数数量
        n_nonzero = np.sum(model.coef_[0] != 0)
        
        results.append({
            'name': name,
            'accuracy': accuracy,
            'n_nonzero': n_nonzero
        })
        
        print(f"{name}: 准确率={accuracy:.4f}, 非零参数={n_nonzero}/{len(model.coef_[0])}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # 准确率对比
    names = [r['name'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    axes[0].bar(names, accuracies)
    axes[0].set_ylabel('准确率')
    axes[0].set_title('不同正则化方法的准确率')
    axes[0].grid(True, alpha=0.3)
    
    # 非零参数数量
    n_nonzeros = [r['n_nonzero'] for r in results]
    axes[1].bar(names, n_nonzeros)
    axes[1].set_ylabel('非零参数数量')
    axes[1].set_title('不同正则化方法的稀疏性')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 运行示例
    example1_binary_classification()
    example2_multiclass()
    example3_regularization()

