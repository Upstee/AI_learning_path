"""
简单项目1：鸢尾花分类
使用KNN分类器对鸢尾花进行分类

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# ========== 1. 加载数据 ==========
print("=" * 60)
print("1. 加载数据")
print("=" * 60)

# 加载鸢尾花数据集
# 这是scikit-learn内置的经典数据集
iris = load_iris()

# 获取特征和标签
X = iris.data      # 特征：花萼长度、花萼宽度、花瓣长度、花瓣宽度
y = iris.target    # 标签：0=Setosa, 1=Versicolor, 2=Virginica

print(f"数据信息:")
print(f"  样本数: {X.shape[0]}")
print(f"  特征数: {X.shape[1]}")
print(f"  特征名称: {iris.feature_names}")
print(f"  类别: {iris.target_names}")
print(f"  类别数: {len(iris.target_names)}")

# 查看前几个样本
print(f"\n前5个样本:")
for i in range(5):
    print(f"  样本{i+1}: {X[i]}, 类别: {iris.target_names[y[i]]}")

# ========== 2. 数据可视化 ==========
print("\n" + "=" * 60)
print("2. 数据可视化")
print("=" * 60)

# 可视化特征分布
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
feature_names = iris.feature_names

for i, (ax, feature_name) in enumerate(zip(axes.flat, feature_names)):
    # 为每个类别绘制直方图
    for j, target_name in enumerate(iris.target_names):
        ax.hist(X[y == j, i], alpha=0.7, label=target_name, bins=20)
    ax.set_xlabel(feature_name)
    ax.set_ylabel('频数')
    ax.set_title(f'{feature_name}的分布')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('iris_feature_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# 可视化特征之间的关系（散点图矩阵）
# 只使用前两个特征进行可视化
plt.figure(figsize=(10, 8))
colors = ['red', 'green', 'blue']
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X[y == i, 0], X[y == i, 1], 
               c=colors[i], label=target_name, alpha=0.6, s=50)
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])
plt.title('鸢尾花数据散点图（前两个特征）')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('iris_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 数据预处理 ==========
print("\n" + "=" * 60)
print("3. 数据预处理")
print("=" * 60)

# 划分训练集和测试集
# test_size=0.3 表示测试集占30%
# random_state=42 确保结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# 特征标准化
# KNN对特征尺度敏感，需要标准化
# StandardScaler将特征缩放到均值为0，标准差为1
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n标准化前（前5个训练样本）:")
print(X_train[:5])
print("\n标准化后（前5个训练样本）:")
print(X_train_scaled[:5])

# ========== 4. 训练模型 ==========
print("\n" + "=" * 60)
print("4. 训练模型")
print("=" * 60)

# 创建KNN分类器
# n_neighbors=5: 选择最近的5个邻居
# weights='uniform': 所有邻居权重相同
# metric='euclidean': 使用欧氏距离
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights='uniform',
    metric='euclidean'
)

# 训练模型
print("训练模型...")
knn.fit(X_train_scaled, y_train)
print("训练完成！")

# ========== 5. 预测和评估 ==========
print("\n" + "=" * 60)
print("5. 预测和评估")
print("=" * 60)

# 预测测试集
y_pred = knn.predict(X_test_scaled)

# 计算准确率
# 准确率 = 正确预测的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 详细分类报告
# 包含精确率、召回率、F1分数等指标
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

# 混淆矩阵
# 显示每个类别的预测情况
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=iris.target_names, 
            yticklabels=iris.target_names)
plt.title('混淆矩阵 - 鸢尾花分类')
plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.tight_layout()
plt.savefig('iris_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 测试不同K值 ==========
print("\n" + "=" * 60)
print("6. 测试不同K值")
print("=" * 60)

# 测试不同的K值
k_values = range(1, 21)
accuracies = []

print("\n测试不同K值:")
for k in k_values:
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train_scaled, y_train)
    y_pred_k = knn_k.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred_k)
    accuracies.append(acc)
    if k <= 5 or k % 5 == 0:
        print(f"  K={k}: 准确率={acc:.4f}")

# 找到最佳K值
best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\n最佳K值: {best_k}, 准确率: {best_acc:.4f}")

# 可视化K值对准确率的影响
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'o-', linewidth=2, markersize=8)
plt.xlabel('K值', fontsize=12)
plt.ylabel('准确率', fontsize=12)
plt.title('K值对准确率的影响', fontsize=14)
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k, color='r', linestyle='--', linewidth=2, label=f'最佳K={best_k}')
plt.legend(fontsize=12)
plt.tight_layout()
plt.savefig('iris_k_values.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 7. 测试新样本 ==========
print("\n" + "=" * 60)
print("7. 测试新样本")
print("=" * 60)

# 使用最佳K值重新训练
knn_best = KNeighborsClassifier(n_neighbors=best_k)
knn_best.fit(X_train_scaled, y_train)

# 新样本（未标准化的原始特征）
new_samples = np.array([
    [5.1, 3.5, 1.4, 0.2],  # 应该是Setosa
    [6.2, 3.4, 5.4, 2.3],  # 应该是Virginica
    [5.9, 3.0, 4.2, 1.5],  # 应该是Versicolor
])

# 标准化新样本
new_samples_scaled = scaler.transform(new_samples)

# 预测
predictions = knn_best.predict(new_samples_scaled)
probabilities = knn_best.predict_proba(new_samples_scaled)

print("\n预测结果:")
for i, sample in enumerate(new_samples):
    pred_name = iris.target_names[predictions[i]]
    print(f"\n样本 {i+1}: {sample}")
    print(f"  预测类别: {pred_name}")
    print(f"  概率分布:")
    for j, target_name in enumerate(iris.target_names):
        print(f"    {target_name}: {probabilities[i][j]:.4f}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. KNN是一个简单而有效的分类算法
2. 特征标准化对KNN很重要
3. K值的选择会影响性能，需要通过实验找到最佳值
4. KNN适合小规模数据，对局部模式敏感
""")

