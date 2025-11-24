"""
使用scikit-learn实现KNN
本示例展示如何使用scikit-learn库快速实现KNN分类和回归
适合小白学习，包含详细注释和解释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, make_regression, load_iris
from sklearn.preprocessing import StandardScaler
import seaborn as sns

# ========== 1. KNN分类 ==========
print("=" * 60)
print("1. KNN分类")
print("=" * 60)
print("""
KNN分类器：
- 找到最近的K个邻居
- 根据多数投票预测类别
- 适合多分类问题
""")

# 生成分类数据
# make_classification 生成分类数据集
X, y = make_classification(
    n_samples=200,      # 200个样本
    n_features=2,       # 2个特征（方便可视化）
    n_informative=2,    # 2个有信息的特征
    n_redundant=0,      # 没有冗余特征
    n_classes=3,        # 3个类别
    random_state=42
)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.3, 
    random_state=42
)

print(f"\n数据信息:")
print(f"  训练集大小: {X_train.shape[0]}")
print(f"  测试集大小: {X_test.shape[0]}")
print(f"  类别数: {len(np.unique(y))}")

# 创建KNN分类器
# n_neighbors: K值，选择最近的K个邻居
# weights: 权重方式
#   - 'uniform': 所有邻居权重相同
#   - 'distance': 距离越近，权重越大
# metric: 距离度量
#   - 'euclidean': 欧氏距离（默认）
#   - 'manhattan': 曼哈顿距离
knn_classifier = KNeighborsClassifier(
    n_neighbors=5,      # K=5
    weights='uniform',   # 均匀权重
    metric='euclidean'   # 欧氏距离
)

# 训练模型
print("\n训练模型...")
knn_classifier.fit(X_train, y_train)

# 预测
y_pred = knn_classifier.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"\n准确率: {accuracy:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# 可视化决策边界（仅适用于2D特征）
if X.shape[1] == 2:
    plt.figure(figsize=(12, 5))
    
    # 左图：训练数据
    plt.subplot(1, 2, 1)
    scatter = plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='viridis', alpha=0.6)
    plt.title('训练数据')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.colorbar(scatter)
    
    # 右图：测试数据和预测
    plt.subplot(1, 2, 2)
    scatter = plt.scatter(X_test[:, 0], X_test[:, 1], c=y_pred, cmap='viridis', alpha=0.6)
    plt.title('测试数据预测')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.colorbar(scatter)
    
    plt.tight_layout()
    plt.savefig('knn_classification.png', dpi=300, bbox_inches='tight')
    plt.show()

# ========== 2. KNN回归 ==========
print("\n" + "=" * 60)
print("2. KNN回归")
print("=" * 60)
print("""
KNN回归器：
- 找到最近的K个邻居
- 计算K个邻居的平均值（或加权平均）作为预测值
- 适合非线性回归问题
""")

# 生成回归数据
X_reg, y_reg = make_regression(
    n_samples=200,
    n_features=1,
    noise=10,
    random_state=42
)

# 划分数据集
X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
    X_reg, y_reg, 
    test_size=0.3, 
    random_state=42
)

# 创建KNN回归器
knn_regressor = KNeighborsRegressor(
    n_neighbors=5,
    weights='distance'  # 距离加权
)

# 训练模型
print("\n训练模型...")
knn_regressor.fit(X_train_r, y_train_r)

# 预测
y_pred_r = knn_regressor.predict(X_test_r)

# 计算R²分数
from sklearn.metrics import r2_score, mean_squared_error
r2 = r2_score(y_test_r, y_pred_r)
mse = mean_squared_error(y_test_r, y_pred_r)

print(f"\nR²分数: {r2:.4f}")
print(f"MSE: {mse:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.scatter(X_test_r, y_test_r, alpha=0.6, label='真实值')
plt.scatter(X_test_r, y_pred_r, alpha=0.6, label='预测值')
plt.xlabel('特征')
plt.ylabel('目标值')
plt.title('KNN回归结果')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_regression.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 3. 不同K值的影响 ==========
print("\n" + "=" * 60)
print("3. 不同K值的影响")
print("=" * 60)
print("""
K值的影响：
- K太小：过拟合，对噪声敏感
- K太大：欠拟合，忽略局部模式
- 需要找到最佳K值
""")

# 使用鸢尾花数据集
iris = load_iris()
X_iris = iris.data
y_iris = iris.target

X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, 
    test_size=0.3, 
    random_state=42
)

# 标准化特征（KNN对特征尺度敏感）
scaler = StandardScaler()
X_train_i_scaled = scaler.fit_transform(X_train_i)
X_test_i_scaled = scaler.transform(X_test_i)

# 测试不同的K值
k_values = range(1, 21)
accuracies = []

print("\n测试不同K值:")
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train_i_scaled, y_train_i)
    y_pred_i = knn.predict(X_test_i_scaled)
    acc = accuracy_score(y_test_i, y_pred_i)
    accuracies.append(acc)
    if k <= 5 or k % 5 == 0:
        print(f"  K={k}: 准确率={acc:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_values, accuracies, 'o-')
plt.xlabel('K值')
plt.ylabel('准确率')
plt.title('K值对准确率的影响')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('knn_k_values.png', dpi=300, bbox_inches='tight')
plt.show()

# 找到最佳K值
best_k = k_values[np.argmax(accuracies)]
best_acc = max(accuracies)
print(f"\n最佳K值: {best_k}, 准确率: {best_acc:.4f}")

# ========== 4. 不同距离度量的对比 ==========
print("\n" + "=" * 60)
print("4. 不同距离度量的对比")
print("=" * 60)

metrics = ['euclidean', 'manhattan', 'minkowski']
metric_accuracies = []

for metric in metrics:
    knn = KNeighborsClassifier(n_neighbors=5, metric=metric)
    knn.fit(X_train_i_scaled, y_train_i)
    y_pred_m = knn.predict(X_test_i_scaled)
    acc = accuracy_score(y_test_i, y_pred_m)
    metric_accuracies.append(acc)
    print(f"  {metric}: {acc:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.bar(metrics, metric_accuracies, color=['skyblue', 'lightgreen', 'lightcoral'])
plt.ylabel('准确率')
plt.title('不同距离度量的对比')
plt.ylim([0, 1])
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('knn_metrics.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 加权vs非加权 ==========
print("\n" + "=" * 60)
print("5. 加权vs非加权")
print("=" * 60)

# 均匀权重
knn_uniform = KNeighborsClassifier(n_neighbors=5, weights='uniform')
knn_uniform.fit(X_train_i_scaled, y_train_i)
y_pred_u = knn_uniform.predict(X_test_i_scaled)
acc_uniform = accuracy_score(y_test_i, y_pred_u)

# 距离加权
knn_distance = KNeighborsClassifier(n_neighbors=5, weights='distance')
knn_distance.fit(X_train_i_scaled, y_train_i)
y_pred_d = knn_distance.predict(X_test_i_scaled)
acc_distance = accuracy_score(y_test_i, y_pred_d)

print(f"均匀权重: {acc_uniform:.4f}")
print(f"距离加权: {acc_distance:.4f}")

# 可视化
plt.figure(figsize=(8, 6))
plt.bar(['均匀权重', '距离加权'], [acc_uniform, acc_distance], 
        color=['skyblue', 'lightgreen'])
plt.ylabel('准确率')
plt.title('加权vs非加权KNN')
plt.ylim([0, 1])
plt.grid(True, axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('knn_weights.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 交叉验证选择最佳K值 ==========
print("\n" + "=" * 60)
print("6. 交叉验证选择最佳K值")
print("=" * 60)
print("""
交叉验证：
- 将数据分为多个折（fold）
- 每次用一折作为验证集，其余作为训练集
- 计算平均性能，选择最佳参数
""")

# 使用交叉验证测试不同的K值
k_range = range(1, 21)
cv_scores = []

for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    # 5折交叉验证
    scores = cross_val_score(knn, X_train_i_scaled, y_train_i, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())
    if k <= 5 or k % 5 == 0:
        print(f"  K={k}: 交叉验证准确率={scores.mean():.4f} (+/- {scores.std()*2:.4f})")

# 找到最佳K值
best_k_cv = k_range[np.argmax(cv_scores)]
best_cv_acc = max(cv_scores)
print(f"\n最佳K值（交叉验证）: {best_k_cv}, 准确率: {best_cv_acc:.4f}")

# 可视化
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, 'o-')
plt.xlabel('K值')
plt.ylabel('交叉验证准确率')
plt.title('交叉验证选择最佳K值')
plt.grid(True, alpha=0.3)
plt.axvline(x=best_k_cv, color='r', linestyle='--', label=f'最佳K={best_k_cv}')
plt.legend()
plt.tight_layout()
plt.savefig('knn_cross_validation.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. KNN适合分类和回归任务
2. K值、距离度量、权重方式都会影响性能
3. 使用交叉验证选择最佳参数
4. 特征标准化对KNN很重要
5. 加权KNN通常效果更好
""")

