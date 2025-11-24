"""
使用scikit-learn实现朴素贝叶斯
本示例展示如何使用scikit-learn库快速实现朴素贝叶斯分类
适合小白学习，包含详细注释和解释
"""

import numpy as np
import pandas as pd
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.datasets import make_classification, load_iris
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 高斯朴素贝叶斯（连续特征） ==========
print("=" * 60)
print("1. 高斯朴素贝叶斯（Gaussian Naive Bayes）")
print("=" * 60)
print("""
适用场景：
- 特征值是连续的（如身高、体重、温度等）
- 假设特征服从高斯（正态）分布
- 适合数值型数据
""")

# 生成示例数据
# make_classification 是scikit-learn提供的生成分类数据的函数
# n_samples: 样本数
# n_features: 特征数
# n_informative: 有信息的特征数（真正影响分类的特征）
# n_redundant: 冗余特征数（与其他特征相关的特征）
# random_state: 随机种子，确保结果可复现
X, y = make_classification(
    n_samples=1000,      # 生成1000个样本
    n_features=4,        # 4个特征
    n_informative=4,     # 4个特征都有信息
    n_redundant=0,       # 没有冗余特征
    random_state=42      # 随机种子，确保每次运行结果相同
)

# 划分训练集和测试集
# test_size=0.2 表示测试集占20%，训练集占80%
# random_state=42 确保划分结果可复现
X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42
)

print(f"\n数据信息:")
print(f"  训练集大小: {X_train.shape}")
print(f"  测试集大小: {X_test.shape}")
print(f"  类别数: {len(np.unique(y))}")

# 创建高斯朴素贝叶斯分类器
# GaussianNB 是scikit-learn提供的高斯朴素贝叶斯实现
gnb = GaussianNB()

# 训练模型
# fit() 方法会根据训练数据学习参数
print("\n训练模型...")
gnb.fit(X_train, y_train)

# 预测
# predict() 方法返回预测的类别
y_pred = gnb.predict(X_test)

# 评估准确率
# accuracy_score 计算准确率：正确预测的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred)
print(f"\n准确率: {accuracy:.4f}")

# 详细分类报告
# classification_report 提供更详细的性能指标
print(f"\n分类报告:")
print(classification_report(y_test, y_pred))

# ========== 2. 多项式朴素贝叶斯（离散特征，如词频） ==========
print("\n" + "=" * 60)
print("2. 多项式朴素贝叶斯（Multinomial Naive Bayes）")
print("=" * 60)
print("""
适用场景：
- 特征值是离散的计数（如词频、点击次数等）
- 适合文本分类、推荐系统等
- 特征值应该是非负整数
""")

# 生成示例数据（模拟词频）
# 这里我们生成非负整数的特征值，模拟词频
np.random.seed(42)
X_mult = np.random.poisson(lam=2, size=(1000, 10))  # 泊松分布生成词频
y_mult = np.random.randint(0, 2, size=1000)  # 二分类

# 划分数据集
X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
    X_mult, y_mult, 
    test_size=0.2, 
    random_state=42
)

# 创建多项式朴素贝叶斯分类器
# alpha: 平滑参数，默认1.0
# fit_prior: 是否学习先验概率，默认True
mnb = MultinomialNB(alpha=1.0)

# 训练模型
print("\n训练模型...")
mnb.fit(X_train_m, y_train_m)

# 预测
y_pred_m = mnb.predict(X_test_m)

# 评估
accuracy_m = accuracy_score(y_test_m, y_pred_m)
print(f"\n准确率: {accuracy_m:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test_m, y_pred_m))

# ========== 3. 伯努利朴素贝叶斯（二值特征） ==========
print("\n" + "=" * 60)
print("3. 伯努利朴素贝叶斯（Bernoulli Naive Bayes）")
print("=" * 60)
print("""
适用场景：
- 特征值是二值的（0或1，出现或不出现）
- 适合文本分类中的"词是否出现"（而不是词频）
- 适合推荐系统中的"用户是否点击"
""")

# 生成二值特征数据
# 这里我们生成0和1的特征值
X_bern = np.random.randint(0, 2, size=(1000, 10))  # 随机生成0和1
y_bern = np.random.randint(0, 2, size=1000)

# 划分数据集
X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
    X_bern, y_bern, 
    test_size=0.2, 
    random_state=42
)

# 创建伯努利朴素贝叶斯分类器
# alpha: 平滑参数
# binarize: 二值化阈值，如果为None，假设输入已经是二值的
bnb = BernoulliNB(alpha=1.0, binarize=None)

# 训练模型
print("\n训练模型...")
bnb.fit(X_train_b, y_train_b)

# 预测
y_pred_b = bnb.predict(X_test_b)

# 评估
accuracy_b = accuracy_score(y_test_b, y_pred_b)
print(f"\n准确率: {accuracy_b:.4f}")
print(f"\n分类报告:")
print(classification_report(y_test_b, y_pred_b))

# ========== 4. 实际应用：鸢尾花分类 ==========
print("\n" + "=" * 60)
print("4. 实际应用：鸢尾花分类")
print("=" * 60)
print("""
鸢尾花数据集是机器学习中的经典数据集
- 150个样本，4个特征（花萼长度、花萼宽度、花瓣长度、花瓣宽度）
- 3个类别（Setosa、Versicolor、Virginica）
- 特征值是连续的，适合使用高斯朴素贝叶斯
""")

# 加载鸢尾花数据集
iris = load_iris()
X_iris = iris.data      # 特征
y_iris = iris.target    # 标签

print(f"\n数据信息:")
print(f"  样本数: {X_iris.shape[0]}")
print(f"  特征数: {X_iris.shape[1]}")
print(f"  类别: {iris.target_names}")

# 划分数据集
X_train_i, X_test_i, y_train_i, y_test_i = train_test_split(
    X_iris, y_iris, 
    test_size=0.2, 
    random_state=42
)

# 创建并训练模型
gnb_iris = GaussianNB()
gnb_iris.fit(X_train_i, y_train_i)

# 预测
y_pred_i = gnb_iris.predict(X_test_i)

# 评估
accuracy_i = accuracy_score(y_test_i, y_pred_i)
print(f"\n准确率: {accuracy_i:.4f}")

# 混淆矩阵
# 混淆矩阵显示每个类别的预测情况
cm = confusion_matrix(y_test_i, y_pred_i)
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
plt.savefig('naive_bayes_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 5. 模型比较 ==========
print("\n" + "=" * 60)
print("5. 不同朴素贝叶斯模型比较")
print("=" * 60)

# 在相同的数据上比较不同模型
# 注意：这里只是为了演示，实际应用中应该根据数据类型选择模型

results = {
    '高斯朴素贝叶斯': accuracy,
    '多项式朴素贝叶斯': accuracy_m,
    '伯努利朴素贝叶斯': accuracy_b,
    '鸢尾花分类': accuracy_i
}

print("\n准确率对比:")
for model_name, acc in results.items():
    print(f"  {model_name}: {acc:.4f}")

# 可视化对比
plt.figure(figsize=(10, 6))
models = list(results.keys())
accuracies = list(results.values())
plt.bar(models, accuracies, color=['skyblue', 'lightgreen', 'lightcoral', 'gold'])
plt.ylabel('准确率')
plt.title('不同朴素贝叶斯模型的准确率对比')
plt.ylim([0, 1])
plt.grid(True, axis='y', alpha=0.3)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('naive_bayes_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("示例完成！")
print("=" * 60)
print("""
总结：
1. 高斯朴素贝叶斯：适合连续特征
2. 多项式朴素贝叶斯：适合离散计数特征（如词频）
3. 伯努利朴素贝叶斯：适合二值特征
4. 根据数据类型选择合适的模型很重要
""")

