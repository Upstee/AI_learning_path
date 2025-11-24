"""
简单项目2：情感分析
使用朴素贝叶斯分类器进行文本情感分析

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 正面情感文本（positive = 1）
positive_texts = [
    "这部电影太棒了，我非常喜欢",
    "服务很好，推荐大家来",
    "产品质量不错，值得购买",
    "今天心情很好",
    "这个餐厅的菜很好吃",
    "学习很有趣",
    "工作很顺利",
    "生活很美好",
]

# 负面情感文本（negative = 0）
negative_texts = [
    "这部电影太糟糕了，浪费时间",
    "服务很差，不推荐",
    "产品质量不好，不值得购买",
    "今天心情很糟糕",
    "这个餐厅的菜很难吃",
    "学习很无聊",
    "工作不顺利",
    "生活很糟糕",
]

# 合并数据和标签
texts = positive_texts + negative_texts
labels = [1] * len(positive_texts) + [0] * len(negative_texts)

print(f"总文本数: {len(texts)}")
print(f"正面情感: {len(positive_texts)}")
print(f"负面情感: {len(negative_texts)}")

# ========== 2. 特征提取 ==========
print("\n" + "=" * 60)
print("2. 特征提取")
print("=" * 60)

# 使用TF-IDF特征
# TF-IDF考虑了词的重要性，比简单词频更好
vectorizer = TfidfVectorizer(max_features=1000)
X = vectorizer.fit_transform(texts)

print(f"特征矩阵形状: {X.shape}")
print(f"词汇表大小: {len(vectorizer.vocabulary_)}")

# ========== 3. 划分数据集 ==========
print("\n" + "=" * 60)
print("3. 划分数据集")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, labels, 
    test_size=0.3, 
    random_state=42
)

print(f"训练集大小: {X_train.shape[0]}")
print(f"测试集大小: {X_test.shape[0]}")

# ========== 4. 训练模型 ==========
print("\n" + "=" * 60)
print("4. 训练模型")
print("=" * 60)

nb = MultinomialNB(alpha=1.0)
nb.fit(X_train, y_train)
print("训练完成！")

# ========== 5. 评估 ==========
print("\n" + "=" * 60)
print("5. 评估")
print("=" * 60)

y_pred = nb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['负面', '正面']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['负面', '正面'], 
            yticklabels=['负面', '正面'])
plt.title('混淆矩阵 - 情感分析')
plt.ylabel('真实情感')
plt.xlabel('预测情感')
plt.tight_layout()
plt.savefig('sentiment_analysis_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 测试新文本 ==========
print("\n" + "=" * 60)
print("6. 测试新文本")
print("=" * 60)

new_texts = [
    "今天天气真好，心情愉快",
    "这个产品太差了，不推荐",
    "服务还可以，一般般",
]

X_new = vectorizer.transform(new_texts)
predictions = nb.predict(X_new)
probabilities = nb.predict_proba(X_new)

print("\n预测结果:")
for i, text in enumerate(new_texts):
    sentiment = "正面" if predictions[i] == 1 else "负面"
    prob_positive = probabilities[i][1]
    prob_negative = probabilities[i][0]
    print(f"\n文本: {text}")
    print(f"  情感: {sentiment}")
    print(f"  负面概率: {prob_negative:.4f}")
    print(f"  正面概率: {prob_positive:.4f}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)

