"""
简单项目1：垃圾邮件检测
使用朴素贝叶斯分类器检测垃圾邮件

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ========== 1. 准备数据 ==========
print("=" * 60)
print("1. 准备数据")
print("=" * 60)

# 简单的邮件数据示例
# 在实际应用中，你需要从文件或数据库中加载真实数据
# 这里我们使用模拟数据来演示

# 正常邮件（ham = 0）
normal_emails = [
    "你好，今天天气真好",
    "明天一起吃饭吧",
    "会议改到下午三点",
    "谢谢你的帮助",
    "周末有什么计划吗",
    "项目进展顺利",
    "明天见",
    "祝你工作顺利",
]

# 垃圾邮件（spam = 1）
spam_emails = [
    "免费获得1000元现金",
    "点击这里立即领取大奖",
    "限时优惠，不要错过",
    "恭喜你中奖了",
    "立即购买，享受折扣",
    "免费试用，无需付费",
    "点击链接领取奖品",
    "限时特价，立即购买",
]

# 合并数据和标签
# 0表示正常邮件，1表示垃圾邮件
emails = normal_emails + spam_emails
labels = [0] * len(normal_emails) + [1] * len(spam_emails)

print(f"总邮件数: {len(emails)}")
print(f"正常邮件: {len(normal_emails)}")
print(f"垃圾邮件: {len(spam_emails)}")

# ========== 2. 特征提取 ==========
print("\n" + "=" * 60)
print("2. 特征提取")
print("=" * 60)

# 方法1：使用词频（CountVectorizer）
# CountVectorizer将文本转换为词频矩阵
# 每一行是一个文档，每一列是一个词，值是词频
print("\n方法1：使用词频特征")
vectorizer_count = CountVectorizer()
X_count = vectorizer_count.fit_transform(emails)

print(f"特征矩阵形状: {X_count.shape}")
print(f"词汇表大小: {len(vectorizer_count.vocabulary_)}")
print(f"前10个词: {list(vectorizer_count.vocabulary_.keys())[:10]}")

# 方法2：使用TF-IDF（可选）
# TF-IDF考虑了词的重要性，不仅仅是词频
print("\n方法2：使用TF-IDF特征")
vectorizer_tfidf = TfidfVectorizer()
X_tfidf = vectorizer_tfidf.fit_transform(emails)

print(f"特征矩阵形状: {X_tfidf.shape}")

# 这里我们使用词频特征
X = X_count
vectorizer = vectorizer_count

# ========== 3. 划分训练集和测试集 ==========
print("\n" + "=" * 60)
print("3. 划分训练集和测试集")
print("=" * 60)

# 将数据分为训练集和测试集
# test_size=0.3 表示测试集占30%
# random_state=42 确保结果可复现
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

# 使用多项式朴素贝叶斯
# 适合词频特征
# alpha=1.0 是平滑参数，防止概率为0
nb = MultinomialNB(alpha=1.0)

# 训练模型
print("训练模型...")
nb.fit(X_train, y_train)
print("训练完成！")

# ========== 5. 预测和评估 ==========
print("\n" + "=" * 60)
print("5. 预测和评估")
print("=" * 60)

# 预测测试集
y_pred = nb.predict(X_test)

# 计算准确率
# 准确率 = 正确预测的样本数 / 总样本数
accuracy = accuracy_score(y_test, y_pred)
print(f"准确率: {accuracy:.4f}")

# 详细分类报告
# 包含精确率、召回率、F1分数等指标
print(f"\n分类报告:")
print(classification_report(y_test, y_pred, target_names=['正常', '垃圾']))

# 混淆矩阵
# 显示每个类别的预测情况
cm = confusion_matrix(y_test, y_pred)
print(f"\n混淆矩阵:")
print(cm)

# 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['正常', '垃圾'], 
            yticklabels=['正常', '垃圾'])
plt.title('混淆矩阵 - 垃圾邮件检测')
plt.ylabel('真实类别')
plt.xlabel('预测类别')
plt.tight_layout()
plt.savefig('spam_detection_confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 测试新邮件 ==========
print("\n" + "=" * 60)
print("6. 测试新邮件")
print("=" * 60)

# 新邮件示例
new_emails = [
    "明天一起吃饭",  # 应该是正常邮件
    "免费领取大奖",  # 应该是垃圾邮件
    "项目进展如何",  # 应该是正常邮件
]

# 转换为特征
X_new = vectorizer.transform(new_emails)

# 预测
predictions = nb.predict(X_new)
probabilities = nb.predict_proba(X_new)

print("\n预测结果:")
for i, email in enumerate(new_emails):
    pred = "垃圾邮件" if predictions[i] == 1 else "正常邮件"
    prob_spam = probabilities[i][1]
    prob_normal = probabilities[i][0]
    print(f"\n邮件: {email}")
    print(f"  预测: {pred}")
    print(f"  正常邮件概率: {prob_normal:.4f}")
    print(f"  垃圾邮件概率: {prob_spam:.4f}")

# ========== 7. 特征重要性分析 ==========
print("\n" + "=" * 60)
print("7. 特征重要性分析")
print("=" * 60)

# 获取每个词对分类的贡献
# 这里我们分析哪些词更可能出现在垃圾邮件中
feature_names = vectorizer.get_feature_names_out()

# 获取每个类别的特征对数概率
# log_prob_ 是 log P(x_i|y)
log_prob_spam = nb.feature_log_prob_[1]  # 垃圾邮件的特征对数概率
log_prob_normal = nb.feature_log_prob_[0]  # 正常邮件的特征对数概率

# 计算差异，找出对垃圾邮件分类最重要的词
# 差异越大，说明这个词越能区分垃圾邮件
diff = log_prob_spam - log_prob_normal

# 找出最重要的10个词（最可能出现在垃圾邮件中）
top_spam_words_idx = np.argsort(diff)[-10:][::-1]
top_spam_words = [feature_names[i] for i in top_spam_words_idx]

print("\n最可能出现在垃圾邮件中的词（前10个）:")
for i, word in enumerate(top_spam_words, 1):
    idx = np.where(feature_names == word)[0][0]
    print(f"  {i}. {word}: 差异={diff[idx]:.4f}")

# 找出最不可能出现在垃圾邮件中的词（最可能出现在正常邮件中）
top_normal_words_idx = np.argsort(diff)[:10]
top_normal_words = [feature_names[i] for i in top_normal_words_idx]

print("\n最可能出现在正常邮件中的词（前10个）:")
for i, word in enumerate(top_normal_words, 1):
    idx = np.where(feature_names == word)[0][0]
    print(f"  {i}. {word}: 差异={diff[idx]:.4f}")

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. 朴素贝叶斯非常适合文本分类任务
2. 词频特征简单有效
3. 模型训练和预测都很快
4. 可以分析哪些词对分类最重要
""")

