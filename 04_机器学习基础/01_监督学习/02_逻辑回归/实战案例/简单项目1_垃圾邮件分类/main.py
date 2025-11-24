"""
简单项目1：垃圾邮件分类
使用逻辑回归构建垃圾邮件分类器
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def load_data():
    """加载数据"""
    # 使用20newsgroups数据集的部分类别作为示例
    # 在实际应用中，应该使用真实的垃圾邮件数据集
    categories = ['alt.atheism', 'soc.religion.christian']
    
    print("加载数据...")
    newsgroups_train = fetch_20newsgroups(subset='train', categories=categories, 
                                          shuffle=True, random_state=42)
    newsgroups_test = fetch_20newsgroups(subset='test', categories=categories, 
                                         shuffle=True, random_state=42)
    
    X_train = newsgroups_train.data
    y_train = newsgroups_train.target
    X_test = newsgroups_test.data
    y_test = newsgroups_test.target
    
    print(f"训练集大小: {len(X_train)}")
    print(f"测试集大小: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test


def extract_features(X_train, X_test):
    """特征提取：TF-IDF向量化"""
    print("\n特征提取...")
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', 
                                lowercase=True, strip_accents='unicode')
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    print(f"特征维度: {X_train_tfidf.shape[1]}")
    
    return X_train_tfidf, X_test_tfidf, vectorizer


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
    
    # 准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, 
                                target_names=['类别0', '类别1']))
    
    # 混淆矩阵
    cm = confusion_matrix(y_test, y_pred)
    print("\n混淆矩阵:")
    print(cm)
    
    # 可视化混淆矩阵
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['类别0', '类别1'],
                yticklabels=['类别0', '类别1'])
    plt.title('混淆矩阵')
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300)
    plt.show()
    
    return y_pred, y_proba


def analyze_features(model, vectorizer, top_n=20):
    """分析重要特征"""
    print("\n分析重要特征...")
    feature_names = vectorizer.get_feature_names_out()
    coefficients = model.coef_[0]
    
    # 获取最重要的特征
    top_positive = np.argsort(coefficients)[-top_n:][::-1]
    top_negative = np.argsort(coefficients)[:top_n]
    
    print(f"\n最重要的 {top_n} 个正特征（倾向于类别1）:")
    for idx in top_positive:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")
    
    print(f"\n最重要的 {top_n} 个负特征（倾向于类别0）:")
    for idx in top_negative:
        print(f"  {feature_names[idx]}: {coefficients[idx]:.4f}")


def main():
    """主函数"""
    print("=" * 50)
    print("垃圾邮件分类项目")
    print("=" * 50)
    
    # 1. 加载数据
    X_train, X_test, y_train, y_test = load_data()
    
    # 2. 特征提取
    X_train_tfidf, X_test_tfidf, vectorizer = extract_features(X_train, X_test)
    
    # 3. 训练模型
    model = train_model(X_train_tfidf, y_train)
    
    # 4. 评估模型
    y_pred, y_proba = evaluate_model(model, X_test_tfidf, y_test)
    
    # 5. 分析特征
    analyze_features(model, vectorizer)
    
    print("\n" + "=" * 50)
    print("项目完成！")
    print("=" * 50)


if __name__ == "__main__":
    main()

