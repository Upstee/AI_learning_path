"""
从零实现朴素贝叶斯分类器
本示例详细展示了如何从零开始实现一个朴素贝叶斯分类器
适合小白学习，包含大量注释和解释
"""

import numpy as np
from collections import defaultdict
import math


class NaiveBayes:
    """
    朴素贝叶斯分类器（多项式版本）
    
    这个类实现了朴素贝叶斯算法的核心功能
    适合文本分类等离散特征问题
    """
    
    def __init__(self, alpha=1.0):
        """
        初始化朴素贝叶斯分类器
        
        参数说明：
        ----------
        alpha : float, 默认=1.0
            平滑参数（拉普拉斯平滑）
            为什么需要平滑？
            - 如果某个特征在训练集中从未出现，概率为0
            - 这会导致整个乘积为0，无法分类
            - 平滑确保所有概率都大于0
        """
        self.alpha = alpha  # 保存平滑参数
        # 存储每个类别的先验概率 P(y)
        # 例如：{'类别1': 0.3, '类别2': 0.7}
        self.prior = {}
        
        # 存储每个类别下每个特征的条件概率 P(x_i|y)
        # 例如：{'类别1': {特征1: 0.2, 特征2: 0.8}, ...}
        self.conditional = {}
        
        # 存储类别列表，方便后续使用
        self.classes = []
    
    def fit(self, X, y):
        """
        训练朴素贝叶斯分类器
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练特征矩阵
            每一行是一个样本，每一列是一个特征
            例如：[[1, 0, 1], [0, 1, 1], ...]
        
        y : array-like, shape = [n_samples]
            训练标签（类别）
            例如：[0, 1, 0, 1, ...]
        
        训练过程：
        --------
        1. 计算每个类别的先验概率 P(y)
        2. 计算每个类别下每个特征的条件概率 P(x_i|y)
        """
        # 将输入转换为numpy数组，方便计算
        X = np.array(X)
        y = np.array(y)
        
        # 获取所有唯一的类别
        # 例如：[0, 1] 或 ['spam', 'ham']
        self.classes = np.unique(y)
        
        # 获取样本数和特征数
        n_samples, n_features = X.shape
        
        print(f"开始训练朴素贝叶斯分类器...")
        print(f"  样本数: {n_samples}")
        print(f"  特征数: {n_features}")
        print(f"  类别数: {len(self.classes)}")
        
        # ========== 步骤1：计算先验概率 P(y) ==========
        print("\n步骤1：计算先验概率 P(y)")
        print("-" * 50)
        
        # 遍历每个类别
        for c in self.classes:
            # 计算类别c的样本数
            # y == c 返回一个布尔数组，True表示该样本属于类别c
            # np.sum() 统计True的数量，即类别c的样本数
            n_class = np.sum(y == c)
            
            # 计算先验概率：类别c的样本数 / 总样本数
            # 这就是 P(y=c)
            self.prior[c] = n_class / n_samples
            
            print(f"  类别 {c}: {n_class}/{n_samples} = {self.prior[c]:.4f}")
        
        # ========== 步骤2：计算条件概率 P(x_i|y) ==========
        print("\n步骤2：计算条件概率 P(x_i|y)")
        print("-" * 50)
        
        # 初始化条件概率字典
        # 为每个类别创建一个字典，存储该类别下每个特征的概率
        self.conditional = {c: {} for c in self.classes}
        
        # 遍历每个类别
        for c in self.classes:
            # 获取属于类别c的所有样本
            # X[y == c] 选择所有标签为c的样本
            X_class = X[y == c]
            
            # 计算类别c中所有特征的总出现次数
            # 例如：如果特征值是词频，这就是该类别的总词数
            total_count = np.sum(X_class)
            
            print(f"\n  类别 {c}:")
            print(f"    总特征计数: {total_count}")
            
            # 遍历每个特征
            for i in range(n_features):
                # 计算特征i在类别c中的总出现次数
                # X_class[:, i] 选择所有样本的特征i
                # np.sum() 求和，得到特征i在类别c中的总次数
                feature_count = np.sum(X_class[:, i])
                
                # 计算条件概率，使用拉普拉斯平滑
                # 公式：P(x_i|y) = (N_{y,i} + alpha) / (N_y + alpha * n_features)
                # 其中：
                #   N_{y,i} = feature_count（特征i在类别y中的次数）
                #   N_y = total_count（类别y中所有特征的总次数）
                #   alpha = self.alpha（平滑参数）
                #   n_features = 特征总数
                prob = (feature_count + self.alpha) / (total_count + self.alpha * n_features)
                
                # 存储条件概率
                self.conditional[c][i] = prob
                
                # 打印前5个特征的概率（避免输出太多）
                if i < 5:
                    print(f"    特征 {i}: {feature_count}/{total_count} -> {prob:.4f}")
        
        print("\n训练完成！")
        return self
    
    def predict_proba(self, X):
        """
        预测每个类别的概率
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            测试特征矩阵
        
        返回：
        ------
        probabilities : array, shape = [n_samples, n_classes]
            每个样本属于每个类别的概率
        """
        # 转换为numpy数组
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 初始化概率矩阵
        # 每一行是一个样本，每一列是一个类别
        probabilities = np.zeros((n_samples, len(self.classes)))
        
        # 遍历每个样本
        for i in range(n_samples):
            # 获取当前样本的特征
            x = X[i]
            
            # 遍历每个类别
            for j, c in enumerate(self.classes):
                # 计算 log P(y=c|x)
                # 使用对数避免数值下溢
                # log P(y|x) = log P(y) + sum(log P(x_i|y))
                
                # 第一部分：log P(y=c)
                log_prob = math.log(self.prior[c])
                
                # 第二部分：sum(log P(x_i|y=c))
                # 遍历每个特征
                for k in range(len(x)):
                    # 获取特征k的值
                    feature_value = x[k]
                    
                    # 如果特征值大于0（特征出现）
                    if feature_value > 0:
                        # 获取条件概率 P(x_k|y=c)
                        cond_prob = self.conditional[c].get(k, self.alpha / (self.alpha * len(x)))
                        
                        # 累加 log P(x_k|y=c)
                        # 注意：这里假设特征值就是出现次数
                        # 如果是二值特征，应该用不同的公式
                        log_prob += feature_value * math.log(cond_prob)
                
                # 存储对数概率（稍后会转换为概率）
                probabilities[i, j] = log_prob
        
        # 将对数概率转换为概率
        # 使用log-sum-exp技巧避免数值问题
        # 公式：exp(log_prob - max(log_prob)) / sum(exp(log_prob - max(log_prob)))
        max_log_probs = np.max(probabilities, axis=1, keepdims=True)
        probabilities = np.exp(probabilities - max_log_probs)
        probabilities = probabilities / np.sum(probabilities, axis=1, keepdims=True)
        
        return probabilities
    
    def predict(self, X):
        """
        预测类别
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            测试特征矩阵
        
        返回：
        ------
        predictions : array, shape = [n_samples]
            预测的类别
        """
        # 获取每个类别的概率
        probabilities = self.predict_proba(X)
        
        # 选择概率最大的类别
        # np.argmax() 返回最大值的索引
        # axis=1 表示在每一行（每个样本）中找最大值
        predictions = self.classes[np.argmax(probabilities, axis=1)]
        
        return predictions


# ========== 示例：使用朴素贝叶斯分类器 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("朴素贝叶斯分类器示例")
    print("=" * 60)
    
    # ========== 创建简单的训练数据 ==========
    # 这是一个文本分类的简化示例
    # 特征表示词频（每个特征是一个词的出现次数）
    # 类别：0=正常邮件，1=垃圾邮件
    
    print("\n1. 准备训练数据")
    print("-" * 60)
    
    # 训练特征：每个样本是一个文档的词频向量
    # 例如：[词1出现次数, 词2出现次数, 词3出现次数, ...]
    X_train = np.array([
        [2, 1, 0, 0, 1],  # 文档1：词1出现2次，词2出现1次，...
        [1, 2, 1, 0, 0],  # 文档2
        [0, 1, 2, 1, 0],  # 文档3
        [1, 0, 1, 2, 1],  # 文档4
        [0, 0, 1, 1, 2],  # 文档5
    ])
    
    # 训练标签：每个样本的类别
    y_train = np.array([0, 0, 1, 1, 1])  # 前2个是正常邮件，后3个是垃圾邮件
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"训练标签: {y_train}")
    print(f"类别: {np.unique(y_train)}")
    
    # ========== 训练模型 ==========
    print("\n2. 训练模型")
    print("-" * 60)
    
    # 创建分类器实例
    # alpha=1.0 是默认的平滑参数
    nb = NaiveBayes(alpha=1.0)
    
    # 训练模型
    nb.fit(X_train, y_train)
    
    # ========== 预测 ==========
    print("\n3. 预测新样本")
    print("-" * 60)
    
    # 测试样本
    X_test = np.array([
        [1, 1, 0, 0, 1],  # 新文档1
        [0, 0, 2, 1, 1],  # 新文档2
    ])
    
    # 预测概率
    probabilities = nb.predict_proba(X_test)
    print(f"\n预测概率:")
    for i, probs in enumerate(probabilities):
        print(f"  样本 {i+1}:")
        for j, c in enumerate(nb.classes):
            print(f"    类别 {c}: {probs[j]:.4f}")
    
    # 预测类别
    predictions = nb.predict(X_test)
    print(f"\n预测类别: {predictions}")
    
    # ========== 解释结果 ==========
    print("\n4. 结果解释")
    print("-" * 60)
    print("""
    结果说明：
    - 每个样本都有一个概率分布，表示属于每个类别的概率
    - 选择概率最大的类别作为预测结果
    - 概率值可以帮助我们了解模型对预测的置信度
    """)
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)

