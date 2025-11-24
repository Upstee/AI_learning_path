"""
从零实现KNN分类器
本示例详细展示了如何从零开始实现一个KNN分类器
适合小白学习，包含大量注释和解释
"""

import numpy as np
from collections import Counter
import math


class KNN:
    """
    K近邻分类器
    
    这个类实现了KNN算法的核心功能
    支持分类任务，可以扩展到回归任务
    """
    
    def __init__(self, k=3, distance_metric='euclidean', weights='uniform'):
        """
        初始化KNN分类器
        
        参数说明：
        ----------
        k : int, 默认=3
            K值，即选择最近的K个邻居
            为什么选择3？
            - 通常选择奇数，避免平票
            - 3是一个常见的起点
            - 可以通过交叉验证选择最佳K值
        
        distance_metric : str, 默认='euclidean'
            距离度量方法
            - 'euclidean': 欧氏距离（直线距离）
            - 'manhattan': 曼哈顿距离（城市街区距离）
            - 'minkowski': 闵可夫斯基距离（通用形式）
        
        weights : str, 默认='uniform'
            权重方式
            - 'uniform': 所有邻居权重相同
            - 'distance': 距离越近，权重越大
        """
        self.k = k
        self.distance_metric = distance_metric
        self.weights = weights
        
        # 存储训练数据
        # X_train: 训练特征矩阵
        # y_train: 训练标签
        self.X_train = None
        self.y_train = None
    
    def _euclidean_distance(self, x1, x2):
        """
        计算欧氏距离
        
        公式：d = sqrt(sum((x1_i - x2_i)^2))
        
        参数说明：
        ----------
        x1 : array-like
            第一个样本的特征向量
        
        x2 : array-like
            第二个样本的特征向量
        
        返回：
        ------
        distance : float
            欧氏距离
        """
        # 转换为numpy数组，方便计算
        x1 = np.array(x1)
        x2 = np.array(x2)
        
        # 计算差的平方
        # (x1 - x2) 计算每个特征的差值
        # **2 计算平方
        diff_squared = (x1 - x2) ** 2
        
        # 求和并开方
        # np.sum() 对所有特征求和
        # np.sqrt() 开平方根
        distance = np.sqrt(np.sum(diff_squared))
        
        return distance
    
    def _manhattan_distance(self, x1, x2):
        """
        计算曼哈顿距离
        
        公式：d = sum(|x1_i - x2_i|)
        
        参数说明：
        ----------
        x1 : array-like
            第一个样本的特征向量
        
        x2 : array-like
            第二个样本的特征向量
        
        返回：
        ------
        distance : float
            曼哈顿距离
        """
        x1 = np.array(x1)
        x2 = np.array(x2)
        
        # 计算绝对差值
        # np.abs() 计算绝对值
        # np.sum() 对所有特征求和
        distance = np.sum(np.abs(x1 - x2))
        
        return distance
    
    def _compute_distance(self, x1, x2):
        """
        根据指定的距离度量计算距离
        
        参数说明：
        ----------
        x1 : array-like
            第一个样本
        
        x2 : array-like
            第二个样本
        
        返回：
        ------
        distance : float
            距离值
        """
        if self.distance_metric == 'euclidean':
            return self._euclidean_distance(x1, x2)
        elif self.distance_metric == 'manhattan':
            return self._manhattan_distance(x1, x2)
        else:
            raise ValueError(f"不支持的距离度量: {self.distance_metric}")
    
    def fit(self, X, y):
        """
        训练KNN分类器
        
        注意：KNN是"懒惰学习"，训练时不做任何计算
        只是存储训练数据，所有计算都在预测时进行
        
        参数说明：
        ----------
        X : array-like, shape = [n_samples, n_features]
            训练特征矩阵
            每一行是一个样本，每一列是一个特征
        
        y : array-like, shape = [n_samples]
            训练标签（类别）
        """
        # 转换为numpy数组
        self.X_train = np.array(X)
        self.y_train = np.array(y)
        
        print(f"KNN分类器训练完成（懒惰学习，只存储数据）")
        print(f"  训练样本数: {len(self.X_train)}")
        print(f"  特征数: {self.X_train.shape[1]}")
        print(f"  类别数: {len(np.unique(self.y_train))}")
        print(f"  K值: {self.k}")
        print(f"  距离度量: {self.distance_metric}")
        print(f"  权重方式: {self.weights}")
        
        return self
    
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
        # 转换为numpy数组
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 存储预测结果
        predictions = []
        
        # 遍历每个测试样本
        for i in range(n_samples):
            # 获取当前测试样本
            x_test = X[i]
            
            # 计算与所有训练样本的距离
            distances = []
            for j in range(len(self.X_train)):
                # 计算距离
                dist = self._compute_distance(x_test, self.X_train[j])
                distances.append((dist, self.y_train[j]))
            
            # 按距离排序，选择最近的K个
            # sorted() 按距离从小到大排序
            # [:self.k] 选择前K个
            k_nearest = sorted(distances, key=lambda x: x[0])[:self.k]
            
            # 提取K个最近邻居的标签
            k_labels = [label for _, label in k_nearest]
            
            # 根据权重方式预测
            if self.weights == 'uniform':
                # 均匀权重：简单投票
                # Counter 统计每个类别出现的次数
                # most_common(1) 返回出现次数最多的类别
                prediction = Counter(k_labels).most_common(1)[0][0]
            elif self.weights == 'distance':
                # 距离权重：距离越近，权重越大
                # 权重 = 1 / 距离（距离为0时，权重为1）
                weighted_votes = {}
                for dist, label in k_nearest:
                    # 避免除零错误
                    weight = 1.0 / (dist + 1e-10)
                    if label not in weighted_votes:
                        weighted_votes[label] = 0
                    weighted_votes[label] += weight
                
                # 选择权重最大的类别
                prediction = max(weighted_votes, key=weighted_votes.get)
            else:
                raise ValueError(f"不支持的权重方式: {self.weights}")
            
            predictions.append(prediction)
        
        return np.array(predictions)
    
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
        X = np.array(X)
        n_samples = X.shape[0]
        
        # 获取所有唯一的类别
        classes = np.unique(self.y_train)
        n_classes = len(classes)
        
        # 初始化概率矩阵
        probabilities = np.zeros((n_samples, n_classes))
        
        # 遍历每个测试样本
        for i in range(n_samples):
            x_test = X[i]
            
            # 计算与所有训练样本的距离
            distances = []
            for j in range(len(self.X_train)):
                dist = self._compute_distance(x_test, self.X_train[j])
                distances.append((dist, self.y_train[j]))
            
            # 选择最近的K个
            k_nearest = sorted(distances, key=lambda x: x[0])[:self.k]
            
            # 计算每个类别的权重
            class_weights = {}
            total_weight = 0
            
            for dist, label in k_nearest:
                if self.weights == 'uniform':
                    weight = 1.0
                elif self.weights == 'distance':
                    weight = 1.0 / (dist + 1e-10)
                else:
                    weight = 1.0
                
                if label not in class_weights:
                    class_weights[label] = 0
                class_weights[label] += weight
                total_weight += weight
            
            # 计算概率
            for j, c in enumerate(classes):
                if c in class_weights:
                    probabilities[i, j] = class_weights[c] / total_weight
                else:
                    probabilities[i, j] = 0.0
        
        return probabilities


# ========== 示例：使用KNN分类器 ==========
if __name__ == "__main__":
    print("=" * 60)
    print("KNN分类器示例")
    print("=" * 60)
    
    # ========== 1. 创建简单的训练数据 ==========
    print("\n1. 准备训练数据")
    print("-" * 60)
    
    # 使用简单的二维数据，方便可视化
    # 特征1和特征2
    X_train = np.array([
        [1, 2],  # 类别0
        [2, 3],  # 类别0
        [3, 1],  # 类别0
        [6, 5],  # 类别1
        [7, 6],  # 类别1
        [8, 4],  # 类别1
    ])
    
    # 标签：0或1
    y_train = np.array([0, 0, 0, 1, 1, 1])
    
    print(f"训练特征形状: {X_train.shape}")
    print(f"训练标签: {y_train}")
    print(f"类别: {np.unique(y_train)}")
    
    # ========== 2. 训练模型 ==========
    print("\n2. 训练模型")
    print("-" * 60)
    
    # 创建KNN分类器
    # k=3 表示选择最近的3个邻居
    # distance_metric='euclidean' 使用欧氏距离
    # weights='uniform' 所有邻居权重相同
    knn = KNN(k=3, distance_metric='euclidean', weights='uniform')
    
    # 训练（实际上只是存储数据）
    knn.fit(X_train, y_train)
    
    # ========== 3. 预测 ==========
    print("\n3. 预测新样本")
    print("-" * 60)
    
    # 测试样本
    X_test = np.array([
        [2, 2],  # 应该预测为类别0
        [7, 5],  # 应该预测为类别1
        [4, 3],  # 中间位置，可能预测为类别0或1
    ])
    
    # 预测类别
    predictions = knn.predict(X_test)
    print(f"\n预测类别: {predictions}")
    
    # 预测概率
    probabilities = knn.predict_proba(X_test)
    print(f"\n预测概率:")
    for i, probs in enumerate(probabilities):
        print(f"  样本 {i+1}: 类别0={probs[0]:.4f}, 类别1={probs[1]:.4f}")
    
    # ========== 4. 测试不同参数 ==========
    print("\n4. 测试不同参数")
    print("-" * 60)
    
    # 测试不同的K值
    for k in [1, 3, 5]:
        knn_k = KNN(k=k)
        knn_k.fit(X_train, y_train)
        pred_k = knn_k.predict(X_test)
        print(f"K={k}: 预测={pred_k}")
    
    # 测试不同的距离度量
    for metric in ['euclidean', 'manhattan']:
        knn_m = KNN(k=3, distance_metric=metric)
        knn_m.fit(X_train, y_train)
        pred_m = knn_m.predict(X_test)
        print(f"距离度量={metric}: 预测={pred_m}")
    
    # 测试加权KNN
    knn_w = KNN(k=3, weights='distance')
    knn_w.fit(X_train, y_train)
    pred_w = knn_w.predict(X_test)
    print(f"加权KNN: 预测={pred_w}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)
    print("""
总结：
1. KNN是"懒惰学习"，训练时只存储数据
2. 预测时需要计算与所有训练样本的距离
3. K值、距离度量、权重方式都会影响结果
4. 适合小规模数据，大规模数据需要优化
""")

