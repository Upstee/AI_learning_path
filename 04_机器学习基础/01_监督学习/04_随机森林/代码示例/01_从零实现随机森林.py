"""
从零实现随机森林（分类）
基于之前实现的决策树
"""

import numpy as np
from collections import Counter
from decision_tree import DecisionTree  # 假设已有决策树实现


class RandomForest:
    """随机森林分类器"""
    
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features='sqrt', random_state=None):
        """
        初始化随机森林
        
        参数:
        n_estimators: 树的数量
        max_depth: 树的最大深度
        min_samples_split: 内部节点最小样本数
        max_features: 每次分割考虑的特征数（'sqrt', 'log2', 或整数）
        random_state: 随机种子
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []  # 存储每棵树使用的特征索引
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def _bootstrap_sample(self, X, y):
        """Bootstrap采样"""
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indices], y[indices], indices
    
    def _get_max_features(self, n_features):
        """获取每次分割考虑的特征数"""
        if self.max_features == 'sqrt':
            return int(np.sqrt(n_features))
        elif self.max_features == 'log2':
            return int(np.log2(n_features))
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        else:
            return n_features
    
    def fit(self, X, y):
        """训练随机森林"""
        X = np.array(X)
        y = np.array(y)
        n_samples, n_features = X.shape
        
        self.trees = []
        self.feature_indices = []
        
        print(f"训练 {self.n_estimators} 棵树...")
        for i in range(self.n_estimators):
            if (i + 1) % 10 == 0:
                print(f"  已完成 {i + 1}/{self.n_estimators} 棵树")
            
            # Bootstrap采样
            X_boot, y_boot, _ = self._bootstrap_sample(X, y)
            
            # 随机选择特征
            max_features = self._get_max_features(n_features)
            feature_idx = np.random.choice(n_features, size=max_features, replace=False)
            X_boot = X_boot[:, feature_idx]
            
            # 训练决策树
            tree = DecisionTree(
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state
            )
            tree.fit(X_boot, y_boot)
            
            self.trees.append(tree)
            self.feature_indices.append(feature_idx)
        
        print("训练完成！")
        return self
    
    def predict(self, X):
        """预测"""
        X = np.array(X)
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_estimators))
        
        # 每棵树进行预测
        for i, (tree, feature_idx) in enumerate(zip(self.trees, self.feature_indices)):
            X_subset = X[:, feature_idx]
            predictions[:, i] = tree.predict(X_subset)
        
        # 多数投票
        final_predictions = []
        for i in range(n_samples):
            votes = predictions[i, :]
            final_predictions.append(Counter(votes).most_common(1)[0][0])
        
        return np.array(final_predictions)
    
    def predict_proba(self, X):
        """预测概率"""
        X = np.array(X)
        n_samples = X.shape[0]
        n_classes = len(np.unique([tree.predict(X[:, idx])[0] 
                                   for tree, idx in zip(self.trees[:1], self.feature_indices[:1])]))
        probabilities = np.zeros((n_samples, n_classes))
        
        # 每棵树进行预测
        for tree, feature_idx in zip(self.trees, self.feature_indices):
            X_subset = X[:, feature_idx]
            pred = tree.predict(X_subset)
            for i, p in enumerate(pred):
                probabilities[i, int(p)] += 1
        
        # 归一化
        probabilities /= self.n_estimators
        return probabilities


# ========== 使用示例 ==========
if __name__ == "__main__":
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 生成数据
    X, y = make_classification(n_samples=1000, n_features=20, n_informative=10,
                               n_redundant=10, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练随机森林
    print("训练随机森林...")
    rf = RandomForest(n_estimators=50, max_depth=10, max_features='sqrt', random_state=42)
    rf.fit(X_train, y_train)
    
    # 预测
    y_pred = rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\n准确率: {accuracy:.4f}")

