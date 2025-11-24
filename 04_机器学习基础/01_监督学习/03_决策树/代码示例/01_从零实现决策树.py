"""
从零实现决策树（分类树）
"""

import numpy as np
import pandas as pd
from collections import Counter


class DecisionTree:
    """决策树分类器"""
    
    def __init__(self, max_depth=10, min_samples_split=2, criterion='gini'):
        """
        初始化决策树
        
        参数:
        max_depth: 最大深度
        min_samples_split: 内部节点最小样本数
        criterion: 分割准则 ('gini' 或 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.tree = None
    
    def _gini(self, y):
        """计算基尼不纯度"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return 1 - sum(p ** 2 for p in probabilities)
    
    def _entropy(self, y):
        """计算信息熵"""
        if len(y) == 0:
            return 0
        counts = Counter(y)
        probabilities = [count / len(y) for count in counts.values()]
        return -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)
    
    def _impurity(self, y):
        """根据criterion计算不纯度"""
        if self.criterion == 'gini':
            return self._gini(y)
        elif self.criterion == 'entropy':
            return self._entropy(y)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")
    
    def _information_gain(self, y, y_left, y_right):
        """计算信息增益"""
        parent_impurity = self._impurity(y)
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        if n == 0:
            return 0
        
        child_impurity = (n_left / n) * self._impurity(y_left) + \
                        (n_right / n) * self._impurity(y_right)
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """找到最佳分割点"""
        best_gain = 0
        best_feature = None
        best_threshold = None
        
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            # 获取该特征的所有唯一值
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # 尝试每个可能的分割点
            for threshold in unique_values:
                # 分割数据
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                # 计算信息增益
                gain = self._information_gain(y, y[left_mask], y[right_mask])
                
                # 更新最佳分割
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_gain
    
    def _majority_class(self, y):
        """返回多数类"""
        counter = Counter(y)
        return counter.most_common(1)[0][0]
    
    def _build_tree(self, X, y, depth=0):
        """递归构建决策树"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))
        
        # 停止条件
        if (depth >= self.max_depth or
            n_samples < self.min_samples_split or
            n_classes == 1):
            return {'leaf': True, 'class': self._majority_class(y)}
        
        # 找到最佳分割
        best_feature, best_threshold, best_gain = self._best_split(X, y)
        
        # 如果信息增益为0，创建叶节点
        if best_gain == 0:
            return {'leaf': True, 'class': self._majority_class(y)}
        
        # 分割数据
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        # 递归构建左右子树
        left_tree = self._build_tree(X[left_mask], y[left_mask], depth + 1)
        right_tree = self._build_tree(X[right_mask], y[right_mask], depth + 1)
        
        return {
            'leaf': False,
            'feature': best_feature,
            'threshold': best_threshold,
            'left': left_tree,
            'right': right_tree
        }
    
    def fit(self, X, y):
        """训练决策树"""
        X = np.array(X)
        y = np.array(y)
        self.tree = self._build_tree(X, y)
        return self
    
    def _predict_sample(self, x, node):
        """对单个样本进行预测"""
        if node['leaf']:
            return node['class']
        
        if x[node['feature']] <= node['threshold']:
            return self._predict_sample(x, node['left'])
        else:
            return self._predict_sample(x, node['right'])
    
    def predict(self, X):
        """预测"""
        X = np.array(X)
        return np.array([self._predict_sample(x, self.tree) for x in X])


# ========== 使用示例 ==========
if __name__ == "__main__":
    # 创建示例数据
    from sklearn.datasets import make_classification
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    # 生成数据
    X, y = make_classification(n_samples=200, n_features=4, n_informative=2,
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 训练决策树
    print("训练决策树（基尼不纯度）...")
    dt_gini = DecisionTree(max_depth=5, criterion='gini')
    dt_gini.fit(X_train, y_train)
    
    # 预测
    y_pred = dt_gini.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")
    
    # 使用信息熵
    print("\n训练决策树（信息熵）...")
    dt_entropy = DecisionTree(max_depth=5, criterion='entropy')
    dt_entropy.fit(X_train, y_train)
    
    y_pred = dt_entropy.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"准确率: {accuracy:.4f}")

