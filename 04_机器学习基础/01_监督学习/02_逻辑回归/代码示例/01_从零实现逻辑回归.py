"""
从零实现逻辑回归
不使用任何机器学习库，从零实现逻辑回归算法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class LogisticRegression:
    """逻辑回归类"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, regularization=None, lambda_reg=0.1):
        """
        初始化逻辑回归模型
        
        参数:
        learning_rate: 学习率
        max_iter: 最大迭代次数
        tol: 收敛容差
        regularization: 正则化类型 ('l1', 'l2', None)
        lambda_reg: 正则化系数
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.regularization = regularization
        self.lambda_reg = lambda_reg
        self.theta = None
        self.cost_history = []
    
    def sigmoid(self, z):
        """
        Sigmoid函数
        
        参数:
        z: 输入值
        
        返回:
        Sigmoid函数值
        """
        # 防止溢出
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def compute_cost(self, X, y):
        """
        计算损失函数（交叉熵）
        
        参数:
        X: 特征矩阵 (m, n+1)
        y: 标签向量 (m,)
        
        返回:
        损失值
        """
        m = X.shape[0]
        h = self.sigmoid(X @ self.theta)
        
        # 交叉熵损失
        cost = -(1/m) * np.sum(y * np.log(h + 1e-15) + (1-y) * np.log(1-h + 1e-15))
        
        # 正则化项
        if self.regularization == 'l2':
            cost += (self.lambda_reg / (2*m)) * np.sum(self.theta[1:]**2)
        elif self.regularization == 'l1':
            cost += (self.lambda_reg / m) * np.sum(np.abs(self.theta[1:]))
        
        return cost
    
    def compute_gradient(self, X, y):
        """
        计算梯度
        
        参数:
        X: 特征矩阵 (m, n+1)
        y: 标签向量 (m,)
        
        返回:
        梯度向量
        """
        m = X.shape[0]
        h = self.sigmoid(X @ self.theta)
        
        # 基本梯度
        gradient = (1/m) * X.T @ (h - y)
        
        # 正则化项
        if self.regularization == 'l2':
            gradient[1:] += (self.lambda_reg / m) * self.theta[1:]
        elif self.regularization == 'l1':
            # L1正则化的次梯度
            gradient[1:] += (self.lambda_reg / m) * np.sign(self.theta[1:])
        
        return gradient
    
    def fit(self, X, y):
        """
        训练模型
        
        参数:
        X: 特征矩阵 (m, n)
        y: 标签向量 (m,)
        """
        # 添加偏置项
        m, n = X.shape
        X_with_bias = np.hstack([np.ones((m, 1)), X])
        
        # 初始化参数
        self.theta = np.zeros(n + 1)
        
        # 梯度下降
        for i in range(self.max_iter):
            # 计算损失
            cost = self.compute_cost(X_with_bias, y)
            self.cost_history.append(cost)
            
            # 计算梯度
            gradient = self.compute_gradient(X_with_bias, y)
            
            # 更新参数
            self.theta -= self.learning_rate * gradient
            
            # 检查收敛
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                print(f"在第 {i+1} 次迭代后收敛")
                break
        
        return self
    
    def predict_proba(self, X):
        """
        预测概率
        
        参数:
        X: 特征矩阵 (m, n)
        
        返回:
        概率向量 (m,)
        """
        m = X.shape[0]
        X_with_bias = np.hstack([np.ones((m, 1)), X])
        return self.sigmoid(X_with_bias @ self.theta)
    
    def predict(self, X, threshold=0.5):
        """
        预测类别
        
        参数:
        X: 特征矩阵 (m, n)
        threshold: 决策阈值
        
        返回:
        预测标签 (m,)
        """
        probabilities = self.predict_proba(X)
        return (probabilities >= threshold).astype(int)
    
    def plot_cost_history(self):
        """绘制损失函数历史"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('损失函数收敛曲线')
        plt.grid(True)
        plt.show()


def plot_decision_boundary(X, y, model):
    """绘制决策边界"""
    # 创建网格
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 预测网格点
    Z = model.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 绘制
    plt.figure(figsize=(10, 8))
    plt.contourf(xx, yy, Z, levels=50, alpha=0.5, cmap='RdYlBu')
    plt.contour(xx, yy, Z, levels=[0.5], colors='black', linewidths=2)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlBu', edgecolors='black')
    plt.xlabel('特征1')
    plt.ylabel('特征2')
    plt.title('逻辑回归决策边界')
    plt.colorbar()
    plt.show()


# 使用示例
if __name__ == "__main__":
    # 生成分类数据
    X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, 
                               n_informative=2, n_clusters_per_class=1, 
                               random_state=42)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 创建和训练模型
    model = LogisticRegression(learning_rate=0.1, max_iter=1000, regularization='l2', lambda_reg=0.1)
    model.fit(X_train, y_train)
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 评估
    accuracy = np.mean(y_pred == y_test)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 绘制损失函数历史
    model.plot_cost_history()
    
    # 绘制决策边界
    plot_decision_boundary(X_train, y_train, model)
    
    # 打印参数
    print(f"\n模型参数: {model.theta}")
    print(f"偏置项: {model.theta[0]:.4f}")
    print(f"权重: {model.theta[1:]}")

