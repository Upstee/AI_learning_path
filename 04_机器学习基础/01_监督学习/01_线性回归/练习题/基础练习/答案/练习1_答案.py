"""
基础练习1答案：从零实现线性回归
"""

import numpy as np
import matplotlib.pyplot as plt


class LinearRegression:
    """线性回归类（从零实现）"""
    
    def __init__(self, learning_rate=0.01, max_iter=1000, tol=1e-6, random_state=42):
        """
        参数:
        - learning_rate: 学习率
        - max_iter: 最大迭代次数
        - tol: 收敛容差
        - random_state: 随机种子
        """
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w = None
        self.b = None
        self.cost_history = []
    
    def fit(self, X, y):
        """训练模型"""
        m, n = X.shape
        
        # 初始化参数
        np.random.seed(self.random_state)
        self.w = np.random.randn(n) * 0.01
        self.b = 0
        
        # 梯度下降迭代
        for i in range(self.max_iter):
            # 预测
            y_pred = X @ self.w + self.b
            
            # 计算损失
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1 / m) * X.T @ (y_pred - y)
            db = (1 / m) * np.sum(y_pred - y)
            
            # 更新参数
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db
            
            # 检查收敛
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < self.tol:
                print(f"在第 {i+1} 次迭代后收敛")
                break
        
        return self
    
    def predict(self, X):
        """预测"""
        if self.w is None or self.b is None:
            raise ValueError("模型尚未训练，请先调用 fit()")
        return X @ self.w + self.b
    
    def score(self, X, y):
        """计算R²分数"""
        y_pred = self.predict(X)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot != 0 else 0


def test_implementation():
    """测试实现"""
    print("=" * 60)
    print("测试从零实现的线性回归")
    print("=" * 60)
    
    # 生成测试数据
    np.random.seed(42)
    X = np.random.randn(100, 1) * 10
    y = 2 * X.flatten() + 1 + np.random.randn(100) * 2
    
    # 创建模型并训练
    model = LinearRegression(learning_rate=0.01, max_iter=1000)
    model.fit(X, y)
    
    # 预测
    y_pred = model.predict(X)
    
    # 评估
    r2 = model.score(X, y)
    mse = np.mean((y - y_pred) ** 2)
    
    print(f"\n评估结果:")
    print(f"  MSE: {mse:.4f}")
    print(f"  R²分数: {r2:.4f}")
    print(f"\n学习到的参数:")
    print(f"  权重 w: {model.w[0]:.4f}")
    print(f"  偏置 b: {model.b:.4f}")
    print(f"\n真实参数:")
    print(f"  权重 w: 2.0")
    print(f"  偏置 b: 1.0")
    print(f"\n误差:")
    print(f"  权重误差: {abs(model.w[0] - 2.0):.4f}")
    print(f"  偏置误差: {abs(model.b - 1.0):.4f}")
    
    # 可视化
    plt.figure(figsize=(12, 5))
    
    # 左图：数据点和拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='数据点')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, 
             label=f'拟合直线: y={model.w[0]:.4f}x+{model.b:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title('线性回归结果')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：损失函数收敛过程
    plt.subplot(1, 2, 2)
    plt.plot(model.cost_history)
    plt.xlabel('迭代次数')
    plt.ylabel('损失')
    plt.title('损失函数收敛过程')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 判断是否通过
    if r2 > 0.9 and abs(model.w[0] - 2.0) < 0.5 and abs(model.b - 1.0) < 0.5:
        print("\n✅ 测试通过！")
    else:
        print("\n❌ 测试未通过，请检查实现")


if __name__ == "__main__":
    test_implementation()

