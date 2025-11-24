"""
从零实现线性回归算法
包含梯度下降优化和最小二乘法两种方法
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


class LinearRegression:
    """线性回归类（从零实现）"""
    
    def __init__(self, method='gradient_descent', learning_rate=0.01, 
                 max_iter=1000, tol=1e-6, random_state=42):
        """
        参数:
        - method: 优化方法，'gradient_descent' 或 'least_squares'
        - learning_rate: 学习率（仅用于梯度下降）
        - max_iter: 最大迭代次数（仅用于梯度下降）
        - tol: 收敛容差（仅用于梯度下降）
        - random_state: 随机种子
        """
        self.method = method
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.w = None
        self.b = None
        self.cost_history = []
    
    def _add_bias(self, X):
        """添加偏置项（在特征矩阵前加一列1）"""
        m = X.shape[0]
        return np.hstack([np.ones((m, 1)), X])
    
    def _least_squares(self, X, y):
        """最小二乘法（解析解）"""
        # 添加偏置项
        X_bias = self._add_bias(X)
        
        # 计算 (X^T X)^(-1) X^T y
        try:
            # 使用伪逆提高数值稳定性
            theta = np.linalg.pinv(X_bias.T @ X_bias) @ X_bias.T @ y
        except np.linalg.LinAlgError:
            # 如果矩阵奇异，使用伪逆
            theta = np.linalg.pinv(X_bias) @ y
        
        # 分离偏置和权重
        self.b = theta[0]
        self.w = theta[1:]
        
        return self
    
    def _gradient_descent(self, X, y):
        """梯度下降法（数值解）"""
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
    
    def fit(self, X, y):
        """训练模型"""
        if self.method == 'least_squares':
            return self._least_squares(X, y)
        elif self.method == 'gradient_descent':
            return self._gradient_descent(X, y)
        else:
            raise ValueError(f"未知的方法: {self.method}")
    
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


def generate_data(n_samples=100, noise=0.1, random_state=42):
    """生成模拟数据"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, 1) * 10
    y = 2 * X.flatten() + 1 + np.random.randn(n_samples) * noise * 10
    return X, y


def visualize_results(X, y, model, title="线性回归结果"):
    """可视化结果"""
    plt.figure(figsize=(12, 5))
    
    # 左图：数据点和拟合直线
    plt.subplot(1, 2, 1)
    plt.scatter(X, y, alpha=0.5, label='数据点')
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, label=f'拟合直线: y={model.w[0]:.4f}x+{model.b:.4f}')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 右图：损失函数收敛过程（仅梯度下降）
    if model.method == 'gradient_descent' and len(model.cost_history) > 0:
        plt.subplot(1, 2, 2)
        plt.plot(model.cost_history)
        plt.xlabel('迭代次数')
        plt.ylabel('损失')
        plt.title('损失函数收敛过程')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def main():
    """主函数"""
    print("=" * 60)
    print("从零实现线性回归")
    print("=" * 60)
    
    # 生成数据
    X, y = generate_data(n_samples=100, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征维度: {X_train.shape[1]}")
    
    # 方法1：最小二乘法
    print("\n" + "-" * 60)
    print("方法1：最小二乘法（解析解）")
    print("-" * 60)
    model_ls = LinearRegression(method='least_squares')
    model_ls.fit(X_train, y_train)
    
    y_pred_ls = model_ls.predict(X_test)
    mse_ls = mean_squared_error(y_test, y_pred_ls)
    r2_ls = model_ls.score(X_test, y_test)
    
    print(f"学习到的参数:")
    print(f"  权重 w: {model_ls.w[0]:.4f}")
    print(f"  偏置 b: {model_ls.b:.4f}")
    print(f"真实参数: w=2.0, b=1.0")
    print(f"\n评估指标:")
    print(f"  MSE: {mse_ls:.4f}")
    print(f"  R²: {r2_ls:.4f}")
    
    # 方法2：梯度下降
    print("\n" + "-" * 60)
    print("方法2：梯度下降（数值解）")
    print("-" * 60)
    model_gd = LinearRegression(
        method='gradient_descent',
        learning_rate=0.01,
        max_iter=1000,
        tol=1e-6
    )
    model_gd.fit(X_train, y_train)
    
    y_pred_gd = model_gd.predict(X_test)
    mse_gd = mean_squared_error(y_test, y_pred_gd)
    r2_gd = model_gd.score(X_test, y_test)
    
    print(f"学习到的参数:")
    print(f"  权重 w: {model_gd.w[0]:.4f}")
    print(f"  偏置 b: {model_gd.b:.4f}")
    print(f"真实参数: w=2.0, b=1.0")
    print(f"\n评估指标:")
    print(f"  MSE: {mse_gd:.4f}")
    print(f"  R²: {r2_gd:.4f}")
    print(f"  迭代次数: {len(model_gd.cost_history)}")
    
    # 可视化
    print("\n" + "-" * 60)
    print("可视化结果")
    print("-" * 60)
    visualize_results(X_train, y_train, model_ls, "最小二乘法结果")
    visualize_results(X_train, y_train, model_gd, "梯度下降结果")
    
    # 对比
    print("\n" + "=" * 60)
    print("方法对比")
    print("=" * 60)
    print(f"{'方法':<20} {'MSE':<15} {'R²':<15}")
    print("-" * 50)
    print(f"{'最小二乘法':<20} {mse_ls:<15.4f} {r2_ls:<15.4f}")
    print(f"{'梯度下降':<20} {mse_gd:<15.4f} {r2_gd:<15.4f}")


if __name__ == "__main__":
    main()

