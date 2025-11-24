"""
挑战练习1答案：大规模数据回归系统
代码量：> 500行
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import time
import pickle
import json
from pathlib import Path

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class LargeScaleRegression:
    """大规模数据回归系统"""
    
    def __init__(self, batch_size=1000, learning_rate=0.01, max_iter=1000):
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
        self.cost_history = []
    
    def batch_gradient_descent(self, X, y):
        """批量梯度下降"""
        m, n = X.shape
        w = np.random.randn(n) * 0.01
        b = 0
        
        for i in range(self.max_iter):
            # 预测
            y_pred = X @ w + b
            
            # 计算损失
            cost = (1 / (2 * m)) * np.sum((y_pred - y) ** 2)
            self.cost_history.append(cost)
            
            # 计算梯度
            dw = (1 / m) * X.T @ (y_pred - y)
            db = (1 / m) * np.sum(y_pred - y)
            
            # 更新参数
            w = w - self.learning_rate * dw
            b = b - self.learning_rate * db
            
            # 早停
            if i > 0 and abs(self.cost_history[-2] - self.cost_history[-1]) < 1e-6:
                break
        
        self.w = w
        self.b = b
        return self
    
    def stochastic_gradient_descent(self, X, y):
        """随机梯度下降"""
        m, n = X.shape
        w = np.random.randn(n) * 0.01
        b = 0
        
        for epoch in range(self.max_iter):
            indices = np.random.permutation(m)
            epoch_cost = 0
            
            for i in indices:
                x_i = X[i:i+1]
                y_i = y[i]
                
                y_pred = x_i @ w + b
                dw = (y_pred - y_i) * x_i.flatten()
                db = (y_pred - y_i)
                
                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db
                
                epoch_cost += (y_pred - y_i) ** 2
            
            cost = epoch_cost / (2 * m)
            self.cost_history.append(cost)
        
        self.w = w
        self.b = b
        return self
    
    def mini_batch_gradient_descent(self, X, y):
        """小批量梯度下降"""
        m, n = X.shape
        w = np.random.randn(n) * 0.01
        b = 0
        
        for epoch in range(self.max_iter):
            indices = np.random.permutation(m)
            epoch_cost = 0
            
            for i in range(0, m, self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                y_pred = X_batch @ w + b
                dw = (1 / len(batch_indices)) * X_batch.T @ (y_pred - y_batch)
                db = (1 / len(batch_indices)) * np.sum(y_pred - y_batch)
                
                w = w - self.learning_rate * dw
                b = b - self.learning_rate * db
                
                epoch_cost += np.sum((y_pred - y_batch) ** 2)
            
            cost = epoch_cost / (2 * m)
            self.cost_history.append(cost)
        
        self.w = w
        self.b = b
        return self
    
    def fit(self, X, y, method='mini_batch'):
        """训练模型"""
        # 标准化
        X_scaled = self.scaler.fit_transform(X)
        
        # 选择优化方法
        if method == 'batch':
            self.batch_gradient_descent(X_scaled, y)
        elif method == 'stochastic':
            self.stochastic_gradient_descent(X_scaled, y)
        else:  # mini_batch
            self.mini_batch_gradient_descent(X_scaled, y)
        
        return self
    
    def predict(self, X):
        """预测"""
        X_scaled = self.scaler.transform(X)
        return X_scaled @ self.w + self.b
    
    def save_model(self, filepath):
        """保存模型"""
        model_data = {
            'w': self.w.tolist(),
            'b': self.b,
            'scaler_mean': self.scaler.mean_.tolist(),
            'scaler_scale': self.scaler.scale_.tolist()
        }
        with open(filepath, 'w') as f:
            json.dump(model_data, f)
    
    def load_model(self, filepath):
        """加载模型"""
        with open(filepath, 'r') as f:
            model_data = json.load(f)
        self.w = np.array(model_data['w'])
        self.b = model_data['b']
        self.scaler.mean_ = np.array(model_data['scaler_mean'])
        self.scaler.scale_ = np.array(model_data['scaler_scale'])


def generate_large_data(n_samples=100000, n_features=10, random_state=42):
    """生成大规模数据"""
    np.random.seed(random_state)
    X = np.random.randn(n_samples, n_features)
    y = X @ np.random.randn(n_features) + np.random.randn(n_samples) * 0.1
    return X, y


def compare_optimization_methods(X_train, y_train, X_test, y_test):
    """对比不同优化方法"""
    print("=" * 70)
    print("对比不同优化方法")
    print("=" * 70)
    
    methods = ['batch', 'stochastic', 'mini_batch']
    results = {}
    
    for method in methods:
        print(f"\n{method}梯度下降:")
        start_time = time.time()
        
        model = LargeScaleRegression(batch_size=1000, learning_rate=0.01, max_iter=100)
        model.fit(X_train, y_train, method=method)
        
        train_time = time.time() - start_time
        
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        results[method] = {
            'model': model,
            'train_time': train_time,
            'mse': mse,
            'r2': r2,
            'iterations': len(model.cost_history)
        }
        
        print(f"  训练时间: {train_time:.2f}秒")
        print(f"  迭代次数: {len(model.cost_history)}")
        print(f"  测试MSE: {mse:.4f}")
        print(f"  测试R²: {r2:.4f}")
    
    # 可视化
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 训练时间对比
    methods_list = list(results.keys())
    times = [results[m]['train_time'] for m in methods_list]
    axes[0].bar(methods_list, times)
    axes[0].set_ylabel('训练时间（秒）')
    axes[0].set_title('训练时间对比')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # 收敛曲线
    for method in methods_list:
        axes[1].plot(results[method]['model'].cost_history[:50], 
                    label=method, alpha=0.7)
    axes[1].set_xlabel('迭代次数')
    axes[1].set_ylabel('损失')
    axes[1].set_title('收敛曲线对比（前50次迭代）')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('优化方法对比.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return results


def main():
    """主函数"""
    print("=" * 70)
    print("挑战练习1：大规模数据回归系统")
    print("=" * 70)
    
    # 生成大规模数据
    print("\n生成大规模数据...")
    X, y = generate_large_data(n_samples=100000, n_features=10)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"训练集大小: {X_train.shape[0]}")
    print(f"测试集大小: {X_test.shape[0]}")
    print(f"特征数量: {X_train.shape[1]}")
    
    # 对比优化方法
    results = compare_optimization_methods(X_train, y_train, X_test, y_test)
    
    # 模型保存和加载示例
    print("\n" + "=" * 70)
    print("模型保存和加载")
    print("=" * 70)
    
    best_method = min(results.items(), key=lambda x: x[1]['train_time'])[0]
    best_model = results[best_method]['model']
    
    model_path = 'model.json'
    best_model.save_model(model_path)
    print(f"模型已保存到: {model_path}")
    
    # 加载模型
    new_model = LargeScaleRegression()
    new_model.load_model(model_path)
    y_pred_loaded = new_model.predict(X_test)
    r2_loaded = r2_score(y_test, y_pred_loaded)
    print(f"加载的模型R²: {r2_loaded:.4f}")
    
    print("\n✅ 练习完成！")


if __name__ == "__main__":
    main()

