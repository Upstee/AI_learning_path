"""
简单项目1：基础房价预测
使用线性回归预测房价
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


def generate_data(n_samples=100, random_state=42):
    """生成模拟房价数据"""
    np.random.seed(random_state)
    
    # 特征：面积（平方米）
    area = np.random.normal(100, 20, n_samples)
    
    # 目标：房价（万元）= 面积 * 0.8 + 基础价格 + 噪声
    price = area * 0.8 + 50 + np.random.normal(0, 10, n_samples)
    
    # 确保价格为正值
    price = np.maximum(price, 20)
    
    return area.reshape(-1, 1), price


def main():
    """主函数"""
    print("=" * 60)
    print("简单项目1：基础房价预测")
    print("=" * 60)
    
    # 1. 生成数据
    X, y = generate_data(n_samples=100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"\n数据信息:")
    print(f"  训练集大小: {X_train.shape[0]}")
    print(f"  测试集大小: {X_test.shape[0]}")
    print(f"  特征: 面积（平方米）")
    
    # 2. 训练模型
    print("\n训练模型...")
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 3. 预测
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # 4. 评估
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"\n模型评估:")
    print(f"  训练集MSE: {train_mse:.2f}")
    print(f"  测试集MSE: {test_mse:.2f}")
    print(f"  训练集R²: {train_r2:.4f}")
    print(f"  测试集R²: {test_r2:.4f}")
    print(f"\n模型参数:")
    print(f"  系数（每平方米价格）: {model.coef_[0]:.2f} 万元/平方米")
    print(f"  截距（基础价格）: {model.intercept_:.2f} 万元")
    
    # 5. 可视化
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, alpha=0.6, label='训练数据')
    plt.scatter(X_test, y_test, alpha=0.6, label='测试数据', marker='x')
    
    # 绘制拟合直线
    X_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = model.predict(X_line)
    plt.plot(X_line, y_line, 'r-', linewidth=2, 
             label=f'拟合直线: y={model.coef_[0]:.2f}x+{model.intercept_:.2f}')
    
    plt.xlabel('面积（平方米）')
    plt.ylabel('房价（万元）')
    plt.title('基础房价预测')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('房价预测结果.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("\n✅ 项目完成！")


if __name__ == "__main__":
    main()

