"""
简单项目1：数据可视化工具
使用Matplotlib和Seaborn创建数据可视化
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class DataVisualizer:
    """数据可视化工具类"""
    
    def __init__(self, data):
        """初始化"""
        if isinstance(data, pd.DataFrame):
            self.df = data
        else:
            raise ValueError("数据必须是Pandas DataFrame")
    
    def plot_line(self, x_col, y_col, title="线图", save_path=None):
        """绘制线图"""
        plt.figure(figsize=(10, 6))
        plt.plot(self.df[x_col], self.df[y_col], linewidth=2)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.title(title)
        plt.grid(True, alpha=0.3)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_scatter(self, x_col, y_col, hue_col=None, title="散点图", save_path=None):
        """绘制散点图"""
        plt.figure(figsize=(10, 6))
        if hue_col:
            sns.scatterplot(data=self.df, x=x_col, y=y_col, hue=hue_col)
        else:
            sns.scatterplot(data=self.df, x=x_col, y=y_col)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_bar(self, x_col, y_col, title="柱状图", save_path=None):
        """绘制柱状图"""
        plt.figure(figsize=(10, 6))
        sns.barplot(data=self.df, x=x_col, y=y_col)
        plt.title(title)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_histogram(self, col, title="直方图", save_path=None):
        """绘制直方图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=col, kde=True)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_heatmap(self, title="热力图", save_path=None):
        """绘制相关性热力图"""
        numeric_df = self.df.select_dtypes(include=[np.number])
        corr = numeric_df.corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    # 创建示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'x': np.arange(100),
        'y': np.random.randn(100).cumsum(),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'value': np.random.randn(100)
    })
    
    print("创建示例数据")
    print(df.head())
    
    # 创建可视化工具
    viz = DataVisualizer(df)
    
    # 绘制不同类型的图表
    print("\n绘制线图...")
    viz.plot_line('x', 'y', title='累积随机游走')
    
    print("\n绘制散点图...")
    viz.plot_scatter('x', 'value', hue_col='category', title='分类散点图')
    
    print("\n绘制柱状图...")
    df_grouped = df.groupby('category')['value'].mean().reset_index()
    viz_grouped = DataVisualizer(df_grouped)
    viz_grouped.plot_bar('category', 'value', title='按类别平均')
    
    print("\n绘制直方图...")
    viz.plot_histogram('value', title='值分布')
    
    print("\n绘制热力图...")
    df_numeric = pd.DataFrame({
        'A': np.random.randn(100),
        'B': np.random.randn(100),
        'C': np.random.randn(100),
        'D': np.random.randn(100)
    })
    viz_numeric = DataVisualizer(df_numeric)
    viz_numeric.plot_heatmap(title='相关性热力图')


if __name__ == "__main__":
    main()

