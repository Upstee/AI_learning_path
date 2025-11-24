"""
简单项目2：统计图表生成器
使用Seaborn创建统计图表
"""

import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# 设置样式
sns.set_style("whitegrid")
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


class StatisticalPlotter:
    """统计图表生成器"""
    
    def __init__(self, data):
        """初始化"""
        self.df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    
    def plot_distribution(self, col, hue=None, title="分布图", save_path=None):
        """绘制分布图"""
        plt.figure(figsize=(10, 6))
        sns.histplot(data=self.df, x=col, hue=hue, kde=True)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_box(self, x_col, y_col, title="箱线图", save_path=None):
        """绘制箱线图"""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=self.df, x=x_col, y=y_col)
        plt.title(title)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_violin(self, x_col, y_col, title="小提琴图", save_path=None):
        """绘制小提琴图"""
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=self.df, x=x_col, y=y_col)
        plt.title(title)
        plt.xticks(rotation=45)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_regression(self, x_col, y_col, title="回归图", save_path=None):
        """绘制回归图"""
        plt.figure(figsize=(10, 6))
        sns.regplot(data=self.df, x=x_col, y=y_col)
        plt.title(title)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_pairplot(self, hue=None, title="成对关系图", save_path=None):
        """绘制成对关系图"""
        sns.pairplot(self.df, hue=hue)
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()


def main():
    """主函数"""
    # 创建示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'category': np.repeat(['A', 'B', 'C'], 50),
        'value1': np.concatenate([
            np.random.normal(10, 2, 50),
            np.random.normal(15, 3, 50),
            np.random.normal(12, 2.5, 50)
        ]),
        'value2': np.random.randn(150) * 5 + 20,
        'value3': np.random.randn(150) * 3 + 15
    })
    
    print("创建示例数据")
    print(df.head())
    
    # 创建统计图表生成器
    plotter = StatisticalPlotter(df)
    
    # 绘制不同类型的统计图表
    print("\n绘制分布图...")
    plotter.plot_distribution('value1', hue='category', title='按类别分布')
    
    print("\n绘制箱线图...")
    plotter.plot_box('category', 'value1', title='按类别箱线图')
    
    print("\n绘制小提琴图...")
    plotter.plot_violin('category', 'value1', title='按类别小提琴图')
    
    print("\n绘制回归图...")
    plotter.plot_regression('value2', 'value3', title='回归分析')
    
    print("\n绘制成对关系图...")
    plotter.plot_pairplot(hue='category')


if __name__ == "__main__":
    main()

