"""
简单项目1：CSV数据分析工具
使用Pandas进行CSV数据分析和处理
"""

import pandas as pd
import numpy as np


class CSVAnalyzer:
    """CSV数据分析器"""
    
    def __init__(self, file_path):
        """初始化，加载数据"""
        self.file_path = file_path
        self.df = None
        self.load_data()
    
    def load_data(self):
        """加载CSV数据"""
        try:
            self.df = pd.read_csv(self.file_path)
            print(f"数据加载成功！")
            print(f"数据形状: {self.df.shape}")
        except Exception as e:
            print(f"加载数据失败: {e}")
    
    def basic_info(self):
        """基本信息"""
        print("\n" + "=" * 50)
        print("数据基本信息")
        print("=" * 50)
        print(f"行数: {self.df.shape[0]}")
        print(f"列数: {self.df.shape[1]}")
        print(f"\n列名: {self.df.columns.tolist()}")
        print(f"\n数据类型:\n{self.df.dtypes}")
        print(f"\n前5行:\n{self.df.head()}")
    
    def missing_values(self):
        """缺失值分析"""
        print("\n" + "=" * 50)
        print("缺失值分析")
        print("=" * 50)
        missing = self.df.isna().sum()
        missing_pct = (missing / len(self.df)) * 100
        missing_df = pd.DataFrame({
            '缺失数量': missing,
            '缺失比例(%)': missing_pct
        })
        print(missing_df[missing_df['缺失数量'] > 0])
    
    def statistics(self):
        """描述性统计"""
        print("\n" + "=" * 50)
        print("描述性统计")
        print("=" * 50)
        print(self.df.describe())
    
    def generate_report(self):
        """生成分析报告"""
        self.basic_info()
        self.missing_values()
        self.statistics()


def main():
    """主函数"""
    # 创建示例数据（如果没有CSV文件）
    sample_data = pd.DataFrame({
        'name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
        'age': [25, 30, 35, 40, 28],
        'salary': [50000, 60000, 70000, 80000, 55000],
        'city': ['NY', 'LA', 'NY', 'LA', 'NY']
    })
    sample_data.to_csv('sample_data.csv', index=False)
    
    # 使用分析器
    analyzer = CSVAnalyzer('sample_data.csv')
    analyzer.generate_report()


if __name__ == "__main__":
    main()

