"""
简单项目2：数据清洗工具
使用Pandas进行数据清洗
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗器"""
    
    def __init__(self, df):
        """初始化"""
        self.df = df.copy()
        self.original_shape = df.shape
    
    def clean_missing(self, strategy='drop', fill_value=0):
        """处理缺失值"""
        print(f"\n处理缺失值 - 策略: {strategy}")
        print(f"缺失值数量: {self.df.isna().sum().sum()}")
        
        if strategy == 'drop':
            self.df = self.df.dropna()
        elif strategy == 'fill':
            self.df = self.df.fillna(fill_value)
        elif strategy == 'mean':
            self.df = self.df.fillna(self.df.mean())
        
        print(f"处理后缺失值数量: {self.df.isna().sum().sum()}")
        return self
    
    def clean_outliers(self, method='zscore', threshold=3):
        """处理异常值"""
        print(f"\n处理异常值 - 方法: {method}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                self.df = self.df[z_scores < threshold]
            elif method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                self.df = self.df[(self.df[col] >= Q1 - 1.5*IQR) & 
                                  (self.df[col] <= Q3 + 1.5*IQR)]
        
        print(f"处理后数据形状: {self.df.shape}")
        return self
    
    def clean_duplicates(self):
        """处理重复值"""
        print(f"\n处理重复值")
        print(f"重复行数: {self.df.duplicated().sum()}")
        self.df = self.df.drop_duplicates()
        print(f"删除后数据形状: {self.df.shape}")
        return self
    
    def get_cleaned_data(self):
        """获取清洗后的数据"""
        return self.df
    
    def print_summary(self):
        """打印清洗摘要"""
        print("\n" + "=" * 50)
        print("数据清洗摘要")
        print("=" * 50)
        print(f"原始数据形状: {self.original_shape}")
        print(f"清洗后数据形状: {self.df.shape}")
        print(f"删除行数: {self.original_shape[0] - self.df.shape[0]}")


def main():
    """主函数"""
    # 创建包含问题的示例数据
    data = pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5, 100],  # 包含缺失值和异常值
        'B': [1, 2, 3, 4, 5, 6],
        'C': [1, 1, 2, 2, 3, 3]  # 包含重复值
    })
    data = pd.concat([data, data.iloc[[0, 1]]], ignore_index=True)  # 添加重复行
    
    print("原始数据:")
    print(data)
    
    # 数据清洗
    cleaner = DataCleaner(data)
    cleaner.clean_missing(strategy='mean')
    cleaner.clean_outliers(method='zscore')
    cleaner.clean_duplicates()
    cleaner.print_summary()
    
    print("\n清洗后的数据:")
    print(cleaner.get_cleaned_data())


if __name__ == "__main__":
    main()

