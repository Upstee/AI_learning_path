"""
简单项目1：数据清洗工具
使用Pandas进行数据清洗
"""

import pandas as pd
import numpy as np


class DataCleaner:
    """数据清洗工具类"""
    
    def __init__(self, df):
        """初始化"""
        self.df = df.copy()
        self.original_shape = df.shape
        self.cleaning_log = []
    
    def clean_missing(self, strategy='mean', threshold=0.5):
        """处理缺失值"""
        print(f"\n处理缺失值 - 策略: {strategy}")
        
        # 统计缺失值
        missing = self.df.isna().sum()
        missing_pct = missing / len(self.df)
        
        print(f"缺失值统计:")
        print(missing[missing > 0])
        
        # 删除缺失值比例高的列
        cols_to_drop = missing_pct[missing_pct > threshold].index
        if len(cols_to_drop) > 0:
            self.df = self.df.drop(columns=cols_to_drop)
            self.cleaning_log.append(f"删除缺失值比例>50%的列: {list(cols_to_drop)}")
            print(f"删除了 {len(cols_to_drop)} 个列")
        
        # 处理剩余缺失值
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        if strategy == 'mean':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].mean())
        elif strategy == 'median':
            self.df[numeric_cols] = self.df[numeric_cols].fillna(self.df[numeric_cols].median())
        elif strategy == 'mode':
            for col in categorical_cols:
                self.df[col] = self.df[col].fillna(self.df[col].mode()[0] if len(self.df[col].mode()) > 0 else 'Unknown')
        elif strategy == 'drop':
            self.df = self.df.dropna()
        
        self.cleaning_log.append(f"使用{strategy}策略处理缺失值")
        return self
    
    def clean_outliers(self, method='iqr'):
        """处理异常值"""
        print(f"\n处理异常值 - 方法: {method}")
        
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if method == 'iqr':
                Q1 = self.df[col].quantile(0.25)
                Q3 = self.df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                outliers_count = ((self.df[col] < lower) | (self.df[col] > upper)).sum()
                self.df[col] = self.df[col].clip(lower=lower, upper=upper)
                if outliers_count > 0:
                    print(f"  {col}: 处理了 {outliers_count} 个异常值")
            elif method == 'zscore':
                z_scores = np.abs((self.df[col] - self.df[col].mean()) / self.df[col].std())
                outliers_count = (z_scores > 3).sum()
                self.df = self.df[z_scores <= 3]
                if outliers_count > 0:
                    print(f"  {col}: 删除了 {outliers_count} 个异常值")
        
        self.cleaning_log.append(f"使用{method}方法处理异常值")
        return self
    
    def clean_duplicates(self):
        """处理重复值"""
        print(f"\n处理重复值")
        duplicates_count = self.df.duplicated().sum()
        print(f"发现 {duplicates_count} 个重复行")
        
        if duplicates_count > 0:
            self.df = self.df.drop_duplicates()
            self.cleaning_log.append(f"删除了 {duplicates_count} 个重复行")
        
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
        print(f"删除列数: {self.original_shape[1] - self.df.shape[1]}")
        print(f"\n清洗步骤:")
        for i, step in enumerate(self.cleaning_log, 1):
            print(f"  {i}. {step}")


def main():
    """主函数"""
    # 创建包含问题的示例数据
    np.random.seed(42)
    data = {
        'A': np.random.normal(100, 15, 100),
        'B': np.random.normal(50, 10, 100),
        'C': np.random.choice(['X', 'Y', 'Z'], 100)
    }
    df = pd.DataFrame(data)
    
    # 添加问题
    df.loc[10:15, 'A'] = np.nan  # 添加缺失值
    df.loc[20:25, 'B'] = 200  # 添加异常值
    df = pd.concat([df, df.iloc[0:5]], ignore_index=True)  # 添加重复值
    
    print("原始数据:")
    print(f"形状: {df.shape}")
    print(f"缺失值: {df.isna().sum().sum()}")
    print(f"重复值: {df.duplicated().sum()}")
    
    # 数据清洗
    cleaner = DataCleaner(df)
    cleaner.clean_missing(strategy='mean')
    cleaner.clean_outliers(method='iqr')
    cleaner.clean_duplicates()
    cleaner.print_summary()
    
    print("\n清洗后的数据预览:")
    print(cleaner.get_cleaned_data().head())


if __name__ == "__main__":
    main()

