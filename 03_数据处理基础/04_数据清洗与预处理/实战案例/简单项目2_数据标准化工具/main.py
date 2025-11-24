"""
简单项目2：数据标准化工具
使用sklearn进行数据标准化和特征工程
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder


class DataScaler:
    """数据标准化工具类"""
    
    def __init__(self, df):
        """初始化"""
        self.df = df.copy()
        self.scalers = {}
    
    def standardize(self, columns=None, method='zscore'):
        """标准化数据"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        if method == 'zscore':
            scaler = StandardScaler()
        elif method == 'minmax':
            scaler = MinMaxScaler()
        elif method == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"不支持的方法: {method}")
        
        self.df[columns] = scaler.fit_transform(self.df[columns])
        self.scalers[method] = scaler
        
        print(f"使用{method}方法标准化了 {len(columns)} 个列")
        return self
    
    def encode_categorical(self, columns=None, method='onehot'):
        """编码分类变量"""
        if columns is None:
            columns = self.df.select_dtypes(include=['object']).columns
        
        if method == 'onehot':
            for col in columns:
                dummies = pd.get_dummies(self.df[col], prefix=col)
                self.df = pd.concat([self.df.drop(columns=[col]), dummies], axis=1)
                print(f"对 {col} 进行了独热编码")
        elif method == 'label':
            le = LabelEncoder()
            for col in columns:
                self.df[col] = le.fit_transform(self.df[col])
                print(f"对 {col} 进行了标签编码")
        
        return self
    
    def transform_features(self, columns=None):
        """特征变换"""
        if columns is None:
            columns = self.df.select_dtypes(include=[np.number]).columns
        
        for col in columns:
            # 对数变换（只对正值）
            if (self.df[col] > 0).all():
                self.df[f'{col}_log'] = np.log(self.df[col])
            # 平方根变换
            if (self.df[col] >= 0).all():
                self.df[f'{col}_sqrt'] = np.sqrt(self.df[col])
            # 平方变换
            self.df[f'{col}_square'] = self.df[col] ** 2
        
        print(f"对 {len(columns)} 个列进行了特征变换")
        return self
    
    def get_processed_data(self):
        """获取处理后的数据"""
        return self.df
    
    def print_summary(self):
        """打印处理摘要"""
        print("\n" + "=" * 50)
        print("数据处理摘要")
        print("=" * 50)
        print(f"数据形状: {self.df.shape}")
        print(f"使用的标准化方法: {list(self.scalers.keys())}")
        print(f"\n处理后数据统计:")
        numeric_cols = self.df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            print(self.df[numeric_cols].describe())


def main():
    """主函数"""
    # 创建示例数据
    np.random.seed(42)
    df = pd.DataFrame({
        'age': np.random.randint(20, 60, 100),
        'income': np.random.randint(30000, 100000, 100),
        'score': np.random.normal(75, 10, 100),
        'category': np.random.choice(['A', 'B', 'C'], 100),
        'city': np.random.choice(['NY', 'LA', 'SF'], 100)
    })
    
    print("原始数据:")
    print(df.head())
    print(f"\n原始数据统计:")
    print(df.describe())
    
    # 数据标准化和特征工程
    scaler = DataScaler(df)
    scaler.standardize(columns=['age', 'income', 'score'], method='zscore')
    scaler.encode_categorical(columns=['category'], method='onehot')
    scaler.transform_features(columns=['score'])
    scaler.print_summary()
    
    print("\n处理后的数据预览:")
    print(scaler.get_processed_data().head())


if __name__ == "__main__":
    main()

