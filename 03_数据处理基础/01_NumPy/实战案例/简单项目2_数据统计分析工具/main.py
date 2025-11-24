"""
简单项目2：数据统计分析工具
使用NumPy进行数据统计分析
"""

import numpy as np


class DataAnalyzer:
    """数据分析器类"""
    
    def __init__(self, data):
        """初始化，加载数据"""
        self.data = np.array(data)
        self.validate_data()
    
    def validate_data(self):
        """验证数据"""
        if self.data.size == 0:
            raise ValueError("数据不能为空")
        if not np.isfinite(self.data).all():
            print("警告：数据中包含非有限值（NaN或Inf）")
    
    def descriptive_stats(self):
        """描述性统计"""
        stats = {
            'count': self.data.size,
            'mean': np.mean(self.data),
            'median': np.median(self.data),
            'std': np.std(self.data),
            'var': np.var(self.data),
            'min': np.min(self.data),
            'max': np.max(self.data),
            'range': np.ptp(self.data),  # peak to peak
            'sum': np.sum(self.data)
        }
        return stats
    
    def percentiles(self, percentiles=[25, 50, 75, 90, 95]):
        """计算百分位数"""
        return {p: np.percentile(self.data, p) for p in percentiles}
    
    def print_report(self):
        """打印统计报告"""
        print("=" * 50)
        print("数据统计分析报告")
        print("=" * 50)
        
        print(f"\n数据概览:")
        print(f"  形状: {self.data.shape}")
        print(f"  数据类型: {self.data.dtype}")
        print(f"  元素总数: {self.data.size}")
        
        print(f"\n描述性统计:")
        stats = self.descriptive_stats()
        for key, value in stats.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        
        print(f"\n百分位数:")
        percentiles = self.percentiles()
        for p, value in percentiles.items():
            print(f"  {p}%: {value:.4f}")


def main():
    """主函数"""
    # 示例数据
    data = np.random.normal(100, 15, 1000)  # 正态分布数据
    
    print("生成示例数据（正态分布，均值=100，标准差=15，样本数=1000）")
    
    # 创建分析器
    analyzer = DataAnalyzer(data)
    
    # 生成报告
    analyzer.print_report()
    
    # 额外分析
    print(f"\n额外分析:")
    print(f"  大于均值的元素数: {np.sum(analyzer.data > analyzer.descriptive_stats()['mean'])}")
    print(f"  小于均值的元素数: {np.sum(analyzer.data < analyzer.descriptive_stats()['mean'])}")
    print(f"  在均值±1标准差内的元素数: {np.sum(np.abs(analyzer.data - analyzer.descriptive_stats()['mean']) <= analyzer.descriptive_stats()['std'])}")


if __name__ == "__main__":
    main()

