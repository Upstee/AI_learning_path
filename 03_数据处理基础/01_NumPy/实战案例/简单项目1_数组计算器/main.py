"""
简单项目1：数组计算器
使用NumPy进行数组运算和统计
"""

import numpy as np


class ArrayCalculator:
    """数组计算器类"""
    
    def __init__(self):
        self.current_array = None
    
    def create_array(self, data):
        """从列表创建数组"""
        try:
            self.current_array = np.array(data)
            print(f"数组创建成功: {self.current_array}")
            return True
        except Exception as e:
            print(f"创建数组失败: {e}")
            return False
    
    def create_special_array(self, array_type, shape):
        """创建特殊数组"""
        try:
            if array_type == 'zeros':
                self.current_array = np.zeros(shape)
            elif array_type == 'ones':
                self.current_array = np.ones(shape)
            elif array_type == 'random':
                self.current_array = np.random.rand(*shape)
            elif array_type == 'arange':
                self.current_array = np.arange(shape[0])
            else:
                print("不支持的类型")
                return False
            print(f"数组创建成功:\n{self.current_array}")
            return True
        except Exception as e:
            print(f"创建数组失败: {e}")
            return False
    
    def show_statistics(self):
        """显示统计信息"""
        if self.current_array is None:
            print("请先创建数组")
            return
        
        print("\n数组统计信息:")
        print(f"形状: {self.current_array.shape}")
        print(f"维度: {self.current_array.ndim}")
        print(f"元素总数: {self.current_array.size}")
        print(f"数据类型: {self.current_array.dtype}")
        print(f"平均值: {np.mean(self.current_array):.4f}")
        print(f"标准差: {np.std(self.current_array):.4f}")
        print(f"最大值: {np.max(self.current_array)}")
        print(f"最小值: {np.min(self.current_array)}")
        print(f"总和: {np.sum(self.current_array)}")
    
    def perform_operation(self, operation, value=None):
        """执行运算"""
        if self.current_array is None:
            print("请先创建数组")
            return None
        
        try:
            if operation == 'add':
                result = self.current_array + value
            elif operation == 'multiply':
                result = self.current_array * value
            elif operation == 'power':
                result = self.current_array ** value
            elif operation == 'sqrt':
                result = np.sqrt(self.current_array)
            elif operation == 'transpose':
                result = self.current_array.T
            elif operation == 'reshape':
                result = self.current_array.reshape(value)
            else:
                print("不支持的操作")
                return None
            
            print(f"运算结果:\n{result}")
            return result
        except Exception as e:
            print(f"运算失败: {e}")
            return None


def main():
    """主函数"""
    calc = ArrayCalculator()
    
    print("=" * 50)
    print("NumPy数组计算器")
    print("=" * 50)
    
    # 示例1：创建数组
    print("\n示例1：创建数组")
    calc.create_array([1, 2, 3, 4, 5])
    calc.show_statistics()
    
    # 示例2：创建特殊数组
    print("\n示例2：创建3x3随机数组")
    calc.create_special_array('random', (3, 3))
    calc.show_statistics()
    
    # 示例3：数组运算
    print("\n示例3：数组运算")
    calc.create_array([1, 2, 3, 4, 5])
    calc.perform_operation('add', 10)
    calc.perform_operation('multiply', 2)
    calc.perform_operation('power', 2)
    
    # 示例4：形状变换
    print("\n示例4：形状变换")
    calc.create_array(list(range(12)))
    calc.perform_operation('reshape', (3, 4))
    calc.perform_operation('transpose')


if __name__ == "__main__":
    main()

