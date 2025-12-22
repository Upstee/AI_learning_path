# 练习1：实现简单RNN

## 📋 练习目标

- 理解RNN的基本结构和工作原理
- 能够从零实现简单的RNN
- 理解隐藏状态的作用
- 能够使用RNN进行序列预测

## 🎯 难度等级

⭐⭐⭐（基础）

## 📝 练习要求

### 任务1：实现RNN单元

实现一个简单的RNN单元类：

```python
class SimpleRNNCell:
    """
    简单的RNN单元
    
    参数:
        input_size: 输入维度
        hidden_size: 隐藏状态维度
    """
    def __init__(self, input_size, hidden_size):
        # TODO: 初始化权重和偏置
        pass
    
    def forward(self, x_t, h_prev):
        """
        前向传播
        
        参数:
            x_t: 当前时间步的输入 (batch_size, input_size)
            h_prev: 上一时间步的隐藏状态 (batch_size, hidden_size)
        
        返回:
            h_t: 当前时间步的隐藏状态 (batch_size, hidden_size)
        """
        # TODO: 实现前向传播
        # h_t = tanh(W_h @ h_prev + W_x @ x_t + b)
        pass
```

**要求**：
1. 正确初始化权重矩阵（使用Xavier初始化）
2. 实现前向传播公式：$h_t = \tanh(W_h h_{t-1} + W_x x_t + b)$
3. 处理批量输入

### 任务2：实现完整RNN

实现一个完整的RNN，能够处理整个序列：

```python
class SimpleRNN:
    """
    简单的RNN网络
    
    参数:
        input_size: 输入维度
        hidden_size: 隐藏状态维度
        output_size: 输出维度
    """
    def __init__(self, input_size, hidden_size, output_size):
        # TODO: 初始化RNN单元和输出层
        pass
    
    def forward(self, X):
        """
        前向传播
        
        参数:
            X: 输入序列 (batch_size, seq_len, input_size)
        
        返回:
            outputs: 所有时间步的输出 (batch_size, seq_len, output_size)
            h_final: 最后一个时间步的隐藏状态 (batch_size, hidden_size)
        """
        # TODO: 实现序列前向传播
        pass
```

**要求**：
1. 正确处理序列数据
2. 保存所有时间步的隐藏状态
3. 计算每个时间步的输出

### 任务3：训练RNN进行序列预测

使用RNN预测简单的序列（如正弦波）：

```python
# 生成训练数据
def generate_sine_wave(seq_length=100):
    """
    生成正弦波序列
    
    返回:
        X: 输入序列 (seq_length-1, 1)
        y: 目标序列 (seq_length-1, 1)
    """
    # TODO: 生成正弦波数据
    pass

# 训练RNN
def train_rnn(model, X, y, epochs=100, lr=0.01):
    """
    训练RNN模型
    
    要求:
    1. 实现损失函数（MSE）
    2. 实现反向传播（简化版，可以使用自动梯度）
    3. 更新参数
    4. 记录训练损失
    """
    # TODO: 实现训练循环
    pass
```

**要求**：
1. 生成正弦波训练数据
2. 实现训练循环
3. 可视化训练过程和预测结果

### 任务4：对比不同隐藏状态维度

测试不同隐藏状态维度对模型性能的影响：

```python
hidden_sizes = [8, 16, 32, 64]

for hidden_size in hidden_sizes:
    # 创建模型
    model = SimpleRNN(input_size=1, hidden_size=hidden_size, output_size=1)
    
    # 训练模型
    # TODO: 训练并记录结果
    
    # 可视化结果
    # TODO: 绘制预测结果
```

**要求**：
1. 测试不同隐藏状态维度
2. 记录训练损失和预测精度
3. 分析隐藏状态维度的影响

## 💡 提示

1. **隐藏状态初始化**：通常初始化为零向量
2. **权重初始化**：使用Xavier初始化，避免梯度消失/爆炸
3. **激活函数**：使用Tanh激活函数
4. **序列处理**：注意处理序列的第一个时间步（没有前一个隐藏状态）

## ✅ 检查清单

完成练习后，检查以下内容：

- [ ] 实现了RNN单元类
- [ ] 实现了完整的RNN网络
- [ ] 能够处理序列数据
- [ ] 实现了训练循环
- [ ] 能够预测序列
- [ ] 理解了隐藏状态的作用
- [ ] 分析了不同参数的影响

## 📚 参考资料

- [RNN原理详解](../理论笔记/RNN原理详解.md)
- [从零实现RNN](../代码示例/01_从零实现RNN.ipynb)（如果已创建）

## 🎓 扩展思考

1. RNN的梯度消失问题是如何产生的？
2. 如何改进RNN以处理更长的序列？
3. 双向RNN和单向RNN有什么区别？

---

**准备好了吗？开始练习吧！** 🚀

