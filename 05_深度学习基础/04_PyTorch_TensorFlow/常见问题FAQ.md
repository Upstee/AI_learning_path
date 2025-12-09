# PyTorch/TensorFlow常见问题FAQ

> **目的**：快速解决学习过程中的常见问题

---

## PyTorch问题

### Q1: 张量和NumPy数组有什么区别？

**A**: 

**张量（Tensor）**：
- 可以放在GPU上
- 支持自动梯度
- 与NumPy数组类似，但功能更强大

**转换**：
```python
# NumPy → Tensor
numpy_array = np.array([1, 2, 3])
tensor = torch.from_numpy(numpy_array)

# Tensor → NumPy
tensor = torch.tensor([1, 2, 3])
numpy_array = tensor.numpy()
```

---

### Q2: 如何将模型移到GPU？

**A**: 

```python
# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 将模型移到GPU
model = model.to(device)

# 将数据移到GPU
X = X.to(device)
y = y.to(device)
```

---

### Q3: 如何保存和加载模型？

**A**: 

```python
# 保存
torch.save(model.state_dict(), 'model.pth')

# 加载
model = MyModel()
model.load_state_dict(torch.load('model.pth'))
```

---

## TensorFlow问题

### Q4: TensorFlow 2.x和1.x有什么区别？

**A**: 

**TensorFlow 2.x**：
- Eager Execution（即时执行）
- Keras作为高级API
- 更简单易用

**TensorFlow 1.x**：
- 静态计算图
- 需要Session
- 更复杂

**建议**：学习TensorFlow 2.x

---

### Q5: 如何将模型移到GPU？

**A**: 

```python
# TensorFlow自动使用GPU（如果可用）
# 检查GPU
print(tf.config.list_physical_devices('GPU'))

# 强制使用CPU
with tf.device('/CPU:0'):
    model = ...
```

---

## 通用问题

### Q6: 选择PyTorch还是TensorFlow？

**A**: 

**PyTorch**：
- ✅ 更灵活，适合研究
- ✅ 动态计算图
- ✅ 调试更容易
- ⚠️ 生产部署稍复杂

**TensorFlow**：
- ✅ 生产环境成熟
- ✅ 静态计算图（可优化）
- ✅ 部署工具丰富
- ⚠️ 学习曲线稍陡

**建议**：
- 研究/学习：PyTorch
- 生产部署：TensorFlow
- 或者：两个都学

---

### Q7: 如何调试模型？

**A**: 

```python
# 1. 打印模型结构
print(model)

# 2. 检查参数
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 3. 检查梯度
for name, param in model.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.norm()}")
```

---

## 📖 更多资源

- **快速上手**：[00_快速上手.md](./00_快速上手.md)
- **学习检查点**：[学习检查点.md](./学习检查点.md)
- **调试技巧**：[../01_神经网络基础/调试技巧和常见错误.md](../01_神经网络基础/调试技巧和常见错误.md)

---

**如果这里没有你遇到的问题，请查看其他资源！** 💪
