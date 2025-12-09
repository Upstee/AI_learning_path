# PCA常见问题FAQ

> **目的**：快速解决学习过程中的常见问题

---

## 概念理解问题

### Q1: PCA和特征选择有什么区别？

**A**: 

**PCA（主成分分析）**：
- 特征变换：创建新特征（主成分）
- 线性组合：主成分是原始特征的线性组合
- 降维：可以减少维度
- 信息保留：保留方差最大的方向

**特征选择**：
- 特征筛选：选择原始特征
- 不创建新特征
- 不能降维（只能减少特征数量）
- 保留原始特征

**示例**：
```python
# PCA：创建新特征
pca = PCA(n_components=2)
X_new = pca.fit_transform(X)  # 新特征

# 特征选择：选择原始特征
from sklearn.feature_selection import SelectKBest
selector = SelectKBest(k=2)
X_selected = selector.fit_transform(X, y)  # 原始特征的子集
```

---

### Q2: 如何选择主成分数量？

**A**: 三种方法：

#### 方法1：累计解释方差（推荐）

```python
# 选择解释95%方差的主成分
pca = PCA(n_components=0.95)
pca.fit(X)
print(f"需要 {pca.n_components_} 个主成分")
```

#### 方法2：固定数量

```python
# 选择前2个主成分
pca = PCA(n_components=2)
pca.fit(X)
```

#### 方法3：肘部法则

```python
# 绘制累计解释方差曲线
pca = PCA()
pca.fit(X)
cumsum = np.cumsum(pca.explained_variance_ratio_)

plt.plot(cumsum)
plt.axhline(y=0.95, color='r', linestyle='--')
plt.xlabel('主成分数量')
plt.ylabel('累计解释方差')
plt.show()

# 找到解释95%方差的点
n_components = np.argmax(cumsum >= 0.95) + 1
```

---

### Q3: PCA会丢失信息吗？

**A**: 

**会丢失信息**，但PCA会：
- 保留最重要的信息（方差最大的方向）
- 丢弃冗余信息（方差小的方向）

**示例**：
```python
# 原始数据：4维
X_original = iris.data  # 4维

# PCA降维：2维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_original)  # 2维

# 解释方差：通常能保留90%+的信息
explained_variance = pca.explained_variance_ratio_.sum()
print(f"保留了 {explained_variance:.2%} 的信息")
```

**权衡**：
- 降维越多，丢失信息越多
- 但可以去除噪声和冗余
- 通常保留80-95%的信息即可

---

### Q4: PCA需要标准化数据吗？

**A**: 

**强烈建议标准化**！

**原因**：
- PCA基于方差，如果特征量纲不同，方差大的特征会主导
- 标准化后，所有特征在相同尺度上

**示例**：
```python
from sklearn.preprocessing import StandardScaler

# 错误：未标准化
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # 可能不准确

# 正确：先标准化
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)  # 更准确
```

---

### Q5: PCA和SVD有什么关系？

**A**: 

**PCA可以通过SVD实现**：

**传统方法**：
1. 计算协方差矩阵
2. 特征值分解
3. 取前k个特征向量

**SVD方法**：
1. 对数据矩阵进行SVD分解
2. 直接得到主成分

**代码对比**：
```python
# 方法1：使用PCA（内部使用SVD）
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 方法2：直接使用SVD
from numpy.linalg import svd
U, s, Vt = svd(X_scaled, full_matrices=False)
X_reduced_svd = U[:, :2] @ np.diag(s[:2])  # 等价结果
```

**SVD优势**：
- 数值稳定性更好
- 计算更高效（大数据）
- scikit-learn的PCA内部使用SVD

---

## 代码实现问题

### Q6: 如何从零实现PCA？

**A**: 核心步骤：

```python
import numpy as np

def pca_from_scratch(X, n_components):
    """从零实现PCA"""
    # 1. 标准化
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_scaled = (X - X_mean) / X_std
    
    # 2. 计算协方差矩阵
    cov_matrix = np.cov(X_scaled.T)
    
    # 3. 特征值分解
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    
    # 4. 排序（按特征值降序）
    idx = eigenvalues.argsort()[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    # 5. 选择前n个主成分
    components = eigenvectors[:, :n_components]
    
    # 6. 投影
    X_reduced = X_scaled @ components
    
    return X_reduced, components, eigenvalues
```

---

### Q7: 如何可视化主成分？

**A**: 

```python
# 1. 可视化主成分权重
components = pca.components_
plt.figure(figsize=(12, 5))

for i in range(2):
    plt.subplot(1, 2, i+1)
    plt.bar(range(len(components[i])), components[i])
    plt.title(f'主成分{i+1}')
    plt.xlabel('原始特征')
    plt.ylabel('权重')

plt.tight_layout()
plt.show()

# 2. 可视化降维结果
plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap='viridis')
plt.xlabel('主成分1')
plt.ylabel('主成分2')
plt.title('PCA降维结果')
plt.show()
```

---

## 实际应用问题

### Q8: PCA在哪些场景中应用？

**A**: 

**常见应用**：

1. **数据可视化**：
   - 高维数据降到2-3维可视化
   - 探索数据分布

2. **特征提取**：
   - 减少特征数量
   - 去除噪声和冗余

3. **数据压缩**：
   - 减少存储空间
   - 加速计算

4. **降维预处理**：
   - 在机器学习前降维
   - 减少过拟合风险

更多场景请参考：[实战场景库.md](./实战场景库.md)

---

### Q9: PCA可以用于分类吗？

**A**: 

**PCA本身不是分类器**，但可以：
- 作为预处理步骤，降维后再分类
- 提高分类性能（去除噪声）
- 加速训练（维度降低）

**示例**：
```python
# 1. PCA降维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)

# 2. 在降维数据上分类
from sklearn.svm import SVC
classifier = SVC()
classifier.fit(X_reduced, y)
```

**注意**：
- PCA是无监督的（不使用标签）
- 如果目标是分类，可以考虑LDA（线性判别分析）

---

## 错误排查

### Q10: 报错"ValueError: n_components must be between 0 and min(n_samples, n_features)"

**A**: 

**原因**：主成分数量设置不当

**解决**：
```python
# 检查数据维度
print(f"样本数: {X.shape[0]}, 特征数: {X.shape[1]}")

# 确保n_components <= min(n_samples, n_features)
n_components = min(2, X.shape[0], X.shape[1])
pca = PCA(n_components=n_components)
```

---

### Q11: 降维后数据无法解释

**A**: 

**原因**：主成分是原始特征的线性组合，不是原始特征

**解决**：
1. **查看主成分权重**：理解每个主成分的含义
2. **使用特征选择**：如果需要可解释性，使用特征选择而不是PCA
3. **可视化主成分**：观察主成分的权重分布

---

## 📖 更多资源

- **快速上手**：[00_快速上手.md](./00_快速上手.md)
- **学习检查点**：[学习检查点.md](./学习检查点.md)
- **实战场景库**：[实战场景库.md](./实战场景库.md)

---

**如果这里没有你遇到的问题，请查看其他资源！** 💪
