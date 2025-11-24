"""
简单项目2：图像分割
使用K-means对图像进行分割

本示例适合小白学习，包含大量详细注释
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from PIL import Image
import os

# ========== 1. 加载图像 ==========
print("=" * 60)
print("1. 加载图像")
print("=" * 60)

# 创建一个简单的测试图像（如果没有真实图像）
# 在实际应用中，你可以加载真实的图像文件
print("生成测试图像...")

# 创建一个简单的彩色图像（200x200像素）
# 包含3个不同颜色的区域
image = np.zeros((200, 200, 3), dtype=np.uint8)
image[0:67, :] = [255, 0, 0]    # 红色区域
image[67:133, :] = [0, 255, 0]  # 绿色区域
image[133:, :] = [0, 0, 255]    # 蓝色区域

# 添加一些噪声，使图像更真实
noise = np.random.randint(-20, 20, image.shape, dtype=np.int16)
image = np.clip(image.astype(np.int16) + noise, 0, 255).astype(np.uint8)

print(f"图像形状: {image.shape}")
print(f"图像数据类型: {image.dtype}")

# 显示原始图像
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.imshow(image)
plt.title('原始图像', fontsize=14)
plt.axis('off')

# ========== 2. 准备数据 ==========
print("\n" + "=" * 60)
print("2. 准备数据")
print("=" * 60)

# 将图像转换为特征向量
# 每个像素的RGB值作为一个样本
height, width, channels = image.shape
# 将图像重塑为 (height*width, channels) 的形状
# 每一行是一个像素的RGB值
X = image.reshape(-1, channels)

print(f"特征矩阵形状: {X.shape}")
print(f"  像素数: {X.shape[0]}")
print(f"  特征数（RGB通道）: {X.shape[1]}")

# 为了加快计算，可以采样部分像素
# 这里我们采样10%的像素
sample_size = int(0.1 * len(X))
X_sample = shuffle(X, random_state=42, n_samples=sample_size)

print(f"采样后样本数: {len(X_sample)}")

# ========== 3. K-means聚类 ==========
print("\n" + "=" * 60)
print("3. K-means聚类")
print("=" * 60)

# 选择K值（颜色数）
# 这里我们选择3，对应原始图像的3个颜色区域
n_colors = 3

print(f"聚类颜色数: {n_colors}")

# 创建K-means聚类器
kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)

# 训练模型（只使用采样数据）
print("训练模型...")
kmeans.fit(X_sample)

# 获取聚类中心（代表颜色）
colors = kmeans.cluster_centers_.astype(np.uint8)
print(f"\n聚类中心（代表颜色）:")
for i, color in enumerate(colors):
    print(f"  颜色 {i}: RGB({color[0]}, {color[1]}, {color[2]})")

# ========== 4. 预测所有像素 ==========
print("\n" + "=" * 60)
print("4. 预测所有像素")
print("=" * 60)

# 对所有像素进行预测
# 预测每个像素属于哪个颜色簇
print("预测所有像素的簇...")
labels = kmeans.predict(X)

# 将标签转换为颜色
# 使用聚类中心的颜色替换原始像素
segmented_image = colors[labels].reshape(height, width, channels)

print("分割完成！")

# ========== 5. 可视化结果 ==========
print("\n" + "=" * 60)
print("5. 可视化结果")
print("=" * 60)

# 显示分割后的图像
plt.subplot(1, 2, 2)
plt.imshow(segmented_image)
plt.title(f'K-means分割结果 (K={n_colors})', fontsize=14)
plt.axis('off')

plt.tight_layout()
plt.savefig('image_segmentation.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 6. 测试不同K值 ==========
print("\n" + "=" * 60)
print("6. 测试不同K值")
print("=" * 60)

# 测试不同的K值
k_values = [2, 3, 5, 8]
fig, axes = plt.subplots(2, 2, figsize=(12, 12))

for idx, k in enumerate(k_values):
    # 训练模型
    kmeans_k = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans_k.fit(X_sample)
    
    # 预测
    labels_k = kmeans_k.predict(X)
    colors_k = kmeans_k.cluster_centers_.astype(np.uint8)
    segmented_k = colors_k[labels_k].reshape(height, width, channels)
    
    # 可视化
    ax = axes[idx // 2, idx % 2]
    ax.imshow(segmented_k)
    ax.set_title(f'K={k}', fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.savefig('image_segmentation_k_values.png', dpi=300, bbox_inches='tight')
plt.show()

print("\n" + "=" * 60)
print("项目完成！")
print("=" * 60)
print("""
总结：
1. K-means可以用于图像分割
2. 将每个像素的RGB值作为特征
3. K值决定分割后的颜色数
4. 可以用于图像压缩和简化
""")

