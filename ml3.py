# TensorFlow 和 tf.keras
import tensorflow as tf

# 辅助库
import numpy as np
import matplotlib.pyplot as plt

# 打印 TensorFlow 的版本
print(tf.__version__)

# 加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist

# 加载训练集和测试集
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 定义类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印训练集和测试集的形状
print(train_images.shape)  # 打印训练集的形状
print(len(train_labels))   # 打印训练集的标签数量
print(train_labels)        # 打印训练集的标签
print(test_images.shape)   # 打印测试集的形状
print(len(test_labels))    # 打印测试集的标签数量

# 显示第一张训练图像
plt.figure()
plt.imshow(train_images[0])  # 显示第一张图像
plt.colorbar()               # 添加颜色条
plt.grid(False)              # 关闭网格
plt.show()

# 将像素值归一化到 [0, 1] 范围
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示前 25 张训练图像及其标签
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)  # 创建 5x5 的子图
    plt.xticks([])           # 移除 x 轴刻度
    plt.yticks([])           # 移除 y 轴刻度
    plt.grid(False)          # 关闭网格
    plt.imshow(train_images[i], cmap=plt.cm.binary)  # 显示图像
    plt.xlabel(class_names[train_labels[i]])         # 显示标签
plt.show()

# 将训练集和测试集的形状扩展为 (28, 28, 1)，以适配卷积层输入
train_images = np.expand_dims(train_images, axis=-1)
test_images = np.expand_dims(test_images, axis=-1)

# 检查新形状
print("训练集形状:", train_images.shape)
print("测试集形状:", test_images.shape)

# 数据增强：通过随机翻转、旋转和缩放增强数据多样性
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),  # 随机水平翻转
    tf.keras.layers.RandomRotation(0.1),      # 随机旋转
    tf.keras.layers.RandomZoom(0.1),          # 随机缩放
])

# 改进后的模型
model = tf.keras.Sequential([
    data_augmentation,  # 数据增强
    tf.keras.layers.Rescaling(1./255, input_shape=(28, 28, 1)),  # 归一化
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),  # 卷积层，提取特征
    tf.keras.layers.MaxPooling2D((2, 2)),  # 最大池化层，降维
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层
    tf.keras.layers.MaxPooling2D((2, 2)),  # 最大池化层
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),  # 卷积层
    tf.keras.layers.Flatten(),  # 展平层，将多维特征展平为一维
    tf.keras.layers.Dense(128, activation='relu'),  # 全连接层
    tf.keras.layers.Dropout(0.5),  # Dropout 层，防止过拟合
    tf.keras.layers.Dense(10)  # 输出层，10 个类别
])

# 编译模型
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # 优化器
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),  # 损失函数
              metrics=['accuracy'])  # 评估指标

# 训练模型
model.fit(train_images, train_labels, epochs=20, validation_split=0.2)  # 训练 20 个轮次，使用 20% 的数据作为验证集

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)  # 在测试集上评估模型
print('\nTest accuracy:', test_acc)  # 打印测试准确率

# 创建带有 Softmax 的概率模型
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])
predictions = probability_model.predict(test_images)  # 对测试集进行预测
print(predictions[0])  # 打印第一张图像的预测结果
print(np.argmax(predictions[0]))  # 打印预测的类别
print(test_labels[0])  # 打印真实类别

# 绘制图像及其预测结果
def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'  # 预测正确为蓝色
    else:
        color = 'red'   # 预测错误为红色

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100 * np.max(predictions_array),
                                         class_names[true_label]),
                                         color=color)

# 绘制预测值的柱状图
def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')  # 预测类别为红色
    thisplot[true_label].set_color('blue')     # 真实类别为蓝色

# 显示第 0 张图像及其预测结果
i = 0
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 显示第 12 张图像及其预测结果
i = 12
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i], test_labels)
plt.show()

# 显示前 15 张测试图像及其预测结果
num_rows = 5
num_cols = 3
num_images = num_rows * num_cols
plt.figure(figsize=(2 * 2 * num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1)
    plot_image(i, predictions[i], test_labels, test_images)
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2)
    plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 从测试集中获取一张图像
img = test_images[1]
print(img.shape)

# 将图像添加到批次中（批次大小为 1）
img = (np.expand_dims(img, 0))
print(img.shape)

# 对单张图像进行预测
predictions_single = probability_model.predict(img)
print(predictions_single)
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()
print(np.argmax(predictions_single[0]))