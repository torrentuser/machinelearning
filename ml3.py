# 导入 TensorFlow 和 tf.keras
import tensorflow as tf

# 导入辅助库
import numpy as np
import matplotlib.pyplot as plt

# 打印 TensorFlow 的版本
print(tf.__version__)

# 加载 Fashion MNIST 数据集
fashion_mnist = tf.keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# 定义类别名称
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# 打印训练集和测试集的形状及标签数量
print(train_images.shape)
print(len(train_labels))
print(train_labels)
print(test_images.shape)
print(len(test_labels))

# 显示训练集中的第一张图像
plt.figure()
plt.imshow(train_images[0])
plt.colorbar()
plt.grid(False)
plt.show()

# 将图像像素值归一化到 0 到 1 之间
train_images = train_images / 255.0
test_images = test_images / 255.0

# 显示前 25 张训练图像及其标签
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[train_labels[i]])
plt.show()

# 构建神经网络模型
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),  # 将 28x28 的图像展平为一维
    tf.keras.layers.Dense(128, activation='relu'),  # 全连接层，128 个神经元，ReLU 激活函数
    tf.keras.layers.Dense(10)  # 输出层，10 个神经元（对应 10 个类别）
])

# 编译模型，指定优化器、损失函数和评估指标
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# 训练模型，迭代 10 个周期
model.fit(train_images, train_labels, epochs=200)

# 在测试集上评估模型性能
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# 创建带有 Softmax 层的概率模型
probability_model = tf.keras.Sequential([model, 
                                         tf.keras.layers.Softmax()])

# 对测试集进行预测
predictions = probability_model.predict(test_images) 
predictions[0]  # 打印第一个测试样本的预测结果
np.argmax(predictions[0])  # 获取预测的类别索引
test_labels[0]  # 打印第一个测试样本的真实标签

# 定义函数：绘制图像及预测结果
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
    color = 'red'  # 预测错误为红色

  plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

# 定义函数：绘制预测值的柱状图
def plot_value_array(i, predictions_array, true_label):
  true_label = true_label[i]
  plt.grid(False)
  plt.xticks(range(10))
  plt.yticks([])
  thisplot = plt.bar(range(10), predictions_array, color="#777777")
  plt.ylim([0, 1])
  predicted_label = np.argmax(predictions_array)

  thisplot[predicted_label].set_color('red')  # 预测的类别为红色
  thisplot[true_label].set_color('blue')  # 真实的类别为蓝色

# 绘制第一个测试样本的预测结果
i = 0
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# 绘制第 12 个测试样本的预测结果
i = 12
plt.figure(figsize=(6,3))
plt.subplot(1,2,1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1,2,2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# 绘制前 15 张测试图像及其预测结果
num_rows = 5
num_cols = 3
num_images = num_rows*num_cols
plt.figure(figsize=(2*2*num_cols, 2*num_rows))
for i in range(num_images):
  plt.subplot(num_rows, 2*num_cols, 2*i+1)
  plot_image(i, predictions[i], test_labels, test_images)
  plt.subplot(num_rows, 2*num_cols, 2*i+2)
  plot_value_array(i, predictions[i], test_labels)
plt.tight_layout()
plt.show()

# 从测试集中获取一张图像
img = test_images[1]
print(img.shape)

# 将图像添加到一个批次中（批次大小为 1）
img = (np.expand_dims(img,0))
print(img.shape)

# 对单张图像进行预测
predictions_single = probability_model.predict(img)
print(predictions_single)

# 绘制单张图像的预测结果
plot_value_array(1, predictions_single[0], test_labels)
_ = plt.xticks(range(10), class_names, rotation=45)
plt.show()

# 打印预测的类别索引
np.argmax(predictions_single[0])