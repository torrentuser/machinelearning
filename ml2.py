# -*- coding: utf-8 -*-

'''在MNIST数据集上训练一个简单的卷积神经网络。'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
# from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D, GlobalAveragePooling2D
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator
# from keras.layers.normalization import BatchNormalization
from keras.models import Model
import keras
from keras.optimizers import Adam

np.random.seed(0xffffffff)

# 加载MNIST数据集
(X_train, y_train), (X_test, y_test) = mnist.load_data()

print("X_train original shape", X_train.shape)
print("y_train original shape", y_train.shape)
print("X_test original shape", X_test.shape)
print("y_test original shape", y_test.shape)
    
# 调整数据形状
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)

# 转换数据类型
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# 归一化数据
X_train /= 255
X_test /= 255
number_of_classes = 10

# 将标签转换为分类格式
Y_train = to_categorical(y_train, number_of_classes)
Y_test = to_categorical(y_test, number_of_classes)
    
# 构建更复杂的模型
model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=(28,28,1)))  # 增加卷积核数量
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))  # 添加Dropout层

model.add(Conv2D(128, (3, 3)))  # 增加卷积层
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024))  # 增加全连接层神经元数量
model.add(Activation('relu'))
model.add(Dropout(0.5))  # 添加Dropout层
model.add(Dense(10))
model.add(Activation('softmax'))
    
# 调整优化器的学习率
optimizer = Adam(learning_rate=0.0005)  # 降低学习率
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 扩展数据增强范围
gen = ImageDataGenerator(
    rotation_range=20,  # 增加旋转范围
    width_shift_range=0.2,  # 增加水平平移范围
    height_shift_range=0.2,  # 增加垂直平移范围
    shear_range=0.5,  # 增加剪切范围
    zoom_range=0.3,  # 增加缩放范围
    horizontal_flip=True  # 添加水平翻转
)

test_gen = ImageDataGenerator()

train_generator = gen.flow(X_train, Y_train, batch_size=64)
test_generator = test_gen.flow(X_test, Y_test, batch_size=64)

# 增加训练轮次
model.fit(train_generator, steps_per_epoch=60000//64, epochs=150,  # 增加到150轮
          validation_data=test_generator, validation_steps=1000//64)
    
# 评估模型
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test_score:',score[0])
print('test accuracy:',score[1])

# 打印日志
with open('./res.txt', 'a+') as f:
    f.write('Test_score:')
    f.write(str(score[0]))
    f.write("\n")
    f.write('test accuracy:')
    f.write(str(score[1]))
    f.write("\n\n")
    
# 保存模型
model.save('./model.h5')
print('Model Saved')

# 读取日志
with open('./res.txt', 'r') as f:
    res = f.read()
    print(res)
print('OK')

# 读取模型
model = keras.models.load_model('./model.h5')
print('Model Loaded')

# 预测
predictions = model.predict(X_test)
print(predictions[0])
print(np.argmax(predictions[0]))
print(y_test[0])
print(predictions[1])
print(np.argmax(predictions[1]))
print(y_test[1])

# 可视化
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.show()