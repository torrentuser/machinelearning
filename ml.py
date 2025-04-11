import numpy as np
# import matplotlib.pyplot as plt

import pandas as pd
# import seaborn as sns

boston = pd.read_csv('/opt/anaconda3/envs/machinelearning/1.csv') # 在此处使用绝对路径导入csv文件
boston.dropna(inplace=True)
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT','RM'])
Y = boston['MEDV']
from sklearn.model_selection import train_test_split

# splits the training and test data set in 80%：20%
# assign random_state to any value. This ensures consistency
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)

with open('./res.txt', 'a+') as f:
    f.write(str(X_train.shape))
    f.write(str(X_test.shape))
    f.write(str(Y_train.shape))
    f.write(str(Y_test.shape))
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
# model evaluation for training set

y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train,y_train_predict)

# 打印日志，日志必须保存至./results/，否则GPU/CPU Job时无法保存
with open('./res.txt', 'a+') as f:
    f.write("The model performance for training set\n")
    f.write("--------------------------------------\n")
    f.write('RMSE is {}'.format(rmse))
    f.write("\n")
    f.write('R2 score is {}'.format(r2))
    f.write("\n\n")

      
# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)

# root mean square error of the model
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))

# r-squared score of the model
r2 = r2_score(Y_test,y_test_predict)

# 打印日志
with open('./res.txt', 'a+') as f:
    f.write("The model performance for testing set\n")
    f.write("-------------------------------------\n")
    f.write('RMSE is {}'.format(rmse))
    f.write("\n")
    f.write('R2 score is {}'.format(r2))
    f.write("\n\n")

print("OK")