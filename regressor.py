import numpy as np
np.random.seed(1337)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Time_Intertval_Error import exact_PB_feature
from Time_Intertval_Error import exact_TIE_feature

from RN16_Test import getList


def exact_aTIE_feature(file, low, high):

    aTIE = []

    TIE = exact_TIE_feature(file, low, high)
    print(TIE.shape)   
    #建模
    model = LinearRegression() 

    #create data
    X_train = np.linspace(0, 1, TIE.shape[1])

    for tmp in TIE:
        Y_train = np.array(tmp) - np.linspace(0, 1150, 24)
        # print(Y_train)

         #model.fit model.score 需要传递二维列表
        X_train = np.array(X_train).reshape(-1,1)
        Y_train = np.array(Y_train).reshape(-1,1)  

        model.fit(X_train, Y_train) 

        # c = model.intercept_
        m = model.coef_

        aTIE.append(abs(m))
    

    return np.array(aTIE)


  





if __name__ == "__main__":
        
    file = 'H:/data/3preamble_RN16_.pickle'
    IQcommplex = getList(file)
    print(IQcommplex.shape)

    # TIE = exact_TIE_feature(file, 0.02, 0.04)
    # print(TIE.shape)
    aTIE = exact_aTIE_feature(file, 0.02, 0.04).flatten()
    print(aTIE.shape)

    print(aTIE[0],aTIE[1])




# #建模
# model = LinearRegression()



# #create data

# X_train = np.linspace(0, 1, TIE.shape[1])
# # np.random.shuffle(X)
# #shuffle
# # Y = 0.25*X + 2.3 #+ np.random.normal(0, 0.05, (200, ))
# Y_train = np.array(TIE[0]) - np.linspace(0, 1150, 24)
# print(Y_train)

# #plot data
# # plt.scatter(X, Y)
# # plt.show()

# # X_train, Y_train = X[:160], Y[:160]
# # X_test, Y_test = X[160:], Y[160:]

# #model.fit model.score 需要传递二维列表
# X_train = np.array(X_train).reshape(-1,1)
# Y_train = np.array(Y_train).reshape(-1,1)

# model.fit(X_train, Y_train)

# c = model.intercept_
# m = model.coef_

# print(m, c) 

# # 可视化
# Y_pred = model.predict(X_train)
# plt.scatter(X_train, Y_train)
# plt.plot(X_train, Y_pred)
# plt.show()








# # import tensorflow

# import os
# os.environ['KERAS_BACKEND']='theano'

# # import keras
# # print(os.environ['KERAS_BACKEND'])


# import os 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# # 导入模块并创建数据
# import numpy as np
# np.random.seed(1337)

# from keras.models import Sequential
# from keras.layers import Dense


# import matplotlib.pyplot as plt


# #create data
# X = np.linspace(0, 1, 2000)
# # np.random.shuffle(X)
# #shuffle
# Y = 0.5*X + 2 #+ np.random.normal(0, 0.05, (200, ))

# #plot data
# # plt.scatter(X, Y)
# # plt.show()

# X_train, Y_train = X[:1600], Y[:1600]
# X_test, Y_test = X[1600:], Y[1600:]


# #建立模型
# model = Sequential()
# model.add(Dense(1, input_dim = 1))

# # 激活模型

# model.compile(optimizer='sgd', loss='mse')

# # 训练模型
# print('Training....')
# for step in range(300):
#     cost = model.train_on_batch(X_train, Y_train)
#     if step % 50 == 0:
#         print("train cost:", cost)


# # 检验模型
# print('\nTesting......')

# cost = model.evaluate(X_test, Y_test, batch_size=20)
# print('test cost:', cost)
# print('metric:', model.metrics_names)
# W, b = model.layers[0].get_weights()
# print(W, b)

# # 可视化
# Y_pred = model.predict(X_test)
# plt.scatter(X_test, Y_test)
# plt.plot(X_test, Y_pred)
# plt.show()



