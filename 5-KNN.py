import _pickle as pickle
from RN16_Test import getList
from plot_data import plot_IQ_timeseries1
from extract_RN16 import get_Ampl
import numpy as np

import sys
import math

def KNN(X_train, Y_train, X_test, k):
    #修改列表为numpy类型
    X_train = np.array(X_train)
    Y_train = np.array(Y_train)
    X_test = np.array(X_test)

    #获得训练、测试数据的长度
    X_train_len = len(X_train)
    X_test_len = len(X_test)
    pre_lable = [] #存储预测标签

    '''
    依次遍历测试数据，计算每个测试数据与训练数据的距离值，排序
    根据前K个投票结果选出预测结果
    '''

    for test_len in range(X_test_len):#测试第一组数据
        dis = []
        for train_len in range(X_train_len):
            temp_dis = abs(sum(X_train[train_len, :] - X_test[test_len, :]))#计算距离----有点问题
            dis.append(temp_dis**0.5)
        

        dis = np.array(dis)
        sort_id = dis.argsort()

        dic = {}
        for i in range(k):
            vlabel = Y_train[sort_id[i]] #对应的标签计数
            dic[vlabel] = dic.get(vlabel,0)+1
            #寻找vlable代表的标签，如果没有返回0并加一，如果已经存在返回改键值对应的值并加一
        max = 0
        for index, v in dic.items():
            if v > max:
                max = v
                maxIndex = index
        
        pre_lable.append(maxIndex)
    print(pre_lable)




if __name__ == "__main__":
    X_train = [[1, 2, 3, 4],
               [5, 6, 7, 8],
               [9, 10, 11, 12],
                [1, 2, 3, 4],
                [5, 6, 7, 8],
                [9, 10, 11, 12]]

    Y_train = [1,2,3,1,2,3]
    X_test = [[1,2,3,4],
                [5,6,7,8]]
    
    KNN(X_train, Y_train, X_test, 2)

    






