import _pickle as pickle
from RN16_Test import getList
from plot_data import plot_IQ_timeseries1
from extract_RN16 import get_Ampl
import numpy as np

import matplotlib.pyplot as plt
import scipy.signal

import sys
import math

from Time_Intertval_Error import exact_TIE_feature_fake
def exact_high_low_state_feature(file, low, high):
    IQcomplex = getList(file)
    print(len(IQcomplex))

    high_low_state = []#每个preamble有12个脉冲，所以包含24个高低状态
    TIE = exact_TIE_feature_fake(file, low, high)

    print("TIE.sahpe", len(TIE))
    count = 0#指示当前是第一个RN16_preable

    for tmp in IQcomplex:
        index_point_array = TIE[count]#脉冲上升和下降沿的index，截取相应的高低状态
        i = 0
        state = [] #每组24个高低状态
        while (i < len(index_point_array)):
            point_1 = index_point_array[i]
            # point_2 = index_point_array[i+1]
            array = tmp[point_1 + 5 : point_1 + 5 + 30]
            state.append(array)
            i = i + 1

        high_low_state.append(state)
        count = count + 1

    return np.array(high_low_state, dtype=complex)

def double_loop(array):
    i = 0
    j = 0
    COV = []
    while(i < len(array)):
        j = i + 1
        while(j < len(array)):    
            z = np.vstack((array[i], array[j]))
            COV.append(z[0][1])
            j = j + 1

        i = i + 1

    return COV

def exact_COV_feature(file, low, high):
    high_low_state = exact_high_low_state_feature(file, low, high)#shape(900, 24,30)
    COV = []
    for tmp in high_low_state:
        #tmp.shape = (24,30),即900个中的一个preamble
        #将24个分成高低两组
        high_state = []
        low_state = []
        i = 1 #计数分类，奇数和偶数各分一组
        for state in tmp:
            if i % 2 == 1:
                high_state.append(state)
            if i % 2 == 0:
                low_state.append(state)
            i = i + 1
        
        high_state_cov = [] #存储高状态的cov值
        low_state_cov = [] # 存储低状态的cov值

        #C(11,12) = 66
        high_state_cov =  double_loop(high_state)
        low_state_cov =   double_loop(low_state)

        COV_tmp = []
        for tmp in high_state_cov:
            COV_tmp.append(tmp)

        for tmp in low_state_cov:
            COV_tmp.append(tmp)

        COV.append(COV_tmp)

    return np.array(COV,dtype=complex)



#定义自相关函数
def xcorr(data):
    length = len(data)
    R = []
    for m in range(0, length):
        sum = 0.0
        for n in range(0, length - m):
            sum = sum + data[n]*data[n+m]
        R.append(sum/length)
    
    return R


def exact_PSD_feature(file):
    IQcommplex = getList(file)
    print(IQcommplex.shape)

    N = 1230 #采样点数
    p = 512#AR的模型阶数

    n = np.linspace(0, N - 1, N)
    PSD = []
    count = 1
    for tmp in IQcommplex:
            
        yn = tmp

        PSD_tmp = []
        #求自相关
        R = xcorr(yn)

        #构建矩阵及其向量
        A = np.ones([p, p], dtype=complex)
        for i in range(0, p):
            for j in range(0, p):
                A[i][j] = R[abs(i - j)]
        
        C = np.ones([p, 1], dtype=complex)
        for i in range(0, p):
            C[i][0] = -R[i + 1]
        
        #求系数a
        B = np.ones([p, 1])

        A_inv = np.linalg.inv(A)
        B = np.dot(A_inv, C)
        BT = B.T
        #将a0 = 1插入到系数向量里
        a = np.insert(BT, 0, [1])

        #求G
        G2 = R[0]
        for i in range(1, p):
            G2 = G2 + B[i][0] * R[i]
        G = np.sqrt(G2)

        #计算频率响应
        w ,h = scipy.signal.freqz(G, a, worN=N)
        Hf = abs(h)
        Sx = Hf**2
        f = w/(np.pi * 2)
        # plt.scatter(f, Sx)
        # plt.title("PSD")

        # plt.xlabel('f')
        # plt.ylabel('Sx')
        # plt.show()
        # print(Hf.shape, Sx.shape)

        count = count + 1
        # if count > 20:
        #     break
        PSD.append(Sx[:20]/math.pow(10,5))
        # print(len(PSD))
        
    
    PSD = np.array(PSD)
    print(PSD.shape)
    # print(PSD)
    return PSD


def sort_Data(file, low, high):
    COV = exact_COV_feature(file, low, high)#(900, 132)
    PSD = exact_PSD_feature(file)#(900, 20)

    feature = []  #COV : 132  PSD : 20
    count = 0 #样本个数

    while(count < len(COV)):
        feature_tmp = []
        for tmp in COV[count]:
            feature_tmp.append(tmp)
        for tmp in PSD[count]:
            feature_tmp.append(tmp)
        feature.append(feature_tmp)
        count = count + 1

    
    feature = np.array(feature)

    print(feature.shape)
    print(feature[0])
    print(feature[1])
    print(feature[2])
    print(feature[3])
    f = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v2/feature2.dat', 'wb')
    pickle.dump(feature, f) 








if __name__ == '__main__':
    file = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/2preamble.pickle'
    IQcommplex = getList(file)
    print(IQcommplex.shape)
    # get_Train_Test_Data(file, 0.015, 0.05)
    # high_low_state = exact_high_low_state_feature(file, 0.015, 0.05)
    # print(len(high_low_state))
    # print(high_low_state[0])

    # cov = exact_COV_feature(file, 0.015, 0.05)
    # print(cov.shape)
    # print(cov[1])
    # print(cov[2])

    sort_Data(file, 0.015, 0.038)