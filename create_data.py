import _pickle as pickle
from RN16_Test import getList
from plot_data import plot_IQ_timeseries1
from extract_RN16 import get_Ampl
import numpy as np

import sys
import math

from Time_Intertval_Error import exact_PB_feature
from Time_Intertval_Error import exact_TIE_feature
from regressor import exact_aTIE_feature
from Time_Intertval_Error import get_Average_Ampl
from Time_Intertval_Error import exact_TIE_feature_fake



def get_Train_Test_Data(file, low, high):
    aTIE = exact_aTIE_feature(file, low, high).flatten()
    PB = exact_PB_feature(file, low, high)

    print(aTIE.shape)
    print(aTIE)
    print(PB.shape)
    print(PB)
    count = 0 #样本个数
    feature = []
    while count < len(aTIE):
        feature_temp = []
        feature_temp.append(aTIE[count])
        feature_temp.append(PB[count])
        feature.append(feature_temp)
        count = count + 1
    
    
    feature = np.array(feature, dtype = np.float)
    f = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/feature19.dat', 'wb')
    pickle.dump(feature, f)







if __name__ == "__main__":

    # file = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/19preamble.pickle'
    # IQcommplex = getList(file)
    # print(IQcommplex.shape)
    # get_Train_Test_Data(file, 0.015, 0.05)

    
    feature_file = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/feature14.dat'
    f_read = open(feature_file, 'rb')
    feature = np.array(pickle.load(f_read), dtype=np.float)
    print(feature.shape)

    f_feature_select = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/feature_select_14.dat', 'wb')
    f_aTIE = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/aTIE14.dat', 'wb')


    feature_select = []
    aTIE = []
    PB  = [] 
    count = 0   


    # for tmp in feature:
    #     aTIE.append(tmp[0])
    #     PB.append(tmp[1])

    # print(np.around(aTIE, 4))
    # print(np.around(PB, 4))    


    for tmp in feature:
        # 6 < tmp[0] and tmp[0] < 7 and
        # and 0.008 < tmp[1] and tmp[1]<0.0099
        if(True):
            feature_select.append(tmp)
            aTIE.append(tmp[0])
           
        count = count + 1

    feature_select = np.array(feature_select)
    aTIE = np.array(aTIE)
    print(feature_select.shape)
    print(aTIE.shape)
    pickle.dump(feature_select,f_feature_select)
    pickle.dump(aTIE, f_aTIE)
    print(np.around(feature_select,4))





    
