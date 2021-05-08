import _pickle as pickle
from RN16_Test import getList
from plot_data import plot_IQ_timeseries1
from extract_RN16 import get_Ampl
import numpy as np

import sys
import math


#提取特征TIE----------初始index为0
def exact_TIE_feature(file, low, high):
     #获取文件中

    IQcomplex = getList(file)
    print(len(IQcomplex))
    
    TIE =[] #存储一个文件中若干个RN16_Preamble的上升沿下降沿的index，其中元素为24个沿（12+12）
    count  = 1
    for tmp in IQcomplex:
        # print(len(tmp))
        index = 0
        TIE_tmp = []
        while index < (len(tmp) - 5):
            TIE_Index = tmp[index : index + 5]
            TIE_Index_after = tmp[index + 5 : index + 5 + 5]

            if len(TIE_tmp) == 24:
                TIE_tmp[0] = 0
                TIE.append(TIE_tmp)
                # print("GET" + str(count))
                # count += 1
                break
            elif len(TIE_tmp) > 24:
                print("false")
                sys.exit(0)
                
            if index >= 5:
                TIE_Index_before = tmp[index - 5 : index]
                if get_Ampl(TIE_Index_before) < low and get_Ampl(TIE_Index_after) < low and get_Ampl(TIE_Index) > high:

                    if len(TIE_tmp) > 0:
                        TIE_tmp.append(index + 2 - TIE_tmp[0])
                        
                    else:
                        TIE_tmp.append(index + 2)

                    index = index + 5
                    continue
            
            else:
                if get_Ampl(TIE_Index_after) < low and get_Ampl(TIE_Index) > high:

                    if len(TIE_tmp) > 0:
                        TIE_tmp.append(index + 2 - TIE_tmp[0])

                    else:
                        TIE_tmp.append(index + 2)

                    index = index + 5
                    continue

            index = index + 1

    return np.array(TIE)




#提取特征TIE----------初始index为真实位置
def exact_TIE_feature_fake(file, low, high):
     #获取文件中

    IQcomplex = getList(file)
    print(len(IQcomplex))
    
    TIE =[] #存储一个文件中若干个RN16_Preamble的上升沿下降沿的index，其中元素为24个沿（12+12）
    count  = 1
    for tmp in IQcomplex:
        # print(len(tmp))
        index = 0
        TIE_tmp = []
        while index < (len(tmp) - 5):
            TIE_Index = tmp[index : index + 5]
            TIE_Index_after = tmp[index + 5 : index + 5 + 5]

            if len(TIE_tmp) == 24:
                # TIE_tmp[0] = 0
                TIE.append(TIE_tmp)
                # print("GET" + str(count))
                # count += 1
                break
            elif len(TIE_tmp) > 24:
                print("false")
                sys.exit(0)
                
            if index >= 5:
                TIE_Index_before = tmp[index - 5 : index]
                if get_Ampl(TIE_Index_before) < low and get_Ampl(TIE_Index_after) < low and get_Ampl(TIE_Index) > high:

                    if len(TIE_tmp) > 0:
                        TIE_tmp.append(index + 2)
                        
                    else:
                        TIE_tmp.append(index + 2)

                    index = index + 5
                    continue
            
            else:
                if get_Ampl(TIE_Index_after) < low and get_Ampl(TIE_Index) > high:

                    if len(TIE_tmp) > 0:
                        TIE_tmp.append(index + 2)

                    else:
                        TIE_tmp.append(index + 2)

                    index = index + 5
                    continue

            index = index + 1

    return np.array(TIE)



#获取高状态或者低状态的平均幅值
def get_Average_Ampl(IQcomplex):
    Ampl = []
    for tmp in IQcomplex:
        ampl_tmp = np.sqrt(tmp.real ** 2 + tmp.imag ** 2) 
        Ampl.append(ampl_tmp)

    return sum(Ampl)/len(Ampl)


#提取特征PB
def exact_PB_feature(file, low, high):
    IQcomplex = getList(file)
    print(len(IQcomplex))
    PB = []#存储一个文件中若干个RN16_Preamble的PB,平均功率，其中元素为24个沿（12+12），即为12个脉冲，计算公式为1/2(PBi_high-PBi_low)^2, 再求12个脉冲的平均
    TIE = exact_TIE_feature_fake(file, low, high)
    count = 0 #指示当前是第一个RN16_preamble
    for tmp in IQcomplex:
        index_point_array = TIE[count]#脉冲上升和下降沿的index，截取相应的高低状态
        # count = count + 1
        i = 0
        PB_tmp = []
        while(i < len(index_point_array) -1):
            point_1 = index_point_array[i]
            point_2 = index_point_array[i+1]

            PB_array = tmp[point_1 + 5 : point_2 - 5]
            average_Ampl = get_Average_Ampl(PB_array)
            PB_tmp.append(average_Ampl)
            i = i + 1
        
        #处理最后一个平均幅值
        point = index_point_array[len(index_point_array) - 1]
        # point_2 = index_point_array[len(index_point_array) - 1]
        PB_array = tmp[point + 5 : point + 5 + 30]
        average_Ampl = get_Average_Ampl(PB_array)
        PB_tmp.append(average_Ampl)

        #利用公式1/2(PBi_high-PBi_low)^2求平均基带功率
        PBi = []#存储每个脉冲的真正幅值，PB_tmp包含了24个高低状态,PBi存储12个脉冲的幅值
        j = 0

        # print(PB_tmp)

        while(j < len(PB_tmp)):
            PBi.append(math.pow((PB_tmp[j] - PB_tmp[j + 1]), 2)*0.5)
            j = j +2

        #得到12个脉冲的基带功率，然后求平均
        
        PB.append(sum(PBi) * 1.0 / len(PBi))

        count = count + 1 #下一个RN16_preamble的TIE列表
    
    return np.array(PB)



if __name__ == '__main__':

    file = 'H:/data/3preamble_RN16_.pickle'

    print('...one...')

    #获取文件中

    # IQcomplex = getList(file)
    # print(len(IQcomplex))


    # count = len(IQcomplex) -1
    # for tmp in IQcomplex:
    #     # plot_IQ_timeseries2(tmp, IQIndex=0, bins=200)
    #     # plot_IQ_timeseries2(IQcomplex[count], IQIndex=0, bins=200)
    #     plot_IQ_timeseries1(tmp, IQIndex=0, bins=1230)
    #     plot_IQ_timeseries1(IQcomplex[count], IQIndex=0, bins=1230)
    #     count -= 1

    # TIE = exact_TIE_feature(file, 0.02, 0.04)
    # print(TIE[0])
    # print(TIE[1])
    # print(TIE[2])

    PB = exact_PB_feature(file, 0.02, 0.04)
    print(PB.shape)
    print(PB)


 



            


                


            



    






