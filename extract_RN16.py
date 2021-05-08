import _pickle as pickle
from load_data import load_data
from plot_data import plot_IQ_timeseries, plot_IQ_timeseries1
import numpy as np


#得到信号的幅度,然后再求得最大点和最小点
def get_Ampl(IQcomplex):
    Ampl = []
    for tmp in IQcomplex:
        ampl_tmp = np.sqrt(tmp.real ** 2 + tmp.imag ** 2) 
        Ampl.append(ampl_tmp)
    
    Ampl_true = max(Ampl) - min(Ampl)
    return Ampl_true


# 抽取整个信号中的整段RN16信号
def get_RN16(filename1, save_to_file_path, number = 700, IQindex = 0, low1 = 0, high1 = 0, noise1 = 0):
    
    fft_size = 1024
    #先从源文件得到复数列表 
    IQcomplex_reader_with_tag = load_data(size=fft_size, filename= filename1)
    # print(len(IQcomplex_reader_with_tag))
    
    # 将提取的RN16片段存入Tag2_93_RN16.pickle文件中
    Tag2_9x_RN16 = open(save_to_file_path, 'wb')

    # 样本总数
    samples_count = len(IQcomplex_reader_with_tag)

    # 遍历整个样本点
    i = IQindex #偏移点
    count = 0
    tmp = 0
    while i <= (samples_count-65800):
        IQcomplex_RN16 = IQcomplex_reader_with_tag[i : i+3700]
        Ampl = get_Ampl(IQcomplex_RN16)#0.03<ampl<0.1

        if Ampl >= low1 and Ampl <= high1:
            IQcomplex_RN16_Before = IQcomplex_reader_with_tag[i : i+200]
            IQcomplex_RN16_Before_1 = IQcomplex_reader_with_tag[i-200 : i]
            Ampl_tmp_1 = get_Ampl(IQcomplex_RN16_Before) #0.03<ampl<0.1
            Ampl_tmp_1_1 = get_Ampl(IQcomplex_RN16_Before_1)#ampl<0.02

            # print(Ampl_tmp_1, Ampl_tmp_1_1)
            condition1 = Ampl_tmp_1 >= low1 and Ampl_tmp_1 <= high1 and Ampl_tmp_1_1< noise1

            IQcomplex_RN16_After = IQcomplex_reader_with_tag[i+3700-200 : i+3700]
            IQcomplex_RN16_After_1 = IQcomplex_reader_with_tag[i+3700 : i+3700+200] 
            Ampl_tmp_2 = get_Ampl(IQcomplex_RN16_After)#0.03<ampl<0.1
            Ampl_tmp_2_2 = get_Ampl(IQcomplex_RN16_After_1)#ampl<0.02
            
            # print(Ampl_tmp_2 , Ampl_tmp_2_2)
            condition2 = Ampl_tmp_2 >= low1 and Ampl_tmp_2 <= high1 and Ampl_tmp_2_2< noise1

            # print('1')
            # if condition1:
            #     print('1')
            # if condition2:
            #     print('2')
            # print('111111')
            if condition1 and condition2:  #RN16波形的幅度大概范围
    
                pickle.dump(IQcomplex_reader_with_tag[i : i+3700], Tag2_9x_RN16)
                i += 30000
                count += 1
                tmp = i
                print('Get' + str(count))
                if(count >= number):
                    break
            else:
                i += 120#
                # print('no Get')
                # if((i - tmp) > 1000 and tmp > 0):
                #     i += 100
                #     tmp = i
              
        else:
            i += 120#
            # if((i - tmp) > 1000 and tmp > 0):
            #         i += 100
            #         tmp = i





if __name__ == '__main__':

    file = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/17_up/1'
    # 标签类型为Tag2_93
    file_path = []
    save_to_file_path = []

    for i in range(1, 10):
        file_name = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/17_up/source'
        file_name = file_name + str(i)
        file_path.append(file_name)

        save_to_file = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/17_up/'
        save_to_file = save_to_file + str(i) + '_RN16_.pickle'
        save_to_file_path.append(save_to_file)

        print(file_name, save_to_file)
    
    # file_path1 = []
    # save_to_file_path1 = []    
    # for i in range(1, 8):
    #     file_name1 = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/1.2m/5/'
    #     file_name1 = file_name1 + str(i)
    #     file_path1.append(file_name1)

    #     save_to_file1 = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/1.2m/5/'
    #     save_to_file1 = save_to_file1 + str(i) + '_RN16_.pickle'
    #     save_to_file_path1.append(save_to_file1)

    #     print(file_name1, save_to_file1)

    # file_path2 = []
    # save_to_file_path2 = [] 
    # for i in range(1, 5):
    #     file_name2 = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/1.2m/6/'
    #     file_name2 = file_name2 + str(i)
    #     file_path2.append(file_name2)

    #     save_to_file2 = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/1.2m/6/'
    #     save_to_file2 = save_to_file2 + str(i) + '_RN16_.pickle'
    #     save_to_file_path2.append(save_to_file2)

    #     print(file_name2, save_to_file2)
    
    
    
    
    
    for i in range(4, 9):
        # get_RN16(file_path_Tag2_93[i], save_to_file_path_Tag2_93[i], IQindex=50*1000)
        get_RN16(file_path[i], save_to_file_path[i], 700, 50*6000, 0.04, 0.1, 0.03)
    

    # for i in range(0, 7):
    #     # get_RN16(file_path_Tag2_93[i], save_to_file_path_Tag2_93[i], IQindex=50*1000)
    #     get_RN16(file_path1[i], save_to_file_path1[i], 50*1000, 0.025, 0.07, 0.025)
        

    # for i in range(0, 4):
    #     # get_RN16(file_path_Tag2_93[i], save_to_file_path_Tag2_93[i], IQindex=50*1000)
    #     get_RN16(file_path2[i], save_to_file_path2[i], 50*1000, 0.02, 0.07, 0.02)


    # for i in range(0, 15):
        # get_RN16(file_path_Tag2_93[i], save_to_file_path_Tag2_93[i], IQindex=50*1000)
        # get_RN16(file_path_Tag2_97[i], save_to_file_path_Tag2_97[i], IQindex=50*1000)
        # pass

'''  
    fft_size = 1024
    # 先通过load_data获得source数据的复数样本C
    file1 = './misc/data/Tag2_93/source3'
    file4 = './misc/data/Tag2_97/source1'
    IQcomplex_reader_with_tag = load_data(size=fft_size, filename= file1)
    # print(len(IQcomplex_reader_with_tag))
    
    # 将提取的RN16片段存入Tag2_93_RN16.pickle文件中

    file_path = './misc/data/Tag2_93/Tag2_93_RN16.pickle'
    file_path_4 = './misc/data/Tag2_97/Tag2_97_RN16.pickle'

    Tag2_93_RN16 = open(file_path, 'wb')

    
    # for complex_tmp in IQcomplex_reader_with_tag:
    # 样本总数
    samples_count = len(IQcomplex_reader_with_tag)

    # 遍历整个样本点
    i = 110*1000
    while i <= (samples_count):
        IQcomplex_RN16 = IQcomplex_reader_with_tag[i : i+1200]
        Ampl = get_Ampl(IQcomplex_RN16)#0.03<ampl<0.1

        if Ampl >= 0.03 and Ampl <= 0.1:
            IQcomplex_RN16_Before = IQcomplex_reader_with_tag[i : i+100]
            IQcomplex_RN16_Before_1 = IQcomplex_reader_with_tag[i-100 : i]
            Ampl_tmp_1 = get_Ampl(IQcomplex_RN16_Before) #0.03<ampl<0.1
            Ampl_tmp_1_1 = get_Ampl(IQcomplex_RN16_Before_1)#ampl<0.02

            # print(Ampl_tmp_1, Ampl_tmp_1_1)
            condition1 = Ampl_tmp_1 >= 0.03 and Ampl_tmp_1 <= 0.1 and Ampl_tmp_1_1< 0.02

            IQcomplex_RN16_After = IQcomplex_reader_with_tag[i+1110 : i+1100+100]
            IQcomplex_RN16_After_1 = IQcomplex_reader_with_tag[i+1200 : i+1200+100] 
            Ampl_tmp_2 = get_Ampl(IQcomplex_RN16_After)#0.03<ampl<0.1
            Ampl_tmp_2_2 = get_Ampl(IQcomplex_RN16_After_1)#ampl<0.02
            
            # print(Ampl_tmp_2 , Ampl_tmp_2_2)
            condition2 = Ampl_tmp_2 >= 0.03 and Ampl_tmp_2 <= 0.1 and Ampl_tmp_2_2< 0.02

            # print('1')
            # if condition1:
            #     print('1')
            # if condition2:
            #     print('2')

            if condition1 and condition2:  #RN16波形的幅度大概范围
    
                pickle.dump(IQcomplex_reader_with_tag[i : i+1200], Tag2_93_RN16)
                i += 12500
                print('Get')
            else:
                i += 1
                # print('no Get')
                

              
        else:
            i += 1
            # print('------')
'''




    

