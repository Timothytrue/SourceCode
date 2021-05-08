import _pickle as pickle
from plot_data import plot_IQ_timeseries, plot_IQ_timeseries1
import matplotlib.pyplot as plt

# from load_data import load_data
import numpy as np
import numpy as np
from extract_RN16 import get_Ampl


def getListFromPickle(filename):
    IQcomplex = []
    with open(filename, 'rb') as f:
        while True:
            try:
                IQcomplex_tmp = pickle.load(f)
                for tmp in IQcomplex_tmp:
                    IQcomplex.append(tmp)
            except:
                break
    
    return np.array(IQcomplex, dtype=complex)


def getList(filename):
    IQcomplex = []
    with open(filename, 'rb') as f:
        while True:
            try:
                IQcomplex_tmp = pickle.load(f)
                IQcomplex.append(IQcomplex_tmp)
            except:
                break
    
    return np.array(IQcomplex, dtype=complex)


'''
从source_RN16_.pickele包含的整个RN16信号截取前导码，
其中前导码占6比特，所有样本数量为1230
'''
def get_RN16_preamble(filename1, save_to_file_path):
    
    #先从源文件得到RN16列表 
    IQcomplex = getList(filename=filename1)
    # print(len(IQcomplex_reader_with_tag))
    
    # 将提取的RN16片段存入Tag2_93_RN16.pickle文件中
    Tag2_9x_RN16_preamble = open(save_to_file_path, 'wb')
    count = 1

    for tmp in IQcomplex :
        # print(count)
        i = 0

        while(i < 300):
            IQcomplex_prefix = tmp[i:i+20]#误差设置为20
            Ampl = get_Ampl(IQcomplex_prefix) 
            if Ampl >= 0.05 and Ampl <= 0.20:
                pickle.dump(tmp[i : i+1230], Tag2_9x_RN16_preamble)
                if(count % 100 == 0 or count > len(IQcomplex) - 2):
                    print(count)
                count += 1
                break
            else:
                i += 1
        
        # if count > 500:
        #     break


'''
将所有文件RN16_preamble转换成一个numpy array
并dump到文件上
'''
def get_Tag_preamble(filename, save_to_file):

    IQcomplex_preamble = []
    for file_tmp in filename:
        with open(file_tmp, 'rb') as f:
            while True:
                try:
                    IQcomplex_tmp = pickle.load(f)
                    IQcomplex_preamble.append(IQcomplex_tmp)
                except:
                    break
    
    

    IQcomplex_preamble = np.array(IQcomplex_preamble, dtype=complex)

    f = open(save_to_file, 'wb')    
    pickle.dump(IQcomplex_preamble, f)

    return IQcomplex_preamble



def plot_IQ_timeseries2(IQcomplex, IQIndex=0, bins = 200):
    plt.figure(figsize=(10, 6))
    num = IQIndex

		
    IQcomplex_tmp_time = np.linspace(0, 400/4 , 400) #横坐标是时间
    IQcomplex_tmp = IQcomplex[100:500] 
    mag = np.sqrt(IQcomplex_tmp.real ** 2 + IQcomplex_tmp.imag ** 2)
    # plt.plot(IQcomplex_tmp_time, mag)
    plt.scatter(IQcomplex_tmp_time, mag)
    plt.xlabel('Time(us)')
    # plt.ylim([0.0, 0.8])
    #plt.xlim([0, 100])
    plt.xticks(np.arange(0, 101, 1))
    plt.show()
    return



if __name__ == '__main__':

    
    #测试提取得波形是否正确

    
    file = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/26_up/1_RN16_.pickle'
    
    file1= 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/41/19preamble.pickle'

    #(1, 2, 3)
    # file2 = './misc/data/Tag2_97/RN16_preamble12.pickle'

    # IQcomplex = getListFromPickle(file1)
    # print(len(IQcomplex))
    Index = 0
    
    # for i in range(10):
    #     plot_IQ_timeseries1(IQcomplex, IQIndex = i*1+0, bins=1000)
    IQcomplex = getList(file1)
    print(len(IQcomplex))


    count = len(IQcomplex) -1
    for tmp in IQcomplex:
        plot_IQ_timeseries2(tmp, IQIndex=0, bins=200)
        plot_IQ_timeseries2(IQcomplex[count], IQIndex=0, bins=200)
        # plot_IQ_timeseries1(tmp, IQIndex=0, bins=1230)
        # plot_IQ_timeseries1(IQcomplex[count], IQIndex=0, bins=1230)
        count -= 1
        






    # IQcomplex1 = getList(file1)
    # print(len(IQcomplex1))

    # count = len(IQcomplex1) -1
    # for tmp in IQcomplex1:
    #     plot_IQ_timeseries1(tmp, IQIndex=0, bins=330)
    #     plot_IQ_timeseries1(IQcomplex1[count], IQIndex=0, bins=330)
    #     count -= 1
        # plot_IQ_timeseries1(tmp, IQIndex=0, bins=1200)
        # plot_IQ_timeseries1(IQcomplex[count-1], IQIndex=0, bins=300)
        # count -=1

    # plot_IQ_timeseries1(IQcomplex1[500], IQIndex=0, bins=330)
   

    
    #提取程序

    # 标签类型为Tag2_93
    file_path = []
    save_to_file_path = []
    for i in range(1, 2):
        file_name = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/10/'
        file_name = file_name + str(i) + '_RN16_.pickle'
        file_path.append(file_name)

        save_to_file = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/10/'
        save_to_file = save_to_file + str(i) + 'preamble.pickle'
        save_to_file_path.append(save_to_file)

        print(file_name, save_to_file)

    
  
    
    # for i in range(0, 4):
    #     get_RN16_preamble(file_path[i], save_to_file_path[i])

    # for i in range(0, 1):
    #     get_RN16_preamble(file, file1)
    
    
    
    
    
    # file_1_tag = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/10/10.pickle'
    # tag_preamble = get_Tag_preamble(save_to_file_path, file_1_tag)
    # print(tag_preamble.shape)

    

