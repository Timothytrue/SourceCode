import numpy as np
import _pickle as pickle
from sklearn import tree

'''
create data
'''
file_path = []
save_path = []#将预处理后的特征存入这个文件上
for i in range(1, 20):
    file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v2/feature'
    file_name = file_name + str(i) + '.dat'
    file_path.append(file_name)

    file_name1 = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v2/feature_pre'
    file_name1 = file_name1 + str(i) + '.dat'
    save_path.append(file_name1)


'''
第一步将文件中的特征(900,152)说明特征是152维
将前面的132维COV特征转为sorted into 40bins
因为是树形结构，所以不需要归一化特征
'''
# len(file_path)
for i in range(1):
    f_read = open(file_path[i], 'rb')
    feature = pickle.load(f_read)
    print(feature.shape)
    f_write = open(save_path[i],'wb')

    for unit in feature:
        feature_COV = unit[:132]
        feature_PSD = unit[132:152]
        
        '''
        将COV的132个特征sorted into 40bins
        '''
        #先转化为实数幅度
        feature_COV_ampl = []
        for tmp in feature_COV:
            ampl = np.sqrt(tmp.real**2 + tmp.imag**2) 
            feature_COV_ampl.append(ampl)
        
        # print(feature_COV_ampl)
        #再归(0,40)范围
        max_value = feature_COV_ampl[0]
        min_value = feature_COV_ampl[0]
        for tmp in feature_COV_ampl:
            if(tmp > max_value):
                max_value = tmp
            if(tmp < min_value):
                min_value = tmp
        

        for i in range(len(feature_COV_ampl)):
            feature_COV_ampl[i] = (feature_COV_ampl[i] - min_value) * 40/ (max_value - min_value)

        # print(feature_COV_ampl)
        #这个是来存储40bins的个数，所以大小就只有40
        feature_COV_bins = []
        
            




