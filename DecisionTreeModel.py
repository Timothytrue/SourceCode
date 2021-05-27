import numpy as np
import _pickle as pickle
from sklearn import tree


def change_COV_into_Bins():
        
    '''
    create data
    '''
    file_path = []
    save_path = []#将预处理后的特征存入这个文件上
    for i in range(1, 20):
        file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature'
        file_name = file_name + str(i) + '_2.dat'
        file_path.append(file_name)

        file_name1 = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature_third'
        file_name1 = file_name1 + str(i) + '.dat'
        save_path.append(file_name1)


    '''
    第一步将文件中的特征(900,152)说明特征是152维
    将前面的132维COV特征转为sorted into 40bins
    因为是树形结构，所以不需要归一化特征
    '''
    # len(file_path)
    for i in range(len(file_path)):
        f_read = open(file_path[i], 'rb')
        feature = pickle.load(f_read)
        print(feature.shape)
        f_write = open(save_path[i],'wb')
        feature_pre = []

        for unit in feature:
            feature_COV = unit[:132]
            feature_PSD = unit[132:152]
            # print(np.round(feature_COV,2))
            # print(type(feature_COV[0]))
            
            '''
            将COV的132个特征sorted into 40bins
            '''
            #先转化为实数幅度
            feature_COV_ampl = []
            feature_PSD_ampl = []
            for tmp in feature_COV:
                ampl = np.sqrt(tmp.real**2 + tmp.imag**2) 
                feature_COV_ampl.append(ampl)

            for tmp in feature_PSD:
                ampl = np.sqrt(tmp.real**2 + tmp.imag**2) 
                feature_PSD_ampl.append(ampl)

            
            # print(np.round(feature_COV_ampl, 2))
            # print(np.round(feature_PSD_ampl, 3))
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

            # print(np.round(feature_COV_ampl, 2))
            #这个是来存储40bins的个数，所以大小就只有40
            feature_COV_bins = []
            for i in range(41):
                feature_COV_bins.append(0)
            
            for tmp in feature_COV_ampl:

                feature_COV_bins[int(tmp)] = feature_COV_bins[int(tmp)] + 1
            
            # print(feature_COV_bins)
            #将COV分布的41维加上原本的PSD20维，总共61维
            feature_pre_tmp = []
            for tmp in feature_COV_bins:
                feature_pre_tmp.append(tmp)
            
            count = 0
            for tmp in feature_PSD_ampl:
                
                feature_pre_tmp.append(tmp)
                count = count + 1
                if(count >= 100):
                    break
            
            # print(len(feature_pre_tmp))
            feature_pre.append(feature_pre_tmp)
        

        feature_pre = np.array(feature_pre)
        # print(np.round(feature_pre[0], 4).shape)
        print(np.round(feature_pre, 4).shape)
        pickle.dump(feature_pre, f_write)


def training():
    '''
    将处理好的特征导入到X中，并建立标签Y
    '''
    file_path = []
    for i in range(1, 20):
        file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature_second'
        file_name = file_name + str(i) + '.dat'
        file_path.append(file_name)

    Tags_preamble = []
    for tmp in file_path:
        f_feature_select = open(tmp,'rb')
        Tag_preamble = pickle.load(f_feature_select)
        # print(Tag_preamble.shape)
        Tags_preamble.append(Tag_preamble)

    '''
    建立特征值到X中
    '''
    X = []
    for tmp in Tags_preamble:
        number = 0
        for a_feature in tmp:
            X.append(a_feature)
            number = number + 1
            #来控制每个标签的前导码个数
            if number >= 10000:
                break

    X = np.array(X)
    print(X.shape)

    '''
    建立标签Y
    '''
    Y = []
    count = 0
    for tmp in Tags_preamble:
        for i in range(len(tmp)):
            if i >= 10000:
                break
            
            Y.append(count)  
        count = count + 1

    Y = np.array(Y)
    print(Y.shape) 

    n_example = X.shape[0]
    n_train = n_example * 0.25
    train_id = np.random.choice(range(0, n_example), size= int(n_train), replace= False)
    test_id = list(set(range(0, n_example)) - set(train_id))

    '''
    set train data set
    set test data set
    '''

    X_train = X[train_id]
    X_test = X[test_id]
    X_train = np.array(X_train)
    X_test = np.array(X_test)
    print(X_train.shape)
    print(X_test.shape)
    Y_train = Y[train_id]
    Y_test = Y[test_id]
    Y_train = np.array(Y_train)
    Y_test = np.array(Y_test)
    print(Y_train.shape)
    print(Y_test.shape)

    '''
    定义模型决策树,其中Criterion = "entropy"-------ID3  C4.5
    定义模型决策树,其中Criterion = "gini"-------CART
    '''
    model = tree.DecisionTreeClassifier("gini")
    model.fit(X_train, Y_train)
    score = model.score(X_test, Y_test)

    print(score)
    return model


def test_for_new_sample():
    '''
    将处理好的特征导入到X中，并建立标签Y
    '''

    file_path = []
    for i in range(1, 20):
        file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature_second'
        file_name = file_name + str(i) + '.dat'
        file_path.append(file_name)

    Tags_preamble = []
    for tmp in file_path:
        f_feature_select = open(tmp,'rb')
        Tag_preamble = pickle.load(f_feature_select)
        # print(Tag_preamble.shape)
        Tags_preamble.append(Tag_preamble)

    '''
    建立特征值到X中
    '''
    X = []
    for tmp in Tags_preamble:
        number = 0
        for a_feature in tmp:
            X.append(a_feature)
            number = number + 1
            #来控制每个标签的前导码个数
            if number >= 10000:
                break

    X = np.array(X)
    print(X.shape)

    '''
    建立标签Y
    '''
    Y = []
    count = 0
    for tmp in Tags_preamble:
        for i in range(len(tmp)):
            if i >= 10000:
                break
            
            Y.append(count)  
        count = count + 1

    Y = np.array(Y)
    print(Y.shape) 

    model = training()
    score = model.score(X, Y)
    print("test: ", score)


if __name__ == '__main__':

    '''
    pre_process_one
    '''
    # change_COV_into_Bins()

    training()

    # test_for_new_sample()


    
