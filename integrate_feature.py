import numpy as np
import _pickle as pickle


def integrate():
    filename = []
    filename.append("D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature19.dat")
    filename.append("D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature19_2.dat")

    feature_integrate = []
    for tmp in filename:
        f_read = open(tmp, 'rb')
        feature = pickle.load(f_read)
        print(feature.shape)
        for unit in feature:
            feature_integrate.append(unit)
    
    feature_integrate = np.array(feature_integrate)
    print(feature_integrate.shape)
    f_write = open("D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v4/feature19_.dat", 'wb')
    pickle.dump(feature_integrate,f_write)


if __name__ =="__main__":
    integrate()
