import numpy as np
import _pickle as cPickle

from plot_data import plot_IQ
from feature_extraction import cal_mag,feature_cal,find_continuous_indexs
from load_data import load_data

if __name__ == '__main__':
    fft_size = 64
    signal_num = 200000

    IQcommplex = cPickle.load(open("mixed_signal_n200000_fft_size_1024.pickle", 'rb'))
    labels = cPickle.load(open('mixed_signal_label_n200000.pickle', 'rb'))

    boundary = find_continuous_indexs((labels)) #this is the atart and end index of each zigbee tag
    while 0 in boundary:
        boundary.remove(0)

    IQcommplex = np.delete(IQcommplex, boundary, axis=0)
    labels = np.delete(labels, boundary, axis=0)

    find_continuous_indexs(labels)
    # print("oringal data shape: "+IQcommplex.shape)
    # print("original label num: " + sum(np.array(labels)==1))

    multiple = int(1024/fft_size)
    labels = labels.repeat(multiple)

    IQcommplex_n = np.empty((len(IQcommplex) * multiple, 2, fft_size))
    for i in range(len(IQcommplex)):
        IQcommplex_n_real = np.split(IQcommplex[i][0], multiple)
        IQcommplex_n_imag = np.split(IQcommplex[i][1], multiple)
        for n in range(multiple):
            IQcommplex_n[i * multiple + n][0] = IQcommplex_n_real[n] #每次i变化，IQcommplex_256_real都会被替换
            IQcommplex_n[i * multiple + n][1] = IQcommplex_n_imag[n]

    print(IQcommplex_n.shape)

    # np.split(IQcommplex, len(IQcommplex)*4)


    file = open('mixed_signal_n' + str(len(IQcommplex_n)) + '_fft_size_' + str(fft_size) + '.pickle', 'wb')
    cPickle.dump(IQcommplex_n, file)
    file.close()

    file = open('mixed_signal_label_n' + str(len(labels)) + '_fft_size_' + str(fft_size) +'.pickle','wb')
    cPickle.dump(labels,file)
    file.close()