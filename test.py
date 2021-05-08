import numpy as np
import _pickle as cPickle
from plot_data import plot_IQ, plot_one_image_spectrum,plot_one_image_time, plot_IQ_timeseries
from feature_extraction import cal_mag,feature_cal,find_continuous_indexs
from load_data import load_data
import math


#
# IQcommplex = cPickle.load(open("mixed_signal_n20000_fft_size_1024.pickle", 'rb'))
# labels = cPickle.load(open('mixed_signal_label.pickle','rb'))
#
# # fft_size = 256
# # multiple = 1024 / fft_size
# # labels.repeat(multiple)
#
# # np.split(Xd,)
# print(IQcommplex[0].shape)
# print(len(IQcommplex))
# IQcommplex_256 = np.empty((len(IQcommplex)*4, 2, 256))
# # np.split(np.array(IQcommplex).flatten(),len(IQcommplex)*4)
# # for IQ in IQcommplex[1]:
# #     IQ_real = np.split(IQ[0], 4)
# #     IQ_imag = np.split(IQ[1], 4)
# #     IQcommplex_256
#
# for i in range(len(IQcommplex)):
#     IQcommplex_256_real  = np.split(IQcommplex[i][0], 4)
#     IQcommplex_256_imag  = np.split(IQcommplex[i][0], 4)
#     for n in range(4):
#         IQcommplex_256[i*4+n][0] = IQcommplex_256_real[n]
#         IQcommplex_256[i*4+n][1] = IQcommplex_256_real[n]
# print(IQcommplex_256[60000])
# # print(len(IQcommplex_256))

## for
# Xd = cPickle.load(open("mixed_signal_n400000_fft_size_256.pickle", 'rb'))
# print(Xd.shape)
# labels = cPickle.load(open('mixed_signal_label_n400000_fft_size_256.pickle','rb'))
# print(labels.shape)

# labels = labels.repeat(4)
#
# file = open('mixed_signal_label_n400000_fft_size_256.pickle', 'wb')
# cPickle.dump(labels, file)
# file.close()

fft_size = 64;
# fft_size = 1024;
# IQcomplex_wifi_with_zigbee = load_data(size=fft_size, filename=r'F:\Data\10.15Exp\withwifi_withoutbackoff_cf_2433M_SR_25M.dat',n=1000)
# IQcomplex_wifi_with_zigbee = load_data(filename=r'F:\Data\10.18Exp\rf_15_cf_2433M_SR_25M.dat', n=1000, size=256)
# IQcomplex_wifi_with_zigbee = load_data(size=fft_size, filename='Data/usrp_25M_data_with_telosb.dat', n=5000)
# IQcomplex_zigbee = load_data(size=fft_size, filename='Data/usrp_25M_data_with_telosb.dat', n=10000)

Index = 3200
# plot_one_image_time(IQcomplex_zigbee, IQIndex=Index, bins=fft_size)
# plot_one_image_spectrum(IQcomplex_zigbee, IQIndex=Index, bins=fft_size)
# plot_IQ_timeseries(IQcomplex_zigbee, IQIndex=Index, bins=fft_size)

# plot_one_image_spectrum(IQcomplex_zigbee, IQIndex=Index, bins=fft_size)

# plot_IQ(IQcomplex_zigbee, IQIndex=Index, bins=fft_size)
# # plot_IQ_time_spectrum(IQcomplex=IQcomplex_wifi_with_zigbee, SpecificIndex=Index, bins=fft_size)
#
# center_freqs_zigbee, bandwidths_zigbee, indexs_wifi_with_zigbee = feature_cal(
#     IQcomplex_zigbee, n=1000, bins=fft_size)
#
# feature_wifi_with_zigbee = np.array([center_freqs_zigbee, bandwidths_zigbee])
#
# print(indexs_wifi_with_zigbee)
# # print(center_freqs_wifi_with_zigbee)
# print(bandwidths_zigbee[383])
#
# print(14979 * 1024 * 8 / 8000000)

# Xd = cPickle.load(open("mixed_signal_n3186784_fft_size_64.pickle", 'rb'))
# print(Xd.shape)
# in_shp = list(Xd.shape[1:])
# print([1]+in_shp)

# labels = cPickle.load(open('mixed_signal_three_label_n200000.pickle', 'rb'))
# print(labels.shape)
# print(labels[110000:110200])

A = [[1, 2, 3],[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4],[2,3,4]]
A = np.array(A)
# for tmp in A:
# B = [1,2,3,4]
# B = np.array(B)

# print(len(B))
# print(abs(sum(A[0 , :] - A[1, :])))
# v = 1
A = [[5, 0.02],[10, 0.01]]
A[0][1] = A[0][1] / 0.01
print(A) 