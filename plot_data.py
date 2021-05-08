import matplotlib.pyplot as plt

from load_data import load_data
import numpy as np
from feature_extraction import cal_mag, feature_cal
from feature_extraction import find_continuous_indexs

def plot_IQ(IQcomplex, IQIndex=0, bins=1024):
    """
    This function plot 10 figures, use IQIndex*100 us IQdata
    :param IQcomplex:
    :param IQIndex: the index num of IQcomplex
    :return:
    """
    plt.figure(figsize=(6, 10))
    num = IQIndex
    for n in range(10):
        plt.subplot(5, 2, n + 1)
        IQcomplex_tmp = IQcomplex[(n + num) * bins:(n + 1 + num) * bins]
        mag, freqs = cal_mag(IQcomplex_tmp)
        plt.plot(freqs, mag)
    plt.show()
    return

def plot_one_image_spectrum(IQcomplex, IQIndex=0, bins=1024):
    # plt.figure(figsize=(6, 10))
    num = IQIndex
    m_fontsize = 20

    IQcomplex_tmp = IQcomplex[(num) * bins:(1 + num) * bins]
    mag, freqs = cal_mag(IQcomplex_tmp)
    plt.plot(freqs, mag)
    plt.xlabel('Frequency(MHz)', fontsize=m_fontsize)
    plt.ylabel('Amplitude(dB)', fontsize=m_fontsize)
    plt.xticks(fontsize=m_fontsize)
    plt.yticks(fontsize=m_fontsize)
    plt.savefig("paper_images/plot_spectrum64.png", bbox_inches='tight')
    plt.show()
    return

def plot_one_image_time(IQcomplex, IQIndex=0, bins=64):
    plt.figure(figsize=(6, 10))
    num = IQIndex

    Qcomplex_tmp_time = np.linspace(0, bins , bins)
    IQcomplex_tmp = IQcomplex[(num) * bins:(1 + num) * bins]
    mag = np.sqrt(IQcomplex_tmp.real ** 2 + IQcomplex_tmp.imag ** 2)
    plt.plot(Qcomplex_tmp_time, mag)
    plt.show()
    return

def plot_IQ_timeseries(IQcomplex, IQIndex=0, bins = 1024*8):
    plt.figure(figsize=(30, 30))
    num = IQIndex
    for n in range(10):
        plt.subplot(5, 2, n + 1)
        # for time analysis

        IQcomplex_tmp_time = np.linspace(0, bins / 20, bins) #横坐标是时间20
        IQcomplex_tmp = IQcomplex[(n + num) * bins:(n + 1 + num) * bins]
        # plot time IQ seperate
        #         plt.plot(IQcomplex_tmp_time,IQcomplex_tmp.real)
        #         plt.plot(IQcomplex_tmp_time,IQcomplex_tmp.imag)

        # plot energy
        #         IQcomplex_tmp_time = np.linspace(0, bins*2/25, bins)
        mag = np.sqrt(IQcomplex_tmp.real ** 2 + IQcomplex_tmp.imag ** 2)
        plt.plot(IQcomplex_tmp_time, mag)
        plt.xlabel('Time(us)')
    plt.show()
    return


def plot_IQ_timeseries1(IQcomplex, IQIndex=0, bins = 1024*2):
    plt.figure(figsize=(10, 6))
    num = IQIndex

		
    IQcomplex_tmp_time = np.linspace(0, bins/20 , bins) #横坐标是时间
    IQcomplex_tmp = IQcomplex[(num) * bins:(1 + num) * bins] 
    mag = np.sqrt(IQcomplex_tmp.real ** 2 + IQcomplex_tmp.imag ** 2)
    plt.plot(IQcomplex_tmp_time, mag)
    plt.xlabel('Time(us)')
    # plt.ylim([0.0, 0.8])
    #plt.xlim([0, 100])
    plt.xticks(np.arange(0, 51, 1))
    plt.show()
    return



def plot_IQ_time_spectrum(IQcomplex, SpecificIndex, bins):
    n = SpecificIndex
    plt.subplot(121)
    # bins = 1024
    IQcomplex_tmp = IQcomplex[(n) * bins:(n + 1) * bins]
    mag, freqs = cal_mag(IQcomplex_tmp)
    plt.plot(freqs, mag)
    plt.xlabel('Freq(MHz)')

    plt.subplot(122)

    IQcomplex_tmp_time = np.linspace(0, bins / 25, bins)
    mag = np.sqrt(IQcomplex_tmp.real ** 2 + IQcomplex_tmp.imag ** 2)
    plt.plot(IQcomplex_tmp_time, mag)
    plt.xlabel('Time(us)')
    plt.tight_layout()

    plt.show()
    return

if __name__=='__main__':
    fft_size = 1024
    # IQcomplex_wifi_with_zigbee = load_data(filename=r'F:\Data\10.15Exp\withwifi_withoutbackoff_cf_2433M_SR_25M.dat',n=1000, size=fft_size)
    # IQcomplex_wifi_with_zigbee = load_data(filename=r'F:\Data\10.18Exp\rf_15_cf_2433M_SR_25M.dat', n=1000, size=256)
    file1 = '/tmp/RN16.dat'
    file2 = 'misc/data/file_source_test'
    file3 = './misc/data/Tag2_93/source1'
    file4 = './misc/data/Tag2_97/source9'
    file5 = './misc/data/Tag2_97/RN16_preamble2.pickle'

    file = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/17_up/source2'
    IQcomplex_reader_with_tag = load_data(size=fft_size, filename= file)
    #misc/data/
    print(len(IQcomplex_reader_with_tag))

    Index = 0
    for i in range(60):
        # plot_IQ_timeseries(IQcomplex_reader_with_tag, IQIndex = i*10+0, bins=1000*10)#40
        plot_IQ_timeseries1(IQcomplex_reader_with_tag, IQIndex = i*1+100, bins=2000)#40
        pass




    # l1 = list()
    
    #plot_IQ(IQcomplex_wifi_with_zigbee, IQIndex=Index, bins=fft_size)
    # plot_IQ_time_spectrum(IQcomplex=IQcomplex_wifi_with_zigbee, SpecificIndex=Index, bins=fft_size)

    #center_freqs_wifi_with_zigbee, bandwidths_wifi_with_zigbee, indexs_wifi_with_zigbee = feature_cal(
        #IQcomplex_wifi_with_zigbee, n=1000, bins=fft_size)

    #feature_wifi_with_zigbee = np.array([center_freqs_wifi_with_zigbee, bandwidths_wifi_with_zigbee])

    #print(indexs_wifi_with_zigbee)
    # print(center_freqs_wifi_with_zigbee)
    #print(bandwidths_wifi_with_zigbee[383])

    #print(14979*1024*8/8000000)


