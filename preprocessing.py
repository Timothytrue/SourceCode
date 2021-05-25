import _pickle as pickle
from plot_data import plot_IQ_timeseries, plot_IQ_timeseries1
import matplotlib.pyplot as plt
import numpy as np
from extract_RN16 import get_Ampl
from RN16_Test import getList
import math




def process_scale_1(read_file, write_file, factor):
    IQcomplex = getList(read_file)
    f_save = open(write_file, 'wb')
    for tmp in IQcomplex:
        preamble = []
        for complex_unit in tmp:
            preamble.append(factor*complex_unit)

        pickle.dump(np.array(preamble), f_save)   


def process_scale(read_file, write_file):
    IQcomplex = getList(read_file)
    f_save = open(write_file, 'wb')
    
    for tmp in IQcomplex:
        preamble = []
        max = math.sqrt(tmp[0].real**2 + tmp[0].imag**2)
        min = math.sqrt(tmp[0].real**2 + tmp[0].imag**2)
        for complex_unit in tmp:
            ampl = math.sqrt(complex_unit.real**2 + complex_unit.imag**2)
            if ampl > max:
                max = ampl
            if ampl < min:
                min = ampl
        
        factor = max - min
        for complex_unit in tmp:
            preamble.append(complex_unit/factor)

        pickle.dump(np.array(preamble), f_save)   

def process_offset(read_file, write_file, factor):
    IQcomplex = getList(read_file)
    f_save = open(write_file, 'wb')
    for tmp in IQcomplex:
        preamble = []
        for complex_unit in tmp:
            real = complex_unit.real
            imag = complex_unit.imag
            #如果实部是复数，即处于第二、第三象限arctan(b/a) +pi
            angle = math.atan(imag/real)
            if real < 0 and imag > 0:
                angle = angle + math.pi
            if real < 0 and imag < 0:
                angle = angle - math.pi
            
            preamble.append(complex_unit + (math.cos(angle) + math.sin(angle)* 1j) * factor)

        pickle.dump(np.array(preamble), f_save)  

if __name__ == "__main__":

    file= 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/7_up/3preamble.pickle'
    IQcomplex = getList(file)
    print(len(IQcomplex))

    # count = len(IQcomplex) -1
    # for tmp in IQcomplex:
    #     plot_IQ_timeseries1(tmp, IQIndex=0, bins=1230)
    #     plot_IQ_timeseries1(IQcomplex[count], IQIndex=0, bins=1230)
    #     count -= 1

    save_to_file= 'E:/RFID/RFID/特征/代码/DeepLearning/misc/ruboust_group/1.2m/7_up/3preamble_V2.pickle'
    # process_offset(file,save_to_file, 2)
    process_scale(file, save_to_file)
    IQcomplex1 = getList(save_to_file)
    print(len(IQcomplex1))
   
  
    count = len(IQcomplex1) -1
    for tmp in IQcomplex1:
        plot_IQ_timeseries1(tmp, IQIndex=0, bins=1230)
        plot_IQ_timeseries1(IQcomplex1[count], IQIndex=0, bins=1230)
        count -= 1


             


