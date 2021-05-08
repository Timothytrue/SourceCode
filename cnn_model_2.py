import os, random
import numpy as np
import _pickle as pickle

from keras.utils import np_utils
import keras.models as models
from keras.layers import Reshape,Dense,Dropout,Activation,Flatten
# from keras.layers
from keras.layers.noise import GaussianNoise
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.regularizers import *
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import keras, sys
# from RN16_Test import getList

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

np.random.seed(2020)
'''
Dataset setup
'''
np.random.seed(2020)
# X = np.ndarray([1, 2, 128]) #dataset, [none, 2, 128]
# Y = np.array([1]) #label, 2 num_classes

'''
load data
'''
# file1 = './misc/data/Tag2_93/Tag93_preamble.pickle'
# file2 = './misc/data/Tag2_97/Tag97_preamble.pickle'
# file_name = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/pickle/'

file_path = []
for i in range(1, 13):
        file_name = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/pickle/'
        file_name = file_name + str(i) + '.pickle'
        file_path.append(file_name)
        # print(file_name)

# Tag93 = open(file1, 'rb')
# Tag97 = open(file2, 'rb')

# Tag93_preamble = pickle.load(Tag93)
# Tag97_preamble = pickle.load(Tag97)
Tags_preamble = []
for tmp in  file_path:
        Tag_file = open(tmp, 'rb')
        Tag_preamble = pickle.load(Tag_file)
        Tags_preamble.append(Tag_preamble)
        # print(Tag_preamble.shape)



Tag = []
# for tmp in Tag93_preamble:
#         Tag.append(tmp)

# for tmp in Tag97_preamble:
#         Tag.append(tmp)

for tmp in Tags_preamble:
        for a_preamble in tmp:
                Tag.append(a_preamble) 

Tag = np.array(Tag)


# print(Tag93_preamble.shape, Tag97_preamble.shape)
print(Tag.shape)


Y = []
# for i in range(9910):
#         Y.append(0)

# for i in range(12829):
#         Y.append(1)

count = 0
for tmp in Tags_preamble:
        for i in range(0, len(tmp)):
                Y.append(count)
        count += 1
        # print(len(tmp))


Y = np.array(Y)  #lable标注
print(Y.shape, Y[1])


X = []   #data
X_Real = []
X_Imag = []
for tmp in Tag:
        X.append(tmp.real)
        X.append(tmp.imag)

# for tmp in Tag:
#         X_Real.append(tmp.real)
#         X_Imag.append(tmp.imag)
# X.append(X_Real)
# X.append(X_Imag)

X = np.array(X)
X = np.reshape(X, (-1, 2, 330))
print(X.shape)
# print(Tag[1].real)
# print(X[1])

#################################################################################################################

n_example = X.shape[0] 
n_train = n_example * 0.85
train_idx = np.random.choice(range(0, n_example), size=int(n_train), replace=False)#训练的个数
test_idx = list(set(range(0, n_example)) - set(train_idx))

print(len(test_idx))
print(n_train)

'''
set train data set
set test data set
'''

X_train = X[train_idx]
X_test = X[test_idx]
X_train = np.array(X_train)
X_test = np.array(X_test)
print(X_train.shape)
print(X_test.shape)




'''
data preprocessing
'''


Y_train = Y[train_idx]
Y_test = Y[test_idx]
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print(Y_test.shape)
print(Y_train.shape)

num_classes = 12# two similar tag in one company
Y_train = np_utils.to_categorical(Y_train, num_classes)
Y_test = np_utils.to_categorical(Y_test, num_classes)
in_shp = list(X_train.shape[1:]) #[2, 300]
print(X_train.shape, in_shp)


'''
build the CNN model
'''

dr = 0.5 #dropout rate(%)
model = models.Sequential()
model.add(Reshape(in_shp + [1], input_shape = in_shp))#change the shape of input
# model.add(Reshape([1] + in_shp, input_shape = in_shp))#change the shape of input
#(none, 1, 2, 300)

'''
conv1+maxpooling
'''
model.add(Convolution2D(
        filters=128,
        kernel_size=(2, 8),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',

))
model.add(Activation('relu'))
#(none, 128, 1, 330)

model.add(MaxPooling2D(
        pool_size=(1, 2),
        strides= 2,
        padding='same',
        data_format='channels_last',


))
#(128, 1, 165)


'''
conv2+maxpooing
'''

model.add(Convolution2D(
        64,
        (1, 16),
        strides=(1, 1),
        padding='same',
        data_format='channels_last',

))
model.add(Activation('relu'))
#(64, 1, 165)

model.add(MaxPooling2D(
        pool_size=(1, 2),
        strides= 2,
        padding='same',
        data_format='channels_last',

))
#(64, 1, 82)


# '''
# conv3+maxpooling
# '''
# model.add(Convolution2D(
#         32,
#         (1, 16),
#         strides=(1, 1),
#         padding='same',
#         data_format='channels_last',

# ))
# model.add(Activation('relu'))
# #(64, 1, 32)

# model.add(MaxPooling2D(
#         pool_size=(1, 2),
#         strides=2,
#         padding='same',
#         data_format='channels_last',

# ))
# #(64, 1, 16)

'''
flatten
'''
model.add(Flatten())#(64*1*82)

'''
FC + relu
'''
model.add(Dense(
        1024,
        activation='relu',
        kernel_initializer='he_normal',
        name = 'dense1',
))


'''
FC + relu
'''
model.add(Dense(
        256,
        activation='relu',
        kernel_initializer='he_normal',
        name = 'dense2',
))


'''
FC + softmax
'''

model.add(Dense(
        num_classes,
        kernel_initializer='he_normal',
        name = 'dense3',

))
model.add(Activation('softmax'))

'''
reshape output
'''
model.add(Reshape([num_classes]))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()


'''
train the model
'''
nb_epoch = 2 #number of epoch to train on
batch_size = 64 #train batch size
count = 1

filepath = 'E:/RFID/RFID/特征/代码/DeepLearning/misc/data/0.9m/pickle/' + str(count) + '.wts.h5'
history = model.fit(
        X_train,
        Y_train,
        batch_size=batch_size,
        epochs=nb_epoch,
        verbose=2,
        validation_data=(X_test, Y_test),
        callbacks=[
                keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, mode='auto'),
                keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=0, mode='auto')

        ]

)

model.load_weights(filepath)

loss, accuracy= model.evaluate(X_test, Y_test)

print(loss, accuracy)