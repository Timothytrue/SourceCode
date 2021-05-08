from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn import datasets
from sklearn import metrics
import _pickle as pickle

# iris = datasets.load_iris()
# iris_x = iris.data
# iris_y = iris.target

# print(len(iris_y))
# X_train, X_test, Y_train, Y_test = train_test_split(iris_x, iris_y, test_size = 0.2, random_state =42)


# X_train = [[1, 2, 3, 4],
#                [5, 6, 7, 8],
#                [9, 10, 11, 12],
#                 [1, 2, 3, 4],
#                 [5, 6, 7, 8],
#                 [9, 10, 11, 12]]

# Y_train = [1,2,3,1,2,3]
# X_test = [[1,2,3,4],[5,6,7,8]]
# Y_test = [1,2]


# print(X_train.shape)
'''
create data
# f_feature_select = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/feature_select_1.dat', 'rb')
# f_aTIE = open('D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/aTIE1.dat', 'rb')
'''

file_path = []
for i in range(1, 20):
    file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/feature'
    # file_name = 'D:/Work/RFID/RFID/特征/代码/DeepLearning/feature/v1/aTIE'
    file_name = file_name + str(i) + '.dat'
    file_path.append(file_name)

Tags_preamble = []
for tmp in file_path:
    f_feature_select = open(tmp, 'rb')
    Tag_preamble = pickle.load(f_feature_select)
    Tags_preamble.append(Tag_preamble)

X = []
for tmp in Tags_preamble:
    number = 0
    for a_feature in tmp:
        X.append(a_feature)
        number = number + 1
        if(number >= 160):
            break

X = np.array(X,dtype=np.float)
print(X.shape)

###############################################################################
'''归一化---feature_select'''
'''对aTIE, PB归一化'''
max_TIE = X[0][0]
min_TIE = X[0][0]
max_PB = X[0][1]
min_PB = X[0][1]

for tmp in X:
    if tmp[0] > max_TIE:
        max_TIE = tmp[0]
    if tmp[0] < min_TIE:
        min_TIE = tmp[0]
    
    if tmp[1] > max_PB:
        max_PB = tmp[1]
    if tmp[1] < min_PB:
        min_PB = tmp[1]

i = 0
while(i < len(X)):
    X[i][0] = (X[i][0] - min_TIE) / (max_TIE - min_TIE)
    X[i][1] = (X[i][1] - min_PB) / (max_PB - min_PB)
    i = i + 1



#############################################################################

Y = []
count = 0
for tmp in Tags_preamble:
    for i in range(0, 160):
        Y.append(count)
    count = count + 1

Y = np.array(Y)
print(Y.shape)

n_example = X.shape[0]
n_train = n_example * 0.85
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

# X_train = np.array(X_train).reshape(-1,1)
# X_test = np.array(X_test).reshape(-1,1)
print(X_train.shape)
print(X_test.shape)

Y_train = Y[train_id]
Y_test = Y[test_id]
Y_train = np.array(Y_train)
Y_test = np.array(Y_test)
print(Y_train.shape)
print(Y_test.shape)



#定义模型
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train,Y_train)


#预测
y_pred_on_train = knn.predict(X_test)

#输出
# print(y_pred_on_train)
# print("------------")
# print(Y_test)
acc = metrics.accuracy_score(Y_test, y_pred_on_train)
print(acc)