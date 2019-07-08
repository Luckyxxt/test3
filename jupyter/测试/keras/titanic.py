'''
根据泰坦尼克号的游客信息，来预测每个人可能生还的概率。重点在于，
1、如何把数据清洗成适合训练和预测的格式，如把性别male／female转换成0/1，把仓位转换成onehot编码，以及缺失的数据通过均值补齐；
2、输出激励函数不是softmax，而是概率输出sigmoid
                                                                        微信号:  tinghai87605025
'''

import numpy as np
import pandas as pd
from sklearn import preprocessing

from keras.models import Sequential
from keras.layers import Dense, Dropout

import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
np.random.seed(10)

#from biostat.mc.vanderbilt.edu/wiki/pub/Main/DataSets/titanic3.xls
filepath = "C:\Users\ting\PycharmProjects\jupyter\data\titanic.xls"
all_df = pd.read_excel(filepath)
#print(all_df[:3])

cols = ["survived", "name", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]
all_df = all_df[cols]
print(all_df[:3])

def preprocess_data(raw_df):
    #clean data
    df = raw_df.drop(['name'], axis=1)
    #print(df[:3])
    #print(raw_df.isnull().sum())

    age_mean = df['age'].mean()
    df['age'] = df['age'].fillna(age_mean)
    fare_mean = df['fare'].mean()
    df['fare'] = df['fare'].fillna(fare_mean)
    df['sex'] = df['sex'].map({'female':0, 'male':1}).astype(int)
    x_onehot_df = pd.get_dummies(data = df, columns=["embarked"])

    #print("*" * 12)
    #print(x_onehot_df.isnull().sum())
    #print(x_onehot_df[:3])

    #transfer to label and features
    ndarray = x_onehot_df.values
    #print(ndarray.shape)
    #print(ndarray[:3])
    label = ndarray[:,0]
    features = ndarray[:, 1:]

    minmax_scale = preprocessing.MinMaxScaler(feature_range=(0, 1))
    scaled_feature = minmax_scale.fit_transform(features)
    #print(scaled_feature[:3])

    return scaled_feature, label


msk = np.random.rand(len(all_df)) < 0.8
train_df = all_df[msk]
test_df = all_df[~msk]

train_features, train_label = preprocess_data(train_df)
test_features, test_label = preprocess_data(test_df)

print(train_features[:3])
print(test_features[:3])


model = Sequential()
model.add(Dense(units = 40,
        input_dim = 9,
        kernel_initializer = "uniform",
        activation = "relu"))

model.add(Dense(units = 30,
        kernel_initializer = "uniform",
        activation = "relu"))

model.add(Dense(units = 1,
        kernel_initializer = "uniform",
        activation = "sigmoid"))

model.compile(loss = "binary_crossentropy",
        optimizer = "adam",
        metrics = ["accuracy"])

train_history = model.fit(x = train_features,
                    y = train_label,
                    validation_split = 0.1,
                    epochs = 30,
                    batch_size = 30,
                    verbose = 2)

def show_train_history(train_history, train, val):
    plt.plot(train_history.history[train])
    plt.plot(train_history.history[val])
    plt.title("Train History")
    plt.ylabel(train)
    plt.xlabel("Epochs")
    plt.legend(["train", "validation"], loc="upper left")
    plt.show()

show_train_history(train_history, "acc", "val_acc")
show_train_history(train_history, "loss", "val_loss")

scores = model.evaluate(x = test_features, y = test_label)
print("    loss: ", scores[0])
print("accuracy: ", scores[1])

jack = pd.Series([0, 'Jack', 3, 'male', 23, 1, 0, 5.0000, 'S'])
rose = pd.Series([1, 'Rose', 1, 'female', 20, 1, 0, 100.0000, 'S'])

jr_df = pd.DataFrame([list(jack), list(rose)], columns = cols)
all_df = pd.concat([all_df, jr_df])
print(all_df[-2:])
all_features, all_label = preprocess_data(all_df)
all_probability = model.predict(all_features)
print(all_probability[:10])

print("*" * 20)
new_df = all_df
new_df.insert(len(all_df.columns), "probability", all_probability)
print(new_df[-2:])
