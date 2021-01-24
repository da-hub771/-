import numpy as np
import pandas as pd
import math
from sklearn.preprocessing import MinMaxScaler

file="tests.xls"
df=pd.read_excel('tests.xls')
x=df[['抓举','30米跑(s)','立定三级跳远()','助跑4—6步跳高()','负重深蹲杠铃()','杠铃半蹲系数','100米(s)']]
y=df[['跳高成绩()']]

x_scaler=MinMaxScaler(feature_range=(-1,1))
y_scaler=MinMaxScaler(feature_range=(-1,1))
x=x_scaler.fit_transform(x)
y=y_scaler.fit_transform(y)

sample_in=x.T
sample_out=y.T

max_epochs=100000
learn_rate=0.5
mse_final=0.01
sample_number=x.shape[0]
input_number=7
output_number=1
hidden_unit_number=14

w1 = 0.5 * np.random.rand(hidden_unit_number, input_number) - 0.1
b1 = 0.5 * np.random.rand(hidden_unit_number, 1) - 0.1

w2 = 0.5 * np.random.rand(output_number, hidden_unit_number) - 0.1
b2 = 0.5 * np.random.rand(output_number, 1) - 0.1

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

mse_history=[]
for i in range(max_epochs):
    hidden_out=sigmoid(np.dot(w1,sample_in)+b1)
    network_out=np.dot(w2,hidden_out)+b2

    err=sample_out-network_out
    mse=np.average(np.square(err))
    mse_history.append(mse)
    if mse<mse_final:
        break

    delta2 = -err
    delta1 = np.dot(w2.T, delta2) * hidden_out * (1 - hidden_out)

    delta_w2 = np.dot(delta2, hidden_out.T)
    delta_b2 = np.dot(delta2, np.ones((sample_number, 1)))

    delta_w1 = np.dot(delta1, sample_in.T)
    delta_b1 = np.dot(delta1, np.ones((sample_number, 1)))

    w2 -= learn_rate * delta_w2
    b2 -= learn_rate * delta_b2
    w1 -= learn_rate * delta_w1
    b1 -= learn_rate * delta_b1
mse_history1g=np.log10(mse_history)

network_out=y_scaler.inverse_transform(network_out.T)
sample_out+y_scaler.inverse_transform(y)


x=([50,3,9.3,2.05,100,2.8,11.2])
print("预测得到的跳高成绩为：",float('%.2f'%np.nanmean(y_scaler.inverse_transform(y))))
    




