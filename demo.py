# data分为3个微物理变量，每个变量具有3个不同高度的数据，每个高度具有258次降水过程的时间记录，
# 2小时过程中要求通过前一小时也就是6帧的数据预测后一小时的数据，每帧数据为256*256的平面。
# input.size = (3, 3, 258, 6, 256, 256)
# output.size = (1, 6, 256, 256)
# 所以每一个批次实际上是(3, 3, 1, 6, 256, 256)
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# data = [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118, 115, 126,
#  141, 135, 125, 149, 170, 170, 158, 133, 114, 140, 145, 150, 178, 163,
#  172, 178, 199, 199, 184, 162, 146, 166, 171, 180, 193, 181, 183, 218,
#  230, 242, 209, 191, 172, 194, 196, 196, 236, 235, 229, 243, 264, 272,
#  237, 211, 180, 201, 204, 188, 235, 227, 234, 264, 302, 293, 259, 229,
#  203, 229, 242, 233, 267, 269, 270, 315, 364, 347, 312, 274, 237, 278,
#  284, 277, 317, 313, 318, 374, 413, 405, 355, 306, 271, 306, 315, 301,
#  356, 348, 355, 422, 465, 467, 404, 347, 305, 336, 340, 318, 362, 348,
#  363, 435, 491, 505, 404, 359, 310, 337, 360, 342, 406, 396, 420, 472,
#  548, 559, 463, 407, 362, 405, 417, 391, 419, 461, 472, 535, 622, 606,
#  508, 461, 390, 432]

data = list(map(float, data))

x = torch.randn((batch_size, 30, 256, 256))

# 测试数据量，训练测试数据划分
test_data_size = 12
train_data = np.array(data[:-test_data_size])
test_data = data[-test_data_size:]

# 归一化
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = scaler.fit_transform(train_data.reshape(-1, 1))
train_data_normalized = torch.FloatTensor(train_data_normalized).view(-1)

# 设置窗口，类似于通过前12个月推出第13个月，通过前6次采样推出第7次采样
train_window = 12
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)
# print(len(train_inout_seq))

# 筛选后12个值
fut_pred = 12
test_inputs = train_data_normalized[-train_window:].tolist()

model.eval()

for i in range(fut_pred):
    seq = torch.FloatTensor(test_inputs[-train_window:])
    with torch.no_grad():
        test_inputs.append(model(seq.to('cuda')).item())

# 逆标准化
actual_predictions = scaler.inverse_transform(np.array(test_inputs[train_window:]).reshape(-1, 1))
# print(actual_predictions)
x = np.arange(132, 144, 1)

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(data)
plt.plot(x, actual_predictions)
plt.show()

plt.title('Month vs Passenger')
plt.ylabel('Total Passengers')
plt.grid(True)
plt.autoscale(axis='x', tight=True)

plt.plot(x, data[-train_window:])
plt.plot(x, actual_predictions)
plt.show()

# bias = []
# print(type(np.array(data[-train_window:])), type(actual_predictions[0]))
# print(loss_function(np.array(data[-train_window:]), actual_predictions))