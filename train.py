import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from utils import create_inout_sequences, get_loaders, data_read
from LSTM import MainNetwork
from tqdm import tqdm

load = True
batch_size = 4
NUM_WORKERS = 0
PIN_MEMORY = True
input_channel = 30
output_channel = 30
epochs = 50
norm_param = {
    'dBZ': [0, 65],
    'ZDR': [-1, 5],
    'KDP': [-1, 6]
}
# 逆归一化
mmin, mmax = norm_param['dBZ']
save_filename = 'model_dBZ_KDP_ZDR_differentPlus_1.pth'

# 引用数据集
train_loader = get_loaders(
            True,
            batch_size,
            NUM_WORKERS,
            PIN_MEMORY,
        )

test_loader = get_loaders(
            False,
            batch_size,
            NUM_WORKERS,
            PIN_MEMORY,
        )

# 设置损失函数，模型，优化器
model = MainNetwork(input_channel, output_channel, batch_size).to('cuda')
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

loss_list = []
sum = 0

if load == True:
    # 定义模型保存路径
    save_path = save_filename

    # 加载模型的参数
    model.load_state_dict(torch.load(save_path))

    # 打印可训练参数
    for name, param in model.named_parameters():
        if param.requires_grad and name in ['a', 'b', 'c']:
            print(name, param)

    model.eval()

    with torch.no_grad():
        for idx, (test_data_dBZ, test_data_KDP, test_data_ZDR, test_val_data) in enumerate(test_loader):
            test_data = []
            test_data.append(test_data_dBZ.to('cuda'))
            test_data.append(test_data_KDP.to('cuda'))
            test_data.append(test_data_ZDR.to('cuda'))
            predict_data = model(test_data)

            original_data = predict_data.to('cpu') * (mmax - mmin) + mmin
            original_val_data = test_val_data.to('cpu') * (mmax - mmin) + mmin

            diff = torch.abs(original_data - original_val_data)
            is_accuracy_channel = diff < 0.5
            num_correct_points_channel = torch.sum(is_accuracy_channel, dim=(2, 3))
            accuracy_channel = num_correct_points_channel / (256 * 256)

        print(f'每个通道的准确率: {accuracy_channel}')

    x = np.arange(0, epochs, 1)

    loss_list = data_read('loss_data.txt')

    plt.plot(x, loss_list)

    plt.show()

    for i in range(1, 4, 1):
        plt.subplot(1, 3, i)

        x = np.arange(0, 10, 1)

        plt.ylabel('accuracy')
        if i == 1:
            plt.xlabel('1km')
            plt.bar(x, accuracy_channel[0][0:10])
        if i == 2:
            plt.xlabel('3km')
            plt.bar(x, accuracy_channel[0][10:20])
        if i == 3:
            plt.xlabel('7km')
            plt.bar(x, accuracy_channel[0][20:30])

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.6,
                        hspace=0.1
                        )
    plt.show()

else:
    for i in range(epochs):
        for idx, (train_data_dBZ, train_data_KDP, train_data_ZDR, val_data) in enumerate(train_loader):
            sum = 0
            optimizer.zero_grad()

            train_data = []
            train_data.append(train_data_dBZ.to('cuda'))
            train_data.append(train_data_KDP.to('cuda'))
            train_data.append(train_data_ZDR.to('cuda'))
            y_pred = model(train_data)

            # 损失值计算及反向传播更新参数
            single_loss = loss_function(y_pred, val_data.to('cuda'))
            single_loss.backward()
            optimizer.step()

        # if i % 25 == 0:
            print(f'epoch: {i:3} [{idx}:{len(train_loader)}] loss: {single_loss.item():10.8f}')

            sum = sum + single_loss.item()

        loss_list.append(sum / len(train_loader))

    # 定义保存路径
    save_path = save_filename

    # 保存模型的参数
    torch.save(model.state_dict(), save_path)

    with open("loss_data.txt", 'w') as train_los:
        train_los.write(str(loss_list))
