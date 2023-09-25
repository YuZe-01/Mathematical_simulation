from torch.utils.data.dataset import Dataset
import os
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
import torch

def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq, train_label))
    return inout_seq

class TrainDataset(Dataset):
    def __init__(self, train):
        super(TrainDataset, self).__init__()
        self.train_len = 100
        self.test_len = 120
        self.filePath_dBZ = "D:\shumo\dBZ"
        self.filePath_KDP = "D:\shumo\KDP"
        self.filePath_ZDR = "D:\shumo\ZDR"
        self.train_path = "traindata"
        self.val_path = "objdata"
        self.trainSet_dBZ = os.listdir(os.path.join(self.filePath_dBZ, self.train_path))
        self.trainSet_KDP = os.listdir(os.path.join(self.filePath_KDP, self.train_path))
        self.trainSet_ZDR = os.listdir(os.path.join(self.filePath_ZDR, self.train_path))
        self.valSet = os.listdir(os.path.join(self.filePath_dBZ, self.val_path))
        self.train_file_dBZ = []
        self.train_file_KDP = []
        self.train_file_ZDR = []
        self.val_file = []

        if train == True:
            for i in range(self.train_len):
                self.train_file_dBZ.append(os.path.join(self.filePath_dBZ, self.train_path, self.trainSet_dBZ[i]))
                self.train_file_KDP.append(os.path.join(self.filePath_KDP, self.train_path, self.trainSet_KDP[i]))
                self.train_file_ZDR.append(os.path.join(self.filePath_ZDR, self.train_path, self.trainSet_ZDR[i]))
                self.val_file.append(os.path.join(self.filePath_dBZ, self.val_path, self.valSet[i]))
        else:
            for i in range(self.train_len, self.test_len):
                self.train_file_dBZ.append(os.path.join(self.filePath_dBZ, self.train_path, self.trainSet_dBZ[i]))
                self.train_file_KDP.append(os.path.join(self.filePath_KDP, self.train_path, self.trainSet_KDP[i]))
                self.train_file_ZDR.append(os.path.join(self.filePath_ZDR, self.train_path, self.trainSet_ZDR[i]))
                self.val_file.append(os.path.join(self.filePath_dBZ, self.val_path, self.valSet[i]))


    def __getitem__(self, item):
        norm_param = {
            'dBZ': [0, 65],
            'ZDR': [-1, 5],
            'KDP': [-1, 6], }

        mmin_dBZ, mmax_dBZ = norm_param['dBZ']
        mmin_KDP, mmax_KDP = norm_param['KDP']
        mmin_ZDR, mmax_ZDR = norm_param['ZDR']

        train_x_dBZ = np.load(self.train_file_dBZ[item]).astype(np.float32)
        train_x_KDP = np.load(self.train_file_KDP[item]).astype(np.float32)
        train_x_ZDR = np.load(self.train_file_ZDR[item]).astype(np.float32)
        val_x = np.load(self.val_file[item]).astype(np.float32)

        train_x_dBZ = (train_x_dBZ - mmin_dBZ) / (mmax_dBZ - mmin_dBZ)
        train_x_KDP = (train_x_KDP - mmin_KDP) / (mmax_KDP - mmin_KDP)
        train_x_ZDR = (train_x_ZDR - mmin_ZDR) / (mmax_ZDR - mmin_ZDR)
        val_x = (val_x - mmin_dBZ) / (mmax_dBZ - mmin_dBZ)

        return train_x_dBZ, train_x_KDP, train_x_ZDR, val_x

    def __len__(self):
        return len(self.train_file_dBZ)

def get_loaders(
    train=True,
    batch_size=4,
    num_workers=4,
    pin_memory=True,
):
    train_ds = TrainDataset(
        train=train,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    # val_ds = TrainDataset(
    #     train=train_dir,
    #     test=True,
    # )
    #
    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=batch_size,
    #     num_workers=num_workers,
    #     pin_memory=pin_memory,
    #     shuffle=False,
    # )

    return train_loader

# 读取存储为txt文件的数据
def data_read(dir_path):
    with open(dir_path, "r") as f:
        raw_data = f.read()
        data = raw_data[1:-1].split(", ")   # [-1:1]是为了去除文件中的前后中括号"[]"

    return np.asfarray(data, float)

if __name__ == '__main__':
    x = TrainDataset()