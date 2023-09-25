import torch
import torch.nn as nn
import torchvision.transforms.functional as TF

class DoubleConv(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        return self.conv(x)

class Resblock(nn.Module):
    def __init__(self, channels):
        super(Resblock, self).__init__()
        self.residual = nn.Sequential(
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(channels)
            )

    def forward(self, x):
        residual = self.residual(x)
        return x + residual

class LSTM(nn.Module):
    def __init__(self, input_size=1024, hidden_layer_size=1024, output_size=1024):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.lstm = nn.LSTM(input_size, hidden_layer_size)
        self.hidden_cell = None

    def forward(self, input_seq):
        # 设定LSTM隐藏层初始值
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size).to('cuda'),
                             torch.zeros(1, 1, self.hidden_layer_size).to('cuda'))
        lstm_out, _ = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1]

class DataEncoder(nn.Module):
    def __init__(self, input_channels):
        super(DataEncoder, self).__init__()
        features = [64, 128, 256, 512]
        self.downs = nn.ModuleList()
        self.pool = nn.ModuleList()
        self.channels = input_channels
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # Encoder
        for feature in features:
            self.downs.append(DoubleConv(self.channels, feature))
            self.pool.append(nn.Conv2d(feature, feature, kernel_size=3, stride=2, padding=1))
            self.channels = feature

    def forward(self, x):
        skip_connections = []

        for down, i in zip(self.downs, range(0, len(self.downs), 1)):
            x = down(x)
            skip_connections.append(x)  # 64*256*256, 128*128*128, 256*64*64, 512*32*32
            x = self.pool[i](x)

        x = self.bottleneck(x)  # 1024*8*8

        return x, skip_connections

class DataDecoder(nn.Module):
    def __init__(self, output_channels, batch_size):
        super(DataDecoder, self).__init__()
        features = [64, 128, 256, 512]
        self.ups = nn.ModuleList()
        self.linear = nn.Linear(features[0], output_channels)
        self.batch_size = batch_size

        # Decoder
        for feature in reversed(features):
            self.ups.append(nn.ConvTranspose2d(
                feature * 2, feature, kernel_size=2, stride=2,
            ))
            self.ups.append(DoubleConv(feature * 2, feature))

    def forward(self, x, connections):
        skip_connections = connections[::-1]  # 64*256*256, 128*128*128, 256*64*64, 512*32*32 颠倒

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx // 2]

            if x.shape != skip_connection.shape:
                skip_connection = TF.resize(skip_connection, size=x.shape[2:])

            x = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx + 1](x)

        x = self.linear(x.view(256, 256, self.batch_size, -1)).reshape(self.batch_size, -1, 256, 256)

        return x

class MainNetwork(nn.Module):
    def __init__(self, input_channels, output_channels, batch_size):
        super(MainNetwork, self).__init__()
        self.encoder = DataEncoder(input_channels)
        self.decoder = DataDecoder(output_channels, batch_size)
        self.a = nn.Parameter(torch.tensor(1.0))
        self.b = nn.Parameter(torch.tensor(0.3))
        self.c = nn.Parameter(torch.tensor(0.3))

    def forward(self, list):

        x = list[0]*self.a + list[1]*self.b + list[2]*self.c

        x, skip_connections = self.encoder(x)

        x = self.decoder(x, skip_connections)

        return x

if __name__ == '__main__':
    batch_size = 4
    model = LSTM()
    x = torch.randn((batch_size, 1024, 4, 4))
    output = model(x)
    print(output.size())