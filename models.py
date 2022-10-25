from tqdm import tqdm
import torch.nn as nn
import torch
from torch.autograd import Variable

class Block(nn.Module):
  def __init__(self,inplace):
    super().__init__()
    self.conv1=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=2,stride=2,padding=0)
    self.conv2=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=4,stride=2,padding=1)
    self.conv3=nn.Conv1d(in_channels=inplace,out_channels=32,kernel_size=8,stride=2,padding=3)
    self.relu=nn.ReLU()

  def forward(self,x):
    x1=self.relu(self.conv1(x))
    x2=self.relu(self.conv2(x))
    x3=self.relu(self.conv3(x))
    x=torch.cat([x1,x3,x3],dim=1)
    return x

class ChronoNet(nn.Module):
  def __init__(self,channel):
    super().__init__()
    self.block1=Block(channel)
    self.block2=Block(96)
    self.block3=Block(96)
    self.gru1=nn.GRU(input_size=96,hidden_size=32,batch_first=True)
    self.gru2=nn.GRU(input_size=32,hidden_size=32,batch_first=True)
    self.gru3=nn.GRU(input_size=64,hidden_size=32,batch_first=True)
    self.gru4=nn.GRU(input_size=96,hidden_size=32,batch_first=True)
    self.gru_linear=nn.Linear(62,1)
    self.flatten=nn.Flatten()
    self.fc1=nn.Linear(32,1)
    self.relu=nn.ReLU()

  def forward(self,x):
    x = x.squeeze()
    x=self.block1(x)
    x=self.block2(x)
    x=self.block3(x)
    x=x.permute(0,2,1)
    gru_out1,_=self.gru1(x)
    gru_out2,_=self.gru2(gru_out1)
    gru_out=torch.cat([gru_out1, gru_out2],dim=2)
    gru_out3,_=self.gru3(gru_out)
    gru_out=torch.cat([gru_out1, gru_out2, gru_out3],dim=2)
    linear_out=self.relu(self.gru_linear(gru_out.permute(0,2,1)))
    gru_out4,_=self.gru4(linear_out.permute(0,2,1))
    x=self.flatten(gru_out4)
    x=self.fc1(x)
    out = torch.sigmoid(x)
    return out


class LSTM(nn.Module):
    def __init__(self, input_size, num_channels, hidden_units):
        super().__init__()
        self.num_channels = num_channels
        self.hidden_units = hidden_units
        self.num_layers = 2
        self.input_size = input_size
        self.inputsize_channels = num_channels * input_size

        self.lstm = nn.LSTM(
            input_size=num_channels,
            hidden_size=hidden_units,
            batch_first=True,
            num_layers=self.num_layers
        )

        self.linear = nn.Linear(in_features=self.hidden_units, out_features=1)
        self.enc = nn.Linear(in_features=self.inputsize_channels, out_features=100*num_channels)

    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, x.shape[1]*x.shape[2])
        x = self.enc(x)
        x = x.view(batch_size, int(x.shape[-1]/self.num_channels), self.num_channels)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_units).requires_grad_()
        lstm_out, (hn, _) = self.lstm(x, (h0, c0))
        out = self.linear(hn[0]).flatten()
        out = torch.sigmoid(out)
        return out