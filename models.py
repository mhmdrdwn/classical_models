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
    gru_out=torch.cat([gru_out1,gru_out2],dim=2)
    gru_out3,_=self.gru3(gru_out)
    gru_out=torch.cat([gru_out1,gru_out2,gru_out3],dim=2)
    linear_out=self.relu(self.gru_linear(gru_out.permute(0,2,1)))
    gru_out4,_=self.gru4(linear_out.permute(0,2,1))
    x=self.flatten(gru_out4)
    x=self.fc1(x)
    out = torch.sigmoid(x)
    return out

class LSTM(nn.Module):
    def __init__(self, output_size, input_size, hidden_size, num_layers, num_channels=21):
        super(LSTM, self).__init__()

        self.num_layers = num_layers
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_channels = num_channels
        self.LSTMs = []
        for i in range(num_channels):
            lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=num_layers, batch_first=True, 
                            bidirectional=True)
            self.LSTMs.append(lstm)
        self.linear = nn.Linear(hidden_size*2, 8)
        self.out_linear = nn.Linear(8*num_channels, output_size)
        self.softmax= nn.LogSoftmax(dim=-1)

    def forward(self, x):
        ch_out = []
        
        for ch_idx in range(self.num_channels):
            x_ch = x[:, ch_idx, :]
            x_ch = x_ch.unsqueeze(1)
            h0 = Variable(torch.zeros(self.num_layers*2, x.size(0), 
                self.hidden_size))
            c0 = Variable(torch.zeros(self.num_layers*2, x.size(0), 
                self.hidden_size))
            lstm_out, (hn, cn) = self.LSTMs[ch_idx](x_ch, (h0, c0))
            #out = lstm_out[:, -1, :]
            out = hn.view(-1, self.hidden_size*2)
            out = self.linear(out)
            ch_out.append(out)
        #print(out.shape)
        ch_out = torch.cat(ch_out, dim=1)
        out = self.out_linear(ch_out)
        #out = self.softmax(out)
        return out
