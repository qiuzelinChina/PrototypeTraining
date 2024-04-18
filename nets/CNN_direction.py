import torch
import torch.nn as nn
from utils.utils import EEG_ch_to_2D
import torch.nn.functional as F

    
    
class CNN_baseline(nn.Module):
    def __init__(self, win_len=128):
        super(CNN_baseline, self).__init__()

        self.conv_layer = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(17,64), padding=(8, 0))
        self.relu = nn.ReLU()
        self.avg_pool = nn.AvgPool2d(kernel_size=(win_len, 1))
        self.fc1 = nn.Linear(in_features=5, out_features=5)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(in_features=5, out_features=2)

    def forward(self, x, env0, env1):
        x = x.unsqueeze(dim=1)
        conv_out = self.conv_layer(x)
        relu_out = self.relu(conv_out)
        avg_pool_out = self.avg_pool(relu_out)
        flatten_out = torch.flatten(avg_pool_out, start_dim=1)
        fc1_out = self.fc1(flatten_out)
        sigmoid_out = self.sigmoid(fc1_out)
        fc2_out = self.fc2(sigmoid_out)

        return fc2_out
    


    
class EEGWaveNet(nn.Module):        # CNN2加上了batch norm
    def __init__(self, win_len=128):
        super(EEGWaveNet, self).__init__()
        self.ch = 9

        self.conv = nn.Sequential(nn.Conv2d(64, self.ch, kernel_size=(3, 3), stride=(1, 1)), nn.Dropout(0), nn.BatchNorm2d(self.ch), nn.ReLU())

       
        
        self.linear = nn.Linear(9*47, 2)

        self.softmax = nn.Softmax(-1)
        self.drop = nn.Dropout(0.3)
        self.sigmoid= nn.Sigmoid()

        
         
              
    def forward(self, x, env0, env1):  	# x: (B, 1, T, channel)
        
        x = self.conv(x)
        
        x = torch.mean(x, 2).flatten(1)

        x = self.linear(x)

        
        return x
    

