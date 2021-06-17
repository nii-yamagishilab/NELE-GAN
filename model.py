import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.utils.spectral_norm as spectral_norm



## Conv1D Generator
class ConvNorm(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1,
                 padding=None, dilation=1, bias=True, w_init_gain='linear'):
        super(ConvNorm, self).__init__()
        if padding is None:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels, out_channels,
                                    kernel_size=kernel_size, stride=stride,
                                    padding=padding, dilation=dilation,
                                    bias=bias)

        torch.nn.init.xavier_uniform_(
            self.conv.weight, gain=torch.nn.init.calculate_gain(w_init_gain))

    def forward(self, signal):
        conv_signal = self.conv(signal)
        return conv_signal


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        This is used to clipping CNN outputs to simulate causuality
        """
        return x[:, :, :-self.chomp_size].contiguous()


class Generator_Conv1D_cLN(nn.Module):
    def __init__(self):
        super(Generator_Conv1D_cLN, self).__init__()
        self.convolutions = nn.ModuleList()
        self.convolutions.append(
            nn.Sequential(
                ConvNorm(64*2, 256,
                         kernel_size=5, stride=1,
                         padding=int((5 - 1)),
                         dilation=1, w_init_gain='tanh'),
                Chomp1d(5-1),
                cLN(256)
        ))

        for i in range(1, 6 - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(256,
                             256,
                             kernel_size=7, stride=1,
                             padding=int((7 - 1)),
                             dilation=1, w_init_gain='tanh'),
                    Chomp1d(7-1),
                    cLN(256)
            ))

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(256, 64,
                         kernel_size=5, stride=1,
                         padding=int((5 - 1)),
                         dilation=1, w_init_gain='linear'),
                Chomp1d(5-1),
                cLN(64)
            ))
        self.LReLU = nn.LeakyReLU(0.3)
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        # print('2 FC!')

    def forward(self,x,y):
        #  x: clean mag, y: noise mag
        inputs = torch.cat((x, y),dim=2) #[B, T, D]
        inputs = inputs.transpose(1, 2).contiguous() #[B, D, T]
        x = inputs
        for i in range(len(self.convolutions) - 1):
            x = self.LReLU(self.convolutions[i](x))
        # output = self.convolutions[-1](x) #[B, D, T]
        output = self.LReLU(self.convolutions[-1](x)) #[B, D, T]
        output = output.transpose(1, 2).contiguous()

        output = self.fc1(output)
        output = self.LReLU(output)
        output = self.fc2(output)

        return torch.exp(3.2*torch.tanh(output))


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(3, 8,(1,1))))
        layers.append(spectral_norm(nn.Conv2d(8, 16, (3,3))))
        layers.append(spectral_norm(nn.Conv2d(16, 32, (5,5))))
        layers.append(spectral_norm(nn.Conv2d(32, 48, (7,7))))
        layers.append(spectral_norm(nn.Conv2d(48, 64, (9,9))))
        self.layers = nn.ModuleList(layers)

        self.GAPool = nn.AdaptiveAvgPool2d((1,1))
        self.LReLU = nn.LeakyReLU(0.3)
        self.fc1 = spectral_norm(nn.Linear(64, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 3)) # Here 3 output nodes corresponding to SIIB, HASPI, and ESTOI

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = self.LReLU(x)

        x = self.GAPool(x)
        B = x.shape[0]
        C = x.shape[1]

        x = x.view(B,C).contiguous()
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x


class Discriminator_Quality(nn.Module):
    def __init__(self):
        super(Discriminator_Quality, self).__init__()
        layers = []
        layers.append(spectral_norm(nn.Conv2d(2, 8,(1,1))))
        layers.append(spectral_norm(nn.Conv2d(8, 16, (3,3))))
        layers.append(spectral_norm(nn.Conv2d(16, 32, (5,5))))
        layers.append(spectral_norm(nn.Conv2d(32, 48, (7,7))))
        layers.append(spectral_norm(nn.Conv2d(48, 64, (9,9))))
        self.layers = nn.ModuleList(layers)

        self.GAPool = nn.AdaptiveAvgPool2d((1,1))
        self.LReLU = nn.LeakyReLU(0.3)
        self.fc1 = spectral_norm(nn.Linear(64, 64))
        self.fc2 = spectral_norm(nn.Linear(64, 16))
        self.fc3 = spectral_norm(nn.Linear(16, 2)) # Here 2 output nodes corresponding to PESQ and ViSQOL

    def forward(self,x):
        for layer in self.layers:
            x = layer(x)
            x = self.LReLU(x)

        x = self.GAPool(x)
        B = x.shape[0]
        C = x.shape[1]

        x = x.view(B,C).contiguous()
        x = self.LReLU(self.fc1(x))
        x = self.LReLU(self.fc2(x))

        x = torch.sigmoid(self.fc3(x))
        return x

class cLN(nn.Module):
    def __init__(self, dimension, eps = 1e-8, trainable=True):
        super(cLN, self).__init__()
        
        self.eps = eps
        if trainable:
            self.gain0 = nn.Parameter(torch.ones(1, dimension, 1))
            self.bias0 = nn.Parameter(torch.zeros(1, dimension, 1))
        else:
            self.gain0 = Variable(torch.ones(1, dimension, 1), requires_grad=False)
            self.bias0 = Variable(torch.zeros(1, dimension, 1), requires_grad=False)

    def forward(self, input):
        # input size: (Batch, Freq, Time)
        # cumulative mean for each time step
        
        batch_size = input.size(0)
        channel = input.size(1)
        time_step = input.size(2)
        
        step_sum = input.sum(1)  # B, T
        step_pow_sum = input.pow(2).sum(1)  # B, T
        cum_sum = torch.cumsum(step_sum, dim=1)  # B, T
        cum_pow_sum = torch.cumsum(step_pow_sum, dim=1)  # B, T
        
        entry_cnt = np.arange(channel, channel*(time_step+1), channel)
        entry_cnt = torch.from_numpy(entry_cnt).type(input.type())
        entry_cnt = entry_cnt.view(1, -1).expand_as(cum_sum)
        
        cum_mean = cum_sum / entry_cnt  # B, T
        cum_var = (cum_pow_sum - 2*cum_mean*cum_sum) / entry_cnt + cum_mean.pow(2)  # B, T
        cum_std = (cum_var + self.eps).sqrt()  # B, T
        
        cum_mean = cum_mean.unsqueeze(1)
        cum_std = cum_std.unsqueeze(1)
        
        x = (input - cum_mean.expand_as(input)) / cum_std.expand_as(input)
        return x * self.gain0.expand_as(x).type(x.type()) + self.bias0.expand_as(x).type(x.type())