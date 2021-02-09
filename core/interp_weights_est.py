# full assembly of the sub-parts to form the complete net

import torch.nn.functional as F
import torch
import torch.nn as nn

from torch.nn.modules.utils import _pair


class Simple(nn.Module):
    def __init__(self, num_ch, out_ch, filter_sz, dilation=None, final_act=nn.Sigmoid(), use_bn=False):
        super().__init__()
        self.__name__ = "Simple"

        assert len(filter_sz) == len(num_ch)

        if dilation is None:
            dilation = [(1, 1)] * len(num_ch)

        self.in_ch = num_ch[0]  # Number of Input channels is added at the beginning of num_ch
        self.num_layers = len(num_ch)-1

        self.conv = nn.ModuleList()
        for i in range(self.num_layers):
            padding = _pair(int(filter_sz[i]//2 + ((filter_sz[i]-1)*(dilation[i]-1))/2))
            if use_bn:
                self.conv.append(nn.Sequential(
                         nn.Conv2d(num_ch[i], num_ch[i+1], filter_sz[i], padding=padding, dilation=dilation[i], stride=1),
                         nn.BatchNorm2d(num_ch[i+1]),
                         nn.ReLU(inplace=True)))
            else:
                self.conv.append(nn.Sequential(
                    nn.Conv2d(num_ch[i], num_ch[i + 1], filter_sz[i], padding=padding, dilation=dilation[i], stride=1),
                    nn.ReLU(inplace=True)))

        padding = _pair(int(filter_sz[-1] // 2 + ((filter_sz[-1] - 1) * (dilation[-1] - 1)) / 2))
        self.out = nn.Conv2d(num_ch[-1], out_ch, filter_sz[-1], padding=padding, dilation=dilation[-1], stride=1)

        if final_act is None:
            self.final_act = nn.Sequential()
        else:
            self.final_act = final_act

    def forward(self, x):
        for i in range(self.num_layers):
            x = self.conv[i](x)
        return self.final_act(self.out(x))


class UNet(nn.Module):
    def __init__(self, num_ch, out_ch, final_act=torch.sigmoid):
        super().__init__()
        self.__name__ = "UNet"

        self.in_ch = num_ch[0]
        self.final_act = final_act

        self.num_downsampling = len(num_ch)-2

        self.encoder = nn.ModuleList([inconv(num_ch[0], num_ch[1])])
        for i in range(1, self.num_downsampling+1):
            self.encoder.append(down(in_ch=num_ch[i], out_ch=num_ch[i+1]))

        self.decoder = nn.ModuleList([up(in_ch1=num_ch[-i-1], in_ch2=num_ch[-i-2], out_ch=num_ch[-i-2], bilinear=False)
                                      for i in range(self.num_downsampling)])

        self.out = outconv(num_ch[1], out_ch)

    def forward(self, x0):
        x_encoder = [x0]

        for i in range(self.num_downsampling+1):
            x_encoder.append(self.encoder[i](x_encoder[i]))

        x_decoder = [x_encoder[-1]]

        for i in range(self.num_downsampling):
            x_decoder.append(self.decoder[i](x_decoder[-1], x_encoder[-i-2]))

        xf = self.final_act(self.out(x_decoder[-1]))

        return xf
        

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch1, in_ch2, out_ch, bilinear=False):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = torch.nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch1, in_ch1, 2, stride=2)

        self.conv = double_conv(in_ch1+in_ch2, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))
        
        # for padding issues, see 
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x


if __name__ == '__main__':
    net = Simple([2,16,16,8], 1, [3,3,3,3,1], [1,1,1,1])
    #net = UNet([3,10,20,30, 40], out_ch=1)
    x = torch.rand((4,2,100,100))
    y = net(x)


