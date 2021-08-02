# https://github.com/singh-hrituraj/PixelCNN-Pytorch/blob/master/Model.py

'''
Code by Hrituraj Singh
Indian Institute of Technology Roorkee
'''

# from MaskedCNN import MaskedCNN
import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedCNN(nn.Conv2d):
    """
    Implementation of Masked CNN Class as explained in A Oord et. al. 
    Taken from https://github.com/jzbontar/pixelcnn-pytorch
    """

    def __init__(self, mask_type, *args, **kwargs):
        self.mask_type = mask_type
        assert mask_type in ['A', 'B'], "Unknown Mask Type"
        super(MaskedCNN, self).__init__(*args, **kwargs)
        self.register_buffer('mask', self.weight.data.clone())

        _, depth, height, width = self.weight.size()
        self.mask.fill_(1)
        if mask_type =='A':
            self.mask[:,:,height//2,width//2:] = 0
            self.mask[:,:,height//2+1:,:] = 0
        else:
            self.mask[:,:,height//2,width//2+1:] = 0
            self.mask[:,:,height//2+1:,:] = 0


    def forward(self, x):
        self.weight.data*=self.mask
        return super(MaskedCNN, self).forward(x)


class PixelCNN(nn.Module):
    """
    Network of PixelCNN as described in A Oord et. al. 
    """
    def __init__(self, in_channels=1, no_layers=8, kernel = 7, channels=64, device=None):
        super(PixelCNN, self).__init__()
        self.no_layers = no_layers
        self.kernel = kernel
        self.channels = channels
        self.layers = {}
        self.device = device

        self.Conv2d_1 = MaskedCNN('A',in_channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_1 = nn.BatchNorm2d(channels)
        self.ReLU_1= nn.ReLU()

        self.Conv2d_2 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_2 = nn.BatchNorm2d(channels)
        self.ReLU_2= nn.ReLU()

        self.Conv2d_3 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_3 = nn.BatchNorm2d(channels)
        self.ReLU_3= nn.ReLU()

        self.Conv2d_4 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_4 = nn.BatchNorm2d(channels)
        self.ReLU_4= nn.ReLU()

        self.Conv2d_5 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_5 = nn.BatchNorm2d(channels)
        self.ReLU_5= nn.ReLU()

        self.Conv2d_6 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_6 = nn.BatchNorm2d(channels)
        self.ReLU_6= nn.ReLU()

        self.Conv2d_7 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_7 = nn.BatchNorm2d(channels)
        self.ReLU_7= nn.ReLU()

        self.Conv2d_8 = MaskedCNN('B',channels,channels, kernel, 1, kernel//2, bias=False)
        self.BatchNorm2d_8 = nn.BatchNorm2d(channels)
        self.ReLU_8= nn.ReLU()

        self.out = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        if x.shape[1] == 108:
            pdm = x[:,49:50].detach()  # (bs, 1, 128, 128)
        elif x.shape[1] == 315:
            pdm = x[:,141:142].detach()  # (bs, 1, 128, 128)
        else:
            raise ValueError("Input not supported")
        
        x = self.Conv2d_1(x)
        x = self.BatchNorm2d_1(x)
        x = self.ReLU_1(x)

        x = self.Conv2d_2(x)
        x = self.BatchNorm2d_2(x)
        x = self.ReLU_2(x)

        x = self.Conv2d_3(x)
        x = self.BatchNorm2d_3(x)
        x = self.ReLU_3(x)

        x = self.Conv2d_4(x)
        x = self.BatchNorm2d_4(x)
        x = self.ReLU_4(x)

        x = self.Conv2d_5(x)
        x = self.BatchNorm2d_5(x)
        x = self.ReLU_5(x)

        x = self.Conv2d_6(x)
        x = self.BatchNorm2d_6(x)
        x = self.ReLU_6(x)

        x = self.Conv2d_7(x)
        x = self.BatchNorm2d_7(x)
        x = self.ReLU_7(x)

        x = self.Conv2d_8(x)
        x = self.BatchNorm2d_8(x)
        x = self.ReLU_8(x)

        x *= pdm
        return self.out(x)
