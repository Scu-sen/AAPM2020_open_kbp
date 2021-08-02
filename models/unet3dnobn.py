# https://github.com/Tencent/MedicalNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial
from IPython.core.debugger import set_trace

__all__ = [
    'UNet', 'unet10', 'unet18', 'unet34', 'unet50', 'unet101',
    'unet152', 'unet200'
]


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
#         self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
#         self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
#         out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
#         out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
#         self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
#         out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
#         out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
#         out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class UNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(UNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
#         self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.upconv1 = nn.Conv3d(512 * block.expansion, 256, kernel_size=3, stride=1, padding=1)
#         self.upbn1 = nn.BatchNorm3d(256)
        self.upconv2 = nn.Conv3d(512, 128, kernel_size=3, stride=1, padding=1)
#         self.upbn2 = nn.BatchNorm3d(128)
        self.upconv3 = nn.ConvTranspose3d(256, 64, kernel_size=4, stride=2, padding=1)
#         self.upbn3 = nn.BatchNorm3d(64)
        self.upconv4 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1)
#         self.upbn4 = nn.BatchNorm3d(64)
        self.upconv5 = nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1)
#         self.upbn5 = nn.BatchNorm3d(32)
        self.upconv6 = nn.Conv3d(32, num_seg_classes, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):  # [1, 1, 128, 128, 128]
#         set_trace()
        x = self.conv1(x)  # [1, 64, 64, 64, 64]
#         x = self.bn1(x)
        x1 = self.relu(x)
        
        x2 = self.maxpool(x1)  # [1, 64, 32, 32, 32]
        x2 = self.layer1(x2)  # [1, 64, 32, 32, 32]
        x3 = self.layer2(x2)  # [1, 128, 16, 16, 16]
        x4 = self.layer3(x3)  # [1, 256, 16, 16, 16]
        x5 = self.layer4(x4)  # [1, 512, 16, 16, 16]
        
        x = F.relu(self.upconv1(x5))  # [1, 256, 16, 16, 16]
        x = torch.cat([x, x4], dim=1)  # [1, 512, 16, 16, 16]
        x = F.relu(self.upconv2(x))  # [1, 128, 16, 16, 16]
        x = torch.cat([x, x3], dim=1)  # [1, 256, 16, 16, 16]
        x = F.relu(self.upconv3(x))  # [1, 64, 32, 32, 32]
        x = torch.cat([x, x2], dim=1)  # [1, 128, 32, 32, 32]
        x = F.relu(self.upconv4(x))  # [1, 64, 64, 64, 64]
        x = torch.cat([x, x1], dim=1)  # [1, 128, 64, 64, 64]
        x = F.relu(self.upconv5(x))  # [1, 32, 128, 128, 128]
        x = F.relu(self.upconv6(x))  # [1, 1, 128, 128, 128]

        return x
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False)) #, nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

class UNet101(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda = False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(UNet101, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)
            
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)
        
        self.upconv1 = nn.Conv3d(512 * block.expansion, 256, kernel_size=3, stride=1, padding=1)
        self.upbn1 = nn.BatchNorm3d(256)
        self.upconv2 = nn.Conv3d(1280, 128, kernel_size=3, stride=1, padding=1)
        self.upbn2 = nn.BatchNorm3d(128)
        self.upconv3 = nn.ConvTranspose3d(640, 64, kernel_size=4, stride=2, padding=1)
        self.upbn3 = nn.BatchNorm3d(64)
        self.upconv4 = nn.ConvTranspose3d(320, 64, kernel_size=4, stride=2, padding=1)
        self.upbn4 = nn.BatchNorm3d(64)
        self.upconv5 = nn.ConvTranspose3d(128, 32, kernel_size=4, stride=2, padding=1)
        self.upbn5 = nn.BatchNorm3d(32)
        self.upconv6 = nn.Conv3d(33, num_seg_classes, kernel_size=3, stride=1, padding=1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x0):  # [1, 1, 128, 128, 128]
#         set_trace()
        x = self.conv1(x0)  # [1, 64, 64, 64, 64]
        x = self.bn1(x)
        x1 = self.relu(x)
        
        x2 = self.maxpool(x1)  # [1, 64, 32, 32, 32]
        x2 = self.layer1(x2)  # [1, 64, 32, 32, 32]
        x3 = self.layer2(x2)  # [1, 128, 16, 16, 16]
        x4 = self.layer3(x3)  # [1, 256, 16, 16, 16]
        x5 = self.layer4(x4)  # [1, 512, 16, 16, 16]
        
        xo = F.relu(self.upbn1(self.upconv1(x5)))  # [1, 256, 16, 16, 16]
        xo = torch.cat([xo, x4], dim=1)  # [1, 512, 16, 16, 16]
        xo = F.relu(self.upbn2(self.upconv2(xo)))  # [1, 128, 16, 16, 16]
        xo = torch.cat([xo, x3], dim=1)  # [1, 256, 16, 16, 16]
        xo = F.relu(self.upbn3(self.upconv3(xo)))  # [1, 64, 32, 32, 32]
        xo = torch.cat([xo, x2], dim=1)  # [1, 128, 32, 32, 32]
        xo = F.relu(self.upbn4(self.upconv4(xo)))  # [1, 64, 64, 64, 64]
        xo = torch.cat([xo, x1], dim=1)  # [1, 128, 64, 64, 64]
        xo = F.relu(self.upbn5(self.upconv5(xo)))  # [1, 32, 128, 128, 128]
        xo = torch.cat([xo, x0], dim=1)  # [1, 128, 64, 64, 64]
        xo = F.relu(self.upconv6(xo))  # [1, 1, 128, 128, 128]

        return xo
    
    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)
    
def unet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = UNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def unet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = UNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def unet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = UNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


def unet50(**kwargs):
    """Constructs a ResNet-50 model.
    """
    model = UNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


def unet101(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = UNet101(Bottleneck, [3, 4, 23, 3], **kwargs)
    return model


def unet152(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = UNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    return model


def unet200(**kwargs):
    """Constructs a ResNet-101 model.
    """
    model = UNet(Bottleneck, [3, 24, 36, 3], **kwargs)
    return model
