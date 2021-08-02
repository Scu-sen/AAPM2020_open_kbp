# https://github.com/lessw2020/res2net-plus/blob/master/res2fg.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Mish(nn.Module):
    def __init__(self):
        super().__init__()
#         print("Mish activation loaded...")

    def forward(self, x):  
        #save 1 second per epoch with no x= x*() and then return x...just inline it.
        return x *( torch.tanh(F.softplus(x))) 
        #return x * tanh(softplus(x)) = 
        #return x * tanh(ln(1 + e^{x}))
        
act_fn = Mish()

def conv(ni, nf, ks=3, stride=1, bias=False):
    return nn.Conv3d(ni, nf, kernel_size=ks, stride=stride, padding=ks//2, bias=bias)

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Res2Block(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=4, dilation=1, scale=4, first_block=False, norm_layer=None, expansion=4):
        """Implements a residual block
        Args:
            inplanes (int): input channel dimensionality
            planes (int): output channel dimensionality
            stride (int): stride used for conv3x3
            downsample (torch.nn.Module): module used for downsampling
            groups: num of convolution groups
            base_width: base width
            dilation (int): dilation rate of conv3x3            
            scale (int): scaling ratio for cascade convs
            first_block (bool): whether the block is the first to be placed in the conv layer
            norm_layer (torch.nn.Module): norm layer to be used in blocks
        """
        super(Res2Block, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        
        self.expansion = expansion
        width = int(planes * (base_width / 64.)) * groups

        self.conv1 = conv1x1(inplanes, int(width * scale))
        self.bn1 = norm_layer(int(width * scale))

        # If scale == 1, single conv else identity & (scale - 1) convs
        nb_branches = max(scale, 2) - 1
        if first_block:
            self.pool = nn.AvgPool3d(kernel_size=3, stride=stride, padding=1)
        if self.expansion >= 1:
            self.convs = nn.ModuleList([conv3x3(width, width, stride, groups, dilation)
                                        for _ in range(nb_branches)])
        else:
            self.convs = nn.ModuleList([nn.ConvTranspose3d(width, width, kernel_size=2, stride=2, padding=dilation//2,
                                                           groups=groups, bias=False, dilation=dilation)
                                        for _ in range(nb_branches)])
        self.bns = nn.ModuleList([norm_layer(width) for _ in range(nb_branches)])
        self.first_block = first_block
        self.scale = scale
        
        self.conv3 = conv1x1(int(width * scale), int(planes * self.expansion))
        
        self.relu = Mish() #nn.ReLU(inplace=False)
        self.bn3 = norm_layer(int(planes * self.expansion))  #bn reverse

        self.downsample = downsample

    def forward(self, x):

        residual = x

        out = self.conv1(x)
        
        out = self.relu(out)
        out = self.bn1(out) #bn reverse

        # Chunk the feature map
        xs = torch.chunk(out, self.scale, dim=1)
        # Initialize output as empty tensor for proper concatenation
        y = 0
        for idx, conv in enumerate(self.convs):
            # Add previous y-value
            if self.first_block:
                y = xs[idx]
            else:
                y += xs[idx]
            y = conv(y)
            y = self.relu(self.bns[idx](y))
            # Concatenate with previously computed values
            out = torch.cat((out, y), 1) if idx > 0 else y
        # Use last chunk as x1
        if self.scale > 1:
            if self.first_block:
                out = torch.cat((out, self.pool(xs[len(self.convs)])), 1)
            else:
                out = torch.cat((out, xs[len(self.convs)]), 1)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)
        
        out += residual
        out = self.relu(out)

        return out

def conv_layer(ni, nf, ks=3, stride=1, zero_bn=False, act=True):
    bn = nn.BatchNorm3d(nf)
    nn.init.constant_(bn.weight, 0. if zero_bn else 1.)
    if act:
        layers = [conv(ni, nf, ks, stride=stride), act_fn, bn]
    else:
        layers = [conv(ni, nf, ks, stride=stride), bn]
        
    
    #if act: layers.append(act_fn)
    return nn.Sequential(*layers)


class Res2Net3D(nn.Module):
    """Implements a Res2Net model as described in https://arxiv.org/pdf/1904.01169.pdf
    Args:
        block (torch.nn.Module): class constructor to be used for residual blocks
        layers (list<int>): layout of layers
        num_classes (int): number of output classes
        zero_init_residual (bool): whether the residual connections should be initialized at zero
        groups (int): number of convolution groups
        width_per_group (int): number of channels per group
        scale (int): scaling ratio within blocks
        replace_stride_with_dilation (list<bool>): whether stride should be traded for dilation
        norm_layer (torch.nn.Module): norm layer to be used
    """

    def __init__(self, block, layers, c_in=3,num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=26, scale=4, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(Res2Net3D, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm3d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.scale = scale
        self.expansion = 4
        #self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
        #                       bias=False)
        #modify stem
        #stem = []
        sizes = [c_in,32,64,64]  #modified per Grankin
        #for i in range(3):
        #    stem.append(conv_layer(sizes[i], sizes[i+1], stride=2 if i==0 else 1))
        
        #stem (initial entry layers)
        self.conv1 = conv_layer(c_in, sizes[1], stride=2)
        self.conv2 = conv_layer(sizes[1],sizes[2])
        self.conv3 = conv_layer(sizes[2],sizes[3])
        
        
        
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        
        #nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        
        self.scale = 1
        self.layerup1 = self._make_uplayer(block, 2048, layers[3])
        self.layerup2 = self._make_uplayer(block, 512, layers[2])
        self.layerup3 = self._make_uplayer(block, 512, layers[1])
        self.layerup4 = self._make_uplayer(block, 128, layers[0])
        self.layerup5 = self._make_uplayer(block, 32, 2)
        
        self.final_conv = nn.Sequential(nn.Conv3d(8, 8, 3, padding=2, dilation=2), Mish(), nn.BatchNorm3d(8), nn.Conv3d(8, 1, 1))
#         self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
#         self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottle2neck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)
    
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.scale, first_block=True, norm_layer=norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                scale=self.scale, first_block=False, norm_layer=norm_layer))

        return nn.Sequential(*layers)
    
    def _make_uplayer(self, block, planes, blocks, stride=1, dilate=False, expansion=0.25):
        self.expansion = expansion
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
#         if stride != 1 or self.inplanes != int(planes * self.expansion):
        downsample = nn.Sequential(
            nn.ConvTranspose3d(self.inplanes, int(planes * self.expansion), kernel_size=2, stride=2),
            norm_layer(int(planes * self.expansion)),
        )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, self.scale, first_block=True,
                            norm_layer=norm_layer, expansion=self.expansion))
        self.inplanes = int(planes * self.expansion)
        self.expansion = 1
        self.base_width = 16
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, self.inplanes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                scale=self.scale, first_block=False, norm_layer=norm_layer, expansion=self.expansion))
        return nn.Sequential(*layers)

    def forward(self, x):
#         set_trace()
        #stem layers
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = self.maxpool(x)
        
        #res2 block layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        #upsample layers
        x = self.layerup1(x)
        x = self.layerup2(x)
        x = self.layerup3(x)
        x = self.layerup4(x)
        x = self.layerup5(x)
        
        x = self.final_conv(x)

        return x
