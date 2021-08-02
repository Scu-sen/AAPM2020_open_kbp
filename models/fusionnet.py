# https://github.com/charleshouston/unet-pytorch/blob/master/models/fusionnet.py

### Class to define 3D Fusion Net.

from typing import Union, Tuple
import numpy as np
import torch
from torch.autograd import Function, Variable
import torch.nn as nn
import torch.nn.functional as F
# from torch.autograd._functions.utils import prepare_onnx_paddings
from torch.nn.modules.utils import _ntuple

# from models.custom_layers import Softmax3d, ReflectionPad3d


def flip(x: Variable, dim: int) -> Variable:
    """Flip torch Variable along given dimension axis."""
    xsize = x.size()
    dim = x.dim() + dim if dim < 0 else dim
    x = x.contiguous().view(-1, *xsize[dim:])
    x = x.view(x.size(0), x.size(1), -1)[:,
            getattr(torch.arange(x.size(1)-1, -1, -1),
                    ('cpu', 'cuda')[x.is_cuda])().long(), :]
    return x.view(xsize)


class ReflectionPad3d(nn.Module):
    """Wrapper for ReflectionPadNd function in 3 dimensions."""
    def __init__(self, padding: Union[int, Tuple[int]]):
        super(ReflectionPad3d, self).__init__()
        self.padding = _ntuple(6)(padding)

    def forward(self, input: Variable) -> Variable:
        return ReflectionPadNd.apply(input, self.padding)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '(' \
            + str(self.padding) + ')'


class ReflectionPadNd(Function):
    """Padding for same convolutional layer."""

    @staticmethod
    def symbolic(g, input: Variable, padding: Union[int, Tuple[int]]):
        paddings = prepare_onnx_paddings(len(input.type().sizes()), pad)
        return g.op("Pad", input, pads_i=paddings, mode_s="reflect")

    @staticmethod
    def forward(ctx: Function, input: Variable, pad: Tuple[int]) -> Variable:
        ctx.pad = pad
        ctx.input_size = input.size()
        ctx.l_inp = len(input.size())
        ctx.pad_tup = tuple([(a, b)
                             for a, b in zip(pad[:-1:2], pad[1::2])]
                            [::-1])
        ctx.l_pad = len(ctx.pad_tup)
        ctx.l_diff = ctx.l_inp - ctx.l_pad
        assert ctx.l_inp >= ctx.l_pad

        new_dim = tuple([sum((d,) + ctx.pad_tup[i])
                         for i, d in enumerate(input.size()[-ctx.l_pad:])])
        assert all([d > 0 for d in new_dim]), 'input is too small'

        # Create output tensor by concatenating with reflected chunks.
        output = input.new(input.size()[:(ctx.l_diff)] + new_dim).zero_()
        c_input = input

        for i, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                chunk1 = flip(c_input.narrow(i, 0, pad[0]), i)
                c_input = torch.cat((chunk1, c_input), i)
            if p[1] > 0:
                chunk2 = flip(c_input.narrow(i, c_input.shape[i]-p[1], p[1]), i)
                c_input = torch.cat((c_input, chunk2), i)
        output.copy_(c_input)
        return output

    @staticmethod
    def backward(ctx: Function, grad_output: Variable) -> Variable:
        grad_input = Variable(grad_output.data.new(ctx.input_size).zero_())
        grad_input_slices = [slice(0, x,) for x in ctx.input_size]

        cg_output = grad_output
        for i_s, p in zip(range(ctx.l_inp)[-ctx.l_pad:], ctx.pad_tup):
            if p[0] > 0:
                cg_output = cg_output.narrow(i_s, p[0],
                                             cg_output.size(i_s) - p[0])
            if p[1] > 0:
                cg_output = cg_output.narrow(i_s, 0,
                                             cg_output.size(i_s) - p[1])
        gis = tuple(grad_input_slices)
        grad_input[gis] = cg_output

        return grad_input, None, None


class Softmax3d(nn.Module):
    """Applies softmax over features for each spatial location.

    Expects a volumetric image of dimensions `(N, C, D, H, W)`.
    """

    def forward(self, input: Variable) -> Variable:
        assert input.dim() == 5, 'Softmax3d requires a 5D Tensor.'
        return F.softmax(input, 1, _stacklevel=5)

    def __repr__(self) -> str:
        return self.__class__.__name__ + '()'


class ResidualLayer(nn.Module):
    """Definition of a FusionNet residual layer."""

    def __init__(self, features_in: int):
        """Initialisation.

        Args:
            features_in: Number of input features to layer.
        """
        super(ResidualLayer, self).__init__()
        self.pad = ReflectionPad3d(padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv3d(features_in, features_in,
                               kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(features_in)
        self.conv2 = nn.Conv3d(features_in, features_in,
                               kernel_size=3)
        self.bn2 = nn.BatchNorm3d(features_in)
        self.conv3 = nn.Conv3d(features_in, features_in,
                               kernel_size=3)
        self.bn3 = nn.BatchNorm3d(features_in)

    def forward(self, x: Variable) -> Variable:
        """Foward pass through layer."""
        residual = x
        out = self.conv1(self.pad(x))
        out = self.bn1(self.relu(out))
        out = self.conv2(self.pad(out))
        out = self.bn2(self.relu(out))
        out = self.conv3(self.pad(out))
        out = self.bn3(self.relu(out))
        out += residual

        return out


class BasicBlock(nn.Module):
    """Definition of basic components of a FusionNet encoder/decoder layer."""

    def __init__(self, features_in: int, features_out: int):
        """Initialisation.

        Args:
            features_in: Number of input feature channels.
            features_out: Number of output feature channels.
        """
        super(BasicBlock, self).__init__()
        self.pad = ReflectionPad3d(padding=1)
        self.conv1 = nn.Conv3d(features_in, features_out,
                               kernel_size=3)
        self.relu = nn.ReLU(inplace=True)
        self.bn1 = nn.BatchNorm3d(features_out)
        self.resid = ResidualLayer(features_out)
        self.conv2 = nn.Conv3d(features_out, features_out,
                               kernel_size=3)
        self.bn2 = nn.BatchNorm3d(features_out)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        out = self.conv1(self.pad(x))
        out = self.relu(out)
        out = self.bn1(out)

        out = self.resid(out)

        out = self.conv2(self.pad(out))
        out = self.relu(out)
        out = self.bn2(out)

        return out


class EncodingLayer(nn.Module):
    """Definition of encoding layer in FusionNet architecture."""
    def __init__(self, features_in: int, first: bool=False,
                 pooling: nn.Module=None):
        """Initialisation.

        Args:
            features_in: Number of input feature channels.
            first: Whether this is the first encoding layer.
            pooling: (Optional) max pooling layer.
        """
        super(EncodingLayer, self).__init__()
        if first:
            features_out = features_in
            features_in = 1 # TODO: adapt for more than one input channel
        else:
            features_in = features_in
            features_out = 2 * features_in
        self.basic = BasicBlock(features_in, features_out)
        self.pooling = pooling

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        if self.pooling is not None:
            x = self.pooling(x)
        out = self.basic(x)
        return out


class BridgeLayer(nn.Module):
    """Definition of deepest (bridge) layer in FusionNet architecture."""
    def __init__(self, features_in: int, pooling: nn.Module):
        """Initialisation.

        Args:
            features_in: Number of input feature channels.
            pooling: Max pooling layer.
        """
        super(BridgeLayer, self).__init__()
        self.features_in = features_in
        self.pooling = pooling
        self.basic = BasicBlock(features_in, features_in * 2)
        self.deconv = nn.ConvTranspose3d(features_in * 2, features_in,
                                         kernel_size=2,
                                         stride=2)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        out = self.pooling(x)
        out = self.basic(out)
        out = self.deconv(out)
        return out


class DecodingLayer(nn.Module):
    """Definition of decoding layer in FusionNet architecture."""
    def __init__(self, features_in: int, last: bool=False):
        """Initialisation.

        Args:
            features_in: Number of input feature channels.
            last: Whether this is the last decoding layer.
        """
        super(DecodingLayer, self).__init__()
        self.basic = BasicBlock(features_in, features_in)
        self.deconv = None
        self.conv_fc = None
        if last:
            self.conv_fc = nn.Conv3d(features_in, 1,
                                     kernel_size=3, padding=1)
#             self.softmax = Softmax3d()
        else:
            self.deconv = nn.ConvTranspose3d(features_in, features_in // 2,
                                             kernel_size=2,
                                             stride=2)

    def forward(self, x: Variable) -> Variable:
        """Forward pass through layer."""
        out = self.basic(x)
        if self.deconv is not None:
            out = self.deconv(out)
        if self.conv_fc is not None:
            # Only final layer.
            out = self.softmax(self.conv_fc(out))
        return out


class FusionNet3d(nn.Module):
    """FusionNet architecture for 3D volumetric images."""

    def __init__(self, input_shape: Tuple[int], res_levels: int,
                 features_root: int):
        """Initialisation.

        Args:
            input_shape: Expected input shape of data `[D, W, H]`.
            res_levels: Number of resolution layers in network.
            features_root: Number of feature maps in first layer.
        """
        super(FusionNet3d, self).__init__()
        assert len(input_shape) == 3, "Expect 3D input image data."
        self.input_shape = input_shape

        self.res_levels = res_levels
        self.features_root = features_root

        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.layers = self._construct_net(self.res_levels)

    def _construct_net(self, res_levels: int) -> nn.ModuleList:
        """Instantiates layers for network.

        Args:
            res_levels: Number of resolution levels in FusionNet architecture.

        Returns:
            A module list of all layers in the network.
        """
        layers = nn.ModuleList([])
        features_in = self.features_root

        # Encoding path.
        for i in range(res_levels - 1):
            if (i+1)%3 != 0:
                layers.append(EncodingLayer(features_in, first=True,
                                            pooling=None))
            else:
                layers.append(EncodingLayer(features_in, pooling=self.pool))
                features_in *= 2

        # Deepest layer.
        layers.append(BridgeLayer(features_in, pooling=self.pool))

        # Decoding path.
        for i in range(res_levels-1, 0, -1):
            if i%3 != 0:
                layers.append(DecodingLayer(features_in, last=True))
            else:
                layers.append(DecodingLayer(features_in))
                features_in //= 2

        return layers

    def forward(self, x: Variable) -> Variable:
        """Forward pass through network.

        Args:
            x: Network input of shape `input_shape`.

        Returns:
            The output from the network.
        """
        assert len(x.shape) == 5, "Expect input in 5D format N, C, D, H, W."
        shortcut_connections = []
        for i, layer in enumerate(self.layers):
            if i > self.res_levels-1:
                # Pixel-wise addition of shortcut connection.
                x += shortcut_connections.pop()

            x = layer(x)

            if i < self.res_levels-1:
                shortcut_connections.append(x)

        return x
