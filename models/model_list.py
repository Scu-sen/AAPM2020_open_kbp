import sys
import importlib
from functools import partial
import torch
import torch.nn as nn
import torchvision
import pretrainedmodels
from models import resnet3d
from models import unet3d
from models import unet3dnobn
from models.modified3dunet import Modified3DUNet
from models import tasednet
from models.fusionnet import FusionNet3d
from models.exp_net_3D import getExpNet
from models.res2net3d import Res2Net3D, Res2Block
from models.unet2d import UNet as UNet2D
import models.unet2drelubn as Unet2dReluBn
import models.unet2dmish as Unet2dMish
import models.deeplabrelubn as DeeplabReluBn
import models.unet2dprelu as Unet2dPReLu
from models.frrn import frrn as FRRN
import segmentation_models_pytorch as smp
from models.unet2dsepconv import UNet as UNet2DSepConv
from models.unet2dcondconv import UNet as UNet2DCondConv
from models.unet2dantialias import UNet as UNet2DAntiAlias
from models.unet2dsqex import UNet as UNet2DSqEx
from models.encdecsepconv import UNet as EncDecSepConv
from models.unet2dgans import UNet as UNet2DGANs
from models.gancer_networks import UnetGenerator as GancerUnetGenerator
from models.unet2dbeamgate import UNet as UNet2DBeamGate
import models.attnr2unet as AttnR2UNet
from models.anhuinet import UNet as AnhuiNet
from models.pixelcnn import PixelCNN
from models.unet2dlast import UNet as UNet2DLast

def weights_init(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        torch.nn.init.xavier_uniform_(m.weight)
        torch.nn.init.zeros_(m.bias)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()

def resnet10(config):
    return resnet3d.resnet10(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def resnet18(config):
    return resnet3d.resnet18(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def resnet34(config):
    return resnet3d.resnet34(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def resnet50(config):
    return resnet3d.resnet50(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def resnet101(config):
    return resnet3d.resnet101(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def resnet152(config):
    return resnet3d.resnet152(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet10(config):
    return unet3d.unet10(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet18(config):
    return unet3d.unet18(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet34(config):
    return unet3d.unet34(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet50(config):
    return unet3d.unet50(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet101(config):
    return unet3d.unet101(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def unet152(config):
    return unet3d.unet152(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def modified3dunet(config):
    return Modified3DUNet(in_channels=1, n_classes=1)

def unet34nobn(config):
    return unet3dnobn.unet34(sample_input_D=128, sample_input_H=128, sample_input_W=128,
                    num_seg_classes=config.num_classes, shortcut_type='B')

def tasedv2(config):
    return tasednet.TASED_v2()

def fusionnet(config):
    return FusionNet3d(input_shape = (128, 128, 128), res_levels = 8, features_root = 1)

def exp_net(config):
    return getExpNet(NoLabels1=1, dilations=[1, 1, 1, 1], isPriv=True, NoLabels2 = 1, withASPP = False)

RESNET_LAYERS = {18: [2, 2, 2, 2], 
                 34: [3, 4, 6, 3], 
                 50: [3, 4, 6, 3],
                 101: [3, 4, 23, 3], 
                 152: [3, 8, 36, 3]}

def res2net(config):
    return Res2Net3D(Res2Block, RESNET_LAYERS[34], c_in=1, num_classes=1)

def unet2d(config):
    print('Unet 2d with drop rate: {}'.format(config.drop_rate))
    net = UNet2D(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=1,
           num_dilated_convs=4, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=False, padding=1, kernel_size=3,group_norm=config.unet2dgngroups)
#     net.apply(weights_init)
    return net

def unet2drelubn(config):
    return Unet2dReluBn.UNet(in_channels=12, out_channels=1, num_hidden_features=[32, 64, 128, 256, 512, 1024], n_resblocks=1,
           num_dilated_convs=4, dropout_min=0, dropout_max=0, gated=False, padding=1, kernel_size=3,group_norm=32)

def unet2dmish(config):
    return Unet2dMish.UNet(in_channels=12, out_channels=1, num_hidden_features=[32, 64, 128, 256, 512, 1024], n_resblocks=1,
           num_dilated_convs=4, dropout_min=0, dropout_max=0, gated=False, padding=1, kernel_size=3,group_norm=32)

def deeplabrelubn(config):
    return DeeplabReluBn.resnet101(pretrained=True, num_classes=1)

def unet2dprelu(config):
    return Unet2dPReLu.UNet(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=[32, 64, 128, 256, 512, 1024], n_resblocks=1,
           num_dilated_convs=4, dropout_min=config.drop_rate/2., dropout_max=config.drop_rate, gated=False, padding=1, kernel_size=3,group_norm=32)

def frrn(config):
    return FRRN(in_channels=config.in_channels, n_classes=config.num_classes, model_type="B",
                group_norm=True if config.layernorm == 'group' else False, n_groups=16)

def smpunet(config):
    return smp.Unet(config.encoder, encoder_weights='imagenet', in_channels=config.in_channels, classes=config.num_classes)

def smppspnet(config):
    return smp.PSPNet(config.encoder, encoder_weights='imagenet', in_channels=config.in_channels, classes=config.num_classes)

def smpdeeplab(config):
    return smp.DeepLabV3(config.encoder, encoder_weights='imagenet', in_channels=config.in_channels, classes=config.num_classes)

def smpdeeplabplus(config):
    return smp.DeepLabV3Plus(config.encoder, encoder_weights='imagenet', in_channels=config.in_channels, classes=config.num_classes)

def smpfpn(config):
    return smp.FPN(config.encoder, encoder_weights='imagenet', in_channels=config.in_channels, classes=config.num_classes)

def unet2dsepconv(config):
    if config.augautoenc is not None:
        package = 'config.old_configs.{}.config'.format(config.augautoenc)
        encconfig = importlib.import_module(package).config
        augautoenc = getattr(sys.modules[__name__], encconfig.model_name)
        augautoenc = augautoenc(encconfig)
        model_ckpt = './model_weights/{}/models/best_loss.pth'.format(encconfig.exp_name)
        print("Loading auto encoder from ", model_ckpt)
        augautoenc.load_state_dict(torch.load(model_ckpt)['model'])
        augautoenc.to(config.device)
        for param in augautoenc.parameters():
            param.requires_grad = False
    else:
        augautoenc = None

    print('Unet 2d separable conv with drop rate: {}'.format(config.drop_rate))
    net = UNet2DSepConv(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation,augautoenc=augautoenc)
#     net.apply(weights_init)
    return net

def unet2dcondconv(config):
    print('Unet 2d conditional conv with drop rate: {}'.format(config.drop_rate))
    net = UNet2DCondConv(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=1, kernel_size=3,group_norm=config.unet2dgngroups)
#     net.apply(weights_init)
    return net

def unet2dantialias(config):
    print('Unet 2d anti alias conv with drop rate: {}'.format(config.drop_rate))
    net = UNet2DAntiAlias(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=1, kernel_size=3,group_norm=config.unet2dgngroups)
    return net

def unet2dsqex(config):
    print('Unet 2d squeeze excitation with drop rate: {}'.format(config.drop_rate))
    net = UNet2DSqEx(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation)
#     net.apply(weights_init)
    return net

def encdecsepconv(config):
    print('Enc Dec sep conv with drop rate: {}'.format(config.drop_rate))
    net = EncDecSepConv(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation)
#     net.apply(weights_init)
    return net

def unet2dgans(config):
    print('GANs sep conv with drop rate: {}'.format(config.drop_rate))
    net = UNet2DGANs(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation)
#     net.apply(weights_init)
    return net

def gancerunet(config):
    return GancerUnetGenerator(108, 1, 7, ngf=64, norm_layer=partial(nn.BatchNorm2d, affine=True), use_dropout=False, gpu_ids=[0])

def unet2dbeamgate(config):
    if config.augautoenc is not None:
        package = 'config.old_configs.{}.config'.format(config.augautoenc)
        encconfig = importlib.import_module(package).config
        augautoenc = getattr(sys.modules[__name__], encconfig.model_name)
        augautoenc = augautoenc(encconfig)
        model_ckpt = './model_weights/{}/models/best_loss.pth'.format(encconfig.exp_name)
        print("Loading auto encoder from ", model_ckpt)
        augautoenc.load_state_dict(torch.load(model_ckpt)['model'])
        augautoenc.to(config.device)
        for param in augautoenc.parameters():
            param.requires_grad = False
    else:
        augautoenc = None

    print('Unet 2d beam gate with drop rate: {}'.format(config.drop_rate))
    net = UNet2DBeamGate(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation,augautoenc=augautoenc)
#     net.apply(weights_init)
    return net

def attnr2unet(config):
    return AttnR2UNet.R2AttU_Net(config.in_channels, config.num_classes)

def anhuinet(config):
    print('Anhui net with drop rate: {}'.format(config.drop_rate))
    net = AnhuiNet(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation,augautoenc=None,dil_dilations=config.unet2d_dil_dilations)
#     net.apply(weights_init)
    return net

def pixelcnn(config):
    print('pixel cnn')
    return PixelCNN(in_channels=config.in_channels)

def unet2dlast(config):
    return UNet2DLast(in_channels=config.in_channels, out_channels=config.num_classes, num_hidden_features=config.unet2dnum_hidden_features, n_resblocks=config.unet2d_n_resblocks,
           num_dilated_convs=config.unet2d_num_dilated_convs, dropout_min=config.drop_rate_min, dropout_max=config.drop_rate, gated=config.unet2d_gated, padding=config.unet2dpadding, kernel_size=config.unet2dkernel_size,group_norm=config.unet2dgngroups,convdilation=config.convdilation)
