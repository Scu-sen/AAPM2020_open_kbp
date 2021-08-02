import argparse
import os
import utils.gancer_util as util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument(
                '--config',
                default=None,
                help=
                'config file'
                )
        self.parser.add_argument(
                '-f', '--foldidx',
                default=0,
                help=
                'which fold to train'
                )
        self.parser.add_argument(
                '--loadj',
                action='store_true',
                help=
                'load jaehee data'
                )
        self.parser.add_argument(
                '--nloadj',
                type=int,
                default=0,
                help=
                'not load jaehee data for these many samples'
                )
        self.parser.add_argument(
                '--debug',
                action='store_true',
                help=
                'debugging'
                )
        self.parser.add_argument(
                '--dataroot',
                default='./data/gancer_cancer',
                help=
                'path to images (should have subfolders trainA, trainB, valA, valB, etc)'
                )
        self.parser.add_argument(
                '--batchSize', type=int, default=8, help='input batch size')
        self.parser.add_argument(
                '--loadSize',
                type=int,
                default=128,
                help='scale images to this size')
        self.parser.add_argument(
                '--fineSize', type=int, default=128, help='then crop to this size')
        self.parser.add_argument(
                '--input_nc',
                type=int,
                default=108,
                help='# of input image channels')
        self.parser.add_argument(
                '--output_nc',
                type=int,
                default=1,
                help='# of output image channels')
        self.parser.add_argument(
                '--ngf',
                type=int,
                default=64,
                help='# of gen filters in first conv layer')
        self.parser.add_argument(
                '--ndf',
                type=int,
                default=64,
                help='# of discrim filters in first conv layer')
        self.parser.add_argument(
                '--nwf',
                type=int,
                default=64,
                help='# of beamlet filters in first conv layer')
        self.parser.add_argument(
                '--which_model_netD',
                type=str,
                default='n_layers',
                help='selects model to use for netD')
        self.parser.add_argument(
                '--which_model_netG',
                type=str,
                default='unet2dsepconveasy',
                help='selects model to use for netG')
        self.parser.add_argument(
                '--which_model_netW',
                type=str,
                default='resnet_temp',
                help='selects model to use for netW')
        self.parser.add_argument(
                '--n_layers_D',
                type=int,
                default=3,
                help='only used if which_model_netD==n_layers')
        self.parser.add_argument(
                '--gpu_ids',
                type=str,
                default='0',
                help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument(
                '--name',
                type=str,
                default='cancer_pix2pix',
                help=
                'name of the experiment. It decides where to store samples and models'
                )
        self.parser.add_argument(
                '--dataset_mode',
                type=str,
                default='slice',
                help=
                'chooses how datasets are loaded. [voxel | slice | aligned | unaligned | single]'
                )
        self.parser.add_argument(
                '--model',
                type=str,
                default='pix2pix',
                help='chooses which model to use. pix2pix,vox2vox, beamlet'
                )
        self.parser.add_argument(
                '--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument(
                '--nThreads',
                type=int,
                default=os.cpu_count(),
                help='# threads for loading data')
        self.parser.add_argument(
                '--checkpoints_dir',
                type=str,
                default='./checkpoints',
                help='models are saved here')
        self.parser.add_argument(
                '--norm',
                type=str,
                default='batch',
                help=
                'instance normalization or batch normalization [batch | instance | batch_3d | instance_3d]'
                )
        self.parser.add_argument(
                '--serial_batches',
                action='store_true',
                help=
                'if true, takes images in order to make batches, otherwise takes them randomly'
                )
        self.parser.add_argument(
                '--display_winsize',
                type=int,
                default=128,
                help='display window size')
        self.parser.add_argument(
                '--display_id',
                type=int,
                default=1,
                help='window id of the web display')
        self.parser.add_argument(
                '--display_port',
                type=int,
                default=8097,
                help='visdom port of the web display')
        self.parser.add_argument(
                '--no_dropout',
                action='store_true',
                help='no dropout for the generator')
        self.parser.add_argument(
                '--max_dataset_size',
                type=int,
                default=float("inf"),
                help=
                'maximum number of samples allowed per dataset. If the directory contains more than the max size, then only a subset is laoded.'
                )
        self.parser.add_argument(
                '--resize_or_crop',
                type=str,
                default='resize_and_crop',
                help=
                'scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]'
                )
        self.parser.add_argument(
                '--no_flip',
                action='store_true',
                help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument(
                '--init_type',
                type=str,
                default='normal',
                help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument(
                '--no_img',
                action='store_true',
                help=
                'if specified, do not convert 1 channel to 3 channel output.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
        util.mkdirs(expr_dir)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')
        return self.opt
