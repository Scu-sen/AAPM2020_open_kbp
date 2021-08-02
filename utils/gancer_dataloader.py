import torch.utils.data
import pudb
import os
import os.path
import random
import torchvision.transforms as transforms
import torch
from scipy.io import loadmat
from skimage.transform import rescale
import pudb
import numpy as np
import pandas as pd
from utils.dataloader2D import KBPDataset2D
from utils.preprocessing2D import get_train_tfms

import torch.utils.data as data
from PIL import Image

#
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
#

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename, filetypes=IMG_EXTENSIONS):
    return any(filename.endswith(extension) for extension in filetypes)


def make_dataset(dir, filetypes=IMG_EXTENSIONS):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname, filetypes):
                path = os.path.join(root, fname)
                images.append(path)

    return images


def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader, extension=None):
        if extension == None:
            extension = IMG_EXTENSIONS
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(extension)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)


class BaseDataset(data.Dataset):

    ''' Datasets all have to follow the given format.'''

    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass


def get_transforms(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        # scale images to opt.loadSize using BICUBIC rule
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)


class SliceDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
#         self.root = opt.dataroot
#         self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        config = opt.config
        data_df = pd.read_csv(config.DATA_CSV_PATH)
        split_train_mask = (data_df['Fold'] != 'Fold{}'.format(opt.foldidx))
        train_df = data_df[split_train_mask & (data_df['Split'] == 'Train')].reset_index(drop=True)
        valid_df = data_df[(~split_train_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
        test_df = data_df[data_df['Split'] == 'Test'].reset_index(drop=True)
        maintest_df = data_df[data_df['Split'] == 'MainTest'].reset_index(drop=True)
        if opt.phase == 'train':
            if config.pseudo_path is not None:
                assert not (config.add_val_pseudo and config.add_val_orig)
                if config.add_val_pseudo:
                    pseudo_df = pd.concat((valid_df, test_df, maintest_df))
                else:
                    pseudo_df = pd.concat((test_df, maintest_df))
                pseudo_df['Id'] = pseudo_df['Id'] + '_pseudo'
                if config.add_val_orig:
                    pseudo_df = pd.concat((pseudo_df, valid_df))
                train_df = pd.concat((train_df, pseudo_df)).reset_index(drop=True)
            data_df = train_df
            tfms = get_train_tfms(opt.config)
            valid = False
        elif opt.phase == 'valid':
            data_df = valid_df
            tfms = None
            valid = True
        if opt.debug: data_df = data_df[:10]
        print("transforms: ", tfms)
        print("valid: ", 'True' if valid else 'False')
        print("Training folds: {}".format(data_df['Fold'].unique()))
        self.dataset2D = KBPDataset2D(opt.config, data_df, transform=tfms, valid=valid)
        # go through directory return os.path for all images
#         slice_filetype = ['.mat']
#         self.AB_paths = sorted(make_dataset(self.dir_AB, slice_filetype))
        # assert self.opt.loadSize == self.opt.fineSize, 'No resize or cropping.'

#     def rgb_to_rgb(self, mat):
#         ''' Images are 3 channel tensors.'''
#         dose_img = mat['dMs']
#         ct_img = mat['iMs']
#         w, h, nc = ct_img.shape

#         # assert w == self.opt.loadSize, 'size mismatch in width'
#         #assert h == self.opt.loadSize, 'size mismatch in height'

#         # scale
#         # TODO: Test this feature
#         # scale_to = int(self.opt.loadSize / w)
#         # new_ct_img = np.zeros((w, h, nc))
#         # new_dose_img = np.zeros((w, h, nc))
#         # for ic in range(nc):
#         #    new_ct_img[:, :, ic] = rescale(ct_img[:, :, ic], scale_to)
#         #    new_dose_img[:, :, ic] = rescale(dose_img[:, :, ic], scale_to)
#         # ct_img = new_ct_img
#         # dose_img = new_dose_img

#         # to handle aaron's weird uint format
#         if dose_img.dtype == np.uint16:
#             dose_img = dose_img / 256
#         if ct_img.dtype == np.uint16:
#             ct_img = ct_img / 256

#         A = transforms.ToTensor()(ct_img).float()
#         B = transforms.ToTensor()(dose_img).float()

#         # ABs are 3-channel. Normalizing to 0.5 mean, 0.5 std
#         A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
#         B = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(B)

#         # flipping augments the dataset by flipping a bunch of the images.
#         if (not self.opt.no_flip) and random.random() < 0.5:
#             idx = [i for i in range(A.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             A = A.index_select(2, idx)
#             B = B.index_select(2, idx)
#         return A, B

#     def rgb_to_gray(self, mat):
#         ''' Reference images are ct scans with contours. output is single
#         channel dose intensity values.'''
#         dose_val = mat['dMs']
#         ct_img = mat['iMs']
#         w, h, nc = ct_img.shape
#         assert (w, h) == dose_val.shape, 'size mismatch between dose and ct'
        
#         if dose_val.dtype == np.uint16:
#             dose_val = dose_val / 256
#         if ct_img.dtype == np.uint16:
#             ct_img = ct_img / 256

#         A = transforms.ToTensor()(ct_img).float()
#         A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A)
        
#         B = torch.from_numpy(dose_val).float()
#         B = B.unsqueeze(0)
#         B.sub_(0.5).div_(0.5)
        
#         # flipping augments the dataset by flipping a bunch of the images.
#         if (not self.opt.no_flip) and random.random() < 0.5:
#             idx = [i for i in range(A.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             A = A.index_select(2, idx)
#             B = B.index_select(2, idx)
#         return A, B

    
#     def gray_to_gray(self, mat):
#         ''' Reference images are ct scans with contours. output is single
#         channel dose intensity values.'''
#         dose_val = mat['dMs']
#         ct_img = mat['iMs']
#         w, h = ct_img.shape
#         assert (w, h) == dose_val.shape, 'size mismatch between dose and ct'
        
#         if dose_val.dtype == np.uint16:
#             dose_val = dose_val / 256
#         if ct_img.dtype == np.uint16:
#             ct_img = ct_img / 256

#         A = torch.from_numpy(ct_img).float()
#         A = A.unsqueeze(0)
#         A.sub_(0.5).div_(0.5)
        
#         B = torch.from_numpy(dose_val).float()
#         B = B.unsqueeze(0)
#         B.sub_(0.5).div_(0.5)
        
#         # flipping augments the dataset by flipping a bunch of the images.
#         if (not self.opt.no_flip) and random.random() < 0.5:
#             idx = [i for i in range(A.size(2) - 1, -1, -1)]
#             idx = torch.LongTensor(idx)
#             A = A.index_select(2, idx)
#             B = B.index_select(2, idx)
#         return A, B

    def __getitem__(self, index):
#         input_nc = self.opt.input_nc
#         output_nc = self.opt.output_nc
        
        img, (target, pdm, sm, vs, idx, is_pseudo) = self.dataset2D[index]
        A = img
        B = target[0:1]

        AB_path = index  # self.AB_paths[index]
#         mat = loadmat(AB_path)

#         if input_nc == 3 and output_nc == 3:
#             A, B = self.rgb_to_rgb(mat)
#         elif input_nc == 3 and output_nc == 1:
#             A, B = self.rgb_to_gray(mat)
#         elif input_nc == 1 and output_nc == 1:
#             A, B = self.gray_to_gray(mat)
#         else:
#             raise NotImplementedError('inappropriate input_nc/output_nc')

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        return len(self.dataset2D)

    def name(self):
        return 'SliceDataset'

class BaseDataLoader():

    ''' All datasets should follow this protocol. '''

    def __init__(self):
        pass

    def initialize(self, opt):
        self.opt = opt
        pass

    def load_data():
        return None

def CreateDataset(opt):
    ''' Gets called by CustomDatasetDataLoader.initialize(). dataset_mode is
    by default unaligned. Dataset has generic structure, inputs are coming
    from opts. Aligned, Unaligned are for A->B (i.e., image-to-image transfer
    type problems, whereas Single is for z->A problems (and testing).
    '''
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    # I've commented out these dataset modes as we do not use them. They may be
    # useful in a later version.
    # elif opt.dataset_mode == 'unaligned':
    #     from data.unaligned_dataset import UnalignedDataset
    #     dataset = UnalignedDataset()
    # elif opt.dataset_mode == 'single':
    #     from data.single_dataset import SingleDataset
    #     dataset = SingleDataset()
    elif opt.dataset_mode == 'slice':
#         from data.slice_dataset import SliceDataset
        dataset = SliceDataset()
    elif opt.dataset_mode == 'voxel':
        from data.voxel_dataset import VoxelDataset
        dataset = VoxelDataset()
    elif opt.dataset_mode == 'beamlet':
        from data.beamlet_dataset import BeamletDataset
        dataset = BeamletDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    ''' Inherited from Base. Carries functions initialize and load_data '''

    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        # Torch Dataloader combines a dataset and sampler, provides settings.
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,  # dataset class
            batch_size=opt.batchSize,  # how many samples/batch to load
            shuffle=True,  # not opt.serial_batches,  # reshuffle per epoch
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            if i >= self.opt.max_dataset_size:  # if more data than want to use
                break
            yield data

def CreateDataLoader(opt):
    ''' Calls CustomDatasetDataLoader and initializes it. '''
    data_loader = CustomDatasetDataLoader()
    print(data_loader.name())
    data_loader.initialize(opt)
    return data_loader
