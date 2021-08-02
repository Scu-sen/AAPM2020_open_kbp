import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from utils.default_dataloader import DefaultDataLoader
from fastprogress.fastprogress import progress_bar as tqdm
import itertools
import time
import multiprocessing as mp
import ctypes
from scipy import ndimage

X_DIST = np.array(list(itertools.product(range(127, -1, -1), range(127, -1, -1))), dtype='float32') - 63.5
Y_DIST = X_DIST[:, 1].reshape(1, 128, 128)
X_DIST = X_DIST[:, 0].reshape(1, 128, 128)

def transform_item2D(tfms, img, target):
    imgch = img.shape[0]
    img = np.concatenate((img, target))
    tf_item = tfms(image=img)
    img = tf_item['image'][:imgch]
    target = tf_item['image'][imgch:imgch+1]
    return img, target

def resample(image, voxel_size, new_spacing=[4.5,4.5,None]):
    if type(voxel_size) == np.ndarray:
        voxel_size = voxel_size[0] if len(voxel_size.shape) == 2 else voxel_size
    if type(new_spacing) == np.ndarray:
        new_spacing = new_spacing[0] if len(new_spacing.shape) == 2 else new_spacing
    if type(voxel_size) == list:
        voxel_size[2] = new_spacing[2] if voxel_size[2] is None else voxel_size[2]
    if type(new_spacing) == list:
        new_spacing[2] = voxel_size[2] if new_spacing[2] is None else new_spacing[2]
    voxel_size = np.array(voxel_size)
    new_spacing = np.array(new_spacing)
    
    assert len(new_spacing) == 3
    new_spacing = np.array(new_spacing)
    resize_factor = voxel_size / new_spacing
    resize_factor = np.concatenate(([1], resize_factor))
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    oz = ndimage.interpolation.zoom(image, real_resize_factor, order=0, mode='nearest')
    
    # Pad with zeros
    pad = (image.shape - np.array(oz.shape)+1)//2
    ozxind = np.s_[:] if pad[1] < 0 else np.s_[pad[1]:pad[1]+oz.shape[1]]
    ozyind = np.s_[:] if pad[2] < 0 else np.s_[pad[2]:pad[2]+oz.shape[2]]
    ozf = np.zeros(np.maximum(image.shape, oz.shape), dtype=image.dtype)
    ozf[:,ozxind,ozyind] = oz

    # Crop to original size
    crop = (ozf.shape - np.array(image.shape))//2
    ozxind = np.s_[:image.shape[1]] if crop[1] <= 0 else np.s_[crop[1]:crop[1]+image.shape[1]]
    ozyind = np.s_[:image.shape[2]] if crop[2] <= 0 else np.s_[crop[2]:crop[2]+image.shape[2]]
    return ozf[:,ozxind,ozyind]

class KBPDataset2DStack(Dataset):
    """OpenKBP 2D dataset."""
    def __init__(self, config, data_df, training=True, valid=False, transform=None):
        print('Using Stack Dataset')
        self.config = config
        self.data_df = data_df
        self.training = training
        self.valid = valid
        self.transform = transform
        
        assert config.axis in [1, 2, 3]
        self.axis = config.axis
        print('Loading data along axis: {}'.format(self.axis))
        
        assert config.notargetreplace is None

        self.mode = 'training_model' if training else 'dose_prediction'
        self.defdataset = DefaultDataLoader(self.data_df['Id'], batch_size=1, shuffle=False, mode_name=self.mode)
        
        assert np.array(config.offset_list).sum() == 0
        assert config.offset_list[len(config.offset_list)//2] == 0
        totalch = 12*(128*len(data_df) + len(config.offset_list)//2*(len(data_df)+1))
        self.imgcache = np.zeros((totalch, 128, 128), dtype='int16')
        self.targetcache = []
        self.voxelcache = []
        self.originalvoxelsz = []

        cache_idx = 12*(len(config.offset_list)//2)
        for i in tqdm(range(len(self.data_df))):
            item = self.defdataset.get_batch(index=i)
            img, pdm, sm = item['ct'], item['possible_dose_mask'], item['structure_masks']
            assert np.array_equal(img, img.astype('int16').astype('float64'))
            assert np.array_equal(pdm, pdm.astype('int16').astype('float64'))
            assert np.array_equal(sm, sm.astype('int16').astype('float64'))
            if self.training:
                target = item['dose'][:,:,:,:,0].astype('float32')
            else:
                target = np.zeros_like(img[:,:,:,:,0])
            voxel = item['voxel_dimensions'].astype('float32')
            self.originalvoxelsz.append(np.copy(voxel))
            
            if config.resample is not None:
                img = resample(img, voxel, config.resample.copy())
                pdm = resample(pdm, voxel, config.resample.copy())
                sm = resample(sm, voxel, config.resample.copy())
                target = resample(target, voxel, config.resample.copy())
                voxel[0,:2] = config.resample[:2]
            
            if config.imgmulpdm:
                img *= pdm
                
            img = img.astype('int16')[0]
            pdm = pdm.astype('int16')[0]
            sm = sm.astype('int16')[0]
            img = np.moveaxis(np.concatenate((img, pdm, sm), axis=3), 3, 0)
            img = np.reshape(np.moveaxis(img, config.axis, 1), (12*128, 128, 128), order='F')

            self.imgcache[cache_idx: cache_idx+12*128] = img
            cache_idx += 12*(128 + len(config.offset_list)//2)
    
            self.targetcache.append(np.ascontiguousarray(target))
            self.voxelcache.append(np.ascontiguousarray(voxel))
        
#         For Profiling
        shared_array_base = mp.Array(ctypes.c_float, 3)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.timeindexing = shared_array
    
    def __len__(self):
        return len(self.data_df) * 128

    def __getitem__(self, total_idx):
#         tb = time.time()  # Profiling
        cache_idx = (total_idx + (total_idx//128)*(len(self.config.offset_list)//2))*12
        item_idx, slc_idx = total_idx//128, total_idx%128
        
        idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idxs[self.axis] = slc_idx
        target = self.targetcache[item_idx][tuple(idxs)].astype('float32')
        voxel_size = self.voxelcache[item_idx].astype('float32')
#         self.timeindexing[0] += time.time() - tb  # Profiling

#         tb = time.time()  # Profiling
        img = self.imgcache[cache_idx: cache_idx+len(self.config.offset_list)*12].astype('float32')
#         self.timeindexing[1] += time.time() - tb  # Profiling
        
#         tb = time.time()  # Profiling
        if self.transform is not None:
            img, target = transform_item2D(self.transform, img, target)
#         self.timeindexing[2] += time.time() - tb  # Profiling

        assert img.shape[0] == len(self.config.offset_list) * 12
        posimg = np.where(self.config.offset_list == 0)[0][0] * 12
        pdm = img[posimg+1:posimg+2]
        sm = img[posimg+2:posimg+12]

        if type(target) == np.ndarray:
            target = np.concatenate((target, target*sm), axis=0)
        else:
            target = torch.cat((target, target*sm), axis=0)
            
        if self.config.addvoxelch:
            voxch = np.repeat(voxel_size[0, :, None], 128*128, axis=1).reshape((3, 128, 128))
            img = np.concatenate((img, voxch))
        
        return img, (target, pdm, sm, voxel_size, total_idx//128)
    
    def get3DImage(self, idx, offset_list=None):
        img = []
        for i in range(idx*128, (idx+1)*128):
            img.append(self.__getitem__(i, offset_list)[0][None])
        return np.concatenate(img)

