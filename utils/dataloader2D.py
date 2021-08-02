import numpy as np
import copy
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

def resample_slice(image, voxel_size, new_spacing=[4.5,4.5], img_shape=(128, 128)):
    if type(voxel_size) == np.ndarray:
        voxel_size = voxel_size[0] if len(voxel_size.shape) == 2 else voxel_size
    if type(new_spacing) == np.ndarray:
        new_spacing = new_spacing[0] if len(new_spacing.shape) == 2 else new_spacing
    voxel_size = np.array(voxel_size)
    new_spacing = np.array(new_spacing)
    
    assert len(voxel_size) == 2
    assert len(new_spacing) == 2
    resize_factor = voxel_size / new_spacing
    resize_factor = np.concatenate(([1], resize_factor))
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    oz = ndimage.interpolation.zoom(image, real_resize_factor, order=0, mode='nearest')
    
    # Pad with zeros
    img_shape = list(oz.shape)[0:1] + list(img_shape)
    pad = (img_shape - np.array(oz.shape)+1)//2
    ozxind = np.s_[:] if pad[1] < 0 else np.s_[pad[1]:pad[1]+oz.shape[1]]
    ozyind = np.s_[:] if pad[2] < 0 else np.s_[pad[2]:pad[2]+oz.shape[2]]
    ozf = np.zeros(np.maximum(img_shape, oz.shape), dtype=image.dtype)
    ozf[:,ozxind,ozyind] = oz

    # Crop to original size
    crop = (ozf.shape - np.array(img_shape))//2
    ozxind = np.s_[:img_shape[1]] if crop[1] <= 0 else np.s_[crop[1]:crop[1]+img_shape[1]]
    ozyind = np.s_[:img_shape[2]] if crop[2] <= 0 else np.s_[crop[2]:crop[2]+img_shape[2]]
    return ozf[:,ozxind,ozyind]

class KBPDataset2D(Dataset):
    """OpenKBP 2D dataset."""
    def __init__(self, config, data_df, training=True, valid=False, transform=None):
        print('Using Concat Dataset')
        self.config = copy.deepcopy(config)
        self.data_df = data_df.copy()
        self.training = training
        self.valid = valid
        self.transform = transform
        
        self.data_df['loadj'] = config.loadjaehee
        if config.loadjaehee and config.nloadjaehee > 0:
            self.data_df.loc[len(self.data_df)-config.nloadjaehee:, 'loadj'] = False
        
        assert config.axis in [1, 2, 3]
        self.axis = config.axis
        print('Loading data along axis: {}'.format(self.axis))

        self.mode = 'training_model' if training else 'dose_prediction'
        self.defdataset = DefaultDataLoader(self.data_df['Id'], batch_size=1, shuffle=False, mode_name=self.mode, pseudo_path=config.pseudo_path)
        
        self.imgcache = []
        self.targetcache = []
        self.pdmcache = []
        self.smcache = []
        self.voxelcache = []
        self.originalvoxelsz = []
        self.jaeheecache = []
        
        self.pos_map = [(i, j) for i in range(len(self.data_df)) for j in range(128)]
        for i in tqdm(range(len(self.data_df))):
            item = self.defdataset.get_batch(index=i)
            img = item['ct'][:,:,:,:,0].astype('int16')
            if self.training:
                target = item['dose'][:,:,:,:,0].astype('float16')
            else:
                target = img.copy()
            pdm = item['possible_dose_mask'][:,:,:,:,0].astype('bool')
            sm = np.moveaxis(item['structure_masks'][0].astype('bool'), -1, 0)
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
            self.imgcache.append(np.ascontiguousarray(img))
            self.targetcache.append(np.ascontiguousarray(target))
            self.pdmcache.append(np.ascontiguousarray(pdm))
            self.smcache.append(np.ascontiguousarray(sm))
            self.voxelcache.append(np.ascontiguousarray(voxel))
            if config.addjaehee and self.data_df.loc[i, 'loadj']:
                assert self.config.axis == 3
                pat_id = self.data_df.loc[i, 'Id'].split('/')[-1].split('_')[1]
                while len(pat_id) < 3:
                    pat_id = '0' + pat_id
                jc = []
                for sl in range(128):
                    jc.append(np.load('./data/data_Jaehee/comb_data/{}_Feature_{}.npy'.format(pat_id, sl)))
                self.jaeheecache.append(np.concatenate(jc, axis=3))
                if self.config.addtargets:
                    assert self.jaeheecache[-1].shape == (27, 128, 128, 128), self.jaeheecache[-1].shape
                else:
                    assert self.jaeheecache[-1].shape == (23, 128, 128, 128), self.jaeheecache[-1].shape
                assert self.jaeheecache[-1].dtype == np.float16
        
        
#         Resampling
        self.notargetreplace = config.notargetreplace
        if self.training and not self.valid and self.notargetreplace is not None:
            self.notargetreplaceweights = {'pdm': [], 'sm': []}
            for i in range(len(self.pdmcache)):
                pdmsum = self.pdmcache[i].sum(tuple([ax for ax in range(4) if ax != config.axis]))
                smsum = self.smcache[i].sum(tuple([ax for ax in range(4) if ax != config.axis]))
                self.notargetreplaceweights['pdm'].append(pdmsum/pdmsum.sum())
                self.notargetreplaceweights['sm'].append(smsum/smsum.sum())
            
            self.pos_map = []
            for i in range(len(self.pdmcache)):
                nzind = np.where(self.notargetreplaceweights['pdm'][i])[0]
                for j in nzind:
                    self.pos_map.append((i, j))
        
#         For Profiling
        shared_array_base = mp.Array(ctypes.c_float, 3)
        shared_array = np.ctypeslib.as_array(shared_array_base.get_obj())
        self.timeindexing = shared_array
    
    def __len__(self):
        return len(self.pos_map)
    
    def _get_adjacent(self, idx, offset):
        item_idx, slc_idx = self.pos_map[idx]
        print(item_idx, slc_idx)
        if (slc_idx + offset) < 0 or (slc_idx + offset) > 127:
            imgpos = np.zeros((1, 128, 128), dtype='float32')
            targetpos = np.zeros((1, 128, 128), dtype='float32')
            pdmpos = np.zeros((1, 128, 128), dtype='float32')
            smpos = np.zeros((10, 128, 128), dtype='float32')
        else:
            idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
            idxs[self.axis] = slc_idx+offset
            imgpos = self.imgcache[item_idx][tuple(idxs)].astype('float32')
            targetpos = self.targetcache[item_idx][tuple(idxs)].astype('float32')
            pdmpos = self.pdmcache[item_idx][tuple(idxs)].astype('float32')
            smpos = self.smcache[item_idx][tuple(idxs)].astype('float32')
        slcpos = np.ones_like(imgpos)*(slc_idx + offset)
        if self.config.pdmsmmulimg:
                pdmpos *= imgpos
                smpos *= imgpos
        return imgpos, targetpos, pdmpos, smpos, slcpos
    
    def join_pos(self, idx, offset_list):
        img = []
        for offset in offset_list:
            imgpos, targetpos, pdmpos, smpos, slcpos = self._get_adjacent(idx, offset)
            img.extend([imgpos, pdmpos, smpos])
        return np.concatenate(img)
    
    def get_offset(self, idx, offset_list):
        item_idx, slc_idx = self.pos_map[idx]
        low_slc, high_slc = slc_idx + offset_list[0], slc_idx + offset_list[-1]
        l = len(offset_list)
        assert l == high_slc - low_slc + 1, "Only supports continuous offsets"
        idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idxs[self.axis] = np.s_[max(0, low_slc): min(128, high_slc+1)]
        if self.config.addjaehee:
            num_ch = 39 if self.config.addtargets else 35
        else:
            num_ch = 12
        img_shape = [num_ch, 128, 128, 128]
        img_shape[self.axis] = l
        img = np.zeros(img_shape, dtype='float32')
        img_idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        img_idxs[self.axis] = np.s_[max(0, -low_slc): min(l, 127+l-high_slc)]
        img[0:1][tuple(img_idxs)] = self.imgcache[item_idx][tuple(idxs)]
        img[1:2][tuple(img_idxs)] = self.pdmcache[item_idx][tuple(idxs)]
        img[2:12][tuple(img_idxs)] = self.smcache[item_idx][tuple(idxs)]
        if self.config.addjaehee:
            if self.data_df.loc[item_idx, 'loadj']:
                img[12:][tuple(img_idxs)] = self.jaeheecache[item_idx][tuple(idxs)]
            else:
                assert self.config.axis == 3
                pat_id = self.data_df.loc[item_idx, 'Id'].split('/')[-1].split('_')[1]
                while len(pat_id) < 3:
                    pat_id = '0' + pat_id
                for js in zip(range(img_idxs[3].start, img_idxs[3].stop), range(idxs[3].start, idxs[3].stop)):
                    img[12:,:,:,js[0]:js[0]+1] = np.load('./data/data_Jaehee/comb_data/{}_Feature_{}.npy'.format(pat_id, js[1]))
        img = np.moveaxis(img, self.axis, 0)
        img = np.reshape(img, (img.shape[0]*img.shape[1], 128, 128))
        return img

    def __getitem__(self, total_idx, offset_list=None):
#         tb = time.time()  # Profiling
        item_idx, slc_idx = self.pos_map[total_idx]
        
        if self.training and not self.valid and self.notargetreplace in ['pdm', 'sm'] and self.notargetreplaceweights['pdm'][item_idx][slc_idx] == 0:
            slc_idx = np.random.choice(np.arange(128), p=self.notargetreplaceweights[self.notargetreplace][item_idx])
            total_idx = item_idx*128 + slc_idx
        
        idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
        idxs[self.axis] = slc_idx
#         img = self.imgcache[item_idx][tuple(idxs)].astype('float32')
        target = self.targetcache[item_idx][tuple(idxs)].astype('float32')
#         pdm = self.pdmcache[item_idx][tuple(idxs)].astype('float32')
#         sm = self.smcache[item_idx][tuple(idxs)].astype('float32')
        voxel_size = self.voxelcache[item_idx].astype('float32')
#         slc = np.ones_like(img)*(total_idx%128)
#         self.timeindexing[0] += time.time() - tb  # Profiling

#             if not self.valid and np.random.rand() < self.config.mixup:
#                 img, target, possible_dose_mask, structure_masks = self.mixup(img, target, possible_dose_mask, structure_masks, total_idx)

#         tb = time.time()  # Profiling
        if offset_list is None:
            if self.config.offset_list is not None:
                offset_list = self.config.offset_list
            elif self.valid:
                offset_list = np.arange(-3, 4)
            else:
                offset_list = sorted(np.random.choice(range(-5, 0), 3, replace=False)) + [0] + sorted(np.random.choice(range(1, 6), 3, replace=False))
        
#         img = self.join_pos(total_idx, offset_list)  # slow
        img = self.get_offset(total_idx, offset_list)
#         self.timeindexing[1] += time.time() - tb  # Profiling
        
#         tb = time.time()  # Profiling
        imgch = img.shape[0]
        img = np.concatenate((img, target))
        
        if self.training and not self.valid and self.config.patchmix is not None and np.random.rand() < self.config.patchmix:
            mix_total_idx = np.random.choice(range(len(self.pos_map)))
            mix_item_idx, mix_slc_idx = self.pos_map[mix_total_idx]
            mix_idxs = [np.s_[:], np.s_[:], np.s_[:], np.s_[:]]
            mix_idxs[self.axis] = mix_slc_idx
            mix_target = self.targetcache[mix_item_idx][tuple(mix_idxs)].astype('float32')
            mix_voxel_size = self.voxelcache[mix_item_idx].astype('float32')
            mix_img = self.get_offset(mix_total_idx, offset_list)
            mix_img = np.concatenate((mix_img, mix_target))

            if np.random.rand() > 0.5:
                mix_img, img = img, mix_img

            assert img.shape == (len(offset_list)*12+1,128, 128)
            assert img.shape == mix_img.shape
            img[:,:,64:] = mix_img[:,:,64:]
            
            voxel_size = (voxel_size + mix_voxel_size)/2.

        if self.transform is not None:
            img = self.transform(image=img)['image']
        target = img[imgch:imgch+1]
        img = img[:imgch]
#         self.timeindexing[2] += time.time() - tb  # Profiling

        # Resample
        if self.config.resample_slice is not None:
            img = resample_slice(img, voxel_size[:,:2], self.config.resample_slice.copy(), (256, 256))
            target = resample_slice(target, voxel_size[:,:2], self.config.resample_slice.copy(), (256, 256))
            voxel_size[0,:2] = self.config.resample_slice
        
        if self.config.addjaehee:
            if self.config.addtargets:
                assert img.shape[0] == len(offset_list) * 39
                posimg = np.where(offset_list == 0)[0][0] * 39
                pdm = img[posimg+1:posimg+2]
                sm = img[posimg+2:posimg+12]
            else:
                assert img.shape[0] == len(offset_list) * 35
                posimg = np.where(offset_list == 0)[0][0] * 35
                pdm = img[posimg+1:posimg+2]
                sm = img[posimg+2:posimg+12]
        else:
            assert img.shape[0] == len(offset_list) * 12
            posimg = np.where(offset_list == 0)[0][0] * 12
            pdm = img[posimg+1:posimg+2]
            sm = img[posimg+2:posimg+12]

        if type(target) == np.ndarray:
            target = np.concatenate((target, target*sm), axis=0)
        else:
            target = torch.cat((target, target*sm), axis=0)
            
        if self.config.addvoxelch:
            voxch = np.repeat(voxel_size[0, :, None], 128*128, axis=1).reshape((3, 128, 128))
            img = np.concatenate((img, voxch))
        
        is_pseudo = True if 'pseudo' in self.data_df.loc[item_idx, 'Id'] else False
        
        return img, (target, pdm, sm, voxel_size, total_idx//128, is_pseudo)
    
    def get3DImage(self, idx, offset_list=None):
        img = []
        for i in range(idx*128, (idx+1)*128):
            img.append(self.__getitem__(i, offset_list)[0][None])
        return np.concatenate(img)
