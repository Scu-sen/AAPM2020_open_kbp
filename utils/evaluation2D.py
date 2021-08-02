import os, glob
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
# from tqdm.notebook import tqdm
from fastprogress.fastprogress import progress_bar as tqdm

from itertools import product as it_product
from utils.general_functions import sparse_vector_function
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_toolbelt.inference import functional as TTAF

from scipy.signal import convolve
from scipy import ndimage

from utils.dataloader2D import X_DIST, Y_DIST, resample

X_DIST = np.concatenate([X_DIST[None, :] for _ in range(128)])
Y_DIST = np.concatenate([Y_DIST[None, :] for _ in range(128)])

div_factor = 3.
gau_filter3 = np.array([[[1, div_factor, div_factor*2, div_factor*4, div_factor*2, div_factor, 1]]])
gau_filter3 /= gau_filter3.sum()

def make_offset_id(off):
    ids = []
    for o in off:
        if o < 0:
            ids.append('m{}'.format(-o))
        else:
            ids.append('{}'.format(o))
    return ''.join(ids)

class EvalDataset(Dataset):
    def __init__(self, ds, offset_list=None):
        self.ds = ds
        self.offset_list = offset_list
        
    def __len__(self):
        return len(self.ds)//128
    
    def __getitem__(self, idx):
        return self.ds.get3DImage(idx, self.offset_list)

def zoom(img, outshape):
    scale = np.array(outshape)/np.array(img.shape)
    out = ndimage.zoom(img, scale, order=0, mode='nearest')
    return out

def QuadrantTTA(net, img):
    '''
    img: (128, 12, 128, 128)
    '''    
    imgtl = zoom(img[:,:,:64,:64], img.shape)
    imgtr = zoom(img[:,:,64:,:64], img.shape)
    imgbl = zoom(img[:,:,:64,64:], img.shape)
    imgbr = zoom(img[:,:,64:,64:], img.shape)
    
    fullout = np.zeros_like(img[:,0:1])
    fullout[:,:,:64,:64] = zoom(net(torch.tensor(imgtl).cuda()).cpu().numpy(), (128, 1, 64, 64))
    fullout[:,:,64:,:64] = zoom(net(torch.tensor(imgtr).cuda()).cpu().numpy(), (128, 1, 64, 64))
    fullout[:,:,:64,64:] = zoom(net(torch.tensor(imgbl).cuda()).cpu().numpy(), (128, 1, 64, 64))
    fullout[:,:,64:,64:] = zoom(net(torch.tensor(imgbr).cuda()).cpu().numpy(), (128, 1, 64, 64))
    
    return fullout
    

def eval3D(net, img, axis):
    '''
    img: (128, 84, 128, 128)
    output: (1, 128, 128, 128)
    '''
    assert img.shape[0] == 128
    assert img.shape[2:] == (128, 128)
#     img = np.moveaxis(img, axis, 0).astype('float32')    # (128, 12, 128, 128)
    
#     imgleft3 = np.concatenate((np.zeros_like(img[0:3]), img))[:128]
#     imgleft2 = np.concatenate((np.zeros_like(img[0:2]), img))[:128]
#     imgleft = np.concatenate((np.zeros_like(img[0:1]), img))[:128]
#     imgright = np.concatenate((img, np.zeros_like(img[0:1])))[1:]
#     imgright2 = np.concatenate((img, np.zeros_like(img[0:2])))[2:]
#     imgright3 = np.concatenate((img, np.zeros_like(img[0:3])))[3:]
#     img = np.concatenate((X_DIST, Y_DIST, imgleft3, imgleft2, imgleft, img, imgright, imgright2, imgright3), axis=1)
#     img = np.concatenate((imgleft3, imgleft2, imgleft, img, imgright, imgright2, imgright3), axis=1)
    
    with torch.no_grad():
        output = net(img.cuda()).detach().cpu().numpy()
#         output = QuadrantTTA(net, img)
    return output

def TTAD4(net, img, axis, shift_by=2, getall=False, conv=True):
    assert img.shape[0] == 128
    assert img.shape[2:] == (128, 128)
    assert getall == False
    
    with torch.no_grad():
        img = img.cuda()
        out = net(img)

        for aug, deaug in zip([TTAF.torch_rot90, TTAF.torch_rot180, TTAF.torch_rot270], [TTAF.torch_rot270, TTAF.torch_rot180, TTAF.torch_rot90]):
            x = deaug(net(aug(img)))
            out += x

        img = TTAF.torch_transpose(img)

#         for aug, deaug in zip([TTAF.torch_rot270], [TTAF.torch_rot90]):
        for aug, deaug in zip([TTAF.torch_none, TTAF.torch_rot90, TTAF.torch_rot180, TTAF.torch_rot270], [TTAF.torch_none, TTAF.torch_rot270, TTAF.torch_rot180, TTAF.torch_rot90]):
            x = deaug(net(aug(img)))
            out += TTAF.torch_transpose(x)

        one_over_8 = float(1.0 / 8.0)
        out *= one_over_8
    return out.cpu()

def TTAvflip(net, img):
    '''
    img = (128, 108, 128, 128)
    output: (128, 1, 128, 128)
    '''
    assert img.shape[0] == 128
    assert img.shape[1:] == (108, 128, 128)
    with torch.no_grad():
        out = net(img)
        flipidxs = np.arange(img.shape[1])
        flipidxs = flipidxs[::-1]
        for i in range(0, img.shape[1], 12):
            flipidxs[i:i+12] = flipidxs[i:i+12][::-1]
        print(flipidxs)
        img = torch.tensor(img.cpu().numpy()[:, flipidxs]).cuda()
        out += net(img)
        out /= 2
    return out

def shiftup(img, by=1):
    orig_shape = img.shape
    img = torch.cat((img, torch.zeros_like(img[:,:,0:by,:])), axis=2)[:,:,by:]
    assert img.shape == orig_shape
    return img

def shiftdown(img, by=1):
    orig_shape = img.shape
    img = torch.cat((torch.zeros_like(img[:,:,0:by,:]), img), axis=2)[:,:,:128]
    assert img.shape == orig_shape
    return img

def shiftleft(img, by=1):
    orig_shape = img.shape
    img = torch.cat((img, torch.zeros_like(img[:,:,:,0:by])), axis=3)[:,:,:,by:]
    assert img.shape == orig_shape
    return img

def shiftright(img, by=1):
    orig_shape = img.shape
    img = torch.cat((torch.zeros_like(img[:,:,:,0:by]), img), axis=3)[:,:,:,:128]
    assert img.shape == orig_shape
    return img


def TTAflip(net, img, axis, shift_by=2, getall=False, conv=True, evalbs=128):
    '''
    img = (128, 84, 128, 128)
    output: (128, 1, 128, 128)
    '''
    assert img.shape[0] == 128
    assert img.shape[2:] == (128, 128)
    outcat = []
    with torch.no_grad():
        img = img.cuda()
        for i in range(0, 128, evalbs):
            out = net(img[i:i+evalbs])

#             by = 1
#             out += shiftdown(net(shiftup(img, by)), by)
#             out += shiftup(net(shiftdown(img, by)), by)
#             out += shiftleft(net(shiftright(img, by)), by)
#             out += shiftright(net(shiftleft(img, by)), by)
#             out /= 5.

#             out += shiftdown(shiftleft(net(shiftup(shiftright(img, by), by)), by), by)
#             out += shiftup(shiftleft(net(shiftdown(shiftright(img, by), by)), by), by)
#             out += shiftup(shiftright(net(shiftdown(shiftleft(img, by), by)), by), by)
#             out += shiftdown(shiftright(net(shiftup(shiftleft(img, by), by)), by), by)
#             out /= 9.

            flipidx = torch.arange(127, -1, -1)
            imgflip = img[i:i+evalbs, :, :, flipidx]
            outflip = net(imgflip)
#             outflip += shiftdown(net(shiftup(imgflip, by)), by)
#             outflip += shiftup(net(shiftdown(imgflip, by)), by)
#             outflip += shiftleft(net(shiftright(imgflip, by)), by)
#             outflip += shiftright(net(shiftleft(imgflip, by)), by)
#             outflip /= 5.

#             outflip += shiftdown(shiftleft(net(shiftup(shiftright(imgflip, by), by)), by), by)
#             outflip += shiftup(shiftleft(net(shiftdown(shiftright(imgflip, by), by)), by), by)
#             outflip += shiftup(shiftright(net(shiftdown(shiftleft(imgflip, by), by)), by), by)
#             outflip += shiftdown(shiftright(net(shiftup(shiftleft(imgflip, by), by)), by), by)
#             outflip /= 9.

            outflip = outflip[:, :, :, flipidx]
#             out = np.maximum(out, outflip) # Doing max improves DVH score
            out = (out + outflip)/2.
            outcat.append(out)
    out = torch.cat(outcat, axis=0)
    assert out.shape == (128, 1, 128, 128)
    return out.cpu()
    hm = 2./(1./out + 1./outflip)
    return hm.cpu()


def TTAadd(net, img, shift_by=2):
    with torch.no_grad():
        out = eval3D(net, img)
        div_tensor = np.ones_like(out)
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], shift_by, 128, 128)), img), axis=1)[:,:128,:,:])
        out[:,:128-shift_by,:,:] += outshift[:,shift_by:,:,:]
        div_tensor[:,:128-shift_by,:,:] += 1
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], shift_by, 128, 128))), axis=1)[:,shift_by:,:,:])
        out[:,shift_by:,:,:] += outshift[:,:128-shift_by,:,:]
        div_tensor[:,shift_by:,:,:] += 1
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], 128, shift_by, 128)), img), axis=2)[:,:,:128,:])
        out[:,:,:128-shift_by,:] += outshift[:,:,shift_by:,:]
        div_tensor[:,:,:128-shift_by,:] += 1
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], 128, shift_by, 128))), axis=2)[:,:,shift_by:,:])
        out[:,:,shift_by:,:] += outshift[:,:,:128-shift_by,:]
        div_tensor[:,:,shift_by:,:] += 1
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], 128, 128, shift_by)), img), axis=3)[:,:,:,:128])
        out[:,:,:,:128-shift_by] += outshift[:,:,:,shift_by:]
        div_tensor[:,:,:,:128-shift_by] += 1
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], 128, 128, shift_by))), axis=3)[:,:,:,shift_by:])
        out[:,:,:,shift_by:] += outshift[:,:,:,:128-shift_by]
        div_tensor[:,:,:,shift_by:] += 1
        
    return out/div_tensor

def TTAalt(net, img, shift_by=2):
    with torch.no_grad():
        out = eval3D(net, img)
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], shift_by, 128, 128)), img), axis=1)[:,:128,:,:])
        out = np.maximum(out, np.concatenate((outshift, np.zeros((1, shift_by, 128, 128))), axis=1)[:,shift_by:,:,:])
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], shift_by, 128, 128))), axis=1)[:,shift_by:,:,:])
        out = np.maximum(out, np.concatenate((np.zeros((1, shift_by, 128, 128)), outshift), axis=1)[:,:128,:,:])
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], 128, shift_by, 128)), img), axis=2)[:,:,:128,:])
        out = np.maximum(out, np.concatenate((outshift, np.zeros((1, 128, shift_by, 128))), axis=2)[:,:,shift_by:,:])
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], 128, shift_by, 128))), axis=2)[:,:,shift_by:,:])
        out = np.maximum(out, np.concatenate((np.zeros((1, 128, shift_by, 128)), outshift), axis=2)[:,:,:128,:])
        
        outshift = eval3D(net, np.concatenate((np.zeros((img.shape[0], 128, 128, shift_by)), img), axis=3)[:,:,:,:128])
        out = np.maximum(out, np.concatenate((outshift, np.zeros((1, 128, 128, shift_by))), axis=3)[:,:,:,shift_by:])
        
        outshift = eval3D(net, np.concatenate((img, np.zeros((img.shape[0], 128, 128, shift_by))), axis=3)[:,:,:,shift_by:])
        out = np.maximum(out, np.concatenate((np.zeros((1, 128, 128, shift_by)), outshift), axis=3)[:,:,:,:128])
        
    return out


class EvaluateDose2D:
    """Evaluate a full dose distribution against the reference dose on the OpenKBP competition metrics"""

    def __init__(self, config, net, data_loader, dose_loader=None, TTA_shift_by=4, offset_lists=[np.arange(-3,4)], conv=True, load_cache=True, store_cache=True, cache_dir='', evalbs=128):
        """
        Prepare the class for evaluating dose distributions
        :param data_loader: a data loader object that loads data from the reference dataset
        :param dose_loader: a data loader object that loads a dose tensor from any dataset (e.g., predictions)
        """
        # Initialize objects
        self.config = config
        self.data_loader = data_loader  # Loads data related to ground truth patient information
        self.dose_loader = dose_loader  # Loads the data for a benchmark dose
        self.TTA_shift_by = TTA_shift_by

        # Initialize objects for later
        self.patient_list = None
        self.roi_mask = None
        self.new_dose = None
        self.reference_dose = None
        self.voxel_size = None
        self.possible_dose_mask = None

        # Set metrics to be evaluated
        self.oar_eval_metrics = ['D_0.1_cc', 'mean']
        self.tar_eval_metrics = ['D_99', 'D_95', 'D_1']

        # Name metrics for data frame
        oar_metrics = list(it_product(self.oar_eval_metrics, self.data_loader.dataset.defdataset.rois['oars']))
        target_metrics = list(it_product(self.tar_eval_metrics, self.data_loader.dataset.defdataset.rois['targets']))

        # Make data frame to store dose metrics and the difference data frame
        self.metric_difference_df = pd.DataFrame(index=self.data_loader.dataset.defdataset.patient_id_list,
                                                 columns=[*oar_metrics, *target_metrics])
        self.reference_dose_metric_df = self.metric_difference_df.copy()
        self.new_dose_metric_df = self.metric_difference_df.copy()
                
        if net is not None:
            net.eval()
        with torch.no_grad():
            self.preds = [np.zeros((1, 128, 128, 128)) for i in range(len(data_loader))]
            for off in offset_lists:
                print("Offset list: ", off)
                off_id = make_offset_id(off)
                CACHE_PATH = './preds/{}/{}{}'.format(config.exp_name, cache_dir, off_id)
                if store_cache and not os.path.exists(CACHE_PATH):
                    os.makedirs(CACHE_PATH)
                
                loaded = False
                if load_cache:
                    if len(glob.glob(os.path.join(CACHE_PATH, '*.npy'))) != len(data_loader):
                        print(CACHE_PATH)
                        print('Not found sufficient files for loading offset {}, predicting...'.format(off_id))
                    else:
                        print('Loading offset {} from cache...'.format(off_id))
                        for i in range(len(data_loader)):
                            pat_id = os.path.basename(data_loader.dataset.data_df.loc[i, 'Id'])
                            curpred = np.load(os.path.join(CACHE_PATH, '{}.npy'.format(pat_id))).astype('float32')
                            self.preds[i] += curpred/len(offset_lists)
                        loaded = True
                if not loaded:
                    evalds = EvalDataset(dose_loader.dataset, off)
                    evaldl = DataLoader(evalds, batch_size=1, shuffle=False, num_workers=2)
                    print('Making predictions from network...')
                    for i, img in enumerate(tqdm(evaldl)):
                        curpred = TTAflip(net, img[0], axis=self.config.axis, shift_by=self.TTA_shift_by, conv=conv, evalbs=evalbs)
                        if type(curpred) == torch.Tensor:
                            curpred = curpred.numpy()
                        curpred = np.moveaxis(curpred, 0, config.axis)  # (1, 128, 128, 128)
                        if conv:
                            print('conv...')
                            curpred = convolve(curpred[0], gau_filter3, mode='same')[None, :]
                        
                        if config.resample is not None:
                            voxel_sz = evaldl.dataset.ds.originalvoxelsz[i]
                            resampled_sz = config.resample.copy()
                            resampled_sz[2] = voxel_sz[0,2]
                            curpred = resample(curpred, np.array(resampled_sz)[None], voxel_sz[0])
                        
                        if store_cache:
                            curpredhalf = curpred.astype('float16')
                            pat_id = os.path.basename(dose_loader.dataset.data_df.loc[i, 'Id'])
                            np.save(os.path.join(CACHE_PATH, '{}.npy'.format(pat_id)), curpredhalf)
                        self.preds[i] += curpred/len(offset_lists)
        print('Done inference! Making metrics...')

    def make_metrics(self, conv=True, get_dfs=False):
        """Calculate a table of
        :return: the DVH score and dose score for the "new_dose" relative to the "reference_dose"
        """
        num_batches = len(self.data_loader)
        dose_score_vec = np.zeros(num_batches)

        # Only make calculations if data_loader is not empty
        if num_batches == 0:
            print('No patient information was given to calculate metrics')
        else:
            # Change batch size to 1
            assert self.data_loader.batch_size == 1  # Loads data related to ground truth patient information
            if self.dose_loader is not None:
                assert self.dose_loader.batch_size == 1  # Loads data related to ground truth patient information

            for idx in tqdm(range(num_batches)):
                # Get roi masks for patient
                self.get_constant_patient_features(idx)
                # Get dose tensors for reference dose and evaluate criteria
                reference_dose = self.get_patient_dose_tensor(self.data_loader)
                if reference_dose is not None:
                    self.reference_dose_metric_df = self.calculate_metrics(self.reference_dose_metric_df, reference_dose)
                else:
                    raise
                # If a dose loader was provided, calculate the score
                if self.data_loader is not None:
                    new_dose = self.predict_patient_dose_tensor(self.data_loader)
                    # Make metric data frames
                    self.new_dose_metric_df = self.calculate_metrics(self.new_dose_metric_df, new_dose)
                    # Evaluate mean absolute error of 3D dose
                    dose_score_vec[idx] = np.sum(np.abs(reference_dose - new_dose*self.possible_dose_mask.flatten())) / np.sum(self.possible_dose_mask)
#                     print(dose_score_vec[idx])
                    # Save metrics at the patient level (this is a template for how DVH stream participants could save
                    # their files
                    # self.dose_metric_df.loc[self.patient_list[0]].to_csv('{}.csv'.format(self.patient_list[0]))
                else:
                    raise

            if self.data_loader is not None:
                dvh_score = np.nanmean(np.abs(self.reference_dose_metric_df - self.new_dose_metric_df).values)
                if get_dfs:
                    return dose_score_vec, self.reference_dose_metric_df, self.new_dose_metric_df
                dose_score = dose_score_vec.mean()
                return dvh_score, dose_score
            else:
                print('No new dose provided. Metrics were only calculated for the provided dose.')
                raise

    def get_patient_dose_tensor(self, data_loader, flatten=True):
        """Retrieves a flattened dose tensor from the input data_loader.
        :param data_loader: a data loader that load a dose distribution
        :return: a flattened dose tensor
        """
        # Load the dose for the request patient
        dose_batch = data_loader.dataset.defdataset.get_batch(patient_list=self.patient_list)
        dose_key = [key for key in dose_batch.keys() if 'dose' in key.lower()][0]  # The name of the dose
        dose_tensor = dose_batch[dose_key][0]  # Dose tensor
        if flatten:
            return dose_tensor.flatten()
        return dose_tensor
    
    def predict_patient_dose_tensor(self, data_loader, flatten=True):
        """Retrieves a flattened dose tensor from the input data_loader.
        :param data_loader: a data loader that load a dose distribution
        :return: a flattened dose tensor
        """
        # Load the dose for the request patient
#         dose_batch = data_loader.dataset.defdataset.get_batch(patient_list=self.patient_list)
        assert len(self.patient_list) == 1
        pat_idx = data_loader.dataset.defdataset.patient_to_index(self.patient_list)[0]
        dose_tensor = self.preds[pat_idx]    # Predicted dose tensor (1, 128, 128, 128)
        if flatten:
            return dose_tensor.flatten()
        return dose_tensor

    def get_constant_patient_features(self, idx):
        """Gets the roi tensor
        :param idx: the index for the batch to be loaded
        """
        # Load the batch of roi mask
        rois_batch = self.data_loader.dataset.defdataset.get_batch(idx)
        self.roi_mask = rois_batch['structure_masks'][0].astype(bool)
        # Save the patient list to keep track of the patient id
        self.patient_list = rois_batch['patient_list']
        # Get voxel size
        self.voxel_size = np.prod(rois_batch['voxel_dimensions'])
        # Get the possible dose mask
        self.possible_dose_mask = rois_batch['possible_dose_mask']


    def calculate_metrics(self, metric_df, dose):
        """
        Calculate the competition metrics
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        # Prepare to iterate through all rois
        roi_exists = self.roi_mask.max(axis=(0, 1, 2))
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/self.voxel_size))  #
        for roi_idx, roi in enumerate(self.data_loader.dataset.defdataset.full_roi_list):
            if roi_exists[roi_idx]:
                roi_mask = self.roi_mask[:, :, :, roi_idx].flatten()
                roi_dose = dose[roi_mask]
                roi_size = len(roi_dose)
                if roi in self.data_loader.dataset.defdataset.rois['oars']:
                    if 'D_0.1_cc' in self.oar_eval_metrics:
                        # Find the fractional volume in 0.1cc to evaluate percentile
                        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc/roi_size * 100
                        metric_eval = np.percentile(roi_dose, fractional_volume_to_evaluate)
                        metric_df.at[self.patient_list[0], ('D_0.1_cc', roi)] = metric_eval
                    if 'mean' in self.oar_eval_metrics:
                        metric_eval = roi_dose.mean()
                        metric_df.at[self.patient_list[0], ('mean', roi)] = metric_eval
                elif roi in self.data_loader.dataset.defdataset.rois['targets']:
                    if 'D_99' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 1)
                        metric_df.at[self.patient_list[0], ('D_99', roi)] = metric_eval
                    if 'D_95' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 5)
                        metric_df.at[self.patient_list[0], ('D_95', roi)] = metric_eval
                    if 'D_1' in self.tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 99)
                        metric_df.at[self.patient_list[0], ('D_1', roi)] = metric_eval

        return metric_df


# def make_submission2D(config, net, data_loader, save_dir):
#     assert data_loader.batch_size == 1
#     with torch.no_grad():
#         for img, (possible_dose_mask, item) in tqdm(data_loader):
#             # Get patient ID and make a prediction
#             pat_id = item['patient_list'][0][0]
#             img = img[0].detach().cpu().numpy()
# #             dose_pred_gy = net(img)
#             dose_pred_gy = TTAflip(net, img)
#             dose_pred_gy = (dose_pred_gy*(dose_pred_gy>=0.)).astype('float64')
#             dose_pred_gy = dose_pred_gy * possible_dose_mask.detach().cpu().numpy().astype('float64')
#             # Prepare the dose to save
#             dose_pred_gy = np.squeeze(dose_pred_gy)
#             dose_to_save = sparse_vector_function(dose_pred_gy)
#             dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
#                                    columns=['data'])
#             dose_df.to_csv('{}/{}.csv'.format(save_dir, pat_id))

