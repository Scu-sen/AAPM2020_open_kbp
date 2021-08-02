import numpy as np
import pandas as pd
# from tqdm.notebook import tqdm
from fastprogress.fastprogress import progress_bar as tqdm

from itertools import product as it_product
from utils.general_functions import sparse_vector_function
from IPython.core.debugger import set_trace

import torch
import torch.nn as nn
import torch.nn.functional as F

from scipy.signal import convolve

from utils.dataloader2D import X_DIST, Y_DIST

X_DIST = np.concatenate([X_DIST[None, :] for _ in range(128)])
Y_DIST = np.concatenate([Y_DIST[None, :] for _ in range(128)])

div_factor = 3.
gau_filter3 = np.array([[[1, div_factor, div_factor*2, div_factor*4, div_factor*2, div_factor, 1]]])
gau_filter3 /= gau_filter3.sum()

def eval3D(net, img):
    '''
    img: (12, 128, 128, 128)
    output: (1, 128, 128, 128)
    '''
    netmain, netrec = net
    img = np.moveaxis(img, 3, 0).astype('float32')    # (128, 12, 128, 128)
    
    imgleft3 = np.concatenate((np.zeros_like(img[0:3]), img))[:128]
    imgleft2 = np.concatenate((np.zeros_like(img[0:2]), img))[:128]
    imgleft = np.concatenate((np.zeros_like(img[0:1]), img))[:128]
    imgright = np.concatenate((img, np.zeros_like(img[0:1])))[1:]
    imgright2 = np.concatenate((img, np.zeros_like(img[0:2])))[2:]
    imgright3 = np.concatenate((img, np.zeros_like(img[0:3])))[3:]
    imgcat = np.concatenate((X_DIST, Y_DIST, imgleft3, imgleft2, imgleft, img, imgright, imgright2, imgright3), axis=1)
    
    with torch.no_grad():
        imgcat = torch.tensor(imgcat).cuda()
        output = netmain(imgcat)
        
        output = eval3DSlicesIter(netrec, img, output, iters=1)
        
        output = output.detach().cpu().numpy()
    return np.moveaxis(output, 0, 3)

def eval3DSlices(net, img):
    '''
    img: (12, 128, 128, 128)
    output: (1, 128, 128, 128)
    '''
    img = np.moveaxis(img, 3, 0).astype('float32')    # (128, 12, 128, 128)
    
    imgleft2 = np.concatenate((np.zeros_like(img[0:2]), img))[:128]
    imgleft = np.concatenate((np.zeros_like(img[0:1]), img))[:128]
#     imgright = np.concatenate((img, np.zeros_like(img[0:1])))[1:]
#     imgright2 = np.concatenate((img, np.zeros_like(img[0:2])))[2:]
#     img = np.concatenate((imgleft2, imgleft, img, imgright, imgright2), axis=1)
    
    with torch.no_grad():
        imgleft2 = torch.tensor(imgleft2).cuda()
        imgleft = torch.tensor(imgleft).cuda()
        img = torch.tensor(img).cuda()  # (128, 12, 128, 128)
        predleft2 = torch.zeros_like(img[0:1, 0:1])  # (1, 1, 128, 128)
        predleft = torch.zeros_like(img[0:1, 0:1])  # (1, 1, 128, 128)
        
        output = torch.zeros_like(img[:,0:1])  # (128, 1, 128, 128)
        for i in range(128):
            imgslice = torch.cat((imgleft2[i:i+1], predleft2, imgleft[i:i+1], predleft, img[i:i+1]), dim=1)
            predleft2 = predleft.clone()
            predleft = net(imgslice)
            predleft = F.relu(predleft)*img[i:i+1,1:2]
            output[i] = predleft.clone()
        output = output.detach().cpu().numpy()
    return np.moveaxis(output, 0, 3)

def eval3DSlicesIter(net, img, output=None, iters=10):
    '''
    img: (12, 128, 128, 128)
    output: (1, 128, 128, 128)
    '''
#     img = np.moveaxis(img, 3, 0).astype('float32')    # (128, 12, 128, 128)
    
    imgleft2 = np.concatenate((np.zeros_like(img[0:2]), img))[:128]
    imgleft = np.concatenate((np.zeros_like(img[0:1]), img))[:128]
#     imgright = np.concatenate((img, np.zeros_like(img[0:1])))[1:]
#     imgright2 = np.concatenate((img, np.zeros_like(img[0:2])))[2:]
#     img = np.concatenate((imgleft2, imgleft, img, imgright, imgright2), axis=1)
    
    with torch.no_grad():
        imgleft2 = torch.tensor(imgleft2).cuda()
        imgleft = torch.tensor(imgleft).cuda()
        img = torch.tensor(img).cuda()  # (128, 12, 128, 128)
        if output is None:
            output = torch.zeros_like(img[:, 0:1])  # (128, 1, 128, 128)
        
        for _ in range(iters):
            outleft2 = torch.cat((torch.zeros_like(output[0:2]), output))[:128]
            outleft = torch.cat((torch.zeros_like(output[0:1]), output))[:128]
            imgi = torch.cat((imgleft2, outleft2, imgleft, outleft, img), dim=1)
            output = net(imgi)
            output = F.relu(output)*img[:,1:2]
#         output = output.detach().cpu().numpy()
#     return np.moveaxis(output, 0, 3)
    return output

def TTAflip(net, img, shift_by=2, getall=False, conv=True):
    '''
    img = (12, 128, 128, 128)
    output: (1, 128, 128, 128)
    '''
    assert img.shape == (12, 128, 128, 128)
    with torch.no_grad():
        out = eval3D(net, img)
#         out = eval3DSlicesIter(net, img)
        imgflip = np.ascontiguousarray(img[:, :, ::-1, :])
        outflip = eval3D(net, imgflip)
#         outflip = eval3DSlicesIter(net, imgflip)
        outflip = np.ascontiguousarray(outflip[:, :, ::-1, :])
#     return np.maximum(out, outflip) # Doing max improves DVH score
    if getall:
        return out, outflip
    return (out + outflip)/2.
    hm = 2./(1./out + 1./outflip)
    if conv:
        hm = convolve(hm[0], gau_filter3, mode='same')[None, :]
    return hm


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

    def __init__(self, config, data_loader, dose_loader=None, TTA_shift_by=4):
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

    def make_metrics(self, net, conv=True):
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
                if self.dose_loader is not None:
                    new_dose = self.predict_patient_dose_tensor(self.dose_loader, net, conv=conv)
                    # Make metric data frames
                    self.new_dose_metric_df = self.calculate_metrics(self.new_dose_metric_df, new_dose)
                    # Evaluate mean absolute error of 3D dose
                    dose_score_vec[idx] = np.sum(np.abs(reference_dose - new_dose*self.possible_dose_mask.flatten())) / np.sum(self.possible_dose_mask)
                    # Save metrics at the patient level (this is a template for how DVH stream participants could save
                    # their files
                    # self.dose_metric_df.loc[self.patient_list[0]].to_csv('{}.csv'.format(self.patient_list[0]))
                else:
                    raise

            if self.dose_loader is not None:
                dvh_score = np.nanmean(np.abs(self.reference_dose_metric_df - self.new_dose_metric_df).values)
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
    
    def predict_patient_dose_tensor(self, data_loader, net, flatten=True, conv=True):
        """Retrieves a flattened dose tensor from the input data_loader.
        :param data_loader: a data loader that load a dose distribution
        :return: a flattened dose tensor
        """
        # Load the dose for the request patient
        dose_batch = data_loader.dataset.defdataset.get_batch(patient_list=self.patient_list)
        with torch.no_grad():
#             Only ct
#             img = torch.tensor(dose_batch['ct'].astype('float32')).to(self.config.device)
#             12 Channel input
            img = np.concatenate((dose_batch['ct'][:,:,:,:,0], dose_batch['possible_dose_mask'][:,:,:,:,0],
                                  np.moveaxis(dose_batch['structure_masks'][0], 3, 0)), axis=0)    # (12, 128, 128, 128)
#             dose_tensor = net(img.squeeze().unsqueeze(0).unsqueeze(0)).squeeze(0).detach().cpu().numpy()  # Predicted dose tensor
            dose_tensor = TTAflip(net, img, shift_by=self.TTA_shift_by, conv=conv)    # Predicted dose tensor
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


def make_submission2D(config, net, data_loader, save_dir):
    assert data_loader.batch_size == 1
    with torch.no_grad():
        for img, (possible_dose_mask, item) in tqdm(data_loader):
            # Get patient ID and make a prediction
            pat_id = item['patient_list'][0][0]
            img = img[0].detach().cpu().numpy()
#             dose_pred_gy = net(img)
            dose_pred_gy = TTAflip(net, img)
            dose_pred_gy = (dose_pred_gy*(dose_pred_gy>=0.)).astype('float64')
            dose_pred_gy = dose_pred_gy * possible_dose_mask.detach().cpu().numpy().astype('float64')
            # Prepare the dose to save
            dose_pred_gy = np.squeeze(dose_pred_gy)
            dose_to_save = sparse_vector_function(dose_pred_gy)
            dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                   columns=['data'])
            dose_df.to_csv('{}/{}.csv'.format(save_dir, pat_id))

