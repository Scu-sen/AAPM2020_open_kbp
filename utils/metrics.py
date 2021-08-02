import numpy as np
import pandas as pd
import torch
from IPython.core.debugger import set_trace

def pred_mean(pred, target, possible_dose_mask, structure_masks, voxel_size, idx):
    return (pred*possible_dose_mask).sum()/possible_dose_mask.sum()

def target_mean(pred, target, possible_dose_mask, structure_masks, voxel_size, idx):
    return (target*possible_dose_mask).sum()/possible_dose_mask.sum()

def dose_score(pred, target, possible_dose_mask, structure_masks, voxel_size, idx):
    return (torch.abs((pred - target)*possible_dose_mask).sum(1).sum(1).sum(1).sum(1) / possible_dose_mask.sum(1).sum(1).sum(1).sum(1)).mean()

def dvh_score(pred, target, possible_dose_mask, structure_masks, voxel_size, idx):
    dvh_metrics = [('D_0.1_cc', 'Brainstem'), ('D_0.1_cc', 'SpinalCord'),('D_0.1_cc', 'RightParotid'),  ('D_0.1_cc', 'LeftParotid'),
          ('D_0.1_cc', 'Esophagus'), ('D_0.1_cc', 'Larynx'), ('D_0.1_cc', 'Mandible'), ('mean', 'Brainstem'), ('mean', 'SpinalCord'),
          ('mean', 'RightParotid'), ('mean', 'LeftParotid'), ('mean', 'Esophagus'), ('mean', 'Larynx'), ('mean', 'Mandible'),
          ('D_99', 'PTV56'), ('D_99', 'PTV63'), ('D_99', 'PTV70'), ('D_95', 'PTV56'), ('D_95', 'PTV63'), ('D_95', 'PTV70'),
          ('D_1', 'PTV56'), ('D_1', 'PTV63'), ('D_1', 'PTV70')]
    pred = pred.view(pred.shape[0], -1).detach().cpu().numpy()
    target = target.view(target.shape[0], -1).detach().cpu().numpy()
    possible_dose_mask = possible_dose_mask.detach().cpu().numpy()
    structure_masks = structure_masks.detach().cpu().numpy().astype('bool')
    voxel_size = np.prod(voxel_size.detach().cpu().numpy())
    
    def calculate_metrics(metric_df, dose, roi_mask_full, df_idx):
        """
        Calculate the competition metrics
        :param metric_df: A DataFrame with columns indexed by the metric name and the structure name
        :param dose: the dose to be evaluated
        :return: the same metric_df that is input, but now with the metrics for the provided dose
        """
        # Prepare to iterate through all rois
        roi_exists = roi_mask_full.max(axis=(1, 2, 3))
        voxels_in_tenth_of_cc = np.maximum(1, np.round(100/voxel_size))  #
        oar_eval_metrics = ['D_0.1_cc', 'mean']
        tar_eval_metrics = ['D_99', 'D_95', 'D_1']
        full_roi_list = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus', 'Larynx', 'Mandible', 'PTV56', 'PTV63', 'PTV70']
        rois_oars = ['Brainstem', 'SpinalCord', 'RightParotid', 'LeftParotid', 'Esophagus', 'Larynx', 'Mandible']
        rois_targets = ['PTV56', 'PTV63', 'PTV70']
        for roi_idx, roi in enumerate(full_roi_list):
            if roi_exists[roi_idx]:
                roi_mask = roi_mask_full[roi_idx, :, :, :].flatten()
                roi_dose = dose[roi_mask]
                roi_size = len(roi_dose)
                if roi in rois_oars:
                    if 'D_0.1_cc' in oar_eval_metrics:
                        # Find the fractional volume in 0.1cc to evaluate percentile
                        fractional_volume_to_evaluate = 100 - voxels_in_tenth_of_cc/roi_size * 100
                        metric_eval = np.percentile(roi_dose, fractional_volume_to_evaluate)
                        metric_df.at[df_idx, ('D_0.1_cc', roi)] = metric_eval
                    if 'mean' in oar_eval_metrics:
                        metric_eval = roi_dose.mean()
                        metric_df.at[df_idx, ('mean', roi)] = metric_eval
                elif roi in rois_targets:
                    if 'D_99' in tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 1)
                        metric_df.at[df_idx, ('D_99', roi)] = metric_eval
                    if 'D_95' in tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 5)
                        metric_df.at[df_idx, ('D_95', roi)] = metric_eval
                    if 'D_1' in tar_eval_metrics:
                        metric_eval = np.percentile(roi_dose, 99)
                        metric_df.at[df_idx, ('D_1', roi)] = metric_eval
        return metric_df
    
    pred_df, target_df = pd.DataFrame(columns=dvh_metrics), pd.DataFrame(columns=dvh_metrics)
    for df_idx in range(target.shape[0]):
        pred_df = calculate_metrics(pred_df, pred[df_idx], structure_masks[df_idx], df_idx)
        target_df = calculate_metrics(target_df, target[df_idx], structure_masks[df_idx], df_idx)
    return torch.tensor(np.nanmean(np.abs(pred_df - target_df).values))
