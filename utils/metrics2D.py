import numpy as np
import pandas as pd
import torch
from IPython.core.debugger import set_trace

class EvalBatchAccumulator:
    def __init__(self, config, target_bs=128, num_metrics=4):
        assert target_bs % config.batch_size == 0
        self.target_bs = target_bs
        self.num_metrics = num_metrics
        self.reset()
        
    def reset(self):
        self.preds, self.targets, self.pdms, self.sms = [], [], [], []
        self.accum_bs = 0
        self.call_recorder = -1

    def __call__(self, pred, target, possible_dose_mask, structure_masks, voxel_size, idx):
        self.call_recorder += 1
        if self.call_recorder%self.num_metrics == 0:
            self.preds.append(pred.cpu())
            self.targets.append(target.cpu())
            self.pdms.append(possible_dose_mask.cpu())
            self.sms.append(structure_masks.cpu())
            self.accum_bs += pred.shape[0]
        if self.accum_bs == self.target_bs:
            assert self.call_recorder // self.num_metrics == (self.target_bs/pred.shape[0]) - 1
            if self.call_recorder % self.num_metrics == 0:
                self.preds = torch.cat(self.preds)
                self.targets = torch.cat(self.targets)
                self.pdms = torch.cat(self.pdms)
                self.sms = torch.cat(self.sms)
            preds, targets, pdms, sms = self.preds, self.targets, self.pdms, self.sms
            idx = torch.zeros_like(idx)
            if self.call_recorder % self.num_metrics == self.num_metrics - 1:
                self.reset()
        else:
            preds, targets, pdms, sms, voxel_size, idx = None, None, None, None, None, None
        return preds, targets, pdms, sms, voxel_size, idx

def pred_mean2D(pred, target, possible_dose_mask, structure_masks, voxel_size, idx, is_pseudo, evalbatchaccum):
    pred, target, possible_dose_mask, structure_masks, voxel_size, idx = evalbatchaccum(pred, target, possible_dose_mask, structure_masks, voxel_size, idx)
    if pred is None:
        return torch.tensor(np.NaN)
    
    assert pred.shape[0] == evalbatchaccum.target_bs
    assert target.shape[0] == evalbatchaccum.target_bs
    pdmsum = possible_dose_mask.sum()
    if pdmsum == 0:
        return pdmsum
    return (pred*possible_dose_mask).sum()/pdmsum

def target_mean2D(pred, target, possible_dose_mask, structure_masks, voxel_size, idx, is_pseudo, evalbatchaccum):
    pred, target, possible_dose_mask, structure_masks, voxel_size, idx = evalbatchaccum(pred, target, possible_dose_mask, structure_masks, voxel_size, idx)
    if pred is None:
        return torch.tensor(np.NaN)
    
    assert pred.shape[0] == evalbatchaccum.target_bs
    assert target.shape[0] == evalbatchaccum.target_bs
    pdmsum = possible_dose_mask.sum()
    if pdmsum == 0:
        return pdmsum
    return (target[:,0:1]*possible_dose_mask).sum()/pdmsum

def dose_score2D(pred, target, possible_dose_mask, structure_masks, voxel_size, idx, is_pseudo, evalbatchaccum):
    pred, target, possible_dose_mask, structure_masks, voxel_size, idx = evalbatchaccum(pred, target, possible_dose_mask, structure_masks, voxel_size, idx)
    if pred is None:
        return torch.tensor(np.NaN)

    assert pred.shape[0] == evalbatchaccum.target_bs
    assert target.shape[0] == evalbatchaccum.target_bs
    pdms = possible_dose_mask.sum(1).sum(1).sum(1)
    if (pdms.sum() == 0):
        return pdms.sum()
    diffsum = torch.abs((pred - target[:,0:1])*possible_dose_mask).sum(1).sum(1).sum(1)
    diffsum = diffsum[pdms != 0]
    pdms = pdms[pdms != 0]
    return ( diffsum / pdms ).mean()

def dvh_score2D(pred, target, possible_dose_mask, structure_masks, voxel_size, idx, is_pseudo, evalbatchaccum):
    pred, target, possible_dose_mask, structure_masks, voxel_size, idx = evalbatchaccum(pred, target, possible_dose_mask, structure_masks, voxel_size, idx)
    if pred is None:
        return torch.tensor(np.NaN)

    assert pred.shape[0] == evalbatchaccum.target_bs
    assert target.shape[0] == evalbatchaccum.target_bs
    dvh_metrics = [('D_0.1_cc', 'Brainstem'), ('D_0.1_cc', 'SpinalCord'),('D_0.1_cc', 'RightParotid'),  ('D_0.1_cc', 'LeftParotid'),
          ('D_0.1_cc', 'Esophagus'), ('D_0.1_cc', 'Larynx'), ('D_0.1_cc', 'Mandible'), ('mean', 'Brainstem'), ('mean', 'SpinalCord'),
          ('mean', 'RightParotid'), ('mean', 'LeftParotid'), ('mean', 'Esophagus'), ('mean', 'Larynx'), ('mean', 'Mandible'),
          ('D_99', 'PTV56'), ('D_99', 'PTV63'), ('D_99', 'PTV70'), ('D_95', 'PTV56'), ('D_95', 'PTV63'), ('D_95', 'PTV70'),
          ('D_1', 'PTV56'), ('D_1', 'PTV63'), ('D_1', 'PTV70')]
    pred = pred.view(1, -1).detach().cpu().numpy()
    target = target[:,0:1].contiguous().view(1, -1).detach().cpu().numpy()
#     possible_dose_mask = possible_dose_mask.detach().cpu().numpy()
    structure_masks = structure_masks.detach().cpu().numpy().astype('bool')
    voxel_size = np.prod(voxel_size.detach().cpu().numpy())
    
#     pred = np.moveaxis(pred, 1, 0)[None, ...]
#     target = np.moveaxis(target, 1, 0)[None, ...]
#     possible_dose_mask = np.moveaxis(possible_dose_mask, 1, 0)[None, ...]
    structure_masks = np.moveaxis(structure_masks, 1, 0)[None, ...]
#     set_trace()
    
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
    try:
        dvh_score = np.nanmean(np.abs(pred_df - target_df).values)
    except:
        dvh_score = np.NaN
    return torch.tensor(dvh_score)
