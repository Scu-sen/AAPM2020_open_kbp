import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from modules.ssim_loss import SSIMLoss
from IPython.core.debugger import set_trace

class KBPLoss(nn.Module):
    def __init__(self, config):
        super(KBPLoss, self).__init__()
        self.config = config
        self.loss_dict = copy.deepcopy(config.loss_dict)
        assert len(self.loss_dict) > 0
        self.lweights = []
        for k,v in self.loss_dict.items():
            assert 'weight' in v
            self.lweights.append(v['weight'])
            del v['weight']
        self.lweights = torch.tensor(self.lweights).cuda()

        self.losses = [globals()[k](**v) for k,v in self.loss_dict.items()]

    def forward(self, inputs, target, possible_dose_mask, structure_masks, voxel_size, idx, is_pseudo):
        loss = self.losses[0](inputs, target, possible_dose_mask, structure_masks, voxel_size)*self.lweights[0]
        for i in range(1, len(self.losses)):
            loss += self.losses[i](inputs, target, possible_dose_mask, structure_masks, voxel_size)*self.lweights[i]
        return loss

class DVHLoss(nn.Module):
    def __init__(self, sm_weight = 1.):
        super().__init__()
        self.sm_weight = sm_weight
        print("DVH Loss with sm weight: ", self.sm_weight)
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = (torch.abs(input-target[:,0:1])*possible_dose_mask).mean()
        dvhloss = torch.abs(input-target[:,1:])*structure_masks    # (bs, 10, 128, 128)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).unsqueeze(3)   # (bs, 1, 1, 1)
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss
    
class SmoothDVHLoss(nn.Module):
    def __init__(self, sm_weight = 1.):
        super().__init__()
        self.sm_weight = sm_weight
        self.sl1 = nn.SmoothL1Loss()
        self.sl2 = nn.SmoothL1Loss()
        print("Smooth DVH Loss with sm weight: ", self.sm_weight)
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.sl1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask)
        dvhloss = self.sl2(input*structure_masks, target[:,1:]*structure_masks)    # (1)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).unsqueeze(3)   # (bs, 1, 1, 1)
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss
    
class SmoothDVHLossMulVoxel(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False):
        super().__init__()
        self.sm_weight = sm_weight
        self.pdm_mul = pdm_mul
        self.dvh_sm_asym = dvh_sm_asym
        self.sl1 = nn.SmoothL1Loss(reduction='none')
        self.sl2 = nn.SmoothL1Loss(reduction='none')
        print("Smooth DVH Loss Mul Voxel with sm weight: {}, pdm mul: {}, dvh_sm_asym: {}".format(sm_weight, pdm_mul, dvh_sm_asym))
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.sl1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask).mean(1).mean(1).mean(1)    # (bs)
        
        sminput = input*structure_masks
        smtarget = target[:,1:]*structure_masks
        if self.dvh_sm_asym:
            sminput[:7] = F.relu(sminput[:7] - smtarget[:7]) + smtarget[:7]
            sminput[7:] = smtarget[7:] - F.relu(smtarget[7:] - sminput[7:])
        dvhloss = self.sl2(sminput, smtarget).mean(1).mean(1).mean(1)    # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        if self.pdm_mul:
            pdmloss = (pdmloss*vs).mean()
        else:
            pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss

class SmoothDVHLossMulVoxelThres(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False, diffzero=None):
        super().__init__()
        self.sm_weight = sm_weight
        self.pdm_mul = pdm_mul
        self.dvh_sm_asym = dvh_sm_asym
        self.diffzero = diffzero
        self.sl1 = nn.SmoothL1Loss(reduction='none')
        self.sl2 = nn.SmoothL1Loss(reduction='none')
        print("Smooth DVH Loss Mul Voxel Thres with sm weight: {}, pdm mul: {}, dvh_sm_asym: {}".format(sm_weight, pdm_mul, dvh_sm_asym))
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        diff = input - target[:,0:1]
        if self.diffzero is not None:
            diff = diff * (torch.abs(diff) > self.diffzero)
        pdmloss = self.sl1(diff*possible_dose_mask, target[:,0:1]*0).mean(1).mean(1).mean(1)    # (bs)
        
        sminput = input*structure_masks
        smtarget = target[:,1:]*structure_masks
        smdiff = sminput - smtarget
        if self.diffzero is not None:
            smdiff = smdiff * (torch.abs(smdiff) > self.diffzero)
        if self.dvh_sm_asym:
            sminput[:7] = F.relu(sminput[:7] - smtarget[:7]) + smtarget[:7]
            sminput[7:] = smtarget[7:] - F.relu(smtarget[7:] - sminput[7:])
        dvhloss = self.sl2(smdiff, smtarget*0).mean(1).mean(1).mean(1)    # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        if self.pdm_mul:
            pdmloss = (pdmloss*vs).mean()
        else:
            pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss

class SmoothDVHLossMulVoxelPDMWeighted(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False):
        super().__init__()
        self.sm_weight = sm_weight
        self.pdm_mul = pdm_mul
        self.dvh_sm_asym = dvh_sm_asym
        self.sl1 = nn.SmoothL1Loss(reduction='none')
        self.sl2 = nn.SmoothL1Loss(reduction='none')
        print("Smooth DVH Loss Mul Voxel with sm weight: {}, pdm mul: {}, dvh_sm_asym: {}".format(sm_weight, pdm_mul, dvh_sm_asym))
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.sl1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask).mean(1).mean(1).mean(1)    # (bs)
        
        sminput = input*structure_masks
        smtarget = target[:,1:]*structure_masks
        if self.dvh_sm_asym:
            sminput[:7] = F.relu(sminput[:7] - smtarget[:7]) + smtarget[:7]
            sminput[7:] = smtarget[7:] - F.relu(smtarget[7:] - sminput[7:])
        dvhloss = self.sl2(sminput, smtarget).mean(1).mean(1).mean(1)    # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        
        pdmweight = possible_dose_mask.mean(1).mean(1).mean(1) # (bs)
        pdmloss = pdmloss*pdmweight
        dvhloss = dvhloss*pdmweight
        
        if self.pdm_mul:
            pdmloss = (pdmloss*vs).mean()
        else:
            pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss
    
class SmoothDVHLossMulVoxelInversePDMWeighted(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False):
        super().__init__()
        self.sm_weight = sm_weight
        self.pdm_mul = pdm_mul
        self.dvh_sm_asym = dvh_sm_asym
        self.sl1 = nn.SmoothL1Loss(reduction='none')
        self.sl2 = nn.SmoothL1Loss(reduction='none')
        print("Smooth DVH Loss Mul Voxel with sm weight: {}, pdm mul: {}, dvh_sm_asym: {}".format(sm_weight, pdm_mul, dvh_sm_asym))
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.sl1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask).mean(1).mean(1).mean(1)    # (bs)
        
        sminput = input*structure_masks
        smtarget = target[:,1:]*structure_masks
        if self.dvh_sm_asym:
            sminput[:7] = F.relu(sminput[:7] - smtarget[:7]) + smtarget[:7]
            sminput[7:] = smtarget[7:] - F.relu(smtarget[7:] - sminput[7:])
        dvhloss = self.sl2(sminput, smtarget).mean(1).mean(1).mean(1)    # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        
        pdmweight = possible_dose_mask.mean(1).mean(1).mean(1) # (bs)
        pdmloss = pdmloss/(1 + pdmweight)
        dvhloss = dvhloss/(1 + pdmweight)
        
        if self.pdm_mul:
            pdmloss = (pdmloss*vs).mean()
        else:
            pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss

class L1Loss(nn.Module):
    def __init__(self, sm_weight = 1., pdm_mul=False, dvh_sm_asym=False):
        super().__init__()
        self.sm_weight = sm_weight
        self.pdm_mul = pdm_mul
        self.dvh_sm_asym = dvh_sm_asym
        self.sl1 = nn.L1Loss(reduction='none')
        self.sl2 = nn.L1Loss(reduction='none')
        print("L1 DVH Loss Mul Voxel with sm weight: {}, pdm mul: {}, dvh_sm_asym: {}".format(sm_weight, pdm_mul, dvh_sm_asym))
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.sl1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask).mean(1).mean(1).mean(1)    # (bs)
        
        sminput = input*structure_masks
        smtarget = target[:,1:]*structure_masks
        if self.dvh_sm_asym:
            sminput[:7] = F.relu(sminput[:7] - smtarget[:7]) + smtarget[:7]
            sminput[7:] = smtarget[7:] - F.relu(smtarget[7:] - sminput[7:])
        dvhloss = self.sl2(sminput, smtarget).mean(1).mean(1).mean(1)    # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        
        pdmweight = possible_dose_mask.mean(1).mean(1).mean(1) # (bs)
        pdmloss = pdmloss*pdmweight
        dvhloss = dvhloss*pdmweight
        
        if self.pdm_mul:
            pdmloss = (pdmloss*vs).mean()
        else:
            pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss

class SmoothL1Loss(nn.Module):
    # http://openaccess.thecvf.com/content_CVPR_2019/papers/Barron_A_General_and_Adaptive_Robust_Loss_Function_CVPR_2019_paper.pdf
    def __init__(self, c=1.):
        super().__init__()
        self.c = c
        print("Smooth L1 Loss")
        
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        diff = (input - target)*possible_dose_mask
        return torch.sqrt((diff/self.c)**2 + 1.).mean() - 1.

class L2Loss(nn.Module):
    def __init__(self, sm_weight = 1.):
        super().__init__()
        self.sm_weight = sm_weight
        self.mse1 = nn.MSELoss(reduction='none')
        self.mse2 = nn.MSELoss(reduction='none')
        print("L2 Loss")
    
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        pdmloss = self.mse1(input*possible_dose_mask, target[:,0:1]*possible_dose_mask).mean(1).mean(1).mean(1)    # (bs)
        dvhloss = self.mse2(input*structure_masks, target[:,1:]*structure_masks).mean(1).mean(1).mean(1)   # (bs)
        vs = torch.prod(voxel_size, dim=2, keepdims=True).squeeze()   # (bs)
        pdmloss = pdmloss.mean()
        dvhloss = (dvhloss*vs).mean()
        return pdmloss + self.sm_weight*dvhloss

class AsymLoss(nn.Module):
    def __init__(self, under_weight=1):
        super().__init__()
        print("Asym Loss with under_weight: ", under_weight)
        self.under_weight = torch.tensor(under_weight).cuda()
        
    def forward(self, input, target, possible_dose_mask, structure_masks, voxel_size):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        diff = (input - target)*possible_dose_mask
        return F.relu(diff).mean() + self.under_weight*F.relu(-diff).mean()
    
class EigenLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Eigen Loss")
        
    def forward(self, input, target, possible_dose_mask):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))
        
        input *= possible_dose_mask
        n_voxels = target.shape[1] * target.shape[2] * target.shape[3] * target.shape[4]
        input[input <= 0] = 0.00001
        target[target <= 0] = 0.00001
        d = torch.log(input) - torch.log(target)
        term_1 = torch.pow(d.view(-1, n_voxels), 2).mean(dim=1).sum()  # voxel wise mean, then batch sum
        term_2 = (torch.pow(d.view(-1, n_voxels).sum(dim=1), 2) / (2 * (n_voxels ** 2))).sum()
        return term_1 - 0.5*term_2

class FocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        print("Focal Loss with gamma = ", gamma)
        self.gamma = gamma
        
    def forward(self, input, target):
        if not (target.size() == input.size()):
            raise ValueError("Target size ({}) must be the same as input size ({})"
                             .format(target.size(), input.size()))

        max_val = (-input).clamp(min=0)
        loss = input - input * target + max_val + \
            ((-max_val).exp() + (-input - max_val).exp()).log()

        invprobs = F.logsigmoid(-input * (target * 2.0 - 1.0))
        loss = (invprobs * self.gamma).exp() * loss
        
        return loss.sum(dim=1).mean()

class FocalLoss2(nn.Module):
    def __init__(self, gamma=2, genus_weight=0.5):
        super().__init__()
        print("Focal Loss 2 Stream with gamma = ", gamma)
        self.fl1 = FocalLoss(gamma = gamma)
        self.fl2 = FocalLoss(gamma = gamma)
        assert genus_weight>=0 and genus_weight<=1
        self.genus_weight = genus_weight

    def forward(self, input, target_genus, target_species):
        loss_genus = self.fl1(input[0], target_genus)
        loss_species = self.fl2(input[1], target_species)
        total_loss = self.genus_weight*loss_genus + (1-self.genus_weight)*loss_species
        return total_loss

class F1Loss(nn.Module):
    def __init__(self):
        super().__init__()
        print("F1 Loss")

    def forward(self, input, target):
        tp = (target*input).sum(0)
        # tn = ((1-target)*(1-input)).sum(0)
        fp = ((1-target)*input).sum(0)
        fn = (target*(1-input)).sum(0)

        p = tp / (tp + fp + 1e-9)
        r = tp / (tp + fn + 1e-9)

        f1 = 2*p*r / (p+r+1e-9)
        f1[f1!=f1] = 0.
        return 1 - f1.mean()


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
        print("Dice Loss")

    def forward(self, input, target):
        input = torch.sigmoid(input)
        smooth = 1.

        iflat = input.view(-1)
        tflat = target.view(-1)
        intersection = (iflat * tflat).sum()
        
        return 1 - ((2.*intersection + smooth) / (iflat.sum() + tflat.sum() + smooth))


class FocalTverskyLoss(nn.Module):
    """
    https://github.com/nabsabraham/focal-tversky-unet/blob/master/losses.py
    Focal Tversky Loss. Tversky loss is a special case with gamma = 1
    """
    def __init__(self, alpha = 0.4, gamma = 0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        print("Focal Tversky Loss with alpha = ", alpha, ", gamma = ", gamma)

    def tversky(self, input, target):
        smooth = 1.
        input = torch.sigmoid(input)

        target_pos = target.view(-1)
        input_pos = input.view(-1)
        true_pos = (target_pos * input_pos).sum()
        false_neg = (target_pos * (1-input_pos)).sum()
        false_pos = ((1-target_pos)*input_pos).sum()
        return (true_pos + smooth)/(true_pos + self.alpha*false_neg + \
                        (1-self.alpha)*false_pos + smooth)

    def forward(self, input, target):
        pt_1 = self.tversky(input, target)
        return (1-pt_1).pow(self.gamma)
