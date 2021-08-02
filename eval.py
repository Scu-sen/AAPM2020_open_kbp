import os, sys
if not os.path.exists('./dl.yml'):
    os.chdir('..')
    
import time
ts = time.time()

import shutil
import glob
import time
import importlib
import argparse
import json
from easydict import EasyDict
import copy
import pprint
from collections import namedtuple
import gc
from IPython.core.debugger import set_trace

import math
import numpy as np
import pandas as pd
pd.set_option('max_colwidth', None)
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

import fastai
from fastai.basic_data import DataBunch
from fastai.vision import Learner
# from modules.blend_data_augmentation import Learner
from fastai.distributed import setup_distrib, num_distrib
from fastai.callbacks import SaveModelCallback

# from tqdm.notebook import tqdm
from fastprogress.fastprogress import progress_bar as tqdm

from functools import partial

import models.model_list as model_list
from modules.ranger913A import Ranger
from modules.radam import RAdam
from modules.train_annealing import fit_with_annealing
import modules.swa as swa
from utils.dataloader import KBPDataset as KBPDataset3D
from utils.dataloader2D import KBPDataset2D
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.losses import KBPLoss
from utils.metrics import dose_score, dvh_score, pred_mean, target_mean
from utils.metrics2D import dose_score2D, dvh_score2D, pred_mean2D, target_mean2D
# from utils.misc import log_metrics, cosine_annealing_lr
from utils.callbacks import SaveBestModel, WandbCallback
# from utils.evaluation import EvaluateDose, make_submission
from utils.preprocessing2D import HorizontalFlip, RandomScale, RandomShift, RandomRotate
from utils.evaluation2D import EvaluateDose2D, eval3D, TTAflip, TTAadd, TTAalt
from utils.interpolate import Interpolate
from utils.general_functions import sparse_vector_function

from albumentations import Compose, to_tuple

from config.config import config

from utils.dataloader import KBPDataset as KBPDataset3D
from utils.dataloader2D import KBPDataset2D

if torch.cuda.is_available():
    cudnn.benchmark = True
    print('Using CUDA')
else:
    print('**** CUDA is not available ****')

import argparse


def save_pred_csvs(net, dl3D, dl2D, offset_lists, args, setname='test', fold=0):
    # Make Predictions
    dose_evaluator = EvaluateDose2D(config, net=net, data_loader=dl3D, dose_loader=dl2D, offset_lists=offset_lists, load_cache=not args.noloadc, store_cache=not args.nostorec, cache_dir='{}/{}/'.format(setname, args.metric), conv=False, evalbs=args.bs)
    
    if not args.nocsv:
        # Save in csv
        if 'maintest' in setname:
            SAVE_DIR = './subm/{}/{}_main_fold{}'.format(config.exp_name, config.exp_name, fold)
        elif 'test' in setname:
            SAVE_DIR = './subm/{}/{}_fold{}'.format(config.exp_name, config.exp_name, fold)
        elif 'localval' in setname:
            SAVE_DIR = './subm/{}/{}_localval_fold{}'.format(config.exp_name, config.exp_name, fold)
        else:
            raise ValueError

        if not os.path.exists(SAVE_DIR):
            os.makedirs(SAVE_DIR)
        with torch.no_grad():
            is_train = dl3D.dataset.training
            dl3D.dataset.training = False
            for i, (_, (possible_dose_mask, item)) in enumerate(tqdm(dl3D)):
                pat_id = item['patient_list'][0][0]
                dose_pred_gy = dose_evaluator.preds[i] # (1, 128, 128, 128)
                assert dose_pred_gy.shape == (1, 128, 128, 128), dose_pred_gy.shape
                dose_pred_gy = (dose_pred_gy*(dose_pred_gy>=0.)).astype('float64')
                dose_pred_gy = dose_pred_gy * possible_dose_mask.detach().cpu().numpy().astype('float64')
                dose_pred_gy = np.squeeze(dose_pred_gy)
                dose_to_save = sparse_vector_function(dose_pred_gy)
                dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                       columns=['data'])
                dose_df.to_csv('{}/{}.csv'.format(SAVE_DIR, pat_id))
            dl3D.dataset.training = is_train

        if not args.nozip:
            # Zip dose to submit
            save_path = shutil.make_archive('{}_{}'.format(SAVE_DIR, args.metric), 'zip', SAVE_DIR)
            print('Saved to: ', '/'.join(save_path.split('/')[-3:]))
        else:
            print('Saved to: ', '/'.join(SAVE_DIR.split('/')[-3:]))
        gc.collect()
        sys.exit()
    return dose_evaluator


def main_eval(config, args, net, data_df, fold):
    print('**********************************')

    print('Loading fold_{} validation split'.format(fold))

    valid_mask = (data_df['Fold'] == 'Fold{}'.format(fold))
    train_df = data_df[(~valid_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
    valid_df = data_df[(valid_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
    test_df = data_df[data_df['Split'] == 'Test'].reset_index(drop=True)
    maintest_df = data_df[data_df['Split'] == 'MainTest'].reset_index(drop=True)

    if net is not None:
        load_path = './model_weights/{}/models/best_{}_fold{}.pth'.format(config.exp_name, args.metric, fold)
        assert config.exp_name in load_path
        print('Loading model from {}'.format(load_path))
        stdict = torch.load(load_path)
        if 'model' in stdict.keys():
            net.load_state_dict(stdict['model'])
        else:
            net.load_state_dict(stdict)
        net.float()
        net.eval()
        print('')

    if not args.noval:
        valid_ds3D = KBPDataset3D(config, valid_df)
        valid_dl3D = DataLoader(valid_ds3D, batch_size=1, shuffle=False, num_workers=config.num_workers)
        if args.nodl:
            valid_dl2D = None
        else:
            valid_ds2D = KBPDataset2D(config, valid_df, valid=True)
            valid_dl2D = DataLoader(valid_ds2D, batch_size=1, shuffle=False, num_workers=config.num_workers)

        dose_evaluator = save_pred_csvs(net, valid_dl3D, valid_dl2D, offset_lists, args, setname='localval_fold{}'.format(fold), fold=fold)
        # print out scores if data was left for a hold out set
        dvh_sc, dose_sc = dose_evaluator.make_metrics()
        print('For this out-of-sample test:\n'
              '\tthe DVH score is {:.3f}\n '
              '\tthe dose score is {:.3f}'.format(dvh_sc, dose_sc))

    if args.test:
        test_ds3D = KBPDataset3D(config, test_df, training=False)
        test_dl3D = DataLoader(test_ds3D, batch_size=1, shuffle=False, num_workers=config.num_workers)
        if args.nodl:
            test_dl2D = None
        else:
            test_ds2D = KBPDataset2D(config, test_df, training=False, valid=True)
            test_dl2D = DataLoader(test_ds2D, batch_size=1, shuffle=False, num_workers=config.num_workers)
        save_pred_csvs(net, test_dl3D, test_dl2D, offset_lists, args, setname='test_fold{}'.format(fold), fold=fold)
        
    if args.maintest:
        test_ds3D = KBPDataset3D(config, maintest_df, training=False)
        test_dl3D = DataLoader(test_ds3D, batch_size=1, shuffle=False, num_workers=config.num_workers)
        if args.nodl:
            test_dl2D = None
        else:
            test_ds2D = KBPDataset2D(config, maintest_df, training=False, valid=True)
            test_dl2D = DataLoader(test_ds2D, batch_size=1, shuffle=False, num_workers=config.num_workers)
        save_pred_csvs(net, test_dl3D, test_dl2D, offset_lists, args, setname='maintest_fold{}'.format(fold), fold=fold)


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('exp', default=None, help='Experiment name for evaluation')
    parser.add_argument('--nodl', action='store_true', default=False, help='Do not load 2d data')
    parser.add_argument('--noval', action='store_true', default=False, help='Do not do validation')
    parser.add_argument('--test', action='store_true', default=False, help='Save test predictions')
    parser.add_argument('--maintest', action='store_true', default=False, help='Save main test predictions')
    parser.add_argument('--noloadc', action='store_true', default=False, help='Do not load cache')
    parser.add_argument('--nostorec', action='store_true', default=False, help='Do not store cache')
    parser.add_argument('--nonet', action='store_true', default=False, help='Do not define the network')
    parser.add_argument('-m', '--metric', default='dose', help='Save test predictions')
    parser.add_argument('-f', '--foldidx', default=0, help='Evaluation fold')
    parser.add_argument('--allfolds', action='store_true', default=False, help='Evaluate all folds')
    parser.add_argument('--nocsv', action='store_true', default=False, help='Do not make csvs')
    parser.add_argument('--nozip', action='store_true', default=False, help='Do not make a zip')
    parser.add_argument('--bs', type=int, default=128, help='Evaluation batch size')
    args = parser.parse_args()
    
    assert 128 % args.bs == 0
    
    if args.exp is not None:
        package = 'config.old_configs.{}.config'.format(args.exp)
        print(package)
        config = importlib.import_module(package).config

    print('Doing inference for {}'.format(config.exp_name))

    data_df = pd.read_csv(config.DATA_CSV_PATH)
    if config.dataclass is not None:
        data_df = data_df[data_df['Type(Full/Head/Unclean/Bad)'] == config.dataclass].reset_index(drop=True)

    if args.nonet:
        net = None
    else:
        Net = getattr(model_list, config.model_name)
        net = Net(config=config).to(config.device)
    
    offset_lists = [config.offset_list]
#     offset_lists = [np.arange(-3,4), [-4, -2, -1, 0, 1, 2, 4], [-4, -3, -1, 0, 1, 3, 4], [-4, -3, -2, 0, 2, 3, 4]]
#     offset_lists = [np.arange(-3,4), [-4, -2, -1, 0, 1, 2, 4], [-4, -3, -1, 0, 1, 3, 4], [-4, -3, -2, 0, 2, 3, 4], [-5, -2, -1, 0, 1, 2, 5], [-5, -3, -1, 0, 1, 3, 5], [-5, -3, -2, 0, 2, 3, 5]]

    if args.allfolds:
        folds = range(5)
    else:
        folds = [args.foldidx]
        
    for fold in folds:
        main_eval(config, args, net, data_df, fold)
        print('')

    print("Total time taken: {:.1f}s".format(time.time() - ts))
