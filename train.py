import os, sys
if not os.path.exists('./dl.yml'):
    os.chdir('..')

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
from pathlib import Path
import psutil
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
from torch.utils.data import Dataset, DataLoader

import fastai
from fastai.basic_data import DataBunch
# from fastai.vision import Learner
from modules.nan_learner import NaNLearner as Learner
# from modules.blend_data_augmentation import Learner
from fastai.distributed import setup_distrib, num_distrib
from fastai.callbacks import SaveModelCallback, CSVLogger

from tqdm import tqdm
# from fastprogress.fastprogress import progress_bar as tqdm

from functools import partial

import models.model_list as model_list
from modules.ranger913A import Ranger
from modules.radam import RAdam
from modules.train_annealing import fit_with_annealing
import modules.swa as swa
from utils.dataloader import KBPDataset
from utils.dataloader2D import KBPDataset2D
# from utils.dataloader2DStack import KBPDataset2DStack
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from utils.default_dataloader import DefaultDataLoader
from utils.losses import KBPLoss
from utils.metrics import dose_score, dvh_score, pred_mean, target_mean
from utils.metrics2D import dose_score2D, dvh_score2D, pred_mean2D, target_mean2D, EvalBatchAccumulator
# from utils.misc import log_metrics, cosine_annealing_lr
from utils.callbacks import SaveBestModel
from utils.evaluation import EvaluateDose, make_submission
from utils.preprocessing2D import HorizontalFlip, RandomScale, RandomShift, RandomRotate, get_train_tfms
from utils.visualization import plot_batch

from albumentations import Compose, to_tuple

import wandb
from wandb.fastai import WandbCallback

def main(config, args):
    if torch.cuda.is_available():
        cudnn.benchmark = True
        print('Using CUDA')
    else:
        print('**** CUDA is not available ****')

    pprint.pprint(config)

    if args.exp is None:
        if not os.path.exists('./config/old_configs/'+config.exp_name):
            os.makedirs('./config/old_configs/'+config.exp_name)
        shutil.copy2('./config/config.py', './config/old_configs/{}/config.py'
                            .format(config.exp_name))
    
    if not os.path.exists('./model_weights/'+config.exp_name):
        os.makedirs('./model_weights/'+config.exp_name)
    if not os.path.exists('./logs/'+config.exp_name):
        os.makedirs('./logs/'+config.exp_name) 

    data_df = pd.read_csv(config.DATA_CSV_PATH)
    if os.path.exists('/content/data'):
        print('On Colab')
        data_df['Id'] = data_df['Id'].apply(lambda x: '/content' + x[1:])

    if config.dataclass is not None:
        data_df = data_df[data_df['Type(Full/Head/Unclean/Bad)'] == config.dataclass].reset_index(drop=True)
    split_train_mask = (data_df['Fold'] != 'Fold{}'.format(args.foldidx))
    train_df = data_df[split_train_mask & (data_df['Split'] == 'Train')].reset_index(drop=True)
    valid_df = data_df[(~split_train_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
    test_df = data_df[data_df['Split'] == 'Test'].reset_index(drop=True)
    maintest_df = data_df[data_df['Split'] == 'MainTest'].reset_index(drop=True)
    
    print("Training with valid fold: ", args.foldidx)
    print(valid_df.head())
    
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
    
    train_tfms = get_train_tfms(config)
    print(train_tfms)
    if config.debug and config.reduce_dataset:
        if config.pseudo_path is not None:
            train_df = pd.concat((train_df[:10], pseudo_df[:10])).reset_index(drop=True)
        else:
            train_df = train_df[:10]
        valid_df = valid_df[:10]
    
#     DatasetClass = KBPDataset2D if psutil.virtual_memory().total < 20e9 else KBPDataset2DStack
    DatasetClass = KBPDataset2D
    train_ds = DatasetClass(config, train_df, transform=train_tfms)
    valid_ds = DatasetClass(config, valid_df, valid=True)

    # valid_dl = DataLoader(valid_ds, batch_size=128, shuffle=False, num_workers=config.num_workers)

    criterion = KBPLoss(config)

    Net = getattr(model_list, config.model_name)

    net = Net(config=config).to(config.device)
    print(net)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters: ", count_parameters(net))
    
    if config.load_model_ckpt is not None:
        print('Loading model from {}'.format(config.load_model_ckpt.format(args.foldidx)))
        net.load_state_dict(torch.load(config.load_model_ckpt.format(args.foldidx))['model'])

    gpu = setup_distrib(config.gpu)
    opt = config.optimizer
    mom = config.mom
    alpha = config.alpha
    eps = config.eps

    if   opt=='adam': opt_func = partial(optim.Adam, betas=(mom,alpha), eps=eps, amsgrad=config.amsgrad)
    elif opt=='adamw': opt_func = partial(optim.AdamW, betas=(mom,alpha), eps=eps)
    elif opt=='radam': opt_func = partial(RAdam, betas=(mom,alpha), eps=eps, degenerated_to_sgd=config.radam_degenerated_to_sgd)
    elif opt=='sgd': opt_func = partial(optim.SGD, momentum=mom, nesterov=config.nesterov)
    elif opt=='ranger': opt_func = partial(Ranger,  betas=(mom,alpha), eps=eps)
    else:
        raise ValueError("Optimizer not recognized")
    print(opt_func)

    data = DataBunch.create(train_ds, valid_ds, bs=config.batch_size, num_workers=config.num_workers)

    # metrics = [dose_score, dvh_score, pred_mean, target_mean]
    metrics = [dose_score2D, dvh_score2D, pred_mean2D, target_mean2D]
    evalbatchaccum = EvalBatchAccumulator(config, target_bs=128, num_metrics=len(metrics))
    learn = (Learner(evalbatchaccum, data, net, wd=config.weight_decay, opt_func=opt_func,
             bn_wd=False, true_wd=True,
             loss_func = criterion,
             metrics=metrics,
             path='./model_weights/{}/'.format(config.exp_name))
            )
    if config.fp16:
        print('Training with mixed precision...')
        learn = learn.to_fp16(dynamic=True)
    else:
        print('Full precision training...')
    if gpu is None: learn.to_parallel()
    elif num_distrib()>1: learn.to_distributed(gpu)
    if config.mixup: learn = learn.mixup(alpha=config.mixup, stack_y=False)
    print("Learn path: ", learn.path)
    best_save_cb = SaveBestModel(learn, config, outfile='_fold{}'.format(args.foldidx))
    logger_cb = CSVLogger(learn)
    logger_cb.path = Path(str(logger_cb.path).replace('model_weights/', 'logs/').replace('.csv', '_fold{}.csv'.format(args.foldidx)))
    callbacks = [best_save_cb, logger_cb]

    if config.teachers is not None:
        package = 'config.old_configs.{}.config'.format(config.teachers)
        teacherconfig = importlib.import_module(package).config
        teachers = []
        for fold in range(5):
            teacher = getattr(model_list, teacherconfig.model_name)
            teacher = teacher(teacherconfig)
            model_ckpt = './model_weights/{}/models/best_dose_fold{}.pth'.format(teacherconfig.exp_name, fold)
            print("Loading teacher {} encoder from {}".format(fold, model_ckpt))
            teacher.load_state_dict(torch.load(model_ckpt)['model'])
            teacher.to(config.device)
            teacher.eval()
            for param in teacher.parameters():
                param.requires_grad = False
            teachers.append(teacher)
    else:
        teachers = None

    if config.wandb:
        wandb.init(project=config.wandb_project, name=config.exp_name)
        wandb_cb = WandbCallback(learn)
        callbacks.append(wandb_cb)

    print(learn.loss_func.config.loss_dict)
    print(learn.opt_func)
    print("Weight decay: ", learn.wd)

    learn.fit_one_cycle(config.epochs, config.lr, callbacks=callbacks, div_factor=config.div_factor, pct_start=config.pct_start,
                        final_div=config.final_div, teachers=teachers)

    best_str = "Best valid loss: {}, dose score: {}, dvh score: {}".format(best_save_cb.best_loss, best_save_cb.best_dose.item(), best_save_cb.best_dvh.item())
    print(best_str)
    f = open("./logs/{}/bestmetrics_fold{}.txt".format(config.exp_name, args.foldidx), "a")
    f.write(best_str)
    f.close()

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--exp', default=None, help='config file to use')
    parser.add_argument('-f', '--foldidx', default=0, type=int, help='fold to run')
    parser.add_argument('--loadj', action='store_true', help='load jaehee data in memory')
    args = parser.parse_args()
    
    from config.config import config
    if args.exp is not None:
        package = 'config.old_configs.{}.config'.format(args.exp)
        print(package)
        config = importlib.import_module(package).config
    
    if args.loadj:
        config.loadjaehee = True
    
    main(config, args)
