import os
import glob
import shutil
import importlib
import numpy as np
import pandas as pd
import argparse
from fastprogress.fastprogress import progress_bar as tqdm
from torch.utils.data import DataLoader

from utils.dataloader import KBPDataset as KBPDataset3D
from utils.dataloader2D import KBPDataset2D
from utils.evaluation2D import EvaluateDose2D
from utils.general_functions import load_file, sparse_vector_function

def get_eval_exp(config, dl, fpath='_valfolds_0_1', setname='localval'):
    return EvaluateDose2D(config, net=net, data_loader=dl, dose_loader=None, offset_lists=[config.offset_list], load_cache=True, store_cache=False, cache_dir='{}{}/{}/'.format(setname, fpath, args.metric))


if __name__=='__main__':
    import time
    ts = time.time()
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('exps', nargs='+', help='Experiment names for ensembling')
    parser.add_argument('-m', '--metric', default='dose', help='which metric to evaluate')
    parser.add_argument('--noval', action='store_true', default=False, help='Do not do validation')
    parser.add_argument('--test', action='store_true', default=False, help='Save test predictions')
    parser.add_argument('--maintest', action='store_true', default=False, help='Evaluate on the maintest dataset')
    args = parser.parse_args()

    assert type(args.exps) == list and len(args.exps) > 0
    args.exps = sorted(args.exps)
    configs = []
    for exp in args.exps:
        package = 'config.old_configs.{}.config'.format(exp)
        print(package)
        configs.append(importlib.import_module(package).config)

    combexpname = '_'.join(args.exps)
    SAVE_DIR = './subm/{}/{}_ensemble'.format(combexpname, combexpname)
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    data_df = pd.read_csv(configs[0].DATA_CSV_PATH)
    if configs[0].dataclass is not None:
        data_df = data_df[data_df['Type(Full/Head/Unclean/Bad)'] == configs[0].dataclass].reset_index(drop=True)

    net = None
    fpath = ''
        
    if not args.noval:
        fold_idx = [int(i) for i in fpath.split('_')[2:]] if fpath != '' else [0, 1]
        print('Loading {} {} validation split'.format(configs[0].exp_name, fold_idx))
        valid_mask = (data_df['Fold'] == 'Fold{}'.format(fold_idx[0]))
        for fidx in fold_idx:
            valid_mask |= (data_df['Fold'] == 'Fold{}'.format(fidx))

        valid_df = data_df[(valid_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
        valid_ds3D = KBPDataset3D(configs[0], valid_df)
        valid_dl3D = DataLoader(valid_ds3D, batch_size=1, shuffle=False, num_workers=configs[0].num_workers)
    
        dose_evaluators = [get_eval_exp(config, valid_dl3D, fpath, setname='localval') for config in configs]

        for pt in range(len(dose_evaluators[0].preds)):
            for exp in range(1, len(dose_evaluators)):
                dose_evaluators[0].preds[pt] += dose_evaluators[exp].preds[pt]
            dose_evaluators[0].preds[pt] /= len(dose_evaluators)

        dvh_sc, dose_sc = dose_evaluators[0].make_metrics()
        print('For this out-of-sample test:\n'
                  '\tthe DVH score is {:.3f}\n '
                  '\tthe dose score is {:.3f}'.format(dvh_sc, dose_sc))
    
    if args.test:
        test_df = data_df[data_df['Split'] == 'Test'].reset_index(drop=True)
        test_ds3D = KBPDataset3D(configs[0], test_df, training=False)
        test_dl3D = DataLoader(test_ds3D, batch_size=1, shuffle=False, num_workers=configs[0].num_workers)
        
        dose_evaluators = [get_eval_exp(config, test_dl3D, fpath, setname='test') for config in configs]
        
        for pt in range(len(dose_evaluators[0].preds)):
            for exp in range(1, len(dose_evaluators)):
                dose_evaluators[0].preds[pt] += dose_evaluators[exp].preds[pt]
            dose_evaluators[0].preds[pt] /= len(dose_evaluators)
        
        for i, (_, (possible_dose_mask, item)) in enumerate(tqdm(test_dl3D)):
            pat_id = item['patient_list'][0][0]
            dose_pred_gy = dose_evaluators[0].preds[i] # (1, 128, 128, 128)
            assert dose_pred_gy.shape == (1, 128, 128, 128), dose_pred_gy.shape
            dose_pred_gy = (dose_pred_gy*(dose_pred_gy>=0.)).astype('float64')
            dose_pred_gy = dose_pred_gy * possible_dose_mask.detach().cpu().numpy().astype('float64')
            dose_pred_gy = np.squeeze(dose_pred_gy)
            dose_to_save = sparse_vector_function(dose_pred_gy)
            dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                                   columns=['data'])
            dose_df.to_csv('{}/{}.csv'.format(SAVE_DIR, pat_id))

        # Zip dose to submit
        save_path = shutil.make_archive('{}_{}'.format(SAVE_DIR, args.metric), 'zip', SAVE_DIR)
        print('Saved to: ', '/'.join(save_path.split('/')[-3:]))

    print("Total time taken: {:.1f}s".format(time.time() - ts))
