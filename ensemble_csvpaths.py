import os
import glob
import glob
import shutil
import importlib
import numpy as np
import pandas as pd
import argparse
from fastprogress.fastprogress import progress_bar as tqdm

from utils.general_functions import load_file, sparse_vector_function


def get_pat_list(path):
    patlist = glob.glob(path + '/*.csv')
    return sorted([p.split('/')[-1] for p in patlist])
    

if __name__=='__main__':
    import time
    ts = time.time()
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('paths', nargs='+', help='Experiment name for evaluation')
    parser.add_argument('--weights', nargs='+', default=None, help='weights to be assigned')
    parser.add_argument('--out', default=None, help='Experiment name for evaluation')
    parser.add_argument('--nozip', action='store_true', default=False, help='Do not make zip')
    args = parser.parse_args()
    
    assert type(args.paths) == list and len(args.paths) > 0
    args.paths = [p[:-1] if p[-1] == '/' else p for p in args.paths]
    
    if args.weights is None:
        args.weights = [1 for _ in range(len(args.paths))]
    else:
        assert len(args.weights) == len(args.paths)
        args.weights = [int(w) for w in args.weights]
        print("Args weights: {}".format(args.weights))
    
    pats = get_pat_list(args.paths[0])
    
    catpath = '_'.join([p.split('/')[-1] for p in args.paths])
    SAVE_DIR = './subm/{}/{}'.format(catpath, catpath)
    if args.out is not None:
        SAVE_DIR = args.out
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    for pt in tqdm(pats):
        pt_preds = []
        for fpath in args.paths:
            pt_fold = '{}/{}'.format(fpath, pt)
            pt_pred_csv = load_file(pt_fold)
            pt_pred = np.zeros((128, 128, 128, 1), dtype='float64')
            np.put(pt_pred, pt_pred_csv['indices'], pt_pred_csv['data'])
            pt_preds.append(pt_pred[None,:,:,:,0])
        pt_preds = np.concatenate(pt_preds)
        assert pt_preds.shape[1:] == (128, 128, 128), pt_preds.shape
        
#         Mean
#         pt_preds = pt_preds.mean(0, keepdims=True)
#         Weighted average
        pt_preds = (pt_preds*np.array(args.weights)[:,None,None,None]).sum(0, keepdims=True) / np.array(args.weights).sum()
        
        assert pt_preds.shape == (1, 128, 128, 128), pt_preds.shape
        
        pt_preds = (pt_preds*(pt_preds>=0.)).astype('float64')
        pt_preds = np.squeeze(pt_preds)
        dose_to_save = sparse_vector_function(pt_preds)
        dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                               columns=['data'])
        dose_df.to_csv('{}/{}'.format(SAVE_DIR, pt))
    
    # Zip dose to submit
    if not args.nozip:
        save_path = shutil.make_archive(SAVE_DIR, 'zip', SAVE_DIR)
        print('Saved to: ', '/'.join(save_path.split('/')[-3:]))
    else:
        print('Saved to: ', '/'.join(SAVE_DIR))
    
    print("Total time taken: {:.1f}s".format(time.time() - ts))
