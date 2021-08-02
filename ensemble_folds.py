import os
import glob
import shutil
import importlib
import numpy as np
import pandas as pd
import argparse
from fastprogress.fastprogress import progress_bar as tqdm

from utils.general_functions import load_file, sparse_vector_function


if __name__=='__main__':
    import time
    ts = time.time()
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('exp', default=None, help='Experiment name for evaluation')
    parser.add_argument('-m', '--metric', default='dose', help='Save test predictions')
    parser.add_argument('--maintest', action='store_true', default=False, help='Evaluate on the maintest dataset')
    args = parser.parse_args()
    
    assert args.exp is not None
    
    if args.maintest:
        pats = ['pt_{}'.format(i) for i in range(241, 341)]
        SAVE_DIR = './subm/{}/{}_main_ensemble'.format(args.exp, args.exp)
    else:
        pats = ['pt_{}'.format(i) for i in range(201, 241)]
        SAVE_DIR = './subm/{}/{}_ensemble'.format(args.exp, args.exp)
    
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)
    
    for pt in tqdm(pats):
        pt_preds = []
        for fold in range(5):
            if args.maintest:
                pt_fold = './subm/{}/{}_main_fold{}/{}.csv'.format(args.exp, args.exp, fold, pt)
            else:
                pt_fold = './subm/{}/{}_fold{}/{}.csv'.format(args.exp, args.exp, fold, pt)
            pt_pred_csv = load_file(pt_fold)
            pt_pred = np.zeros((128, 128, 128, 1), dtype='float64')
            np.put(pt_pred, pt_pred_csv['indices'], pt_pred_csv['data'])
            pt_preds.append(pt_pred[None,:,:,:,0])
        pt_preds = np.concatenate(pt_preds)
        assert pt_preds.shape == (5, 128, 128, 128), pt_preds.shape
        
        # Mean
        pt_preds = pt_preds.mean(0, keepdims=True)
        
        assert pt_preds.shape == (1, 128, 128, 128), pt_preds.shape
        
        pt_preds = (pt_preds*(pt_preds>=0.)).astype('float64')
#         np.save(pt, pt_preds)
#         continue
        pt_preds = np.squeeze(pt_preds)
        dose_to_save = sparse_vector_function(pt_preds)
        dose_df = pd.DataFrame(data=dose_to_save['data'].squeeze(), index=dose_to_save['indices'].squeeze(),
                               columns=['data'])
        dose_df.to_csv('{}/{}.csv'.format(SAVE_DIR, pt))
    
    # Zip dose to submit
    save_path = shutil.make_archive('{}_{}'.format(SAVE_DIR, args.metric), 'zip', SAVE_DIR)
    print('Saved to: ', '/'.join(save_path.split('/')[-3:]))
    
    print("Total time taken: {:.1f}s".format(time.time() - ts))
