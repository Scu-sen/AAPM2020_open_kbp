import os
import psutil
import shutil
import time
import numpy as np
import pandas as pd
from config.config import config
from config.train_options import TrainOptions
from utils.gancer_dataloader import CreateDataLoader
from models.gancer_models import create_model
from utils.gancer_visualizer import Visualizer
from utils.dataloader2D import KBPDataset2D
from utils.losses import KBPLoss
from utils.metrics2D import dose_score2D, dvh_score2D, pred_mean2D, target_mean2D, EvalBatchAccumulator
from fastprogress.fastprogress import progress_bar as tqdm
import torch
import pudb

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

opt = TrainOptions().parse()
opt.config = config
opt.input_nc = opt.config.in_channels
opt.batchSize = opt.config.batch_size
opt.which_model_netG = opt.config.model_name
opt.config.nloadjaehee = opt.nloadj

import getpass
if getpass.getuser() == 'kagglep100' or opt.loadj:
    opt.config.loadjaehee = True

model = create_model(opt)

if opt.resume:
    print("Resuming...")
    model.netG.load_state_dict(torch.load('./model_weights/{}/models/best_dose_fold{}.pth'.format(config.exp_name, opt.foldidx))['model'])

data_loader = CreateDataLoader(opt)  # inits CustomDatasetDataLoader
dataset = data_loader.load_data()  # returns CustomDatasetDataLoader obj
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

# visualizer = Visualizer(opt)
total_steps = 0

if not os.path.exists('./config/old_configs/'+config.exp_name):
    os.makedirs('./config/old_configs/'+config.exp_name)
shutil.copy2('./config/config.py', './config/old_configs/{}/config.py'
                    .format(config.exp_name))

data_df = pd.read_csv(config.DATA_CSV_PATH)
split_train_mask = (data_df['Fold'] != 'Fold{}'.format(opt.foldidx))
valid_df = data_df[(~split_train_mask) & (data_df['Split'] == 'Train')].reset_index(drop=True)
if psutil.virtual_memory().total < 35e9:
    opt.config.loadjaehee = False
valid_ds = KBPDataset2D(opt.config, valid_df, valid=True)
valid_dl = torch.utils.data.DataLoader(valid_ds, batch_size=opt.batchSize, shuffle=False, num_workers=int(opt.nThreads))
loss_func = KBPLoss(config)
history_df = []
if not os.path.exists('./model_weights/{}/models'.format(config.exp_name)):
    os.makedirs('./model_weights/{}/models'.format(config.exp_name))
if not os.path.exists('./logs/{}'.format(config.exp_name)):
    os.makedirs('./logs/{}'.format(config.exp_name))
best_loss, best_dose, best_dvh = np.inf, np.inf, np.inf

evalbatchaccum = EvalBatchAccumulator(config, target_bs=128, num_metrics=4)

for epoch in range(opt.epoch_count, opt.niter + opt.niter_decay + 1):
    epoch_start_time = time.time()
    iter_data_time = time.time()
    epoch_iter = 0

    for i, data in enumerate(tqdm(dataset, total=len(dataset.dataloader))):
        iter_start_time = time.time()
        if total_steps % opt.print_freq == 0:
            t_data = iter_start_time - iter_data_time
#         visualizer.reset()
        total_steps += opt.batchSize
        epoch_iter += opt.batchSize
        model.set_input(data)
        model.optimize_parameters()
#         if total_steps % opt.display_freq == 0:
#             save_result = total_steps % opt.update_html_freq == 0
#             visualizer.display_current_results(model.get_current_visuals(),
#                                                epoch, save_result)

#         if total_steps % opt.print_freq == 0:
#             errors = model.get_current_errors()
#             t = (time.time() - iter_start_time) / opt.batchSize
#             visualizer.print_current_errors(epoch, epoch_iter, errors, t,
#                                             t_data)
#             if opt.display_id > 0:
#                 visualizer.plot_current_errors(
#                     epoch,
#                     float(epoch_iter) / dataset_size, opt, errors)

#         if total_steps % opt.save_latest_freq == 0:
#             print('saving the latest model (epoch {}, total_steps {})'.format(
#                 epoch, total_steps))
#             model.save('latest')

        iter_data_time = time.time()
#     if epoch % opt.save_epoch_freq == 0:
#         print('saving the model at the end of epoch {}, iters {}'.format(
#             epoch, total_steps))
#         model.save('latest')
#         model.save(epoch)
    
    epoch_loss = 0
    epoch_pm, epoch_tm, epoch_dsc, epoch_dvh = 0, 0, 0, 0
    model.netG.eval()
    with torch.no_grad():
        for data in valid_dl:
            images, (target, pdm, sm, vs, idx, is_pseudo) = data
            images, target, pdm, sm, vs, idx, is_pseudo = images.cuda(), target.cuda(), pdm.cuda(), sm.cuda(), vs.cuda(), idx.cuda(), is_pseudo.cuda()
            pred = model.netG(images)
            epoch_loss += loss_func(pred, target.cuda(), pdm, sm, vs, idx, is_pseudo).cpu().numpy()
            pm = pred_mean2D(pred, target, pdm, sm, vs, idx, is_pseudo, evalbatchaccum)
            tm = target_mean2D(pred, target, pdm, sm, vs, idx, is_pseudo, evalbatchaccum)
            dsc = dose_score2D(pred, target, pdm, sm, vs, idx, is_pseudo, evalbatchaccum)
            dvh = dvh_score2D(pred, target, pdm, sm, vs, idx, is_pseudo, evalbatchaccum)
            if pm != pm:
                assert tm != tm and dsc != dsc and dvh != dvh
            else:
                epoch_pm += pm.cpu().numpy()
                epoch_tm += tm.cpu().numpy()
                epoch_dsc += dsc.cpu().numpy()
                epoch_dvh += dvh.cpu().numpy()
    epoch_loss /= len(valid_dl)
    epoch_pm /= len(valid_ds.data_df)
    epoch_tm /= len(valid_ds.data_df)
    epoch_dsc /= len(valid_ds.data_df)
    epoch_dvh /= len(valid_ds.data_df)
    history_df.append({'Epoch': epoch,
                       'Valid Loss': round(epoch_loss, 4),
                       'Dose Score': round(epoch_dsc, 4),
                       'DVH Score': round(epoch_dvh, 4),
                       'Pred Mean': round(epoch_pm, 4),
                       'Target Mean': round(epoch_tm, 4),
                       'Time': round(time.time() - epoch_start_time, 1)
                      })
    if best_loss > epoch_loss:
        best_loss = epoch_loss
        torch.save({'model': model.netG.state_dict()}, './model_weights/{}/models/best_loss_fold{}.pth'.format(config.exp_name, opt.foldidx))
    if best_dose > epoch_dsc:
        best_dose = epoch_dsc
        torch.save({'model': model.netG.state_dict()}, './model_weights/{}/models/best_dose_fold{}.pth'.format(config.exp_name, opt.foldidx))
    if best_dvh > epoch_dvh:
        best_dvh = epoch_dvh
        torch.save({'model': model.netG.state_dict()}, './model_weights/{}/models/best_dvh_fold{}.pth'.format(config.exp_name, opt.foldidx))

    pd.DataFrame(history_df).to_csv('./logs/{}/history_fold{}.csv'.format(config.exp_name, opt.foldidx), index=False)
    best_str = "Best valid loss: {}, dose score: {}, dvh score: {}".format(best_loss, best_dose, best_dvh)
    f = open("./logs/{}/bestmetrics_fold{}.txt".format(config.exp_name, opt.foldidx), "w")
    f.write(best_str)
    f.close()

    print('End of epoch {} / {} \t Valid loss: {} \t Time taken {} sec'.format(
        epoch, opt.niter + opt.niter_decay,
        str(round(epoch_loss, 4)),
        str(round(time.time() - epoch_start_time, 1))))

    model.netG.train()
    model.update_learning_rate()
