import os
import torch
import torch.nn as nn
from fastai.torch_core import Tensor, MetricsList, Any
from fastai.basic_train import Learner, LearnerCallback
from fastai.callbacks.mixup import MixUpLoss

from IPython.core.debugger import set_trace

import wandb
import fastai
from fastai.callbacks import TrackerCallback
from pathlib import Path
import random
try:
    import matplotlib
    matplotlib.use('Agg')  # non-interactive backend (avoid tkinter issues)
    import matplotlib.pyplot as plt
except:
    print('Warning: matplotlib required if logging sample image predictions')

class SaveBestModel(LearnerCallback):
    def __init__(self, learn:Learner, config, outfile=''):
        super().__init__(learn)
        self.learn = learn
        self.config = config
        self.best_loss = None
        self.best_dose = None
        self.best_dvh = None
        self.outfile = outfile
        if not os.path.exists('./model_weights/{}/models'.format(config.exp_name)):
            os.makedirs('./model_weights/{}/models'.format(config.exp_name))

    def on_epoch_end(self, epoch: int, smooth_loss: Tensor, last_metrics: MetricsList, **kwargs: Any) -> bool:
        loss, dose, dvh, pred_mean, target_mean = last_metrics
        if self.best_dose is None or dose < self.best_dose:
            self.best_dose = dose
            if self.config.model_ckpt_metrics['dose']:
                torch.save({'model': self.learn.model.state_dict()}, './model_weights/{}/models/best_dose{}.pth'.format(self.config.exp_name, self.outfile))
        if self.best_dvh is None or dvh < self.best_dvh:
            self.best_dvh = dvh
            if self.config.model_ckpt_metrics['dvh']:
                torch.save({'model': self.learn.model.state_dict()}, './model_weights/{}/models/best_dvh{}.pth'.format(self.config.exp_name, self.outfile))
        if self.best_loss is None or loss < self.best_loss:
            self.best_loss = loss
            if self.config.model_ckpt_metrics['loss']:
                torch.save({'model': self.learn.model.state_dict()}, './model_weights/{}/models/best_loss{}.pth'.format(self.config.exp_name, self.outfile))


'''W&B Callback for fast.ai

This module hooks fast.ai Learners to Weights & Biases through a callback.
Requested logged data can be configured through the callback constructor.

Examples:
    WandbCallback can be used when initializing the Learner::

        from wandb.fastai import WandbCallback
        [...]
        learn = Learner(data, ..., callback_fns=WandbCallback)
        learn.fit(epochs)
    
    Custom parameters can be given using functools.partial::

        from wandb.fastai import WandbCallback
        from functools import partial
        [...]
        learn = Learner(data, ..., callback_fns=partial(WandbCallback, ...))
        learn.fit(epochs)

    Finally, it is possible to use WandbCallback only when starting
    training. In this case it must be instantiated::

        learn.fit(..., callbacks=WandbCallback(learn))

    or, with custom parameters::

        learn.fit(..., callbacks=WandBCallback(learn, ...))
'''

class WandbCallback(TrackerCallback):

    # Record if watch has been called previously (even in another instance)
    watch_called = False

    def __init__(self,
                 learn,
                 log="gradients",
                 save_model=True,
                 monitor=None,
                 mode='auto',
                 input_type=None,
                 validation_data=None,
                 predictions=36,
                 seed=12345):
        """WandB fast.ai Callback

        Automatically saves model topology, losses & metrics.
        Optionally logs weights, gradients, sample predictions and best trained model.

        Args:
            learn (fastai.basic_train.Learner): the fast.ai learner to hook.
            log (str): "gradients", "parameters", "all", or None. Losses & metrics are always logged.
            save_model (bool): save model at the end of each epoch. It will also load best model at the end of training.
            monitor (str): metric to monitor for saving best model. None uses default TrackerCallback monitor value.
            mode (str): "auto", "min" or "max" to compare "monitor" values and define best model.
            input_type (str): "images" or None. Used to display sample predictions.
            validation_data (list): data used for sample predictions if input_type is set.
            predictions (int): number of predictions to make if input_type is set and validation_data is None.
            seed (int): initialize random generator for sample predictions if input_type is set and validation_data is None.
        """

        # Check if wandb.init has been called
        if wandb.run is None:
            raise ValueError(
                'You must call wandb.init() before WandbCallback()')

        # Adapted from fast.ai "SaveModelCallback"
        if monitor is None:
            # use default TrackerCallback monitor value
            super().__init__(learn, mode=mode)
        else:
            super().__init__(learn, monitor=monitor, mode=mode)
        self.save_model = save_model
        self.model_path = Path(wandb.run.dir) / 'bestmodel.pth'

        self.log = log
        self.input_type = input_type
        self.best = None

        # Select items for sample predictions to see evolution along training
        self.validation_data = validation_data
        if input_type and not self.validation_data:
            wandbRandom = random.Random(seed)  # For repeatability
            predictions = min(predictions, len(learn.data.valid_ds))
            indices = wandbRandom.sample(range(len(learn.data.valid_ds)),
                                         predictions)
            self.validation_data = [learn.data.valid_ds[i] for i in indices]

    def on_train_begin(self, **kwargs):
        "Call watch method to log model topology, gradients & weights"

        # Set self.best, method inherited from "TrackerCallback" by "SaveModelCallback"
        super().on_train_begin()

        # Ensure we don't call "watch" multiple times
        if not WandbCallback.watch_called:
            WandbCallback.watch_called = True

            # Logs model topology and optionally gradients and weights
            wandb.watch(self.learn.model, log=self.log)

    def on_epoch_end(self, epoch, smooth_loss, last_metrics, **kwargs):
        "Logs training loss, validation loss and custom metrics & log prediction samples & save model"

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            current = self.get_monitor_value()
            if current is not None and self.operator(current, self.best):
                print(
                    'Better model found at epoch {} with {} value: {}.'.format(
                        epoch, self.monitor, current))
                self.best = current

                # Save within wandb folder
                with self.model_path.open('wb') as model_file:
                    self.learn.save(model_file)

        # Log sample predictions
        if self.validation_data:
            pred_log = []

            for x, y in self.validation_data:
                pred = self.learn.predict(x)

                # scalar -> likely to be a category
                if not pred[1].shape:
                    pred_log.append(
                        wandb.Image(
                            x.data,
                            caption='Ground Truth: {}\nPrediction: {}'.format(
                                y, pred[0])))

                # most vision datasets have a "show" function we can use
                elif hasattr(x, "show"):
                    # log input data
                    pred_log.append(
                        wandb.Image(x.data, caption='Input data', grouping=3))

                    # log label and prediction
                    for im, capt in ((pred[0], "Prediction"),
                                     (y, "Ground Truth")):
                        # Resize plot to image resolution
                        # from https://stackoverflow.com/a/13714915
                        my_dpi = 100
                        fig = plt.figure(frameon=False, dpi=my_dpi)
                        h, w = x.size
                        fig.set_size_inches(w / my_dpi, h / my_dpi)
                        ax = plt.Axes(fig, [0., 0., 1., 1.])
                        ax.set_axis_off()
                        fig.add_axes(ax)

                        # Superpose label or prediction to input image
                        x.show(ax=ax, y=im)
                        pred_log.append(wandb.Image(fig, caption=capt))
                        plt.close(fig)

                # likely to be an image
                elif hasattr(y, "shape") and (
                    (len(y.shape) == 2) or
                    (len(y.shape) == 3 and y.shape[0] in [1, 3, 4])):

                    pred_log.extend([
                        wandb.Image(x.data, caption='Input data', grouping=3),
                        wandb.Image(pred[0].data, caption='Prediction'),
                        wandb.Image(y.data, caption='Ground Truth')
                    ])

                # we just log input data
                else:
                    pred_log.append(wandb.Image(x.data, caption='Input data'))

            wandb.log({"Prediction Samples": pred_log}, commit=False)

        # Log losses & metrics
        # Adapted from fast.ai "CSVLogger"
        logs = {
            name: stat
            for name, stat in list(
                zip(self.learn.recorder.names, [epoch, smooth_loss] +
                    last_metrics))
        }
        logs['learning_rate'] = self.learn.opt.lr
        logs['momentum'] = self.learn.opt.mom
        wandb.log(logs)

    def on_train_end(self, **kwargs):
        "Load the best model."

        if self.save_model:
            # Adapted from fast.ai "SaveModelCallback"
            if self.model_path.is_file():
                with self.model_path.open('rb') as model_file:
                    self.learn.load(model_file, purge=False)
                    print('Loaded best saved model from {}'.format(
                        self.model_path))


# https://github.com/oguiza/DataAugmentation/blob/master/ImageDataAugmentation.py
class CutMixCallback(LearnerCallback):
    "Callback that creates the cutmixed input and target."
    def __init__(self, learn:Learner, α:float=1., stack_y:bool=True, true_λ:bool=True):
        super().__init__(learn)
        self.α,self.stack_y,self.true_λ = α,stack_y,true_λ

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies cutmix to `last_input` and `last_target` if `train`."
        if not train: return
        λ = np.random.beta(self.α, self.α)
        λ = max(λ, 1- λ)
        shuffle = torch.randperm(last_target.size(0)).to(last_input.device)
        x1, y1 = last_input[shuffle], last_target[shuffle]
        #Get new input
        last_input_size = last_input.shape
        bbx1, bby1, bbx2, bby2 = rand_bbox(last_input.size(), λ)
        new_input = last_input.clone()
        new_input[:, ..., bby1:bby2, bbx1:bbx2] = last_input[shuffle, ..., bby1:bby2, bbx1:bbx2]
        λ = last_input.new([λ])
        if self.true_λ:
            λ = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (last_input_size[-1] * last_input_size[-2]))
            λ = last_input.new([λ])
        if self.stack_y:
            new_target = torch.cat([last_target.unsqueeze(1).float(), y1.unsqueeze(1).float(),
                                    λ.repeat(last_input_size[0]).unsqueeze(1).float()], 1)
        else:
            if len(last_target.shape) == 2:
                λ = λ.unsqueeze(1).float()
            new_target = last_target.float() * λ + y1.float() * (1-λ)
        return {'last_input': new_input, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


def rand_bbox(last_input_size, λ):
    '''lambd is always between .5 and 1'''

    W = last_input_size[-1]
    H = last_input_size[-2]
    cut_rat = np.sqrt(1. - λ) # 0. - .707
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def cutmix(learn:Learner, α:float=1., stack_x:bool=False, stack_y:bool=True, true_λ:bool=True) -> Learner:
    "Add mixup https://arxiv.org/pdf/1905.04899.pdf to `learn`."
    learn.callback_fns.append(partial(CutMixCallback, α=α, stack_y=stack_y, true_λ=true_λ))
    return learn


class RicapLoss(nn.Module):
    "Adapt the loss function `crit` to go with ricap data augmentations."

    def __init__(self, crit, reduction='mean'):
        super().__init__()
        if hasattr(crit, 'reduction'):
            self.crit = crit
            self.old_red = crit.reduction
            setattr(self.crit, 'reduction', 'none')
        else:
            self.crit = partial(crit, reduction='none')
            self.old_crit = crit
        self.reduction = reduction

    def forward(self, output, target):
        if target.ndim == 2:
            c_ = target[:, 1:5]
            W_ = target[:, 5:]
            loss = [W_[:, k] * self.crit(output, c_[:, k].long()) for k in range(4)]
            d = torch.mean(torch.stack(loss))
        else: d = self.crit(output, target)
        if self.reduction == 'mean': return d.mean()
        elif self.reduction == 'sum': return d.sum()
        return d

    def get_old(self):
        if hasattr(self, 'old_crit'): return self.old_crit
        elif hasattr(self, 'old_red'):
            setattr(self.crit, 'reduction', self.old_red)
            return self.crit

class RicapCallback(LearnerCallback):
    "Callback that creates the ricap input and target."
    def __init__(self, learn:Learner, β:float=.3, stack_y:bool=True):
        super().__init__(learn)
        self.β,self.stack_y = β,stack_y

    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = RicapLoss(self.learn.loss_func)

    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies ricap to `last_input` and `last_target` if `train`."
        if not train: return
        I_x, I_y = last_input.size()[2:]
        w = int(np.round(I_x * np.random.beta(self.β, self.β)))
        h = int(np.round(I_y * np.random.beta(self.β, self.β)))
        w_ = [w, I_x - w, w, I_x - w]
        h_ = [h, h, I_y - h, I_y - h]
        cropped_images = {}
        bs = last_input.size(0)
        c_ = torch.zeros((bs, 4)).float().to(last_input.device)
        W_ = torch.zeros(4).float().to(last_input.device)
        for k in range(4):
            idx = torch.randperm(bs)
            x_k = np.random.randint(0, I_x - w_[k] + 1)
            y_k = np.random.randint(0, I_y - h_[k] + 1)
            cropped_images[k] = last_input[idx][:, :, x_k:x_k + w_[k], y_k:y_k + h_[k]]
            c_[:, k] = last_target[idx].float()
            W_[k] = w_[k] * h_[k] / (I_x * I_y)
        patched_images = torch.cat(
            (torch.cat((cropped_images[0], cropped_images[1]), 2),
             torch.cat((cropped_images[2], cropped_images[3]), 2)), 3)  #.cuda()
        if self.stack_y:
                new_target = torch.cat((last_target[:,None].float(), c_,
                                        W_[None].repeat(last_target.size(0), 1)), dim=1)
        else:
            new_target = c_ * W_
        return {'last_input': patched_images, 'last_target': new_target}

    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()


def ricap(learn:Learner, stack_y:bool=True) -> Learner:
    "Add ricap https://arxiv.org/pdf/1811.09030.pdf to `learn`."
    learn.callback_fns.append(partial(RicapCallback, stack_y=stack_y))
    return learn

