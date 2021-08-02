from fastai.torch_core import *
from fastai.basic_data import *
from fastai.callback import CallbackList, Callback, SmoothenValue
from fastai.data_block import *
from fastai.basic_train import BasicLearner, Learner, loss_batch, LearnerCallback
from fastai.callbacks.one_cycle import OneCycleScheduler
from utils.metrics2D import EvalBatchAccumulator
from IPython.core.debugger import set_trace

__all__ = ['NaNLearner', 'validate']

def _get_init_state(): return {'epoch':0, 'iteration':0, 'num_batch':0, 'skip_validate': False}

@dataclass
class CallbackHandler():
    "Manage all of the registered `callbacks` and `metrics`, smoothing loss by momentum `beta`."
    callbacks:CallbackList=None
    metrics:CallbackList=None
    beta:float=0.98
    evalbatchaccum:EvalBatchAccumulator=None

    def __post_init__(self)->None:
        "Initialize smoother and learning stats."
        self.callbacks = ifnone(self.callbacks, [])
        self.metrics = ifnone(self.metrics, [])
        self.metrics = [(met if isinstance(met, Callback) else AverageMetric(met, self.evalbatchaccum)) for met in self.metrics]
        self.callbacks = sorted(self.callbacks, key=lambda o: getattr(o, '_order', 0))
        self.smoothener = SmoothenValue(self.beta)
        self.state_dict:Dict[str,Union[int,float,Tensor]]=_get_init_state()

    def _call_and_update(self, cb, cb_name, **kwargs)->None:
        "Call `cb_name` on `cb` and update the inner state."
        new = ifnone(getattr(cb, f'on_{cb_name}')(**self.state_dict, **kwargs), dict())
        for k,v in new.items():
            if k not in self.state_dict:
                raise Exception(f"{k} isn't a valid key in the state of the callbacks.")
            else: self.state_dict[k] = v
    
    def __call__(self, cb_name, call_mets=True, **kwargs)->None:
        "Call through to all of the `CallbakHandler` functions."
        if call_mets: 
            for met in self.metrics: self._call_and_update(met, cb_name, **kwargs)
        for cb in self.callbacks: self._call_and_update(cb, cb_name, **kwargs)

    def set_dl(self, dl:DataLoader):
        "Set the current `dl` used."
        if hasattr(self, 'cb_dl'): self.callbacks.remove(self.cb_dl)
        if isinstance(dl.dataset, Callback):
            self.callbacks.append(dl.dataset)
            self.cb_dl = dl.dataset

    def on_train_begin(self, epochs:int, pbar:PBar, metrics:MetricFuncList)->None:
        "About to start learning."
        self.state_dict = _get_init_state()
        self.state_dict.update(dict(n_epochs=epochs, pbar=pbar, metrics=metrics))
        names = [(met.name if hasattr(met, 'name') else camel2snake(met.__class__.__name__)) for met in self.metrics]
        self('train_begin', metrics_names=names)
        if self.state_dict['epoch'] != 0:
            self.state_dict['pbar'].first_bar.total -= self.state_dict['epoch']
            for cb in self.callbacks: cb.jump_to_epoch(self.state_dict['epoch'])

    def on_epoch_begin(self)->None:
        "Handle new epoch."
        self.state_dict['num_batch'],self.state_dict['stop_training'] = 0,False
        self('epoch_begin')

    def on_batch_begin(self, xb:Tensor, yb:Tensor, train:bool=True)->Tuple[Any,Any]:
        "Handle new batch `xb`,`yb` in `train` or validation."
        self.state_dict.update(dict(last_input=xb, last_target=yb, train=train, 
            stop_epoch=False, skip_step=False, skip_zero=False, skip_bwd=False))
        self('batch_begin', call_mets = not self.state_dict['train'])
        return self.state_dict['last_input'], self.state_dict['last_target']

    def on_loss_begin(self, out:Tensor)->Any:
        "Handle start of loss calculation with model output `out`."
        self.state_dict['last_output'] = out
        self('loss_begin', call_mets=False)
        return self.state_dict['last_output']

    def on_backward_begin(self, loss:Tensor)->Tuple[Any,Any]:
        "Handle gradient calculation on `loss`."
        self.smoothener.add_value(loss.float().detach().cpu())
        self.state_dict['last_loss'], self.state_dict['smooth_loss'] = loss, self.smoothener.smooth
        self('backward_begin', call_mets=False)
        return self.state_dict['last_loss'], self.state_dict['skip_bwd']

    def on_backward_end(self)->Any:
        "Handle end of gradient calculation."
        self('backward_end', call_mets=False)
        return self.state_dict['skip_step']

    def on_step_end(self)->Any:
        "Handle end of optimization step."
        self('step_end', call_mets=False)
        return self.state_dict['skip_zero']

    def on_batch_end(self, loss:Tensor)->Any:
        "Handle end of processing one batch with `loss`."
        self.state_dict['last_loss'] = loss
        self('batch_end', call_mets = not self.state_dict['train'])
        if self.state_dict['train']:
            self.state_dict['iteration'] += 1
            self.state_dict['num_batch'] += 1
        return self.state_dict['stop_epoch']

    def on_epoch_end(self, val_loss:Tensor)->bool:
        "Epoch is done, process `val_loss`."
        self.state_dict['last_metrics'] = [val_loss] if val_loss is not None else [None]
        self('epoch_end', call_mets = val_loss is not None)
        self.state_dict['epoch'] += 1
        return self.state_dict['stop_training']

    def on_train_end(self, exception:Union[bool,Exception])->None:
        "Handle end of training, `exception` is an `Exception` or False if no exceptions during training."
        self('train_end', exception=exception)
        
    @property
    def skip_validate(self): return self.state_dict['skip_validate']

class AverageMetric(Callback):
    "Wrap a `func` in a callback for metrics computation."
    def __init__(self, func, evalbatchaccum):
        # If func has a __name__ use this one else it should be a partial
        name = func.__name__ if hasattr(func, '__name__') else func.func.__name__
        self.func, self.name = func, name
        self.world = num_distrib()
        self.evalbatchaccum = evalbatchaccum

    def on_epoch_begin(self, **kwargs):
        "Set the inner value to 0."
        self.val, self.count = 0.,0

    def on_batch_end(self, last_output, last_target, **kwargs):
        "Update metric computation with `last_output` and `last_target`."
        if not is_listy(last_target): last_target=[last_target]
        val = self.func(last_output, *last_target, self.evalbatchaccum)
        if val == val:
            self.count += first_el(last_target).size(0)
            if self.world:
                val = val.clone()
                dist.all_reduce(val, op=dist.ReduceOp.SUM)
                val /= self.world
            self.val += first_el(last_target).size(0) * val.detach().cpu()

    def on_epoch_end(self, last_metrics, **kwargs):
        "Set the final result in `last_metrics`."
        return add_metrics(last_metrics, self.val/self.count)

class MixUpCallback(LearnerCallback):
    "Callback that creates the mixed-up input and target."
    def __init__(self, learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True):
        super().__init__(learn)
        self.alpha,self.stack_x,self.stack_y = alpha,stack_x,stack_y
    
    def on_train_begin(self, **kwargs):
        if self.stack_y: self.learn.loss_func = MixUpLoss(self.learn.loss_func)
        
    def on_batch_begin(self, last_input, last_target, train, **kwargs):
        "Applies mixup to `last_input` and `last_target` if `train`."
        if not train: return
        lambd = np.random.beta(self.alpha, self.alpha, last_input.size(0))
        lambd = np.concatenate([lambd[:,None], 1-lambd[:,None]], 1).max(1)
        lambd = last_input.new(lambd)
        shuffle = torch.randperm(last_input.size(0)).to(last_input.device)
        x1 = last_input[shuffle]
        y1 = [lt[shuffle] for lt in last_target]
        if self.stack_x:
            new_input = [last_input, last_input[shuffle], lambd]
        else: 
            out_shape = [lambd.size(0)] + [1 for _ in range(len(x1.shape) - 1)]
            new_input = (last_input * lambd.view(out_shape) + x1 * (1-lambd).view(out_shape))
        if self.stack_y:
            new_target = torch.cat([last_target[:,None].float(), y1[:,None].float(), lambd[:,None].float()], 1)
        else:
#             if len(last_target.shape) == 2:
#                 lambd = lambd.unsqueeze(1).float()
            new_target = []
            for i in range(len(last_target)):
                lt, nt = last_target[i], y1[i]
                out_shape = [lambd.size(0)] + [1 for _ in range(len(lt.shape) - 1)]
                new_target.append(lt.float() * lambd.view(out_shape) + nt.float() * (1-lambd).view(out_shape))
        return {'last_input': new_input, 'last_target': new_target}  
    
    def on_train_end(self, **kwargs):
        if self.stack_y: self.learn.loss_func = self.learn.loss_func.get_old()

def validate(model:nn.Module, dl:DataLoader, loss_func:OptLossFunc=None, cb_handler:Optional[CallbackHandler]=None,
             pbar:Optional[PBar]=None, average=True, n_batch:Optional[int]=None)->Iterator[Tuple[Union[Tensor,int],...]]:
    "Calculate `loss_func` of `model` on `dl` in evaluation mode."
    model.eval()
    with torch.no_grad():
        val_losses,nums = [],[]
        if cb_handler: cb_handler.set_dl(dl)
        for xb,yb in progress_bar(dl, parent=pbar, leave=(pbar is not None)):
            if cb_handler: xb, yb = cb_handler.on_batch_begin(xb, yb, train=False)
            val_loss = loss_batch(model, xb, yb, loss_func, cb_handler=cb_handler)
            val_losses.append(val_loss)
            if not is_listy(yb): yb = [yb]
            nums.append(first_el(yb).shape[0])
            if cb_handler and cb_handler.on_batch_end(val_losses[-1]): break
            if n_batch and (len(nums)>=n_batch): break
        nums = np.array(nums, dtype=np.float32)
        if average: return (to_np(torch.stack(val_losses)) * nums).sum() / nums.sum()
        else:       return val_losses

def fit(epochs:int, learn:BasicLearner, callbacks:Optional[CallbackList]=None, metrics:OptMetrics=None, evalbatchaccum:EvalBatchAccumulator=None, teachers:Optional[list]=None)->None:
    "Fit the `model` on `data` and learn using `loss_func` and `opt`."
    assert len(learn.data.train_dl) != 0, f"""Your training dataloader is empty, can't train a model.
        Use a smaller batch size (batch size={learn.data.train_dl.batch_size} for {len(learn.data.train_dl.dataset)} elements)."""
    cb_handler = CallbackHandler(callbacks, metrics, evalbatchaccum=evalbatchaccum)
    pbar = master_bar(range(epochs))
    cb_handler.on_train_begin(epochs, pbar=pbar, metrics=metrics)

    exception=False
    try:
        for epoch in pbar:
            learn.model.train()
            cb_handler.set_dl(learn.data.train_dl)
            cb_handler.on_epoch_begin()
            for xb,yb in progress_bar(learn.data.train_dl, parent=pbar):
                if teachers is not None and yb[5].sum() > 0:
                    with torch.no_grad():
                        tpreds = [t(xb) for t in teachers]
                        tpreds = torch.cat([tp[None] for tp in tpreds]).mean(0)
                        tpreds *= yb[1]  # multiply with pdm
                        tpreds = torch.cat((tpreds, tpreds*yb[2]), axis=1)
                        yb[0][yb[5]] = tpreds[yb[5]]
                xb, yb = cb_handler.on_batch_begin(xb, yb)
                loss = loss_batch(learn.model, xb, yb, learn.loss_func, learn.opt, cb_handler)
                if cb_handler.on_batch_end(loss): break

            if not cb_handler.skip_validate and not learn.data.empty_val:
                val_loss = validate(learn.model, learn.data.valid_dl, loss_func=learn.loss_func,
                                       cb_handler=cb_handler, pbar=pbar)
            else: val_loss=None
            if cb_handler.on_epoch_end(val_loss): break
    except Exception as e:
        exception = e
        raise
    finally: cb_handler.on_train_end(exception)

class NaNLearner(Learner):
    def __init__(self, evalbatchaccum, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.evalbatchaccum = evalbatchaccum
        
    def fit(self, epochs:int, lr:Union[Floats,slice]=defaults.lr,
            wd:Floats=None, callbacks:Collection[Callback]=None, teachers:Optional[list]=None)->None:
        "Fit the model on this learner with `lr` learning rate, `wd` weight decay for `epochs` with `callbacks`."
        lr = self.lr_range(lr)
        if wd is None: wd = self.wd
        if not getattr(self, 'opt', False): self.create_opt(lr, wd)
        else: self.opt.lr,self.opt.wd = lr,wd
        callbacks = [cb(self) for cb in self.callback_fns + listify(defaults.extra_callback_fns)] + listify(callbacks)
        fit(epochs, self, metrics=self.metrics, callbacks=self.callbacks+callbacks, evalbatchaccum=self.evalbatchaccum,
            teachers=teachers)
    
    def validate(self, dl=None, callbacks=None, metrics=None):
        "Validate on `dl` with potential `callbacks` and `metrics`."
        dl = ifnone(dl, self.data.valid_dl)
        metrics = ifnone(metrics, self.metrics)
        cb_handler = CallbackHandler(self.callbacks + ifnone(callbacks, []), metrics, evalbatchaccum=self.evalbatchaccum)
        cb_handler.on_train_begin(1, None, metrics); cb_handler.on_epoch_begin()
        val_metrics = validate(self.model, dl, self.loss_func, cb_handler)
        cb_handler.on_epoch_end(val_metrics)
        return cb_handler.state_dict['last_metrics']
    
    def fit_one_cycle(learn:Learner, cyc_len:int, max_lr:Union[Floats,slice]=defaults.lr,
                      moms:Tuple[float,float]=(0.95,0.85), div_factor:float=25., pct_start:float=0.3, final_div:float=None,
                      wd:float=None, callbacks:Optional[CallbackList]=None, tot_epochs:int=None, start_epoch:int=None,
                      teachers:Optional[list]=None)->None:
        "Fit a model following the 1cycle policy."
        max_lr = learn.lr_range(max_lr)
        callbacks = listify(callbacks)
        callbacks.append(OneCycleScheduler(learn, max_lr, moms=moms, div_factor=div_factor, pct_start=pct_start,
                                           final_div=final_div, tot_epochs=tot_epochs, start_epoch=start_epoch))
        learn.fit(cyc_len, max_lr, wd=wd, callbacks=callbacks,teachers=teachers)
    
    def mixup(learn:Learner, alpha:float=0.4, stack_x:bool=False, stack_y:bool=True) -> Learner:
        "Add mixup https://arxiv.org/abs/1710.09412 to `learn`."
        learn.callback_fns.append(partial(MixUpCallback, alpha=alpha, stack_x=stack_x, stack_y=stack_y))
        return learn
