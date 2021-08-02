from fastai.vision import Learner
from fastai.callbacks import TrainingPhase, GeneralScheduler
from fastai.callback import annealing_cos

def fit_with_annealing(learn:Learner, num_epoch:int, lr:float=1e-3, annealing_start:float=0.7,
                    callbacks:list=None)->None:
    n = len(learn.data.train_dl)
    anneal_start = int(n*num_epoch*annealing_start)
    phase0 = TrainingPhase(anneal_start).schedule_hp('lr', lr)
    phase1 = TrainingPhase(n*num_epoch - anneal_start).schedule_hp('lr', lr, anneal=annealing_cos)
    phases = [phase0, phase1]
    sched = GeneralScheduler(learn, phases)
    learn.callbacks.append(sched)
    learn.fit(num_epoch, callbacks=callbacks)
