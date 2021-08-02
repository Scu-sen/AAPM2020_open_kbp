import os
import numpy as np
import torch
from easydict import EasyDict
config = EasyDict()


config.exp_name = "run1"
config.model_name = "unet2dsepconv"
config.encoder = None
config.load_model_ckpt = None  # "./model_weights/run125/models/best_dose_fold{}.pth"
config.axis = 3
config.addjaehee = True
config.loadjaehee = False
config.in_channels = 315
config.num_classes = 1
config.desc = "add jaehee data"
config.dataclass = None
config.pseudo_path = './data/pseudo-dose-js/'
config.add_val_orig = False
config.add_val_pseudo = False
config.imgmulpdm = False
config.pdmsmmulimg = False
config.offset_list = np.arange(-4, 5)
config.resample = None  # spatial dimension resampling
config.resample_slice = None  # spatial dimension resampling while dataloading
config.notargetreplace = None  # (None/'pdm'/'sm'/'exclude'): replace 0 target with a weighted pdm/sm slice or exclude them
config.addvoxelch = False
config.pretrained = True
config.wandb = True
config.wandb_project = "openkbp-2020"
config.gpu = 0
config.fp16 = False
config.num_workers = os.cpu_count()
config.one_hot_labels = False
config.debug = False
config.reduce_dataset = False
# config.debug = True
# config.reduce_dataset = True

config.DATA_CSV_PATH = './data/random_split5_class.csv'

config.optimizer = "radam"
config.batch_size = 8
config.lr = 1e-1
config.lr_finetune = 1e-4
config.weight_decay = 0 #1e-1
config.alpha = 0.99
config.mom = 0.9 # Momentum
config.eps = 1e-6
config.nesterov = True
config.radam_degenerated_to_sgd = False
config.amsgrad = True
config.sched_type = "one_cycle" # LR schedule type (one_cycle/flat_and_anneal)

config.loss_dict = EasyDict()
# config.loss_dict.L1Loss = {'weight': 1.0, 'sm_weight': 3, 'pdm_mul': False, 'dvh_sm_asym': False}
config.loss_dict.SmoothDVHLossMulVoxel = {'weight': 1.0, 'sm_weight': 3, 'pdm_mul': False, 'dvh_sm_asym': False}

# # config.augment_prob = 0.9
# config.blend_params = {
#     'size': .15,            # range(0.-1.) You can indicate the size of the patched area/s
#     'alpha': 1.,           # This is used to define a proba distribution
#     'fixed_proba': 0,      # This overrides alpha proba distribution. Will fix the % of the image that is modified
#     'grid': True,          # Determine if patches may overlap or not. With True they do not overlap
#     'blend_type': 'cut',   # Options: 'zero', 'noise', 'mix', 'cut', 'random'
#     'same_size': False,     # All patches may have the same size or not
#     'same_crop': False,    # Cropping patches are from the same subregion as input patches (only with 'mix' and 'cut')
#     'same_image': False,   # Cropping patches will be from the same or different images (only with 'mix' and 'cut')
# }

config.swa = False
config.cosine_annealing = False
config.drop_rate = 0.75
config.drop_rate_min = config.drop_rate/2
config.unet2dgngroups = 32  # 0=bn, default=32
config.unet2dnum_hidden_features = (np.array([32, 64, 128, 256, 512, 1024])*2).astype('int')  # default=[32, 64, 128, 256, 512, 1024]
config.unet2d_num_dilated_convs = 4  # default=4
config.unet2d_gated = False  # default = False
config.unet2d_n_resblocks = 1  # default = 1
config.layernorm = 'group'
config.unet2dpadding = 1  # default = 1
config.unet2dkernel_size = 3  # default = 3
config.convdilation = 1  # default = 1 
config.div_factor = 25. # 1 cycle param 
config.pct_start = 0.3 # 1 cycle param
config.final_div = None # 1 cycle param

config.epochs = 60
config.mixup = 0.
config.patchmix = None
config.ricap = 0    # 0.3
config.ann_start = 0.7 # Annealing start

# Data augmentations
config.horizontalflip = {'axis': config.axis}
config.randomscale  = {'scale_limit': 0.2}
config.randomrotate = {'max_angle': 3, 'p': 0.2}
config.randomshift  = {'shift_limit': 4}
config.randomrotate90  = None  # {'p': 0.25}
config.noise_multiplier = None  # {'multiplier': (0.95, 1.05)}
config.gaussianblur = None  # {'sigma_limit': 1.0, 'blur_channels': False, 'p': 0.75}  # default = None
config.randomcrop = None  # {'crop_limit': 0.1, 'p': 0.8}
config.verticalflip  = None  # {'num_channels': config.in_channels, 'p': 0.5}
config.augautoenc = None
config.teachers = None  # 'run164'

config.lrfinder = False # Run learning rate finder
config.dump = 0 # Print model; don't train"
config.log_file = "./logs/{}".format(config.exp_name) # Log file name
config.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config.model_ckpt_metrics = {'loss': True, 'dose': True, 'dvh': True}

if config.debug:
    config.wandb = False
