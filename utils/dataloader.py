import numpy as np
from torch.utils.data import Dataset, DataLoader
from utils.default_dataloader import DefaultDataLoader


def trasform_item(tfms, img, target, possible_dose_mask, structure_masks):
    mask = np.zeros((12, 128, 128, 128), dtype='float32')
    mask[0] = target
    mask[1] = possible_dose_mask
    mask[2:] = structure_masks
    
    tf_item = tfms(image=img, mask=mask)
    img = tf_item['image']
    target = tf_item['mask'][0:1]
    possible_dose_mask = tf_item['mask'][1:2]
    structure_masks = tf_item['mask'][2:]
    return img, target, possible_dose_mask, structure_masks

class KBPDataset(Dataset):
    """OpenKBP dataset."""
    def __init__(self, config, data_df, training=True, transform=None, load_cached=False):
        self.config = config
        self.data_df = data_df
        self.training = training
        self.transform = transform
        self.load_cached = load_cached

        self.mode = 'training_model' if training else 'dose_prediction'
        self.defdataset = DefaultDataLoader(self.data_df['Id'], batch_size=1, shuffle=False, mode_name=self.mode)
        
    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if self.load_cached:
            img, (target, possible_dose_mask, structure_masks, voxel_size, idx) = self.load_idx(idx)
        else:
            item = self.defdataset.get_batch(index=idx)
            img = item['ct'][:,:,:,:,0].astype('float32')
#             img = np.clip(img, 0, 4075.)
#             img /= 4000.

            possible_dose_mask = item['possible_dose_mask'][:,:,:,:,0].astype('float32')
            structure_masks = np.moveaxis(item['structure_masks'][0].astype('float32'), -1, 0)
            voxel_size = item['voxel_dimensions']
            if self.training:
                target = item['dose'][:,:,:,:,0].astype('float32')
        if self.transform is not None:
            img, target, possible_dose_mask, structure_masks = trasform_item(self.transform,
                        img, target, possible_dose_mask, structure_masks)
        
        img = np.concatenate((img, possible_dose_mask, structure_masks), axis=0)
        
        if self.training:
            return img, (target, possible_dose_mask, structure_masks, voxel_size, idx)
        assert self.load_cached == False, "Please set load_cached to false for test time"
        return img, (possible_dose_mask, item)
    
    def load_idx(self, idx):
        pt_id = self.data_df.loc[idx, 'Id'].split('/')[-1]
        img = np.load('./data/train-pats-npy/imgs/{}.npy'.format(pt_id)).astype('float32')
        target = np.load('./data/train-pats-npy/targets/{}.npy'.format(pt_id)).astype('float32')
        possible_dose_mask = np.load('./data/train-pats-npy/possible_dose_masks/{}.npy'.format(pt_id)).astype('float32')
        structure_masks = np.load('./data/train-pats-npy/structure_masks/{}.npy'.format(pt_id)).astype('float32')
        voxel_size = np.load('./data/train-pats-npy/voxel_sizes/{}.npy'.format(pt_id)).astype('float32')
        return img, (target, possible_dose_mask, structure_masks, voxel_size, idx)


def get_data_bunch(config):
    split_df = pd.read_csv(config.split_csv)
    train_df = split_df[split_df['Fold'] != config.fold].reset_index(drop=True)
    valid_df = split_df[split_df['Fold'] == config.fold].reset_index(drop=True)
    test_df = pd.read_csv(config.test_csv)
    test_df['ImageId'] = 'test_images/' + test_df['ImageId'] + '.jpg'
    # test_df = test_df.loc[1:1].reset_index(drop=True)

    if config.reduce_dataset:
        train_df = train_df.head(200)

    train_ds = CarDataset(config, train_df, training=True)
    valid_ds = CarDataset(config, valid_df, training=False)
    test_ds = CarDataset(config, test_df, training=False)

    data = ImageDataBunch.create(train_ds, valid_ds, test_ds, bs=config.batch_size, num_workers=config.num_workers)

    return data
