import numpy as np
from data.hdf5_clips import UnlabelledClips
from torch.utils.data import Dataset
import torch

import os
import pickle

class ANTsDataset(Dataset):
    def __init__(
        self,
        override_length=32 * 10,
        ants_params_path="/data/vision/polina/users/layjain/notebooks/reg_pretr_params.pkl",
    ):
        root = os.path.join(
            "/data/vision/polina/users/layjain/pickled_data/train_val_split_4a", "train"
        )
        save_file = f"/data/vision/polina/users/layjain/pickled_data/pretraining/ivus_train_len_30.h5"
        self.clip_dataset = UnlabelledClips(
            root,
            frames_per_clip=30,
            transform=None,
            cached=True,
            save_img_size=256,
            save_file=save_file,
            one_item_only=False,
            override_length=override_length,
        )
        self.override_length = override_length

        with open(ants_params_path, 'rb') as fh:
            all_thetas, _, _, all_vxs, all_vys = pickle.load(fh)
            self.all_thetas = np.array(all_thetas)
            self.all_vxs, self.all_vys = np.array(all_vxs), np.array(all_vys)            

    def __len__(self):
        return self.override_length

    def __getitem__(self, idx):
        clip = self.clip_dataset.__getitem__(idx)
        thetas = self.all_thetas[idx]
        v_xs, v_ys = self.all_vxs[idx], self.all_vys[idx]
        

