import sys
sys.path.append("/data/vision/polina/users/layjain/ivus-temporal")

from data.hdf5_clips import UnlabelledClips
from torch.utils.data import DataLoader


mode="train"
clip_len = 30

dataset = UnlabelledClips(
    root = f"/data/vision/polina/users/layjain/pickled_data/train_val_split_4a/{mode}",
    frames_per_clip=clip_len,
    transform=None,
    cached=True,
    save_img_size=256,
    save_file=f"/data/vision/polina/users/layjain/pickled_data/pretraining/ivus_{mode}_len_{clip_len}_del.h5"
)

dataloader = DataLoader(dataset, batch_size=8, pin_memory=True, num_workers = 8, shuffle=True)

print("Loading from dataloader")
images = next(iter(dataloader))

print(images.shape) # 8=B x 30=T x 256=H x 256=W x 1=C torch
