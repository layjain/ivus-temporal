import os
import torch
import pandas as pd
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import utils
from models.registration_model import RegistrationModel
from data.hdf5_clips import UnlabelledClips
import registration

from utils import timer_func
import time

DATAFRAME = []

def get_dataloader(args, mode="train"):
    root = os.path.join(args.data_path, mode)
    save_file = f"/data/vision/polina/users/layjain/pickled_data/pretraining/ivus_{mode}_len_{args.clip_len}.h5"
    dataset = UnlabelledClips(root, frames_per_clip=args.clip_len, transform=None, cached=args.use_cached_dataset, save_img_size=args.img_size, save_file=save_file)
    print(f"Dataset size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=args.batch_size, pin_memory=True, num_workers = args.workers, shuffle=True)
    return dataloader

def train_one_epoch(model, criterion, optimizer, lr_scheduler, dataloader, args):
    model.train()

    running_mean_loss = 0.0
    t1s, t2s, t3s = [], [], []
    # most time spent on dataloader
    for step in tqdm(range(args.batches_per_epoch)):
        t0 = time.time()
        images = next(iter(dataloader)) # B x T x H x W x C=1, [0, 255]
        images = images.transpose(2, 4).to(args.device) # B x T x C x W x H
        t1 = time.time() - t0
        images_tr, template = model(images)
        loss = criterion(images_tr, template)
        t2 = time.time() - t0 - t1

        running_mean_loss = running_mean_loss * (step/(step+1)) + loss.item()/(step+1) # loss already mean-reduced

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        t3 = time.time() - t0 - t1 - t2

        t1s.append(t1); t2s.append(t2); t3s.append(t3)
    t1 = sum(t1s)/len(t1s); t2 = sum(t2s)/len(t2s); t3 = sum(t3s)/len(t3s)
    diags = {'loss':running_mean_loss, 'lr':lr_scheduler.get_last_lr()[0], 't1':t1, 't2':t2, 't3':t3}
    return diags

def main(args):
    model = RegistrationModel(args)
    print(f"Loaded model with {utils.count_parameters(model) / 10**6}M parameters")
    model = model.to(args.device)

    vis = utils.visualize.Visualize(args) if args.visualize else None
    train_dataloader = get_dataloader(args)
    criterion = registration.registration_loss.get_loss(args)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, total_steps=args.batches_per_epoch * args.epochs, pct_start=0.1, anneal_strategy='cos') # OneCycelCos

    def save_model_checkpoint(epoch):
        if args.output_dir:
            save_dir = os.path.join(args.output_dir, 'saved_models')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': args}
            torch.save(
                checkpoint,
                os.path.join(save_dir, 'model_{}.pth'.format(epoch)))

    for epoch in range(args.epochs):
        train_diags = train_one_epoch(model, criterion, optimizer, lr_scheduler, train_dataloader, args)
        diags = train_diags.copy()
        diags['epoch'] = epoch
        print(f"Epoch [{epoch}] : Train: {train_diags}")
        if vis is not None:
            vis.wandb_init(model)
            vis.log(diags)
        DATAFRAME.append(diags)
        save_model_checkpoint(epoch)
        
    df = pd.DataFrame(DATAFRAME)
    df.to_pickle(os.path.join(args.output_dir, 'dataframe.pkl'))

if __name__=="__main__":
    args = registration.registration_arguments.registration_args()
    print(args)
    print("CUDA availability:", torch.cuda.is_available())
    main(args)
