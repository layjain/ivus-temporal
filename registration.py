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

    for step in tqdm(range(args.batches_per_epoch)):
        images = next(iter(dataloader)) # B x T x H x W x C=1, [0, 255]
        images = images.transpose(2, 4).to(args.device) # B x T x C x W x H
        images_tr, template = model(images)
        loss = criterion(images_tr, template)

        running_mean_loss = running_mean_loss * (step/(step+1)) + (loss.item()/images.shape[0])/(step+1)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    diags = {'loss':running_mean_loss, 'lr':lr_scheduler.get_last_lr()[0]}
    return diags

def main(args):
    model = RegistrationModel(args)
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
