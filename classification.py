import os
import time
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler
from torchvision.ops import focal_loss

import utils
from models.encoder_classifier import EncoderClassifier2D
from data.hdf5_clips import LabelledClips

DATAFRAME = []

def get_batch(malapposed_loader, normal_loader):
    malapposed_batch = malapposed_loader.__iter__().next() # B//2 x T x H x W x 1
    normal_batch = normal_loader.__iter__().next()

    batch = torch.cat([malapposed_batch, normal_batch], dim=0) # B x T x H x W x 1
    labels = torch.tensor([1] * malapposed_batch.shape[0] + [0] * normal_batch.shape[0]) # B,
    permutation = torch.randperm(batch.shape[0])
    batch, labels = batch[permutation], labels[permutation]
    return batch, labels

def get_dataloaders(args, img_size, clip_len, mode):
    if mode not in ("train", "val"):
        raise ValueError(f"Invalid dataset mode {mode}")

    mode_to_transform = {
        "train": utils.augs.TrainTransform(
            aug_list=args.classification_augs, img_size=img_size, deterministic_intensity=args.det_in, concat_struts=args.concat_struts
        ),
        "val": utils.augs.ValTransform(
            aug_list=args.classification_augs, img_size=img_size, concat_struts=args.concat_struts
        ),
    }
    transform = mode_to_transform[mode]

    root = os.path.join(args.data_path, f"fold_{args.fold}", mode)
    save_file = os.path.join(
        root, f"labelled_clips_delta_{args.delta_frames}_len_{clip_len}_sz_{img_size}.h5"
    )

    malapposed_dataset, normal_dataset = LabelledClips(
        root=root,
        frames_per_clip=clip_len,
        delta_frames=args.delta_frames,
        transform=transform,
        cached=args.use_cached_dataset,
        save_file=save_file,
        save_img_size=img_size,
    ).create_datasets()

    shuffle = (True if mode=="train" else False)
    if mode=="train":
        batch_size = args.batch_size//2
    else:
        batch_size = args.batch_size # No batch construction during eval
    malapposed_loader = DataLoader(malapposed_dataset, batch_size=batch_size, pin_memory=True, num_workers = args.workers//4, shuffle=shuffle)
    normal_loader = DataLoader(normal_dataset, batch_size=batch_size, pin_memory=True, num_workers = args.workers//4, shuffle=shuffle)

    return malapposed_loader, normal_loader

def get_criterion(args):
    '''
    criterion: (Bx2) x {0,1}^B --> R
    '''
    if args.loss=="CE":
        return nn.CrossEntropyLoss(weight=torch.tensor([1., args.false_positive_weight]).to(args.device), reduction='sum')

    elif args.loss=="focal":
        def focal_criterion(feats, labels):
            inputs = feats[:,1]-feats[:,0]
            inpts=inputs.float()
            labels=labels.float()
            return focal_loss.sigmoid_focal_loss(inputs, labels, alpha=args.focal_alpha, gamma=args.focal_gamma, reduction='sum')
        return focal_criterion

def main(args):    
    model = EncoderClassifier2D(args)
    model = model.to(args.device)

    vis = utils.visualize.Visualize(args) if args.visualize else None

    mal_train_loader, normal_train_loader = get_dataloaders(args, args.img_size, args.clip_len, "train")
    mal_val_loader, normal_val_loader = get_dataloaders(args, args.img_size, args.clip_len, "val")

    criterion = get_criterion(args)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    lr_milestones = [args.batches_per_epoch * m for m in args.lr_milestones]
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=lr_milestones, gamma=args.lr_gamma)

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
        val_diags = val_one_epoch(model, criterion, mal_val_loader, normal_val_loader, args)
        train_diags = train_one_epoch(model, criterion, optimizer, lr_scheduler, mal_train_loader, normal_train_loader, args)
        val_diags = {f'val_{k}':v for k,v in val_diags.items()}
        train_diags = {f'train_{k}':v for k,v in train_diags.items()}

        print(f"Epoch [{epoch}] : Train: {train_diags['train_loss']}, Val: {val_diags['val_loss']}")

        diags=utils.dict_merge(val_diags, train_diags, assert_unique_keys=True)
        if vis is not None:
            vis.wandb_init(model)
            vis.log(diags)
        DATAFRAME.append(diags)
        save_model_checkpoint(epoch)
    
    df = pd.DataFrame(DATAFRAME)
    df.to_pickle(os.path.join(args.output_dir, 'dataframe.pkl'))

def get_tp_fp_tn_fn(preds, labels):
    normal_preds = preds[torch.nonzero(torch.where(labels==0, 1, 0))]
    fp = sum(normal_preds).item()
    tn = normal_preds.shape[0] - fp

    mal_preds = preds[torch.nonzero(torch.where(labels==1, 1, 0))]
    tp = sum(mal_preds).item()
    fn = mal_preds.shape[0] - tp

    diags = {'tp':tp, 'fp':fp, 'tn':tn, 'fn':fn}

    return diags

def train_one_epoch(model, criterion, optimizer, lr_scheduler, malapposed_loader, normal_loader, args):

    model.train()

    running_mean_loss = 0
    stats = {'tp':0,'fp':0,'tn':0, 'fn':0}

    for step in tqdm(range(args.batches_per_epoch)):
        images, labels = get_batch(malapposed_loader, normal_loader)
        images = images.to(args.device); labels=labels.to(args.device)
        feats = model(images) # B x 2
        loss = criterion(feats, labels)

        running_mean_loss = running_mean_loss * (step/(step+1)) + (loss.item()/feats.shape[0])/(step+1)

        preds = torch.argmax(feats, dim=1)
        _stats = get_tp_fp_tn_fn(preds=preds, labels=labels)
        utils.dict_sum(stats, _stats) # Mutates

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

    diags = utils.compute_classification_stats(tp=stats['tp'], fp=stats['fp'], tn=stats['tn'], fn=stats['fn'])
    diags['loss']=running_mean_loss; diags['lr']=lr_scheduler.get_last_lr()[0]
    return utils.dict_merge(diags, stats)

def val_one_epoch(model, criterion, malapposed_loader, normal_loader, args):
    model.eval()

    with torch.no_grad():
        n_correct = 0; n_wrong = 0
        running_mean_mal_loss = 0
        for step, images in tqdm(enumerate(malapposed_loader)):
            images = images.to(args.device)
            feats = model(images) # B x 2
            loss = criterion(feats, torch.tensor([1]*feats.shape[0]).to(args.device))
            running_mean_mal_loss = running_mean_mal_loss * (step/(step+1)) + (loss.item()/feats.shape[0])/(step+1)

            preds = torch.argmax(feats, dim=1)
            n_correct += (preds==1).sum().item()
            n_wrong += (preds==0).sum().item()
        tp, fn=n_correct, n_wrong

        n_correct = 0; n_wrong = 0
        running_mean_normal_loss = 0
        for step, images in tqdm(enumerate(normal_loader)):
            images = images.to(args.device)
            feats = model(images) # B x 2
            loss = criterion(feats, (torch.tensor([0]*feats.shape[0])).to(args.device))
            running_mean_normal_loss = running_mean_normal_loss * (step/(step+1)) + (loss.item()/feats.shape[0])/(step+1)

            preds = torch.argmax(feats, dim=1)
            n_correct += (preds==0).sum().item()
            n_wrong += (preds==1).sum().item()
        tn, fp = n_correct, n_wrong

        diags=utils.compute_classification_stats(tp=tp, fp=fp, tn=tn, fn=fn, matrix_savepath=None)
        diags['mal_loss']=running_mean_mal_loss; diags['normal_loss']=running_mean_normal_loss
        diags['loss']=(running_mean_mal_loss*len(malapposed_loader)+running_mean_normal_loss*len(normal_loader))/(len(malapposed_loader)+len(normal_loader))
    return diags


if __name__=="__main__":
    args = utils.arguments.classification_args()
    print(args)
    print("CUDA availability:", torch.cuda.is_available())
    main(args)


