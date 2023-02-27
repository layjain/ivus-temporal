import argparse
import os
import torch
import random
import utils

def registration_args():
    parser = argparse.ArgumentParser(description='IVUS clip Registration')

    # Data
    parser.add_argument("--data-path", type=str, default='/data/vision/polina/users/layjain/pickled_data/folded_malapposed_runs')
    parser.add_argument("--delta-frames", type=int, default=100)
    parser.add_argument("--clip-len", type=int, default=6)
    parser.add_argument("--use-cached-dataset",  dest="use_cached_dataset", help="Use cached Dataset", action='store_true')
    
    # Model and Methodology
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--encoder-type", default='resnet18', type=str, help='resnet18|resnet50')
    parser.add_argument("--mlp-encoding-size", default=128, type=int)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--img-size', default=256, type=int)
    parser.add_argument("--head-depth", type=int, default=0)
    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')

    # Loss
    parser.add_argument("--loss", type=str, default='CE')
    parser.add_argument("--focal-alpha", type=float, default=0.8)
    parser.add_argument("--focal-gamma", type=float, default=3)

    # Optimization
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument("--false-positive-weight", type=float, default=1.)
    parser.add_argument('--batches-per-epoch', default=1000, type=int, help='no. of training batches per epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr-scheduler', type=str, default='MultiStepLR', help='LR Scheduler')
    parser.add_argument('--lr-milestones', nargs='+', default=[], type=int, help='decrease lr on milestones')
    parser.add_argument('--lr-gamma', default=0.3, type=float, help='decrease lr by a factor of lr-gamma')

    # Mode + Visualizations
    parser.add_argument('--name', default='', type=str, help='')
    parser.add_argument('--output-dir', default='auto', help='path where to save')
    parser.add_argument( "--fast-test", dest="fast_test", help="", action="store_true", )
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize with wandb')
    
    # Augmentations
    parser.add_argument("--classification-augs", default=['norm'], nargs='+', type=str, help='data-augmentations to use while training the classifier')
    parser.add_argument("--det-in", dest="det_in", default=False, action='store_true', help='deterministic intensity augs')
    parser.add_argument("--concat-struts", dest="concat_struts", default=False, action='store_true', help='concatenate strut segmentations')

    parser.add_argument("--project-name", default='TemporalClassification', type=str)

    args = parser.parse_args()

    if args.concat_struts:
        args.model_in_channels = 2 * args.clip_len
    else:
        args.model_in_channels = args.clip_len

    if args.fast_test:
        args.batch_size = 4
        args.workers = 0
        args.batches_per_epoch = 2
        args.name='fast_test'
        args.visualize=False
        args.device='cpu'

    # Make the output-dir
    keys={
        "epochs":"epochs", "delta_frames":"delta","classification_augs":"aug","clip_len":"len","lr":"lr","head_depth":"mlp","concat_struts":"struts"
    }
    name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in sorted(keys)])
    import datetime
    dt = datetime.datetime.today()
    args.group_number=args.name
    args.name = "%s-%s-%s_%s" % (str(dt.month), str(dt.day), args.name, name)
    args.output_dir = "checkpoints/classification/%s/%s/fold_%s" % (args.group_number, args.name, args.fold)
    args.wandb_name = f"{args.name}-fold_{args.fold}"

    if os.path.exists(args.output_dir):
        print("Cleaning the output-dir")
        os.system(f"rm -r {args.output_dir}")
    os.makedirs(args.output_dir)

    return args
