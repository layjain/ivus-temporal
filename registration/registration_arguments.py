import argparse
import os

def registration_args():
    parser = argparse.ArgumentParser(description='IVUS clip Registration')

    # Data
    parser.add_argument("--data-path", type=str, default='/data/vision/polina/users/layjain/pickled_data/train_val_split_4a')
    parser.add_argument("--clip-len", type=int, default=30)
    parser.add_argument("--use-cached-dataset",  dest="use_cached_dataset", help="Use cached Dataset", action='store_true')
    parser.add_argument("--mnist",  dest="mnist", help="Use MNIST data", action='store_true')
    
    # Model and Methodology
    parser.add_argument("--encoder-type", default='resnet18', type=str, help='resnet18|resnet50')
    parser.add_argument("--mlp-encoding-size", default=256, type=int)
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=8, type=int)
    parser.add_argument('--workers', default=10, type=int, metavar='N', help='number of data loading workers (default: 16)')
    parser.add_argument('--img-size', default=256, type=int)
    parser.add_argument("--head-depth", type=int, default=2)
    parser.add_argument('--remove-layers', default=['layer4'], help='layer[1-4]')
    parser.add_argument('--transforms', default=['Translation', 'Rotation'], nargs='+', choices=['Translation', 'Rotation'])
    parser.add_argument('--new-template', default=False, action='store_true', help='use mean(x_tr) as the template base')
    parser.add_argument('--no-unet', default=False, action='store_true', help='as if Unet output is all zeros')

    # Loss
    parser.add_argument("--loss", type=str, default='MSE')
    parser.add_argument("--loss-hyperparams", type=float, nargs='+')

    # Optimization
    parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
    parser.add_argument('--batches-per-epoch', default=15, type=int, help='no. of training batches per epoch')
    parser.add_argument('--lr', default=1e-3, type=float, help='initial learning rate')
    parser.add_argument('--lr-scheduler', default='OneCycleCos', type=str, help='initial learning rate')
    parser.add_argument('--one-clip-only', default=False, action='store_true', help='force overfit')
    parser.add_argument('--zero-init', default=False, action='store_true', help='initialize the last layer of localizer & unet (if any) to 0')

    # Mode + Visualizations
    parser.add_argument('--name', default='', type=str, help='')
    parser.add_argument( "--fast-test", dest="fast_test", help="", action="store_true", )
    parser.add_argument('--visualize', default=False, action='store_true', help='visualize with wandb')
    parser.add_argument("--project-name", default='TemporalRegistration', type=str)

    # Augmentations
    parser.add_argument('--aug-list', default=[], nargs='*', type=str, help='augmentations to apply')

    args = parser.parse_args()

    args.model_in_channels = args.clip_len

    if args.mnist:
        args.img_size = 28

    if args.fast_test:
        # args.batch_size = 4
        # args.workers = 0
        args.batches_per_epoch = 2
        args.name='fast_test'
        args.visualize=False
        args.device='cpu'

    # Make the output-dir
    keys={
        "epochs":"epochs","clip_len":"len","lr":"lr","head_depth":"mlp","loss":"loss","no_unet":"nounet"
    }
    name = '-'.join(["%s%s" % (keys[k], getattr(args, k) if not isinstance(getattr(args, k), list) else '-'.join([str(s) for s in getattr(args, k)])) for k in sorted(keys)])
    import datetime
    dt = datetime.datetime.today()
    args.group_number=args.name
    args.name = "%s-%s-%s_%s" % (str(dt.month), str(dt.day), args.name, name)
    args.output_dir = "checkpoints/registration/%s/%s" % (args.group_number, args.name)
    args.wandb_name = f"{args.name}"

    if os.path.exists(args.output_dir):
        print("Cleaning the output-dir")
        os.system(f"rm -r {args.output_dir}")
    os.makedirs(args.output_dir)

    return args
