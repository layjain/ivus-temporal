import os
import pandas as pd

EXPERIMENT = 4


# Get the best epoch
METRIC = 'val_balanced_accuracy'
START_EPOCH = 25
MAX_EPOCHS = 500
fold_to_epoch = {}
experiment_dir = f"/data/vision/polina/users/layjain/ivus-temporal/checkpoints/classification/{EXPERIMENT}"
for fold in range(4):
    found = False
    for run_dir in os.listdir(experiment_dir):
        dataframe_path = os.path.join(experiment_dir, run_dir, f'fold_{fold}/dataframe.pkl')
        if os.path.exists(dataframe_path):
            found=True
            break
    if not found:
        raise ValueError(f"Run Not Found: experiment {EXPERIMENT} fold {fold}")
    df = pd.read_pickle(dataframe_path)
    if len(df) < MAX_EPOCHS:
        raise ValueError("Insufficient No. of Epochs") 
    df['epoch']=range(len(df)); df=df.loc[df['epoch']<MAX_EPOCHS].copy()
    df.set_index('epoch', inplace=True)
    df['ema'] = df[METRIC].ewm(halflife=5).mean()
    df.loc[df.index<START_EPOCH, 'ema'] = None
    best_epoch = df['ema'].idxmax()
    fold_to_epoch[fold] = best_epoch
   
