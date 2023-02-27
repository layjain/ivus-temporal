import pandas as pd
from sklearn.linear_model import LinearRegression
import os

DF_PATHS = {}
for experiment in range(1, 10):
    experiment_dir = f"/data/vision/polina/users/layjain/ivus-temporal/checkpoints/classification/{experiment}"
    if not os.path.exists(experiment_dir):
        continue
    for fold in range(4):
        found = False
        for run_dir in os.listdir(experiment_dir):
            dataframe_path = os.path.join(experiment_dir, run_dir, f'fold_{fold}/dataframe.pkl')
            if os.path.exists(dataframe_path):
                found=True
                DF_PATHS[(experiment, fold)]=dataframe_path
                break


for experiment in [2]:
    for fold in [0,1,2,3]:
        df = pd.read_pickle(DF_PATHS[(experiment, fold)])
        LR = LinearRegression()
        result=LR.fit(y=df['val_loss'], X=df[['val_mal_loss','val_normal_loss']])
        score=result.score(y=df['val_loss'], X=df[['val_mal_loss','val_normal_loss']])
        coefs=result.coef_

        print(f"E{experiment}F{fold} Score:{score} Coef_:{coefs} Ratio:{coefs[1]/coefs[0]}")