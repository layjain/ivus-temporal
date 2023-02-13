import monai
import glob
import pandas as pd
import numpy as np
import argparse
import torch
import torchvision.transforms.functional as F
from tqdm import tqdm
from PIL import Image
import os

parser = argparse.ArgumentParser(description="ray's segmentation model")
parser.add_argument("--device", default='cuda')
args = parser.parse_args()


model = torch.load("/data/vision/polina/users/layjain/ivus-temporal/struts/model_10600.pt").to(args.device)
model.eval()

# Get the Segmentations
for filepath in glob.glob("/data/vision/polina/users/layjain/pickled_data/malapposed_runs/*.pkl"):
    filename = filepath.split('/')[-1].split('.')[0]
    if filename != "PD2ZWGIA":
        continue
    video = pd.read_pickle(filepath) # TxHxWx1
    segmentation_dir = f"/data/vision/polina/users/layjain/ivus-temporal/struts/segmentations/{filename}"

    if os.path.exists(segmentation_dir):
        os.system(f"rm -r {segmentation_dir}")
    os.makedirs(segmentation_dir)

    for idx in tqdm(range(len(video))): # H x W x 1
        image = video[idx]
        image = np.expand_dims((image/255).squeeze(), axis=(0,1)).astype(np.float32) # 1 x 1 x H x W
        image = torch.from_numpy(image).to(args.device)
        image = F.resize(image, (256, 256))
        
        logits = model(image)
        segmentation = torch.sigmoid(logits).detach().to('cpu').numpy().squeeze() # H x W
        segmentation = (segmentation * 255).astype(np.uint8)
        im = Image.fromarray(segmentation)
        im.save(os.path.join(segmentation_dir, f"{idx}.jpg"))
    break

