'''
1. Check if the transforms work
2. Check if transforms are differentiable
'''

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.append("/data/vision/polina/users/layjain/ivus-temporal")
from models.stns import rigid_stn


B, T, H, W = 3, 2, 256, 256

# Input Batch
orig_img = Image.open('/data/vision/polina/users/layjain/notebooks/square.jpg').resize((H, W))
orig_img = np.asarray(orig_img)[...,0] # H x W
np_batch = np.stack([np.stack([orig_img]*T)] * B) # B x T x H x W
np_batch = np.expand_dims(np_batch, 2) # B x T x 1 x H x W
x = torch.tensor(np_batch.astype(np.float32), requires_grad=False)

# Parameters
p_1 = [0, 0, 0] + [0.2, 0, 0] # batch 1
p_2 = [0,0,np.pi/4] + [0, 0.2, 0] # batch 2
p_3 = [0,0,-np.pi/4] + [0.2,0.2,0] # batch 3
p = np.stack([p_1, p_2, p_3])
p = torch.tensor(p.astype(np.float32), requires_grad=True)

# STN
stn = rigid_stn.RigidSTN()
x_tr = stn.transform(x, p)

# Check if they work
def display_batch(x, savepath="stn_tests.jpg"):
    '''
    x: torch
    '''
    savepath=os.path.join("results/profiling/", savepath)
    x = x.squeeze().detach().numpy()
    B, T, H, W = x.shape
    f, axarr = plt.subplots(B, T)
    for i in range(B):
        for j in range(T):
            axarr[i, j].imshow(x[i][j])

    plt.savefig(savepath)
    plt.close()
    
display_batch(x_tr)

# Check differentiability
print(p.grad)
optimizer = torch.optim.AdamW([p], lr=1e-2)
losses = []
for epoch in tqdm(range(200)):
    optimizer.zero_grad()
    x_tr = stn.transform(x, p)
    loss = torch.norm(x_tr-x)
    losses.append(loss.item())
    loss.backward()
    optimizer.step()

plt.plot(losses); plt.savefig("results/profiling/stn_tests_loss.png"); plt.close()