'''
1. Loss should be invariant on transposing
2. Loss should be minimum on alignment
'''

from PIL import Image
import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import sys
from tqdm import tqdm

sys.path.append("/data/vision/polina/users/layjain/ivus-temporal")
from models.stns import rigid_stn, parameters
import registration

B, T, H, W = 3, 2, 256, 256

# Input Batch
orig_img = Image.open('/data/vision/polina/users/layjain/notebooks/grid_srch_out.jpg').resize((H, W))
orig_img = np.asarray(orig_img) # H x W
orig_img = orig_img/np.max(orig_img) # A bad way to make [0,1]
np_batch = np.stack([np.stack([orig_img]*T)] * B) # B x T x H x W
np_batch = np.expand_dims(np_batch, 2) # B x T x 1 x H x W
x = torch.tensor(np_batch.astype(np.float32), requires_grad=False)

# Parameters
p = parameters.Parameters(
    translation=torch.tensor([[0,0,]+[0.5,0],[0,0]+[0,0.5],[0,0,]+[0.2,0.2]],dtype=torch.float32, requires_grad=True), # B=3 x (2*T=2)
    rotation=torch.tensor([[0,0],[1/4, 0],[-1/4, 0]],dtype=torch.float32, requires_grad=True), # B=3 x T=2
)

stn = rigid_stn.RigidSTN()
x_tr = stn.transform(x, p).squeeze()

# Check that identical images have CC -1
WIN = 9
loss_1, cross, I_var, J_var, cc = registration.registration_loss.CC_loss(None, x.squeeze(), template=x.squeeze(), return_var=True, win=WIN)
assert torch.isclose(cross, I_var).all().item()
assert torch.isclose(cross, J_var).all().item()
assert torch.isclose(J_var, I_var).all().item()
assert torch.isclose(J_var*I_var, cross*cross).all().item()

plt.hist(cross.flatten().detach().numpy(), label='cross/I_var/J_var', alpha=0.5, bins=10)
plt.hist(cc.flatten().detach().numpy(), label='cc', alpha=0.5, bins=10)
plt.legend(); plt.savefig(f"results/profiling/registration_loss_study_CC_{WIN}_dists.png"); plt.clf()

plt.imshow(cc, cmap='gray')
plt.savefig(f"results/profiling/registration_loss_study_CC_{WIN}.png"); plt.clf()

image = torch.cat([torch.from_numpy(orig_img), cross, I_var, J_var, cc],dim=1).detach().numpy(); plt.imshow(image, cmap='gray')
plt.savefig(f"results/profiling/registration_loss_study_CC_{WIN}_variances.png"); plt.clf()

loss_2 = registration.registration_loss.CC_loss(None, x_tr, template=x_tr, win=WIN)
print(loss_1, loss_2)


loss_1 = registration.registration_loss.CC_loss(None, x_tr, template=torch.mean(x_tr, dim=1, keepdim=True), win=WIN)
x_tr = x_tr.transpose(-1, -2)
loss_2 = registration.registration_loss.CC_loss(None, x_tr, template=torch.mean(x_tr, dim=1, keepdim=True), win=WIN)
print(loss_1, loss_2)

