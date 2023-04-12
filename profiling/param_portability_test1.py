'''
Test parameter portability between ANTs and STN using a single image
'''
# 1. Create two images related by simple transforms
import os
from PIL import Image
import numpy as np
import torch
import sys

sys.path.append("/data/vision/polina/users/layjain/ivus-temporal")
os.environ['PATH'] = os.environ['PATH'] + os.pathsep + "/data/vision/polina/users/layjain/ivus-temporal/registration/install/bin"

from models.stns import rigid_stn, parameters
import registration

import matplotlib.pyplot as plt

# a. translation_y
# a = np.zeros((256, 256))
# a[128-30:128+30, :] = 1
# a = a.astype(np.uint8)
# im = Image.fromarray(a)
# im.save('fixed.jpg')
# a = np.zeros((256, 256))
# a[128-30+50:128+30+50, :] = 1
# a = a.astype(np.uint8)
# im = Image.fromarray(a)
# im.save('moving.jpg')

# b. translation_x
# a = np.zeros((256, 256))
# a[128-30:128+30, :] = 1
# a = a.astype(np.uint8)
# im = Image.fromarray(a)
# im.save('fixed.jpg')
# a = np.zeros((256, 256))
# a[128-30:128+30, 50:] = 1
# a = a.astype(np.uint8)
# im = Image.fromarray(a)
# im.save('moving.jpg')

# c. rotation
# a = np.zeros((256, 256))
# a[128-20:128+20, 128-60:128] = 255
# a[128-20:128+20, 128:128+60] = 125
# a = a.astype(np.uint8)
# im = Image.fromarray(a)
# im.save('fixed.jpg')
# im = im.rotate(90)
# im.save('moving.jpg')

# d. both
a = np.zeros((256, 256))
a[128-20:128+20, 128-60:128] = 255
a[128-20:128+20, 128:128+60] = 125
a = a.astype(np.uint8)
im = Image.fromarray(a)
im.save('fixed.jpg')
im = im.rotate(30)
im = im.transform(
    size=im.size,
    method=Image.AFFINE,
    data=(1, 0, 30, 0, 1, 50)
)
im.save('moving.jpg')

# 2. Use ANTS to find transform
os.system(f"antsRegistration -d 2 \
        -o [test_del,del.nii.gz] \
        -m MeanSquares[fixed.jpg , moving.jpg , 1] \
        -t Rigid[0.01] \
        -c 100x100x100 \
        -s 32x16x1 \
        --float \
        -v \
        -f 4x2x1 > results/profiling/param_portability_1.ants")

# 3. Get params
def get_theta(cosine, sin):
    '''
    theta in (-pi, pi]
    '''
    theta = np.arccos(cosine) # [0, pi]
    if sin < 0:
        theta = -1 * theta
    return theta

from scipy.io import loadmat
annots = loadmat("test_del0GenericAffine.mat")
print(annots)
mat = annots['AffineTransform_float_2_2']
cosine, sin, _, _, v_x, v_y = mat[:, 0]
v_x = v_x/(256/2)
v_y = v_y/(256/2)
theta = get_theta(cosine, sin)
print(f"Parsed ANTs parameters: cos={cosine} sin={sin} vx={v_x} vy={v_y} theta={theta}")

# 4. Convert to STN params

def ants_parameters_to_stn_parameters(theta, v_x, v_y, cosine, sin):
    theta = -theta/np.pi
    t_x = cosine+sin+v_x-1
    t_y = -sin+cosine+v_y-1
    translation = torch.tensor([[t_x, t_y]], dtype=torch.float32) # B x 2T
    rotation = torch.tensor([[theta]], dtype=torch.float32) # B x T
    p = parameters.Parameters(translation=translation, rotation=rotation)
    return p

p = ants_parameters_to_stn_parameters(theta, v_x, v_y, cosine, sin)
print(f"STN Params: {str(p)}")

# 5. Transform the input using antsApplyTransforms
os.system(f"antsApplyTransforms -d 2 \
        -i moving.jpg \
        -r fixed.jpg \
        -o del_antsApply.nii.gz \
        -t test_del0GenericAffine.mat \
        -v 1 \
         > results/profiling/param_portability_1.applied")

# 6. Transform the input accordingly manually

moving_image = Image.open("moving.jpg")
moving_image = np.asarray(moving_image).astype(np.float32) # H x W
x = torch.tensor(moving_image).unsqueeze(0).unsqueeze(0).unsqueeze(0) # B=1 x T=1 x ONE=1 x H x W
stn = rigid_stn.RigidSTN()
x_tr = stn.transform(x, p)
moving_image_tr = x_tr.squeeze().numpy() # H x W
fixed_image = np.asarray(Image.open("fixed.jpg")).astype(np.float32) # H x W

# Display
import SimpleITK as sitk
sitk_t1 = sitk.ReadImage("del.nii.gz")
t1 = sitk.GetArrayFromImage(sitk_t1)

ants_applied = sitk.ReadImage("del_antsApply.nii.gz")
ants_applied = sitk.GetArrayFromImage(ants_applied)

plt.figure(figsize = (22, 5))
plt.subplot(1,5,1)
plt.imshow(fixed_image, cmap='gray'); plt.title('fixed.jpg')
plt.subplot(1,5,2)
plt.imshow(moving_image, cmap='gray'); plt.title('moving.jpg')
plt.subplot(1,5,3)
plt.imshow(moving_image_tr, cmap='gray'); plt.title('x_tr')
plt.subplot(1,5,4)
plt.imshow(t1, cmap='gray'); plt.title('ANTs output')
plt.subplot(1,5,5)
plt.imshow(ants_applied, cmap='gray'); plt.title('ANTs ApplyTransforms Output')
plt.savefig("results/profiling/param_portability_1.jpg"); plt.close()