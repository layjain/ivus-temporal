import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader

class MNISTClipDataset(Dataset):
    def __init__(self, clip_length=30, one_item_only=False, transform=None, override_length=None):
        self.override_length = override_length
        self.transform = transform
        self.one_item_only = one_item_only

        if self.override_length is not None:
            assert isinstance(self.override_length, int)
        
        # load MNIST dataset
        mnist_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        self.mnist_dataset = datasets.MNIST(root='../data', train=True, download=True, transform=mnist_transforms)
        self.clip_length = clip_length

        # get indices of all images with label=1
        self.indices = torch.where(self.mnist_dataset.targets == 1)[0]

    def __len__(self):
        if self.override_length is not None:
            return self.override_length
        # return number of clips
        return len(self.indices) // self.clip_length

    def __getitem__(self, idx):
        # get indices of images in this clip
        if self.one_item_only:
            idx = 0
        clip_indices = self.indices[idx*self.clip_length:(idx+1)*self.clip_length]

        # load images and stack them into a tensor
        clip = torch.stack([self.mnist_dataset[i][0] for i in clip_indices], dim=0) # T x 1 x 28 x 28
        clip = clip.transpose(1, 3).numpy() # T x 28 x 28 x 1
        if self.transform is not None:
            clip = self.transform(clip)
        return clip