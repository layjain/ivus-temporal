'''
Verify few simple things:
1. Dataloader iterator Ends
2. Batches are returned in different orders
3. Batches can be incomplete
'''

from torch.utils.data import Dataset
from torch.utils.data import DataLoader

class DummyDataset(Dataset):
    def __init__(self):
        pass

    def __getitem__(self, index):
        return index

    def __len__(self):
        return 31

dataset = DummyDataset()
dataloader = DataLoader(dataset, batch_size=3, pin_memory=True, num_workers = 2, shuffle=True)


# for item in dataloader:
#     print(item)
#     print()

for _ in range(20):
    print(next(iter(dataloader)))
    print()