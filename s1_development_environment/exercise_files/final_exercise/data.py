import torch
import glob
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os
class CorruptedMNIST(Dataset):
    def __init__(self, images, labels) -> None:
        self.images = images
        self.labels = labels
        
    def __getitem__(self, idx):
        return self.images[idx, :], self.labels[idx]
    
    def __len__(self):
        return self.labels.size(0)

def display_image(img, label):
    plt.imshow(img, cmap='gray')
    plt.title(f"Image with label {label}")
    plt.show()

def mnist(batch_size: int = 32):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    filepaths = glob.glob('C:\\Users\\micha\\OneDrive\\Skrivebord\\MLOps\\dtu_mlops\\data\\corruptmnist\\*.pt')
    # filepaths = glob.glob('data\\corruptmnist\\*.pt')
    print("cwd: ", os.getcwd())
    print(f"n files: {len(filepaths)}")
    train_images = torch.cat([torch.load(path) for path in filepaths if 'train_images' in path])
    train_targets = torch.cat([torch.load(path) for path in filepaths if 'train_target' in path])
    test_targets = torch.cat([torch.load(path) for path in filepaths if 'test_target' in path])
    test_images = torch.cat([torch.load(path) for path in filepaths if 'test_images' in path])
    # display image
    # idx = torch.randint(low=0, high=train_images.size(0), size=(1,)).item()
    # display_image(train_images[idx], train_targets[idx])
    
    # create datasets
    train_dataset = CorruptedMNIST(train_images.view(train_images.size(0), -1), train_targets)
    test_dataset = CorruptedMNIST(test_images.view(test_images.size(0), -1), test_targets)
    # create dataloaders and return
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

if __name__ == '__main__':
    train, test = mnist()
    _x, _y = next(iter(train))
    print(f"\nImage batch has dimension: {_x.size()}\nLabels have dimension: {_y.size()}\n")