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

def NormalizeImages(train_images, test_images):
    train_mean = torch.mean(train_images) # mean over images
    train_std = torch.std(train_images) # mean over images
    train_images =  (train_images - train_mean) / train_std
    test_images = (test_images - train_mean) / train_std
    return train_images, test_images

def LoadAndPreprocess(save: bool = False):
    print("Processing data...")
    root_path = 'C:\\Users\\micha\\OneDrive\\Skrivebord\\dtu_mlops'
    filepaths = glob.glob(f'{root_path}\\Corrupted_MNIST\\data\\raw\\*.pt')
    # filepaths = glob.glob('data\\corruptmnist\\*.pt')
    train_images = torch.cat([torch.load(path) for path in filepaths if 'train_images' in path])
    train_targets = torch.cat([torch.load(path) for path in filepaths if 'train_target' in path])
    test_targets = torch.cat([torch.load(path) for path in filepaths if 'test_target' in path])
    test_images = torch.cat([torch.load(path) for path in filepaths if 'test_images' in path])
    
    # display image
    idx = torch.randint(low=0, high=train_images.size(0), size=(1,)).item()
    display_image(train_images[idx], train_targets[idx])
    
    # normalize images 
    train_images_normalized, test_images_normalized = NormalizeImages(train_images, test_images)
    
    if save:
        print(f'Data saved to {root_path}\\Corrupted_MNIST\\data\\processed\\')
        torch.save(train_targets, f'{root_path}\\Corrupted_MNIST\\data\\processed\\train_targets.pt')
        torch.save(test_targets, f'{root_path}\\Corrupted_MNIST\\data\\processed\\test_targets.pt')
        torch.save(train_images_normalized, f'{root_path}\\Corrupted_MNIST\\data\\processed\\train_images.pt')
        torch.save(test_images_normalized, f'{root_path}\\Corrupted_MNIST\\data\\processed\\test_images.pt')
    
    # print(train_images[0])
    # print("SIZEEEE", train_images.size())
    return train_images_normalized, test_images_normalized,  train_targets, test_targets

def mnist(batch_size: int = 32):
    """Return train and test dataloaders for MNIST."""
    # exchange with the corrupted mnist dataset
    train_images, test_images, train_targets, test_targets = LoadAndPreprocess(save=False)
    # create datasets
    train_dataset = CorruptedMNIST(train_images.view(train_images.size(0), -1), train_targets)
    test_dataset = CorruptedMNIST(test_images.view(test_images.size(0), -1), test_targets)
    # create dataloaders and return
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)
    return train_dataloader, test_dataloader

# if __name__ == '__main__':
#     train, test = mnist()
#     _x, _y = next(iter(train))
#     print(f"\nImage batch has dimension: {_x.size()}\nLabels have dimension: {_y.size()}\n")

if __name__ == '__main__':
    # Get the data and process it
    LoadAndPreprocess(save=True)
    
    

