# %% imports
import torch
from torchvision import transforms,datasets
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision.transforms import v2

# %% Noisy MNIST dataset
class Noisy_MNIST(Dataset):
    # initialization of the dataset
    def __init__(self, split,data_loc,transform,noise=0.5):
        # save the input parameters
        self.split = split
        self.data_loc = data_loc
        self.noise = noise
        self.transform = transform

        Train = (self.split in ['train'])

        # get the original MNIST dataset
        Clean_MNIST = datasets.MNIST(self.data_loc, train=Train, download=True)
        # reshuffle the test set to have digits 0-9 at the start
        data = Clean_MNIST.data.unsqueeze(1)
        targets = Clean_MNIST.targets # Grab the targets explicitly so slice becomes possible

        # Adequate splitting logic:
        # 1. Training encompasses the first 50k      
        if self.split == 'train':
            # Keep the first 50,000 for training
            data = data[:50000]
            targets = targets[:50000]
        # 2. Validation holds the remaining 10k
        elif self.split == 'valid':
            data = data[50000:]
            targets = targets[50000:]

        # 3. Test    
        elif self.split == 'test':
            idx = torch.load(r'Assignment 1\Denoise-handwritten-digits-using-FCN\assignment1_data\assignment1_data\test_idx.tar')
            data[:, :] = data[idx, :]
          
        # reshape and normalize
        resizer = transforms.Resize(32)
        resized_data = resizer(data)*1.0
        normalized_data = 2 *(resized_data/255) - 1

        # create the data
        self.Clean_Images = normalized_data
        self.Noisy_Images = normalized_data + torch.randn(normalized_data.size()) * self.noise
        self.Labels = Clean_MNIST.targets

    def __len__(self):
        return self.Labels.size(0)

    # create a a method that retrieves a single item form the dataset
    def __getitem__(self, idx):
        clean_image = self.Clean_Images[idx,:,:,:]
        noisy_image = self.Noisy_Images[idx,:,:,:]
        label = self.Labels[idx]
        if self.transform:
            clean_image,noisy_image,label = self.transform((clean_image,noisy_image,label))
        return clean_image,noisy_image,label

# %% dataloader for the Noisy MNIST dataset
def create_dataloaders(data_loc, batch_size,transform=None):

    # 1. Assigning the data containers to their adequate set (train, val and test);
    Noisy_MNIST_train = Noisy_MNIST("train", data_loc,transform)
    Noisy_MNIST_test = Noisy_MNIST("test" , data_loc,transform)
    Noisy_MNIST_validation = Noisy_MNIST("valid",data_loc,transform)

    # 2. Create loaders;
    Noisy_MNIST_train_loader = DataLoader(Noisy_MNIST_train,
    batch_size=batch_size, shuffle=True, drop_last=False)
    Noisy_MNIST_test_loader = DataLoader(Noisy_MNIST_test ,
    batch_size=batch_size, shuffle=False, drop_last=False)
    Noisy_MNIST_validation_loader = DataLoader(Noisy_MNIST_validation, 
    batch_size=batch_size, shuffle=True, drop_last=False)
    return Noisy_MNIST_train_loader, Noisy_MNIST_test_loader, Noisy_MNIST_validation_loader

# def flattener(data:tuple):
#     f = nn.Flatten(start_dim=0)
#     flattened = [f(item) if isinstance(item, torch.Tensor) else item for item in data[:-1]]
#     return tuple(flattened) + (data[-1],)
    
# basisc_transf = v2.Compose([
#     # transforms.ToTensor(), <- Data is of tensor nature by definition
#     flattener
# ])

# %% test if the dataloaders work
if __name__ == "__main__":
    # define parameters
    data_loc = '../5LSL0-Datasets' #change the datalocation to something that  works for you
    batch_size = 64
    # get dataloader
    train_loader, test_loader, val_loader = create_dataloaders(data_loc, batch_size)
    # get some examples
    examples = enumerate(test_loader)
    _, (x_clean_example, x_noisy_example, labels_example) = next(examples)
    # use these example images througout the assignment as the first 10 correspond to the digits 0-9
    # show the examples in a plot
    plt.figure(figsize=(12,3))
    for i in range(10):
        plt.subplot(2,10,i+1)
        plt.imshow(x_clean_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.subplot(2,10,i+11)
        plt.imshow(x_noisy_example[i,0,:,:],cmap='gray')
        plt.xticks([])
        plt.yticks([])
    plt.tight_layout()
    plt.savefig("data_examples.png",dpi=300,bbox_inches='tight')
    plt.show()
