# torch
import torch
import torch.optim
import torch.utils.data
# torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# built-in
import numpy as np


# Important: We use CIFAR-10 from torchvision in our experiments.
# # The dataset has been previously preprocessed following https://github.com/tscohen/gconv_experiments/tree/master/gconv_experiments (Cohen & Welling, 2016)
# def get_dataset(batch_size, num_workers):
#     # Load dataset
#     train_set = np.load('data/train_all.npz')
#     test_set = np.load('data/test.npz')
#     train_data = train_set['data']
#     train_labels = train_set['labels']
#     test_data = test_set['data']
#     test_labels = test_set['labels']
#     # Tensorize
#     test_data = torch.from_numpy(test_data)
#     test_lbls = torch.from_numpy(test_labels)
#     train_data = torch.from_numpy(train_data)
#     train_lbls = torch.from_numpy(train_labels)
#
#     # Create TensorDataset and DataLoader objects
#     test_dataset = torch.utils.data.TensorDataset(test_data.type(torch.FloatTensor), test_lbls.type(torch.LongTensor))
#     test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=batch_size)
#     train_dataset = torch.utils.data.TensorDataset(train_data.type(torch.FloatTensor), train_lbls.type(torch.LongTensor))
#     dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
#                    'validation': test_loader}
#
#     return dataloaders, test_loader


# Based on torchvision datasets
def get_dataset(batch_size, augmentation, num_workers):
    # Create transformations
    # ----------------------
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # Transformation for val and test
    transf_test = transforms.Compose([transforms.ToTensor(),
                                     normalize,
                                     ])
    # Transformation for train
    if augmentation:
        transf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
    else:
        transf_train = transf_test
    # ----------------------
    # Download dataset and create dataloaders
    train_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=True, transform=transf_train, download=True),
                                               batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(datasets.CIFAR10(root='./data', train=False, transform=transf_test),
                                             batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dataloaders = {'train': train_loader,
                   'validation': test_loader}
    # Return dataloaders
    return dataloaders, test_loader
