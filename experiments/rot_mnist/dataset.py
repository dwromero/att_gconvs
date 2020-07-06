# torch
import torch
import torch.optim
import torch.utils.data
# built-in
import numpy as np


# Taken from https://github.com/tscohen/gconv_experiments/tree/master/gconv_experiments (Cohen & Welling, 2016)
def preprocess_mnist_data(train_data, test_data, train_labels, test_labels):
    train_mean = np.mean(train_data)  # compute mean over all pixels
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data)
    train_data /= train_std
    test_data /= train_std
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    # Return preprocessed dataset
    return train_data, test_data, train_labels, test_labels


def get_dataset(batch_size, num_workers):
    # Load dataset
    train_set = np.load('data/train_all.npz')     # TODO: More flexible
    test_set = np.load('data/test.npz')
    train_data = train_set['data']
    train_labels = train_set['labels']
    test_data = test_set['data']
    test_labels = test_set['labels']

    # Preprocess dataset
    train_data, test_data, train_labels, test_labels = preprocess_mnist_data(
        train_data, test_data, train_labels, test_labels)
    test_data = torch.from_numpy(test_data)
    test_lbls = torch.from_numpy(test_labels)
    train_data = torch.from_numpy(train_data)
    train_lbls = torch.from_numpy(train_labels)

    # Create TensorDataset and DataLoader objects
    test_dataset = torch.utils.data.TensorDataset(test_data.type(torch.FloatTensor), test_lbls.type(torch.LongTensor))
    test_loader = torch.utils.data.DataLoader(test_dataset, shuffle=False, num_workers=4, batch_size=batch_size)
    train_dataset = torch.utils.data.TensorDataset(train_data.type(torch.FloatTensor), train_lbls.type(torch.LongTensor))
    dataloaders = {'train': torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
                   'validation': test_loader}

    return dataloaders, test_loader


