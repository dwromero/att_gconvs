import os
import numpy as np
# torch
import torch
import torch.optim
import torch.utils.data
# torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
# built-in
import numpy as np


def get_pcam_data(datadir):
    assert os.path.exists(datadir), 'Datadir does not exist: %s' % datadir

    train_dataset = _load_dataset(datadir, 'train')
    test_dataset = _load_dataset(datadir, 'test')
    val_dataset = _load_dataset(datadir, 'valid')

    return (train_dataset['x'], train_dataset['y'].flatten(), train_dataset), (test_dataset['x'], test_dataset['y'].flatten(), test_dataset), (val_dataset['x'], val_dataset['y'].flatten(), val_dataset)


def _load_dataset(datadir, setname):
    # The images:
    filename_x = os.path.join(datadir, 'camelyonpatch_level_2_split_' + setname + '_x.h5')
    x = _load_PCAM_H5(filename_x)
    # The labels (whether it is a centered tumor patch or not)
    filename_y = os.path.join(datadir, 'camelyonpatch_level_2_split_' + setname + '_y.h5')
    y = _load_PCAM_H5(filename_y)
    # The meta data
    filename_meta = os.path.join(datadir, 'camelyonpatch_level_2_split_' + setname + '_meta.csv')
    tumor_patch, center_tumor_patch, wsi = _load_PCAM_CSV(filename_meta)
    # Store all data in a python library
    dataset = {'x': x, 'y': y, 'tumor_patch': tumor_patch, 'center_tumor_patch': center_tumor_patch, 'wsi': wsi}
    return dataset


def _load_PCAM_H5(file, key=None):
    import h5py
    with h5py.File(file, 'r') as f:
        if key == None:
            data = f.get(list(f.keys())[0])[()]
        else:
            data = f.get(key)[()]
    return data


def _load_PCAM_CSV(file):
    import csv
    tumor_patch = []
    center_tumor_patch = []
    wsi = []
    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        for row in csv_reader:
            tumor_patch += [row["tumor_patch"] == 'True']
            center_tumor_patch += [row["center_tumor_patch"] == 'True']
            wsi += [row["wsi"]]
    return tumor_patch, center_tumor_patch, wsi

def get_dataset(batch_size, augmentation, num_workers):
    # ------------
    # Create Image Folder structure (if necessary)
    # train_data, test_data, train_labels, test_labels = preprocess_data(train_data, test_data, train_labels, test_labels)
    # create_ImageFolder_structure(train_data, train_labels, test_data, test_labels, val_data, val_labels)
    # ------------
    # Get mean and stddev of the dataset (Given below)
    # mean = get_mean('data/train')
    # std_dev = get_std('data/train', mean) # TODO: Wise to append both datasets? If so, need to recalculate
    normalize = transforms.Normalize(mean=[0.701, 0.538, 0.692], std=[0.235, 0.277, 0.213])
    """
    Mean is: [0.700756   0.5383578  0.69162056]
    StdDev is: [0.23497812 0.27740911 0.21289459]
    """
    transf_test = transforms.Compose([transforms.ToTensor(),
                                     normalize,
                                     ])
    if augmentation:
        transf_train = transforms.Compose([transforms.RandomHorizontalFlip(),
                                        transforms.RandomCrop(32, 4),
                                        transforms.ToTensor(),
                                        normalize,
                                        ])
    else:
        transf_train = transf_test
    # ----------------------
    # Create Dataset and Dataloaders
    train_dataset = datasets.ImageFolder(root='./data/train', transform=transf_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

    val_dataset = datasets.ImageFolder(root='./data/valid', transform=transf_test)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    test_dataset=datasets.ImageFolder(root='./data/test', transform=transf_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dataloaders = {'train': train_loader,
                   'validation': val_loader}
    # Return dataloaders
    return dataloaders, test_loader


# Based on https://github.com/tscohen/gconv_experiments/tree/master/gconv_experiments (Cohen & Welling, 2016)
def preprocess_data(train_data, test_data, train_labels, test_labels):
    train_mean = np.mean(train_data, axis=(0,1,2))
    train_data -= train_mean
    test_data -= train_mean
    train_std = np.std(train_data, axis=(0,1,2))
    train_data = train_data/train_std
    test_data = test_data/train_std
    train_data = train_data.astype(np.float32)
    test_data = test_data.astype(np.float32)
    train_labels = train_labels.astype(np.int32)
    test_labels = test_labels.astype(np.int32)
    # Return preprocessed dataset
    return train_data, test_data, train_labels, test_labels


def create_ImageFolder_structure(train_data, train_labels, test_data, test_labels, val_data, val_labels):
    from PIL import Image

    # train data
    path = 'data/train/'
    len_train = train_data.shape[0]
    for i in range(len_train):
        im = Image.fromarray(train_data[i])
        if train_labels[i] == 1:
            im.save(path + 'tumor/'+str(i)+'.png')
        if train_labels[i] == 0:
            im.save(path + 'no_tumor/' + str(i) + '.png')

    #val_data
    path = 'data/valid/'
    len_val = val_data.shape[0]
    for i in range(len_val):
        im = Image.fromarray(val_data[i])
        if val_labels[i] == 1:
            im.save(path + 'tumor/'+str(i)+'.png')
        if val_labels[i] == 0:
            im.save(path + 'no_tumor/' + str(i) + '.png')

    #test_data
    path = 'data/test/'
    len_test = test_data.shape[0]
    for i in range(len_test):
        im = Image.fromarray(test_data[i])
        if test_labels[i] == 1:
            im.save(path + 'tumor/'+str(i)+'.png')
        if test_labels[i] == 0:
            im.save(path + 'no_tumor/' + str(i) + '.png')


def get_std(path, mean=None):
    import os
    from PIL import Image

    # get mean
    if mean is None: mean = get_mean(path)

    # Running values
    sum_x_min_xmean = np.zeros(3, dtype=np.float64)
    N_pixels = 0
    # Get data
    list_dirs = os.listdir(path)
    for pth in list_dirs:
        list_imgs = os.listdir(path + '/' + pth)
        for img_path in list_imgs:
            img_path_ext = path + '/' + pth + '/' + img_path
            image = np.asarray(Image.open(img_path_ext)) / 255.
            sum_x_min_xmean = sum_x_min_xmean + np.sum(np.abs(image - mean)**2, axis=(0,1))
            N_pixels = N_pixels + (image.shape[0] * image.shape[1])

    std_dev = np.sqrt(sum_x_min_xmean / N_pixels)
    print('StdDev is:', std_dev)
    return std_dev

def get_mean(path):
    import os
    from PIL import Image

    # Running values
    sum = np.zeros(3, dtype=np.float64)
    N_pixels = 0
    # Get data
    list_dirs = os.listdir(path)
    for pth in list_dirs:
        list_imgs = os.listdir(path + '/' + pth)
        for img_path in list_imgs:
            img_path_ext = path + '/' + pth + '/' + img_path
            image = np.asarray(Image.open(img_path_ext)) / 255.
            sum = sum + np.sum(image, axis=(0,1))
            N_pixels = N_pixels + (image.shape[0] * image.shape[1])

    mean = sum / N_pixels
    print('Mean is:', mean)
    return mean








