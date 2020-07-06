# surfsara
import sys
#sys.path.extend(['/scratch/'])  # uncomment line if in surfsara. add /scratch/ to dataset path as well.
# torch
import torch
import torch.nn as nn
# built-in
import copy
import numpy as np
import random
import os
# models
import experiments.pcam.models as models
import experiments.pcam.parser as parser
import experiments.pcam.dataset as dataset
import experiments.pcam.trainer as trainer, test
# logger
from experiments.logger import Logger

import torchvision.transforms as transforms
import torchvision.datasets as datasets

def main(args):
    # Parse arguments
    args = copy.deepcopy(args)

    # Fix seeds for reproducibility and comparability
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.benchmark = True

    # Select model and parameters. Both are inline with the baseline parameters.
    if 'densenet' in args.model:
        args.epochs = 100
        args.weight_decay = 1e-4
        args.optim = 'adam'
        args.batch_size = 64
        args.lr = 1e-3
        # Instantiate model
        model = models.DenseNet(n_channels=26)
        if args.model == 'p4_densenet':
            model = models.P4DenseNet(n_channels=13)
        elif args.model == 'f_att_p4_densenet':
            model = models.fA_P4DenseNet(n_channels=13)
        elif args.model == 'p4m_densenet':
            model = models.P4MDenseNet(n_channels=9)
        elif args.model == 'f_att_p4m_densenet':
            model = models.P4MDenseNet(n_channels=9)

    # Define the device to be used and move model to that device ( :0 required for multiGPU)
    args.device = 'cuda:0' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    # Check if multi-GPU available and if so, use the available GPU's
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)  # Required for multi-GPU
    model.to(args.device)

    # Define transforms and create dataloaders
    dataloaders, test_loader = dataset.get_dataset(batch_size=args.batch_size, augmentation=args.augment, num_workers=4)

    # Create model directory and instantiate args.path
    model_directory(args)

    # Train the model
    if not args.pretrained:
        # Create logger
        sys.stdout = Logger(args)
        # Print arguments (Sanity check)
        print(args)
        # Train the model
        import datetime
        print(datetime.datetime.now())
        trainer.train(model, dataloaders, test_loader, args)

    # Test the model
    if args.pretrained: model.load_state_dict(torch.load(args.path))
    trainer.test(model, test_loader, args.device)


def model_directory(args):
    # Create name from arguments
    comment = "model_{}_optim_{}_lr_{}_wd_{}_seed_{}/".format(args.model, args.optim, args.lr, args.weight_decay, args.seed)
    if args.extra_comment is not "": comment = comment[:-1] + "_" + args.extra_comment + comment[-1]
    # Create directory
    modeldir = "./saved/" + comment
    os.makedirs(modeldir, exist_ok=True)
    # Add the path to the args
    args.path = modeldir + "model.pth"


if __name__ == '__main__':
    main(parser.parse_args())




















