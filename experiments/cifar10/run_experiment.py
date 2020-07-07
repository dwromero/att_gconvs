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
import experiments.cifar10.models as models
import experiments.cifar10.parser as parser
import experiments.cifar10.dataset as dataset
import experiments.cifar10.trainer as trainer
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
    if 'allcnnc' in args.model:
        args.epochs = 350
        args.batch_size = 128
        args.optim = 'sgd'
        args.weight_decay = 1e-3
        assert (args.lr in [0.01, 0.05, 0.1, 0.25]), "lr:({}) not used in the \'AllCNNC\' baseline. Must be one of [0.01, 0.05, 0.1, 0.25].".format(args.lr)
        # Instantiate model
        model = models.AllCNNC()
        # Check that learning rate in the set of learning rates used in the baseline
        # if args.model == 'att_p4_allcnnc':        # Non feassible due to CUDA memory requirements.
        #     model = models.A_P4AllCNNC()
        # if args.model == 'sp_att_p4_allcnnc':
        #     model = models.A_Sp_P4AllCNNC()
        # if args.model == 'ch_att_p4_allcnnc':
        #     model = models.A_Ch_P4AllCNNC()
        if args.model == 'p4_allcnnc':
            model = models.P4AllCNNC()
        if args.model == 'f_att_p4_allcnnc':
            model = models.fA_P4AllCNNC()
        if args.model == 'p4m_allcnnc':
            model = models.P4MAllCNNC()
        if args.model == 'f_att_p4m_allcnnc':
            model = models.fA_P4MAllCNNC()

    if 'resnet44' in args.model:
        args.epochs = 300
        args.batch_size = 128
        args.lr = 0.05
        args.optim = 'sgd'
        args.weight_decay = 0.0
        # Instantiate model
        model = models.ResNet()
        if args.model == 'p4m_resnet44':
            model = models.P4MResNet()
        if args.model == 'f_att_p4m_resnet44':
            model = models.fA_P4MResNet()

    # Define the device to be used and move model to that device ( :0 required for multiGPU)
    args.device = 'cuda:0' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    # Check if multi-GPU available and if so, use the available GPU's
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
    model = nn.DataParallel(model)      # Required for multi-GPU
    model.to(args.device)

    # Define transforms and create dataloaders
    dataloaders, test_loader = dataset.get_dataset(batch_size=args.batch_size, augmentation=args.augment, num_workers=4)

    # Create model directory and instantiate args.path
    model_directory(args)

    # Train the model
    if not args.pretrained:
        # Create logger
        # sys.stdout = Logger(args)
        # Print arguments (Sanity check)
        print(args)
        # Train the model
        import datetime
        print(datetime.datetime.now())
        trainer.train(model, dataloaders, args)

    # Test the model
    if args.pretrained: model.load_state_dict(torch.load(args.path))
    test(model, test_loader, args.device)


# ----------------------------------------------------------------------------
def test(model, test_loader, device):
    # send model to device
    model.eval()
    model.to(device)

    # Summarize results
    lbls = []
    pred = []
    correct = 0
    total = 0

    with torch.no_grad():
        # Iterate through data
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # Print results
    print('Accuracy of the network on the {} test images: {}'.format(total, (100 * correct / total)))
    # Return results
    return correct/total, lbls, pred


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
