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
import experiments.rot_mnist.models as models
import experiments.rot_mnist.parser as parser
import experiments.rot_mnist.dataset as dataset
import experiments.rot_mnist.trainer as trainer
# logger
from experiments.logger import Logger


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
    args.weight_decay = 1e-4
    args.lr = 0.001
    args.optim = 'adam'
    args.batch_size = 128
    # Select model and define remaining parameters
    if args.model == 'z2cnn':
        model = models.Z2CNN()
        args.epochs = 300
    elif args.model == 'p4cnn':
        model = models.P4CNN()
        args.epochs = 100
    elif args.model == 'att_p4cnn':
        model = models.A_P4CNN()
        args.epochs = 100
    elif args.model == 'sp_att_p4cnn':
        # Just spatial attention
        model = models.A_Sp_P4CNN()
        args.epochs = 100
    elif args.model == 'ch_att_p4cnn':
        # Just channel attention
        model = models.A_Ch_P4CNN()
        args.epochs = 100
    elif args.model == 'f_att_p4cnn':
        model = models.fA_P4CNN()
        args.epochs = 100
    # REBUTTAL EXPERIMENTS
    elif args.model == 'RomHog_att_p4cnn':
        model = models.RomHog_fA_P4CNN()
        args.epochs = 100

    # Define the device to be used and move model to that device
    args.device = 'cuda:0' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    model.to(args.device)

    # Define transforms and create dataloaders
    dataloaders, test_loader = dataset.get_dataset(batch_size=args.batch_size, num_workers=4)

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
