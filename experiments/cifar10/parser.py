import argparse


def _get_parser():
    # Running settings
    parser = argparse.ArgumentParser(description='cifar10 experiments.')
    # Parse
    parser.add_argument('--model', type=str, default='p4_allcnnc', metavar='M', help='type of model to use.')
    parser.add_argument('--lr', default=1e-3, type=float, metavar='LR', help='initial learning rate (default: 1e-3)')
    parser.add_argument('--weight_decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)')
    parser.add_argument("--device", type=str, default="cuda", help="Where to deploy the model {cuda, cpu}")
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pre-trained model. If false, the model will be trained.')
    parser.add_argument('--augment', default=False, action='store_true', help='If false, no data augmentation will be used.')
    parser.add_argument('--extra_comment', type=str, default="")

    # Return parser
    return parser


def parse_args():
    parser = _get_parser()
    return parser.parse_args()
