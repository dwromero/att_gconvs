import argparse


def _get_parser():
    # Running settings
    parser = argparse.ArgumentParser(description='rot-mnist experiments.')
    # Parse
    parser.add_argument('--model', type=str, default='z2cnn', metavar='M', help='type of model to use {z2cnn, p4cnn, attp4cnn}')
    parser.add_argument("--device", type=str, default="cuda", help="Where to deploy the model {cuda, cpu}")
    parser.add_argument('--seed', type=int, default=0, metavar='S', help='random seed (default: 0)')
    parser.add_argument('--pretrained', default=False, action='store_true', help='use pre-trained model. If false, the model will be trained.')
    parser.add_argument('--extra_comment', type=str, default="")
    # Return parser
    return parser


def parse_args():
    parser = _get_parser()
    return parser.parse_args()
