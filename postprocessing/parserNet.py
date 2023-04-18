
import argparse


def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-tit', '--title',
                        type=str,
                        help='title of plot',
                        default=None)
    parser.add_argument('-exp', '--experiment',
                        type=str,
                        help='directory in where samples and models will be saved',
                        default='../experiment')

    parser.add_argument('-dt', '--direction',
                        type=str,
                        help='States in which dircetion the training is done. Either DifToCrohn or CrohnToDif',
                        default='CrohnToDif')

    parser.add_argument('-ch', '--channel',
                        type=int,
                        help='Only use with channel size 1. States which channel is used.',
                        default=None)

    parser.add_argument('-ep', '--epoch',
                        type=int,
                        help='Only use with a pretrained network. States at which epoch to continue the learning process.',
                        default=0)

    parser.add_argument('-sig', '--sigmoid',
                        type=bool,
                        help='After addition of the map and the input, the sigmoid is added. This prevents negative numbers in the output',
                        default=False)

    parser.add_argument('-tor', '--torch',
                        type=str,
                        help='Using .pt data instead of .tif data',
                        default=False)

    parser.add_argument('-ln', '--lambdaNorm',
                        type=int,
                        help='Lambda Norm. Default is 100',
                        default=100)

    parser.add_argument('-dat', '--data',
                        type=str,
                        help='folder of the preprocessed data',
                        default='/home/baumgartner/mpaulikat71/Code/vagan-pytorch-martin/src/data/Preprocessed')

    parser.add_argument('-pre', '--pretrained',
                        type=str,
                        help='folder of the pretrained network',
                        default=None)

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        help='input batch size',
                        default=1)

    parser.add_argument('-nc', '--channels_number',
                        type=int,
                        help='input image channels',
                        default=61)

    parser.add_argument('-ngf', '--num_filters_g',
                        type=int,
                        help='number of filters for the first layer of the generator',
                        default=16)

    parser.add_argument('-ndf', '--num_filters_d',
                        type=int,
                        help='number of filters for the first layer of the discriminator',
                        default=16)

    parser.add_argument('-nep', '--nepochs',
                        type=int,
                        help='number of epochs to train for',
                        default=400)

    parser.add_argument('-dit', '--d_iters',
                        type=int,
                        help='number of discriminator iterations per each generator iter, default=5',
                        default=5)

    parser.add_argument('-lrG', '--learning_rate_g',
                        type=float,
                        help='learning rate for generator, default=1e-3',
                        default=1e-3)

    parser.add_argument('-lrD', '--learning_rate_d',
                        type=float,
                        help='learning rate for discriminator, default=1e-3',
                        default=1e-3)

    parser.add_argument('-b1', '--beta1',
                        type=float,
                        help='beta1 for adam. default=0.0',
                        default=0.0)

    parser.add_argument('-seed', '--manual_seed',
                        type=int,
                        help='input for the manual seeds initializations',
                        default=7)

    parser.add_argument('--cuda',
                        action='store_true',
                        help='enables cuda',
                        default=True)

    parser.add_argument('--train',
                        type=int,
                        help='percentage of the dataset for training',
                        default=70)

    parser.add_argument('--eval',
                        type=int,
                        help='percentage of the dataset for evaluation',
                        default=15)
                        
    parser.add_argument('--test',
                        type=int,
                        help='percentage of the dataset for testing',
                        default=15)

    return parser
