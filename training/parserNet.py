import argparse

def get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('-ng', '--numberGraphics',
                        type=int,
                        help='Number of graphic cards which should be used.',
                        default=1)

    parser.add_argument('-dt', '--direction',
                        type=str,
                        help='States in which dircetion the training is done. Either NormalToAbnormal or AbnormalToNormal.',
                        default='NormalToAbnormal')

    parser.add_argument('-logn', '--logname',
                        type=str,
                        help='Name of the log on Wandb.ai',
                        default=None)

    parser.add_argument('-ln', '--lambdaNorm',
                        type=int,
                        help='Lambda Norm. Default is 10',
                        default=10)

    parser.add_argument('-sig', '--sigmoid',
                        type=bool,
                        help='After addition of the map and the input, the sigmoid is added. This prevents negative numbers in the output',
                        default=False)

    parser.add_argument('-pro', '--project',
                        type=str,
                        help='Projectname in WandB',
                        default=None)

    parser.add_argument('-dat', '--data_dir',
                        type=str,
                        help='folder of the data',
                        default=None)

    parser.add_argument('-mf', '--model_folder',
                        type=str,
                        help='directory in where models will be saved',
                        default='../Models/')
    
    parser.add_argument('-sf', '--save_folder', 
                        type=str,
                        help='folder in which the images will be saved',
                        default=None)

    parser.add_argument('-bs', '--batch_size',
                        type=int,
                        help='input batch size',
                        default=12)

    parser.add_argument('-nc', '--channels_number',
                        type=int,
                        help='input image channels',
                        default=3)

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

    parser.add_argument('--train',
                        type=int,
                        help='percentage of the dataset for training',
                        default=100)

    parser.add_argument('--eval',
                        type=int,
                        help='percentage of the dataset for evaluation',
                        default=0)
                        
    parser.add_argument('--test',
                        type=int,
                        help='percentage of the dataset for testing',
                        default=0)
    
    parser.add_argument('--encoder_name',
                        type=str,
                        help='name of the encoder. If None, the default UNet will be used',
                        default="vaganEncoder")
    
    parser.add_argument('--critic_name',
                        type=str,
                        help='name of the critic. If None, the default C3DFCN will be used',
                        default="C3DFCN")

    return parser