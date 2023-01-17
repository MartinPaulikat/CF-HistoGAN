from __future__ import print_function
import os
import shutil

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint

from parserNet import get_parser
from dataLoader import lightningLoader
from cfHistoGAN import cfHistoGAN
from pytorch_lightning.callbacks import LearningRateMonitor

def init_seed(opt):
        '''
        Disable cudnn to maximize reproducibility
        '''
        torch.cuda.cudnn_enabled = False
        torch.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed(opt.manual_seed)
        cudnn.benchmark = True


def init_experiment(opt):
    if opt.experiment is None:
        opt.experiment = '../samples'
    try:
        shutil.rmtree(opt.experiment)
    except:
        pass
    if not os.path.isdir(opt.experiment):
        os.makedirs(opt.experiment)
    if not os.path.isdir(opt.experiment + '/images'):
        os.makedirs(opt.experiment + '/images')
    if not os.path.isdir(opt.experiment + '/models'):
        os.makedirs(opt.experiment + '/models')


def main():
    #important to save things on the slurm server. Delete the follwoing line if you dont use slurm
    del os.environ["SLURM_JOB_NAME"]

    options = get_parser().parse_args()

    #you can also use Tensorboard or other by pythorch lightning supported logger
    logger = WandbLogger(project=options.project, name=options.logname)
    lr_monitor = LearningRateMonitor(logging_interval='epoch')
    if not options.pretrained:
        init_experiment(options)
        init_seed(options)

    seed_everything(42, workers=True)

    dataloader = lightningLoader(options, options.data)

    model = cfHistoGAN(options)

    #model saver
    checkpoint_callback = ModelCheckpoint(
        dirpath=options.experiment + '/models',
        save_last=True,
        save_top_k=-1,
        every_n_epochs=25,
    )

    trainer = Trainer(
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        max_epochs=options.nepochs,
        gpus=options.numberGraphics,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
    )

    #train the network
    trainer.fit(model, datamodule=dataloader)

if __name__ == '__main__':
    main()