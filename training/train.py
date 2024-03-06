from __future__ import print_function
import os

import torch
import torch.backends.cudnn as cudnn
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning import Trainer, seed_everything

from parserNet import get_parser
from dataLoader import lightningLoader
from cfHistoGAN import cfHistoGAN

def init_seed(opt):
        '''
        Disable cudnn to maximize reproducibility
        '''
        torch.cuda.cudnn_enabled = False
        torch.manual_seed(opt.manual_seed)
        torch.cuda.manual_seed(opt.manual_seed)
        cudnn.benchmark = True


def main():
    #important to save things on the slurm server. Delete the following line if you dont use slurm
    del os.environ["SLURM_JOB_NAME"]

    options = get_parser().parse_args()

    if options.sigmoid == 'False':
        options.sigmoid = False

    #you can also use Tensorboard or other by pythorch lightning supported logger
    logger = WandbLogger(project=options.project, name=options.logname)
    init_seed(options)

    seed_everything(42, workers=True)

    dataloader = lightningLoader(data_dir=options.data_dir, batch_size=options.batch_size)

    model = cfHistoGAN(options)

    #create the save folder if it does not exist
    os.makedirs(options.save_folder, exist_ok=True)

    trainer = Trainer(
        logger=logger,
        max_epochs=options.nepochs,
        num_nodes=options.numberGraphics,
        enable_checkpointing=True,
        reload_dataloaders_every_n_epochs=True,
        num_sanity_val_steps=0,
        limit_val_batches=None,
    )

    #train the network
    trainer.fit(model, datamodule=dataloader)
    trainer.save_checkpoint(options.model_folder + options.logname + '.ckpt')
    trainer.test(model, datamodule=dataloader)

if __name__ == '__main__':
    main()