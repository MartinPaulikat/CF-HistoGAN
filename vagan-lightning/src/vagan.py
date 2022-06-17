from tkinter import ON
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from torch import autograd
from models.critics import C3DFCN
from models.mask_generators import UNet
import tifffile as tiff
from imageSaver import Saver
import numpy as np
import csv

class VaGAN(LightningModule):
    def __init__(
        self,
        opt,
        LAMBDA=10
    ):
        super().__init__()
        self.net_g, self.net_d = self.init_model(opt)
        
        self.optimizer_g, self.optimizer_d = self.init_optimizer(opt, self.net_g, self.net_d)

        self.net_g.apply(self.weights_init)
        self.net_d.apply(self.weights_init)

        self.opt = opt
        self.LAMBDA = LAMBDA
        self.LAMBDA_NORM = opt.lambdaNorm

        self.step = 0
        self.trainStep = 0

        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []

        self.first = True

    #################
    ##Init funtions##
    #################

    def weights_init(self, m):
        '''
        Initialize cnn weithgs.
        '''
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            m.weight.data = torch.nn.init.kaiming_normal_(m.weight.data, 2)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

    def init_model(self, opt):
        '''
        Initialize generator and disciminator
        '''
        net_g = UNet(opt.channels_number, opt.num_filters_g)
        net_d = C3DFCN(opt.channels_number, opt.num_filters_d)
        return net_g, net_d

    def init_optimizer(self, opt, net_g, net_d):
        '''
        Initialize optimizers
        '''
        optimizer_g = optim.Adam(net_g.parameters(), lr=opt.learning_rate_g, betas=(
            opt.beta1, 0.9), weight_decay=1e-5)
        optimizer_d = optim.Adam(net_d.parameters(), lr=opt.learning_rate_d, betas=(
            opt.beta1, 0.9), weight_decay=1e-5)

        return optimizer_g, optimizer_d

    def configure_optimizers(self):
        '''
        gen_sch = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(self.optimizer_g, [400, 800], 0.1),
            'interval': 'epoch'
        }
        disc_sch = {
            'scheduler': torch.optim.lr_scheduler.MultiStepLR(self.optimizer_d, [400, 800], 0.1),
            'interval': 'epoch'
        }
        
        return (
            {'optimizer':self.optimizer_g, 'frequency': 1, 'lr_scheduler':gen_sch},
            {'optimizer':self.optimizer_d, 'frequency': 100, 'lr_scheduler':disc_sch},
        )
        '''
        
        return (
            {'optimizer':self.optimizer_g, 'frequency': 1},
            {'optimizer':self.optimizer_d, 'frequency': 100},
        )
        

    #function taken from: https://github.com/Adi-iitd/AI-Art/blob/6969e0a64bdf6a70aa741c29806fc148de93595c/src/CycleGAN/CycleGAN-PL.py#L680
    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=Fasle for all the networks to avoid unnecessary computations
        Parameters:
            nets (network list)   -- a list of networks
            requires_grad (bool)  -- whether the networks require gradients or not
        """

        if not isinstance(nets, list): nets = [nets]
        for net in nets:
            for param in net.parameters():
                param.requires_grad = requires_grad

    #################
    ##Loss funtions##
    #################

    def generatorLoss(self, map, fake, discriminator, LAMBDA):

        loss = discriminator(fake).mean()

        gen_loss = torch.abs(map).mean() 
        
        totalLoss = loss + LAMBDA*gen_loss

        return totalLoss

    def discriminatorLoss(self, real, fake):
        return real.mean() - fake.mean()

    def calc_gradient_penalty(self, discriminator, real_data, fake_data, LAMBDA):
        '''
        Calculate gradient penalty as in  "Improved Training of Wasserstein GANs"
        https://github.com/caogang/wgan-gp
        '''
        bs, ch, h, w = real_data.shape

        alpha = torch.rand(bs, 1, device=self.device)
        alpha = alpha.expand(bs, int(real_data.nelement()/bs)).contiguous().view(bs, ch, h, w)


        interpolates = torch.tensor(alpha * real_data + ((1 - alpha) * fake_data), device=self.device, requires_grad=True)

        disc_interpolates = discriminator(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                grad_outputs=torch.ones(disc_interpolates.size(), device=self.device),
                                create_graph=True, retain_graph=True, only_inputs=True)[0]
        gradients = gradients.view(gradients.size(0), -1)

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * LAMBDA
        return gradient_penalty

    def trainStepD(self, inputToBeFaked, real, discriminator, generator, opt, LAMBDA, calc_grad=True):

        err_d_real = discriminator(real)

        # train with sum (anomaly + anomaly map)
        anomaly_map = generator(inputToBeFaked, sigmoid=opt.sigmoid)

        if opt.sigmoid:
            img_sum = anomaly_map
        else:
            img_sum = inputToBeFaked + anomaly_map
        
        err_d_anomaly_map = discriminator(img_sum)
        cri_loss = self.discriminatorLoss(err_d_real, err_d_anomaly_map)

        if calc_grad:
            cri_loss += self.calc_gradient_penalty(discriminator, inputToBeFaked, img_sum, LAMBDA)

        return cri_loss

    def trainStepG(self, inputImage, net_g, discriminator, opt, LAMBDA):

        anomaly_map = net_g(inputImage, sigmoid=opt.sigmoid)

        if opt.sigmoid:
            output = anomaly_map
            anomaly_map = output - inputImage
        else:
            output = anomaly_map + inputImage

        gen_loss = self.generatorLoss(anomaly_map, output, discriminator, LAMBDA)

        return gen_loss

    def training_step(self, batch, batch_idx, optimizer_idx):
        '''
        Run the trainig algorithm.
        '''

        anomaly, healthy = batch

        if self.opt.direction == 'CrohnToDif':
            dataToFake = healthy
            dataReal = anomaly

        elif self.opt.direction == 'DifToCrohn':
            dataToFake = anomaly
            dataReal = healthy

        if optimizer_idx == 0:
            ############################
            # Update G forward network
            ############################
            #disable discriminator parameters to prevent optimizations
            self.set_requires_grad([self.net_d], requires_grad = False)
            self.set_requires_grad([self.net_g], requires_grad = True)

            #forward generator
            err_g = self.trainStepG(dataToFake, self.net_g, self.net_d, self.opt, self.LAMBDA_NORM)

            return {'loss': err_g}

        elif optimizer_idx == 1:
            ############################
            # Update D backward network
            ############################
            #reactivate discriminator parameters
            self.set_requires_grad([self.net_g], requires_grad = False)
            self.set_requires_grad([self.net_d], requires_grad = True)

            #forward discriminator
            err_d  = self.trainStepD(dataToFake, dataReal, self.net_d, self.net_g, self.opt, self.LAMBDA)

            self.trainStep += 1

            return {'loss': err_d}

    def training_epoch_end(self, outputs):
        
        self.step += 1

        avg_loss = sum([torch.stack([x['loss'] for x in outputs[i]]).mean().item() / 2 for i in range(2)])
        g_loss = sum([torch.stack([x['loss'] for x in outputs[0]]).mean().item()])
        d_loss = sum([torch.stack([x['loss'] for x in outputs[1]]).mean().item()])

        self.log('train/total_loss', avg_loss, on_epoch=True)
        self.log('train/g_mean_loss', g_loss, on_epoch=True)
        self.log('train/d_mean_loss', d_loss, on_epoch=True)

        

        self.losses.append(avg_loss)
        self.G_mean_losses.append(g_loss)
        self.D_mean_losses.append(d_loss)

        #change frequency after 25 epochs and in each 100 epoch
        if self.step == 25:
            self.trainer.optimizer_frequencies = [1,5]
        if self.step % 99 == 0:
            self.trainer.optimizer_frequencies = [1, 100]
        if self.step % 100 == 0:
            self.trainer.optimizer_frequencies = [1,5]

        #visualise the networks on full images
        if self.opt.oneImage and self.step % 25 == 0:
            if self.opt.torch:
                oneImage = torch.load(self.opt.oneImage, device=self.device)
            else:
                oneImage = torch.from_numpy(tiff.imread(self.opt.oneImage)).to(device=self.device)

            oneImage = torch.reshape(oneImage, (1, self.opt.channels_number, np.shape(oneImage)[-2], np.shape(oneImage)[-1]))
            oneImageMap = self.net_g(oneImage, sigmoid=self.opt.sigmoid)

            if self.opt.sigmoid:
                oneImageMap = oneImageMap - oneImage
            
            #save these tensors on to the server
            Saver.saveHEAsTiff(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'forward', self.first)
            Saver.saveAsTiff(oneImage.cpu().detach().numpy(), oneImageMap.cpu().detach().numpy(), self.opt.experiment + '/images', self.step, 'forward', self.first)

            #print the Disc losses as csv with corresponding epoch. Only for experiments
            csvFile = open(self.opt.experiment + '/discLoss.csv' , 'a')
            writer = csv.writer(csvFile)
            writer.writerow([self.step, d_loss])

            if self.first:
                self.first = False

        return None

    def validation_step(self, batch, batch_idx):
        
        anomaly, healthy = batch

        #forward generator
        err_g = self.trainStepG(anomaly, self.net_g, self.net_d, self.opt, self.LAMBDA_NORM)

        #forward discriminator
        err_d  = self.trainStepD(anomaly, healthy, self.net_d, self.net_g, self.opt, self.LAMBDA, calc_grad=False)


        val_avg_loss = (err_d + err_g) / 2

        self.log('val/total_loss', val_avg_loss, on_epoch=True)
        self.log('val/g_loss', err_g, on_epoch=True)
        self.log('val/d_loss', err_d, on_epoch=True)