
from pytorch_lightning import LightningModule
import torch
import torch.optim as optim
from torch import autograd
from imageSaver import Saver
import gc
from models.model_handler import init_model

class cfHistoGAN(LightningModule):
    def __init__(
        self,
        opt,
        LAMBDA=10,
    ):
        super().__init__()

        self.net_g, self.net_d = init_model(opt)

        self.optimizer_g, self.optimizer_d = self.init_optimizer(opt, self.net_g, self.net_d)

        self.opt = opt
        self.LAMBDA = LAMBDA
        self.LAMBDA_NORM = opt.lambdaNorm

        self.epoch = 0

        self.trainStep = 0

        self.losses = []
        self.G_mean_losses = []
        self.D_mean_losses = []

        self.first = True

        self.meansOut = []
        self.meansOtherClass = []
        self.firstTwo = True
        self.i = 0
        self.i2 = 0
        self.counter = 0

        self.countImages = 0

        #This property activates manual optimization.
        self.automatic_optimization = False

        self.frequencyD = 100

        self.outputs = [[], []]

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
        return self.optimizer_g, self.optimizer_d
        
    #function taken from: https://github.com/Adi-iitd/AI-Art/blob/6969e0a64bdf6a70aa741c29806fc148de93595c/src/CycleGAN/CycleGAN-PL.py#L680
    @staticmethod
    def set_requires_grad(nets, requires_grad = False):

        """
        Set requies_grad=False for all the networks to avoid unnecessary computations
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

    def trainStepD(self, inputToBeFaked, real, discriminator, generator, opt, LAMBDA, calc_grad=True, getMap=False):

        err_d_real = discriminator(real)

        # train with sum (anomaly + anomaly map)
        anomaly_map = generator(inputToBeFaked, sigmoid=opt.sigmoid)
        #anomaly_map = generator(inputToBeFaked)

        if opt.sigmoid:
            img_sum = anomaly_map
            anomaly_map = img_sum - inputToBeFaked
        else:
            img_sum = inputToBeFaked + anomaly_map
        
        err_d_anomaly_map = discriminator(img_sum)
        cri_loss = self.discriminatorLoss(err_d_real, err_d_anomaly_map)

        if calc_grad:
            cri_loss += self.calc_gradient_penalty(discriminator, inputToBeFaked, img_sum, LAMBDA)

        if getMap:
            return cri_loss, anomaly_map
        else:
            return cri_loss

    def trainStepG(self, inputImage, generator, discriminator, opt, LAMBDA, getMap=False):
    
        anomaly_map = generator(inputImage, sigmoid=opt.sigmoid)
        #anomaly_map = generator(inputImage)

        if opt.sigmoid:
            output = anomaly_map
            anomaly_map = output - inputImage
        else:
            output = anomaly_map + inputImage

        gen_loss = self.generatorLoss(anomaly_map, output, discriminator, LAMBDA)
        
        if getMap:
            return gen_loss, anomaly_map
        else:
            return gen_loss


    def training_step(self, batch, batch_idx):
        '''
        Run the trainig algorithm.
        '''

        normal, abnormal = batch

        if self.opt.direction == 'AbnormalToNormal':
            dataToFake = abnormal
            dataReal = normal

        elif self.opt.direction == 'NormalToAbnormal':
            dataToFake = normal
            dataReal = abnormal

        #count number of iterations at epoch 0
        if self.epoch == 0:
            self.i += 1
            
        self.trainStep += 1

        if self.trainStep % self.frequencyD == 0:
            ############################
            # Update G forward network
            ############################
            #disable discriminator parameters to prevent optimizations
            self.set_requires_grad([self.net_d], requires_grad = False)
            self.set_requires_grad([self.net_g], requires_grad = True)

            self.optimizer_g.zero_grad()
            err_g = self.trainStepG(dataToFake, self.net_g, self.net_d, self.opt, self.LAMBDA_NORM)
            self.manual_backward(err_g)
            self.optimizer_g.step()

            self.outputs[0].append({'loss': err_g.detach()})
        else:
            ############################
            # Update D backward network
            ############################
            #reactivate discriminator parameters
            self.set_requires_grad([self.net_g], requires_grad = False)
            self.set_requires_grad([self.net_d], requires_grad = True)

            self.optimizer_d.zero_grad()
            err_d = self.trainStepD(dataToFake, dataReal, self.net_d, self.net_g, self.opt, self.LAMBDA)
            self.manual_backward(err_d)
            self.optimizer_d.step()

            self.outputs[1].append({'loss': err_d.detach()})

        #save the images
        if self.epoch % 25 == 0:
            for i in range(len(dataToFake)):
                Saver.saveAsPng(
                    InputTensor=dataToFake[i],
                    MapTensor=self.net_g(dataToFake[i].unsqueeze(0)).squeeze(),
                    saveFolder=self.opt.save_folder,
                    epoch=self.epoch, 
                    imageNumber=str(self.countImages),
                    first=self.first,
                    sigmoid=self.opt.sigmoid,
                    mode='train')
                self.countImages += 1

        if self.epoch == 1:
            self.first = False

        torch.cuda.empty_cache()
        
    def on_train_epoch_end(self):

        self.i2 = 0
        self.countImages = 0
        
        self.epoch += 1

        #if the discriminator loss is empty, append 0
        if len(self.outputs[1]) == 0:
            self.outputs[1].append({'loss': torch.tensor(0.0).to(self.device)})
        #if the generator loss is empty, append 0
        if len(self.outputs[0]) == 0:
            self.outputs[0].append({'loss': torch.tensor(0.0).to(self.device)})

        avg_loss = sum([torch.stack([x['loss'] for x in self.outputs[i]]).mean().item() / 2 for i in range(2)])
        g_loss = sum([torch.stack([x['loss'] for x in self.outputs[0]]).mean().item()])
        d_loss = sum([torch.stack([x['loss'] for x in self.outputs[1]]).mean().item()])

        self.log('train/total_loss', avg_loss, on_epoch=True)
        self.log('train/g_mean_loss', g_loss, on_epoch=True)
        self.log('train/d_mean_loss', d_loss, on_epoch=True)

        self.losses.append(avg_loss)
        self.G_mean_losses.append(g_loss)
        self.D_mean_losses.append(d_loss)

        #change frequency after 25 epochs and in each 100 epoch
        if self.epoch == 25:
            self.frequencyD = 5
        if self.epoch % 99 == 0:
            self.frequencyD = 100
        if self.epoch % 100 == 0:
            self.frequencyD = 5

        torch.cuda.empty_cache()
        gc.collect()

        return None

    def validation_step(self, batch, batch_idx):
        normal, abnormal = batch

        if self.opt.direction == 'AbnormalToNormal':
            dataToFake = abnormal

        elif self.opt.direction == 'NormalToAbnormal':
            dataToFake = normal

        #save the images
        if self.epoch % 25 == 0:

            #export the first image to wandb
            Saver.saveToLogger(
                InputTensor=dataToFake[0],
                MapTensor=self.net_g(dataToFake[0].unsqueeze(0)).squeeze(),
                logger=self,
                epoch=self.epoch,
                sigmoid=self.opt.sigmoid,
                first=self.firstTwo)
            
            for i in range(len(dataToFake)):
                Saver.saveAsPng(
                    InputTensor=dataToFake[i],
                    MapTensor=self.net_g(dataToFake[i].unsqueeze(0)).squeeze(),
                    saveFolder=self.opt.save_folder,
                    epoch=self.epoch,
                    imageNumber=str(self.countImages),
                    first=self.firstTwo,
                    sigmoid=self.opt.sigmoid,
                    mode='eval')
                self.countImages += 1

        if self.epoch == 1:
            self.firstTwo = False

        torch.cuda.empty_cache()

        return None
    
    def test_step(self, batch, batch_idx):

        normal, abnormal = batch

        if self.opt.direction == 'AbnormalToNormal':
            dataToFake = abnormal

        elif self.opt.direction == 'NormalToAbnormal':
            dataToFake = normal

        #save the images
        for i in range(len(dataToFake)):
            Saver.saveAsPng(
                InputTensor=dataToFake[i],
                #MapTensor=self.net_g(dataToFake[i].unsqueeze(0), sigmoid=self.opt.sigmoid).squeeze().cpu().detach().numpy(),
                MapTensor=self.net_g(dataToFake[i].unsqueeze(0)).squeeze(),
                saveFolder=self.opt.save_folder,
                imageNumber=str(self.countImages),
                sigmoid=self.opt.sigmoid,
                mode='test')
            self.countImages += 1