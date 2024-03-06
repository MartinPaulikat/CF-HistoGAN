from torchvision.models import efficientnet_b5, efficientnet_b4, efficientnet_b3, efficientnet_b2, efficientnet_b1, efficientnet_b0
from torchvision.models import resnet101, resnet50, resnet34, resnet18
#import C3DFCN from the file critics in the same directory
from models.critics import C3DFCN
from models.mask_generators import UNet
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

class HistoGANUnet(smp.Unet):
    def __init__(self, activation='tanh', *args, **kwargs):
        super(HistoGANUnet, self).__init__(*args, **kwargs)
        if activation == 'sigmoid':
            self.final_activation = nn.Sigmoid()
        elif activation == 'tanh':
            self.final_activation = nn.Tanh()
        else:
            raise ValueError("Unsupported activation function. Choose 'sigmoid' or 'tanh'")

    def forward(self, x):
        x_original = x
        x = super(HistoGANUnet, self).forward(x)
        
        if self.final_activation == nn.Tanh:
            output = x
        elif self.final_activation == nn.Sigmoid:
            output = x + x_original
        output = x + x_original
        
        return self.final_activation(output)

def init_model(opt):
    '''
    Initialize generator and disciminator
    '''

    activation = 'sigmoid' if opt.sigmoid else 'tanh'

    if opt.encoder_name == "vaganEncoder":
        net_g = UNet(opt.channels_number, opt.num_filters_g)
    else:
        net_g = HistoGANUnet(
            encoder_name=opt.encoder_name,
            in_channels=opt.channels_number,
            classes=opt.channels_number,
            activation=activation)

    if opt.critic_name == 'C3DFCN':
        net_d = C3DFCN(opt.channels_number, opt.num_filters_d)
    elif opt.critic_name == 'resnet101':
        net_d = resnet101()
        net_d.fc = torch.nn.Linear(2048, 1)
    elif opt.critic_name == 'resnet50':
        net_d = resnet50()
        net_d.fc = torch.nn.Linear(2048, 1)
    elif opt.critic_name == 'resnet34':
        net_d = resnet34()
        net_d.fc = torch.nn.Linear(512, 1)
    elif opt.critic_name == 'resnet18':
        net_d = resnet18()
        net_d.fc = torch.nn.Linear(512, 1)
    elif opt.critic_name == 'efficientnet-b5':
        net_d = efficientnet_b5()
        net_d.classifier[-1] = torch.nn.Linear(2048, 1)
    elif opt.critic_name == 'efficientnet-b4':
        net_d = efficientnet_b4()
        net_d.classifier[-1] = torch.nn.Linear(1792, 1)
    elif opt.critic_name == 'efficientnet-b3':
        net_d = efficientnet_b3()
        net_d.classifier[-1] = torch.nn.Linear(1536, 1)
    elif opt.critic_name == 'efficientnet-b2':
        net_d = efficientnet_b2()
        net_d.classifier[-1] = torch.nn.Linear(1408, 1)
    elif opt.critic_name == 'efficientnet-b1':
        net_d = efficientnet_b1()
        net_d.classifier[-1] = torch.nn.Linear(1280, 1)
    elif opt.critic_name == 'efficientnet-b0':
        net_d = efficientnet_b0()
        net_d.classifier[-1] = torch.nn.Linear(1280, 1)
        
    return net_g, net_d
