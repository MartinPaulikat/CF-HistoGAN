import numpy as np
from PIL import Image
import os
import pytorch_lightning as pl
import wandb
import torchvision.transforms as transforms

#saves torch tensors as tiff or png at the experiment folder
class Saver(pl.LightningModule):
    def saveAsPng(InputTensor, MapTensor, saveFolder, imageNumber, mode, sigmoid, epoch=None, first=None):

        InputTensor = InputTensor.cpu().detach().numpy()
        MapTensor = MapTensor.cpu().detach().numpy()

        if sigmoid:
            map = (MapTensor - InputTensor)*255
            mapPos = map.copy()
            mapPos[mapPos < 0] = 0
            mapNeg = map.copy()
            mapNeg[mapNeg > 0] = 0
            #imgMapPos are all the values that are positive
            imgMapPos = Image.fromarray(mapPos.astype(np.uint8).transpose(1, 2, 0))
            imgMapNeg = Image.fromarray(mapNeg.astype(np.uint8).transpose(1, 2, 0))
            imgOut = Image.fromarray((MapTensor*255).astype(np.uint8).transpose(1, 2, 0))
        else:
            map = MapTensor*255
            mapPos = map.copy()
            mapPos[mapPos < 0] = 0
            mapNeg = map.copy()
            mapNeg[mapNeg > 0] = 0
            out = MapTensor + InputTensor
            out[out > 1] = 1
            out[out < 0] = 0
            imgMapPos = Image.fromarray(mapPos.astype(np.uint8).transpose(1, 2, 0))
            imgMapNeg = Image.fromarray(mapNeg.astype(np.uint8).transpose(1, 2, 0))
            imgOut = Image.fromarray((out*255).astype(np.uint8).transpose(1, 2, 0))
            
        imgIn = Image.fromarray((InputTensor*255).astype(np.uint8).transpose(1, 2, 0))

        if mode == 'train' or mode == 'eval':
            #save the input only if it is the first call of this function
            if first:
                os.makedirs(saveFolder + mode + '/input/', exist_ok=True)
                path = saveFolder + mode + '/input/' + imageNumber + '.png'
                imgIn.save(path)
            os.makedirs(saveFolder + mode + '/' + str(epoch) + '/out', exist_ok=True)
            os.makedirs(saveFolder + mode + '/' + str(epoch) + '/mapPos', exist_ok=True)
            os.makedirs(saveFolder + mode + '/' + str(epoch) + '/mapNeg', exist_ok=True)
            path = saveFolder + mode + '/' + str(epoch) + '/mapPos/' + imageNumber + '.png'
            imgMapPos.save(path)
            path = saveFolder + mode + '/' + str(epoch) + '/mapNeg/' + imageNumber + '.png'
            imgMapNeg.save(path)
            path = saveFolder + mode + '/' + str(epoch) + '/out/' + imageNumber + '.png'
            imgOut.save(path)
        else:
            #create the test folder
            os.makedirs(saveFolder + 'test/input', exist_ok=True)
            os.makedirs(saveFolder + 'test/mapPos', exist_ok=True)
            os.makedirs(saveFolder + 'test/mapNeg', exist_ok=True)
            os.makedirs(saveFolder + 'test/out', exist_ok=True)

            #save the input only if it is the first call of this function
            path = saveFolder + 'test/input/' + imageNumber + '.png'
            imgIn.save(path)
            path = saveFolder + 'test/mapPos/' + imageNumber + '.png'
            imgMapPos.save(path)
            path = saveFolder + 'test/mapNeg/' + imageNumber + '.png'
            imgMapNeg.save(path)
            path = saveFolder + 'test/out/' + imageNumber + '.png'
            imgOut.save(path)

    def saveToLogger(InputTensor, MapTensor, logger, epoch, sigmoid, first=None):
        #downsample the images to 256x256 using torchvision.transforms.Resize
        downsample = transforms.Resize([256, 256])
        InputTensor = downsample(InputTensor)
        MapTensor = downsample(MapTensor)

        InputTensor = InputTensor.cpu().detach().numpy()
        MapTensor = MapTensor.cpu().detach().numpy()        

        if sigmoid:
            imgOut = Image.fromarray((MapTensor*255).astype(np.uint8).transpose(1, 2, 0))
            imgMap = Image.fromarray(((MapTensor - InputTensor)*255).astype(np.uint8).transpose(1, 2, 0))
        else:
            output = MapTensor + InputTensor
            output[output > 1] = 1
            output[output < 0] = 0

            imgMap = Image.fromarray((MapTensor*255).astype(np.uint8).transpose(1, 2, 0))
            imgOut = Image.fromarray((output*255).astype(np.uint8).transpose(1, 2, 0))
            
        imgIn = Image.fromarray((InputTensor*255).astype(np.uint8).transpose(1, 2, 0))
        
        if first:
            logger.logger.experiment.log({"input_sample": wandb.Image(imgIn)}, step=epoch)
        logger.logger.experiment.log({"map_sample": wandb.Image(imgMap)}, step=epoch)
        logger.logger.experiment.log({"output_sample": wandb.Image(imgOut)}, step=epoch)
