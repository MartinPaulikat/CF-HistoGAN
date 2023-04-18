import tifffile as tiff
import numpy as np
from PIL import Image
import os

#saves torch tensors as tiff or png at the experiment folder
class Saver:
    def saveAsPng(InputTensor, MapTensor, nameAddition, saveFolder):

        rgbInput = np.zeros((np.shape(InputTensor)[-2],np.shape(InputTensor)[-1],3), dtype=np.uint8)
        rgbInput[..., 0] = InputTensor[0, -3, ...] * 256
        rgbInput[..., 1] = InputTensor[0, -2, ...] * 256
        rgbInput[..., 2] = InputTensor[0, -1, ...] * 256

        imgIn = Image.fromarray(rgbInput)

        rgbMap = np.zeros((np.shape(InputTensor)[-2],np.shape(InputTensor)[-1],3), dtype=np.uint8)
        rgbMap[..., 0] = MapTensor[0, -3, ...] * 256
        rgbMap[..., 1] = MapTensor[0, -2, ...] * 256
        rgbMap[..., 2] = MapTensor[0, -1, ...] * 256

        imgMap = Image.fromarray(rgbMap)
        imgOut = Image.fromarray(rgbInput + rgbMap)

        #save the input only if it is the first call of this function
        path = saveFolder + '/input_samplesHE_' + str(nameAddition) + '.png'
        imgIn.save(path)
        path = saveFolder + '/map_samplesHE_' + str(nameAddition) + '.png'
        imgMap.save(path)
        path = saveFolder + '/output_samplesHE_' + str(nameAddition) + '.png'
        imgOut.save(path)

    def saveAsTiff(InputTensor, MapTensor, nameAddition, saveFolder):
        if not os.path.isdir(saveFolder):
            os.makedirs(saveFolder)
        imgIn = np.array(InputTensor)
        imgMap = np.array(MapTensor)
        imgOut = imgIn + imgMap

        #save the input only if it is the first call of this function
        path = saveFolder + '/input_samples_' + str(nameAddition) + '.tif'
        tiff.imsave(path, imgIn)
        path = saveFolder + '/map_samples_' + str(nameAddition) + '.tif'
        tiff.imsave(path, imgMap)
        path = saveFolder + '/output_samples_' + str(nameAddition) + '.tif'
        tiff.imsave(path, imgOut)

    def saveMaxIntensity(InputTensor, MapTensor, nameAddition, saveFolder):

        inp = np.max(np.array(InputTensor), axis=0)
        img = np.max(np.array(MapTensor), axis=0)

        imgAnomaly = Image.fromarray(img)
        imgOut = Image.fromarray(inp + img)
        imgIn = Image.fromarray(inp)

        path = saveFolder + '/max_map_samples_' + str(nameAddition) + '.tif'
        imgAnomaly.save(path)
        path = saveFolder + '/max_output_samples_' + str(nameAddition) + '.tif'
        imgOut.save(path)
        path = saveFolder + '/max_input_samples_' + str(nameAddition) + '.tif'
        imgIn.save(path)