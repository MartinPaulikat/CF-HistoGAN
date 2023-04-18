import tifffile as tiff
import numpy as np
import os
from dataLoader import CRCPrep
from cfHistoGAN import VaGAN
from parserNet import get_parser


def createPlot(data, name):
    he = data[-3:,...]
    path = name + ".tif"
    tiff.imwrite(path, he)

def createColoredChannels(input, map, outputName, folder):

    workingMap = map.copy()
    output = input + map

    createPlot(input, folder + outputName + "Input")

    #do the coloring for the output image
    #remove the last 3 channels as they are HE

    createPlot(output, folder + outputName + "Output")

    #do here the same stuff for the maps
    #split the map into negative and positive values

    mapPositive = workingMap.copy()
    mapNegative = workingMap.copy()
    mapPositive[mapPositive < 0] = 0
    mapNegative[mapNegative > 0] = 0
    mapNegative *= -1

    #create the plots for the maps
    createPlot(mapPositive, folder + outputName + "MapPositive")
    createPlot(mapNegative, folder + outputName + "MapNegative")

def combineNets(data, anomalyMapNumpy, name, folder):

    #calculate the mean of all the maps
    meanMap = np.mean(anomalyMapNumpy, axis=0)

    #calculate the mean of all data
    meanData = np.mean(data, axis=0)

    createColoredChannels(meanData, meanMap, name, folder)



def processVAGAN(models, options, dataLoaderIter, dataLen, resultFolder):
    i = 0
    while i < dataLen:
        i += 1
        anomaly, healthy = dataLoaderIter.next()

        if options.direction == 'DifToCrohn':
            data = anomaly
        else:
            data = healthy

        anomalyMapConc = None
        dataConc = None
        
        for model in models:
            anomalyMap = model.net_g(data, options.sigmoid)

            if options.sigmoid:
                anomalyMap = anomalyMap - data

            anomalyMapNumpy = anomalyMap.detach().numpy()
            dataNumpy = data.detach().numpy()
            if anomalyMapConc is None:
                anomalyMapConc = anomalyMapNumpy
                dataConc = dataNumpy
            else:
                anomalyMapConc = np.concatenate((anomalyMapConc, anomalyMapNumpy), axis=0)
                dataConc = np.concatenate((dataConc, dataNumpy), axis=0)

        combineNets(dataConc, anomalyMapConc, str(i), resultFolder)


def main():
    options = get_parser().parse_args()

    dataCRC = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCrcFull/"
    dataCTCL = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCtclFull/"
    workingPath = "../"
    modelPath = "models/"
    results = "../resultsHE/"

    netsFolder = os.listdir(workingPath)
    for netsDir in netsFolder:
        if netsDir != "scripts" and netsDir != "results" and netsDir != "csvs" and netsDir!= "resultsHE":
            if netsDir == "DTC" or netsDir == "CTD":
                options.data = dataCRC
                options.channels_number = 61
                
            else:
                options.data = dataCTCL
                options.channels_number = 62

            if netsDir == "DTC" or netsDir == "NTR":
                options.direction = "DifToCrohn"
            elif netsDir == "CTD" or netsDir == "RTN":
                options.direction = "CrohnToDif"

            crcObject = CRCPrep(options.data, options.train, options.eval, options.test, batchsize=options.batch_size, useTorch=options.torch)
            trainGroup, _, _ = crcObject.returnLoaders()

            netsPaths = os.listdir(workingPath + netsDir + "/" + modelPath)
            models = []
            for net in netsPaths:
                dataLoaderIter = iter(trainGroup)

                if net != "last.ckpt":
                    networkPath = workingPath + netsDir + "/" + modelPath + "/" + net
                    model = VaGAN.load_from_checkpoint(checkpoint_path=networkPath, opt=options)
                    models.append(model)

            dataLoaderIter = iter(trainGroup)

            resultFolder = results + netsDir + "/"

            processVAGAN(models, options, dataLoaderIter, len(trainGroup), resultFolder)


if __name__ == '__main__':
    main()