from asyncore import write
import os
from dataLoader import CRCPrep
from cfHistoGAN import VaGAN
from parserNet import get_parser
import torch
import numpy as np
import csv
from scipy import stats

def statTests(meansIn, meansOut, meansOtherClass, name):
    unpairedResult = stats.ttest_ind(meansIn, meansOtherClass)
    pairedResult = stats.ttest_rel(meansIn, meansOut)

    with open(name, 'w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['', 'Unpaired p-value', 'Paired p-value'])
        for channel in range(meansIn.shape[1]):
            writer.writerow([channel+1, unpairedResult[1][channel], pairedResult[1][channel]])

def combineNets(data, anomalyMapNumpy):
    #calculate the mean of all the maps
    meanMap = np.mean(anomalyMapNumpy, axis=0)
    #calculate the mean of all data
    meanData = np.mean(data, axis=0)

    return meanMap, meanData

def processVAGAN(models, options, dataLoaderIter, dataLen, resultFolder):

    #save the means
    meansIn = np.zeros((dataLen, options.channels_number))
    meansOut = np.zeros((dataLen, options.channels_number))
    meansOtherClass = np.zeros((dataLen, options.channels_number))

    i = 0
    while i < dataLen:
        i += 1
        anomaly, healthy = dataLoaderIter.next()

        if options.direction == 'DifToCrohn':
            data = anomaly
            otherClassdata = healthy.detach().numpy()
        else:
            data = healthy
            otherClassdata = anomaly.detach().numpy()

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

        combineNets(dataConc, anomalyMapConc)

        #save the means for:
        for channel in range(anomalyMapNumpy.shape[1]):
            #   - input
            meansIn[i, channel] = np.mean(dataNumpy[0, channel])
            #   - output
            meansOut[i, channel] = np.mean(dataNumpy[0, channel] + anomalyMapNumpy[0, channel])
            #   - other class
            meansOtherClass[i, channel] = np.mean(otherClassdata[0, channel])

        i += 1
    
    #call the stats tests function
    statTests(meansIn, meansOut, meansOtherClass, resultFolder + "statsTests.csv")

def main():
    options = get_parser().parse_args()

    dataCRC = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCrcFull/"
    dataCTCL = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCtclFull/"
    workingPath = "../"
    modelPath = "models/"
    results = "../results/"

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