import numpy as np
import csv
import os
from dataLoader import CRCPrep
from cfHistoGAN import VaGAN
from parserNet import get_parser

def getabsoluteChange(model, options, dataLoaderIter, dataLen):

    absoluteChange = None
    meanChange = None

    i = 0
    while i < dataLen:
        i += 1
        anomaly, healthy = dataLoaderIter.next()

        if options.direction == 'DifToCrohn':
            data = anomaly
        else:
            data = healthy
        
        anomalyMap = model.net_g(data, options.sigmoid)

        if options.sigmoid:
            anomalyMap = anomalyMap - data

        anomalyMapNumpy = anomalyMap.detach().numpy()

        map = anomalyMapNumpy[0,1:anomalyMapNumpy.shape[1]-4]
        if absoluteChange is None:
            absoluteChange = np.zeros(map.shape[0])
            meanChange = np.zeros(map.shape[0])
            for channel in range(map.shape[0]):
                entry = np.sum(np.absolute(map[channel]))                
                absoluteChange[channel] = entry
                meanChange[channel] = np.mean(map[channel])
        else:
            for channel in range(map.shape[0]):
                entry = np.sum(np.absolute(map[channel]))
                absoluteChange[channel] += entry
                meanChange[channel] += np.mean(map[channel])

    return absoluteChange, meanChange

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
            absoluteChange = None
            meanChange = None
            for net in netsPaths:
                dataLoaderIter = iter(trainGroup)

                if net != "last.ckpt":
                    networkPath = workingPath + netsDir + "/" + modelPath + "/" + net
                    model = VaGAN.load_from_checkpoint(checkpoint_path=networkPath, opt=options)
                    if absoluteChange is None:
                        absoluteChange, meanChange = getabsoluteChange(model, options, dataLoaderIter, len(trainGroup))
                    else:
                        l1Entry, meanEntry = getabsoluteChange(model, options, dataLoaderIter, len(trainGroup))

                        absoluteChange += l1Entry
                        meanChange += meanEntry
                        
            meanChange /= np.max(meanChange)
            absoluteChange /= np.max(absoluteChange)

            resultFolder = results + netsDir + "/"

            #write results to csv
            with open(resultFolder + "absoluteChangeAndMeanChange.csv", 'w') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(["Channel", "absoluteChange", "MeanChange"])
                for i in range(len(absoluteChange)):
                    writer.writerow([i+1, absoluteChange[i], meanChange[i]])

if __name__ == '__main__':
    main()