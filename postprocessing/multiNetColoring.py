import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import tifffile as tiff
import csv
from matplotlib.colors import ListedColormap
from matplotlib import colors
import os
from dataLoader import CRCPrep
from cfHistoGAN import VaGAN
from parserNet import get_parser
import torch

def getabsoluteChange(model, options, dataLoaderIter, dataLen):

    absoluteChange = None

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
            for channel in range(map.shape[0]):
                entry = np.sum(np.absolute(map[channel]))
                absoluteChange[channel] = entry
        else:
            for channel in range(map.shape[0]):
                entry = np.sum(np.absolute(map[channel]))
                absoluteChange[channel] += entry

    absoluteChange /= dataLen

    return absoluteChange

def getTopChannels(absoluteChange, N):

    topList = []
    topListChannels = []

    for channel in range(absoluteChange.shape[0]):
        entry = absoluteChange[channel]
        if len(topList) < N:
            topList.append(entry)
            topListChannels.append(channel + 2)
        else:
            if min(topList) < entry:
                topList[topList.index(min(topList))] = entry
                topListChannels[topList.index(min(topList))] = channel + 2

    return topListChannels

def processVAGAN(models, options, dataLoaderIter, absoluteChange, dataLen, resultFolder, csvFileName):
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

        combineNets(dataConc, anomalyMapConc, absoluteChange, str(i), resultFolder, csvFileName)

def combineNets(data, anomalyMapNumpy, absoluteChange, name, folder, csvFileName):

    #calculate the mean of all the maps
    meanMap = np.mean(anomalyMapNumpy, axis=0)

    #calculate the mean of all data
    meanData = np.mean(data, axis=0)

    createColoredChannels(meanData, meanMap, csvFileName, name, folder, absoluteChange)

def createPlot(data, topListChannels, saveName, colorNames, labels, threshold, multiplicator):

    N = len(colorNames)

    #create own colormap
    cmap = ListedColormap(colorNames)
    chosencolors = np.zeros((N,4))
    counter = 0
    for colorName in colorNames:
        chosencolors[counter] = colors.to_rgba(colorName)
        counter += 1

    bounds = np.linspace(0, N, N+1)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)

    # Assemble image
    #change this so that N is the number of plots
    #create 8 subplots
    fig, axs = plt.subplots(2, 4, figsize=(40,20), dpi=200)
    fig.subplots_adjust(wspace=0, hspace=0)
    for i in range(N):
        channel = data[..., int(topListChannels[i]-1)]
        channel[channel < threshold] = 0
        #set pixels under threshhold to 0 to get less noise
        c = chosencolors[None, None, i, :]
        c[-1] *= multiplicator  # reduce alpha channel to account for overlapping channels
        channel_rgba = channel[..., None] * np.tile(c, (data.shape[0], data.shape[1], 1))
        axs[0, 0].imshow(channel_rgba)

        #x=[1,0,1,0,1,0,1]
        #y=[0,1,1,2,2,3,3]

        x = (i+1) % 2
        y = (i+1) // 2

        #remove axis labels
        axs[x, y].set_xticklabels([])
        axs[x, y].set_yticklabels([])
        axs[x, y].set_xticks([])
        axs[x, y].set_yticks([])

        axs[x, y].imshow(channel_rgba)

    axs[0, 0].set_xticklabels([])
    axs[0, 0].set_yticklabels([])
    axs[0, 0].set_xticks([])
    axs[0, 0].set_yticks([])
    
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array([])
    mappable.set_clim(+0.5, N+0.5)
    cbar = plt.colorbar(mappable, ax=axs, fraction=0.023, pad=0.02, shrink=0.5)
    cbar.set_ticks(np.linspace(0, N, N))
    cbar.ax.set_yticklabels(labels)
    plt.savefig(saveName)

    plt.close('all')


def createColoredChannels(input, map, csvFileName, outputName, folder, absoluteChange):

    workingMap = map.copy()
    output = input + map

    N = 7 #number of channels
    multiplicator = 3.5 #multiplicator for the alpha channel
    threshold = 0.1 #threshold for the channels


    topListChannels =  getTopChannels(absoluteChange, N)

    #get the labels out of the csv
    csvFile = open(csvFileName)
    csvreader = csv.reader(csvFile)

    labels = [None] * N
    fullLabels = []
    counter = 0
    #the first row is the header. But we are already skipping 2 rows, so its ok
    for row in csvreader:
        if counter != 0:
            fullLabels.append(row[1])
        if counter in topListChannels:
            #get the index
            index = topListChannels.index(counter)
            labels[index] = row[1]
        counter += 1

    counter = 0
    colorNames = ["red", "green", "blue", "black", "magenta", "cyan", "yellow"]
    #do the coloring for the input image
    #remove the last 3 channels as they are HE
    codex = input[:input.shape[0]-3,...]
    codex = np.swapaxes(np.swapaxes(codex, 0, 2), 0, 1)

    createPlot(codex, topListChannels, folder + outputName + "Input", colorNames, labels, threshold, multiplicator)

    #do the coloring for the output image
    #remove the last 3 channels as they are HE

    codex = output[:output.shape[0]-3,...]
    codex = np.swapaxes(np.swapaxes(codex, 0, 2), 0, 1)

    createPlot(codex, topListChannels, folder + outputName + "Output", colorNames, labels, threshold, multiplicator)

    #do here the same stuff for the maps
    #split the map into negative and positive values
    workingMap = workingMap[:workingMap.shape[0]-3,...]
    workingMap = np.swapaxes(np.swapaxes(workingMap, 0, 2), 0, 1)

    mapPositive = workingMap.copy()
    mapNegative = workingMap.copy()
    mapPositive[mapPositive < 0] = 0
    mapNegative[mapNegative > 0] = 0
    mapNegative *= -1

    #create the plots for the maps
    createPlot(mapPositive, topListChannels, folder + outputName + "MapPositive", colorNames, labels, threshold, multiplicator)
    createPlot(mapNegative, topListChannels, folder + outputName + "MapNegative", colorNames, labels, threshold, multiplicator)

def main():
    options = get_parser().parse_args()

    dataCRC = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCrcFull/"
    dataCTCL = "/mnt/qb/work/baumgartner/mpaulikat71/Code/data/PreprocessedCtclFull/"
    csvCRC = "../csvs/channelNamesCRCShort.csv"
    csvCTCL = "../csvs/channelNamesCTCLShort.csv"
    workingPath = "../"
    modelPath = "models/"
    results = "../results/"
    csvFileName = ""

    netsFolder = os.listdir(workingPath)
    for netsDir in netsFolder:
        if netsDir != "scripts" and netsDir != "results" and netsDir != "csvs" and netsDir!= "resultsHE":
            if netsDir == "DTC" or netsDir == "CTD":
                options.data = dataCRC
                options.channels_number = 61
                csvFileName = csvCRC
                
            else:
                options.data = dataCTCL
                options.channels_number = 62
                csvFileName = csvCTCL

            if netsDir == "DTC" or netsDir == "NTR":
                options.direction = "DifToCrohn"
            elif netsDir == "CTD" or netsDir == "RTN":
                options.direction = "CrohnToDif"

            crcObject = CRCPrep(options.data, options.train, options.eval, options.test, batchsize=options.batch_size, useTorch=options.torch)
            trainGroup, _, _ = crcObject.returnLoaders()

            netsPaths = os.listdir(workingPath + netsDir + "/" + modelPath)
            absoluteChange = None
            models = []
            counter = 0
            for net in netsPaths:
                dataLoaderIter = iter(trainGroup)

                if net != "last.ckpt":
                    networkPath = workingPath + netsDir + "/" + modelPath + "/" + net
                    model = VaGAN.load_from_checkpoint(checkpoint_path=networkPath, opt=options)
                    models.append(model)
                    counter += 1
                    if absoluteChange is None:
                        absoluteChange = getabsoluteChange(model, options, dataLoaderIter, len(trainGroup))
                    else:
                        absoluteChange += getabsoluteChange(model, options, dataLoaderIter, len(trainGroup))

            absoluteChange /= counter
            dataLoaderIter = iter(trainGroup)

            resultFolder = results + netsDir + "/"

            processVAGAN(models, options, dataLoaderIter, absoluteChange, len(trainGroup), resultFolder, csvFileName)

if __name__ == '__main__':
    main()