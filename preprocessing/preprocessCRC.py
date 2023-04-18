'''
author: Martin Paulikat

script to preprocess the crc data.
The images are drawn out of hyperstacks folders and handE folders....

output: folder, which contains a folder for each patient. These folders themself contain the processed images and a file named label.txt containing the patients label
'''

from pathlib import Path
import tifffile as tiff
import numpy as np
import os
from skimage.transform import resize


#labels are hard coded
labels = [1,0,0,0,0,1,0,0,0,1,1,1,1,0,0,0,1,0,1,1,1,0,0,1,0,0,0,1,1,0,0,1,1,1,1]
#size of smaller matrices
smallerSize = 256
#stride
stride = 128
#downsample to
downScale = 128
#remove reduntant, or empty channels
channelsToRemove = [1,2,3,4,8,12,16,20,24,28,32,36,40,44,48,52,56,60,64,68,69,72,73,76,77,80,81,84,85,86,87,88,89,90]
#or name a single channel
single = None
#single = 12
#folders
preprocessedFolder = 'Preprocessed256To128Without1.1/'
path = Path('.')
subdirectories = [str(folder) for folder in path.iterdir() if folder.is_dir()]

#load hyperstack folder paths
endingHyperstack = 'hyperstack'
hyperstackDirectories = [folder for folder in subdirectories if endingHyperstack in folder]
hyperStackPaths = [Path(folder) for folder in hyperstackDirectories]

#load HE folder paths
endingHE = 'HandE'
hEDirectories = [folder for folder in subdirectories if endingHE in folder]
hEPaths = [Path(folder) for folder in hEDirectories]

#create folder for the preprocessed data. If it already exists, end the programm
if not os.path.exists(preprocessedFolder):
    os.makedirs(preprocessedFolder)
else:
    print('folder with preprocessed data already exists')
    quit()

#count number of files for each patient
savecounters = np.zeros(35, dtype='uint16')

switch = 0

for hyperStackItem, hEItem in zip(hyperStackPaths,hEPaths):
    #load file paths
    hyperStackFileNames = [str(file) for file in hyperStackItem.iterdir()]
    hEFileNames = [str(file) for file in hEItem.iterdir()]

    counter = 0
    for hyperStackFile, hEFile in zip(hyperStackFileNames,hEFileNames):

        #remove the first one
        if counter == 0 and switch == 0:
            counter += 1
            switch = 1
        else:
            print(hyperStackFile)

            #the current patient is counter//2+1
            patient = counter // 2 + 1
            counter += 1
            #read hyperstack image
            hyperStackImage = tiff.imread(hyperStackFile)
            #reshape, so that we only have the channels
            image = np.reshape(hyperStackImage, (92,1440,1920))
            #drop channels defined at the beginning
            image = np.delete(image,channelsToRemove,axis=0)
            
            if single:
                image = image[single]
                image = np.reshape(image, (1, 1440, 1920))
            #change to float32ss
            image = image.astype('float32')
            #normalize data. Normalize each channel separately
            for item in range(0,len(image)):
                image[item] = image[item] / np.amax(image[item])

            #read hE image
            hEImage = tiff.imread(hEFile)
            #change to float32
            hEImage = hEImage.astype('float32')
            #get the last 3 channels of the HE
            hEImage = hEImage[1,1:4,...]
            #normalize hE seperatelly
            hEImage = hEImage / np.amax(hEImage)
            #append to the hyperstack matrix
            image = np.append(image,hEImage, axis=0)

            #create folder for patient (unless there is already one)
            if not os.path.exists(preprocessedFolder + str(patient)):
                os.makedirs(preprocessedFolder + str(patient))

            
            #use for splitting into patches
            #split matrix up into multiple smaller matrices
            for y in np.arange(0, image.shape[2]-1-smallerSize, stride):
                for x in np.arange(0, image.shape[1]-1-smallerSize, stride):
                    smallermatrix = image[...,x:x+smallerSize,y:y+smallerSize]
                    downscalledMatrix = np.zeros((len(smallermatrix), downScale, downScale), dtype='float32')
                    
                    for item in range(0,len(smallermatrix)):
                        #downscale
                        downscalledMatrix[item, ...] = resize(smallermatrix[item], (downScale, downScale), anti_aliasing=True)
                    savecounters[patient-1] += 1
                    #save
                    tiff.imsave(preprocessedFolder + str(patient) + '/' + str(savecounters[patient-1]) + '.tif', downscalledMatrix)

            #use for full size preprocessing
            #savecounters[patient-1] += 1
            #tiff.imsave(preprocessedFolder + str(patient) + '/' + str(savecounters[patient-1]) + '.tif' ,image)
            #using torch
            #torch.save(image, preprocessedFolder + str(patient) + '/' + str(savecounters[patient-1]) + '.pt')

            #save correpsonding label as txt
            with open(preprocessedFolder + str(patient) + '/label.txt', 'w') as f:
                f.write(str(labels[patient-1]))
