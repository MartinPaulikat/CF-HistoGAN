'''
author: Martin Paulikat

script to preprocess the ctcl data.
The images are drawn out of hyperstacks folders and handE folders....

output: folder, which contains a folder for each patient. These folders themself contain the processed images and a file named label.txt containing the patients label
'''

from pathlib import Path
import tifffile as tiff
import numpy as np
import os
from skimage.transform import resize

#size of smaller matrices
smallerSize = 256
#stride
stride = 32
#downsample to
downScale = 128
preprocessedFolder = 'PreprocessedCtclWithout3.1/'

#the patients are unbalanced. Therefor we need to hard code their order
patients = [1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,6,6,6,6,7,7,7,7,8,8,8,8,8,8,9,9,9,9,10,10,11,11,11,11,12,12,12,12,6,6,13,13,13,13,13,13,14,14,14,14,14,14,14,14,14,14,9,9,13,13]
#labels are hard coded
labels = [1,1,0,0,1,0,0,0,1,1,1,0,1,0]
#remove reduntant, or empty channels
channelsToRemove = [1,2,3,4,5,6,7,8,12,16,20,24,28,32,36,37,40,41,44,45,48,49,52,53,56,57,60,61,64,65,68,69,72,73,76,77,80,81,84,85,88,89,92,93,96,97,100,101,104,105,108,109,110,111,112,113,114]
#or name a single channel
single = None
#single = 12
#folders
path = Path('.')
subdirectories = [str(folder) for folder in path.iterdir() if folder.is_dir()]

#load hyperstack folder paths
endingHyperstack = 'hyperstack'
hyperstackDirectories = [folder for folder in subdirectories if endingHyperstack in folder]
hyperStackPaths = [Path(folder) for folder in hyperstackDirectories]

#create folder for the preprocessed data. If it already exists, end the programm
if not os.path.exists(preprocessedFolder):
    os.makedirs(preprocessedFolder)
else:
    print('folder with preprocessed data already exists')
    quit()

#count number of files for each patient
savecounters = np.zeros(35, dtype='uint16')

#read big H&E file
hEBig = tiff.imread('HandE.tif')

switch = 0

for hyperStackItem in hyperStackPaths:
    #load file paths
    hyperStackFileNames = [str(file) for file in hyperStackItem.iterdir()]
    counter = 0
    for hyperStackFile in hyperStackFileNames:

        #remove the first one
        if counter == 8 and switch == 0:
            counter += 1
            switch = 1
        else:
            print(hyperStackFile)

            patient = patients[counter]
            counter = counter + 1
            #read hyperstack image
            hyperStackImage = tiff.imread(hyperStackFile)
            #reshape, so that we only have the channels
            image = np.reshape(hyperStackImage, (116, 1440, 1920))
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

            #get correct hE patch
            x = counter % 10
            y = counter // 10
            hEImage = hEBig[1:4, 0+y*1440:(y+1)*1440, 0+x*1920:(x+1)*1920]

            #change to float32
            hEImage = hEImage.astype('float32')
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
            #tiff.imwrite(preprocessedFolder + str(patient) + '/' + str(savecounters[patient-1]) + '.tif' ,image)
            #using torch
            #torch.save(image, preprocessedFolder + str(patient) + '/' + str(savecounters[patient-1]) + '.pt')

            #save correpsonding label as txt
            with open(preprocessedFolder + str(patient) + '/label.txt', 'w') as f:
                f.write(str(labels[patient-1]))