'''
author: Martin Paulikat
Data loader. 
'''
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import os
import random
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

import numpy as np
import torch

class RandomPairedImageFolder(Dataset):
    def __init__(self, root, transform=None):
        self.normal_dir = os.path.join(root, 'normal')
        self.abnormal_dir = os.path.join(root, 'abnormal')
        self.normal_images = os.listdir(self.normal_dir)
        self.abnormal_images = os.listdir(self.abnormal_dir)
        self.root = root

        self.transform = transform

        random.shuffle(self.normal_images)
        random.shuffle(self.abnormal_images)

    def square_crop(self, image):
        width, height = image.shape[1], image.shape[0]
        dim = [height, width]
        #process crop width and height for max available dimension
        crop_width = dim[0] if dim[0]<image.shape[1] else image.shape[1]
        crop_height = dim[1] if dim[1]<image.shape[0] else image.shape[0] 
        mid_x, mid_y = int(width/2), int(height/2)
        cw2, ch2 = int(crop_width/2), int(crop_height/2) 
        crop_img = image[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

        return crop_img

    def __getitem__(self, index):
        # Pair images based on the shuffled lists
        root = os.path.join(self.root, 'images')
        normal_image_path = os.path.join(root, self.normal_images[index % len(self.normal_images)])
        abnormal_image_path = os.path.join(root, self.abnormal_images[index % len(self.abnormal_images)])

        normal_image = Image.open(normal_image_path).convert('RGB')
        abnormal_image = Image.open(abnormal_image_path).convert('RGB')
        #transform the images to numpy arrays
        normal_image = np.array(normal_image)
        abnormal_image = np.array(abnormal_image)

        normal_image = self.square_crop(normal_image)
        abnormal_image = self.square_crop(abnormal_image)

        if self.transform is not None:
            normal_image = self.transform(image=normal_image)["image"]
            abnormal_image = self.transform(image=abnormal_image)["image"]
            
        #divide by 255 to normalize between 0 and 1
        #normal_image = normal_image / 255
        #abnormal_image = abnormal_image / 255

        normal_image = normal_image.to(dtype=torch.float32)
        abnormal_image = abnormal_image.to(dtype=torch.float32)

        return normal_image, abnormal_image

    def __len__(self):
        return min(len(self.normal_images), len(self.abnormal_images))

    def on_epoch_end(self):
        # Shuffle the lists again at the end of each epoch
        self.normal_images = random.shuffle(self.normal_images)
        self.abnormal_images = random.shuffle(self.abnormal_images)

class lightningLoader(LightningDataModule):
    def __init__(self, data_dir: str = "path/to/dir", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size

    def setup(self, stage: str):
        self.transform = A.Compose([
            #do random crop if you didnt split the data into patches beforehand
            #A.RandomCrop(256, 256),
            ToTensorV2(),
        ])
        self.testTransform = A.Compose([
            A.PadIfNeeded(min_height=None, min_width=None, pad_height_divisor=32, pad_width_divisor=32),
            ToTensorV2(),
        ])

        self.paired_dataset = RandomPairedImageFolder(root=self.data_dir, transform=self.transform)
        self.test_dataset = RandomPairedImageFolder(root=self.data_dir, transform=self.testTransform)

    def train_dataloader(self):
        return DataLoader(self.paired_dataset, batch_size=self.batch_size, num_workers=11)

    def val_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=11)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=11)

    def predict_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1)