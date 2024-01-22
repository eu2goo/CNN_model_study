import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from torchvision.utils import make_grid

# transformation to the image dataset:
transform = transforms.Compose([transforms.Resize((227, 227)),
                                transforms.ToTensor()
                                ])
train_dir = "/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/train/"
test_dir = "/kaggle/input/alzheimers-dataset-4-class-of-images/Alzheimer_s Dataset/test/"

# Applying the already definded transformation to the dataset
train_dataset = datasets.ImageFolder(train_dir, transform=transform)
test_dataset = datasets.ImageFolder(test_dir, transform=transform)

# Loading the training dataset in a way that optimizes the training process
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Get one batch of images
images, labels = next(iter(train_loader))

# number of images you want to display
num_images = 4

class_names = train_dataset.classes

class_names = train_dataset.classes
class_counts = Counter([label for _, label in train_dataset])
