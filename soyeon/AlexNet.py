from sklearn.metrics import roc_auc_score
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch.nn.functional as F
import torchvision.models as models
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


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        # Image input_size=(3, 227, 227)
        self.layers = nn.Sequential(
            # input_size=(96, 55, 55)
            nn.Conv2d(in_channels=3, out_channels=96,
                      kernel_size=(11, 11), stride=4, padding=0),
            nn.ReLU(),
            # input_size=(96, 27, 27)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # input_size=(256, 27, 27)
            nn.Conv2d(in_channels=96, out_channels=256,
                      kernel_size=(5, 5), stride=1, padding=2),
            nn.ReLU(),
            # input_size=(256, 13, 13)
            nn.MaxPool2d(kernel_size=3, stride=2),
            # input_size=(384, 13, 13)
            nn.Conv2d(in_channels=256, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(384, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=384,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(256, 13, 13)
            nn.Conv2d(in_channels=384, out_channels=256,
                      kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(),
            # input_size=(256, 6, 6)
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(in_features=256*6*6, out_features=4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(in_features=4096, out_features=4096),
            nn.ReLU(),
            nn.Linear(in_features=4096, out_features=4),
        )

    def forward(self, x):
        x = self.layers(x)
        x = x.view(-1, 256*6*6)
        x = self.classifier(x)
        return x
