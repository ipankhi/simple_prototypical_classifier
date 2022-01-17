import torch
from torch.utils import data
import pandas as pd
import albumentations
from albumentations import pytorch as AT
from tqdm import tqdm
import math
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
import random
import torch.nn.functional as F
import torchvision.models as models
from torch import nn
#from torchsummary import summary
from collections import OrderedDict
import torch.optim as optim
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

from torch.utils.data import DataLoader



class Dataset(data.Dataset):
      'Characterizes a dataset for PyTorch'
      def __init__(self, dff, transforms =  None):
            'Initialization'
            self.transforms = transforms
            self.dff=dff
            data = []
            label = []
            lb = -1
            p = []
            self.wnids = []
            for index in range(len(dff)):
                  X = self.dff.iloc[index]['pixelss']
                  X = np.array(X).reshape(48,48,1)
                  data.append(X)
                  y = self.dff.iloc[index]['emotion']
                  label.append(y)
            self.data = data
            self.label = label

      def __len__(self):
            return len(self.data)
      def __getitem__(self, i):
            image, label = self.data[i], self.label[i]
            #image = self.transform(image)
            return image, label

#returns X and y 

#training_generator = data.DataLoader(training_set, **params)

#validation_set = Dataset(df_valid, transforms=None)
#validation_generator = data.DataLoader(validation_set, **params)

#test_set = Dataset(df_test, transforms=None)
#test_generator = data.DataLoader(test_set, **params)




"""     

for image, label in trainset:
      train_sampler = CategoriesSampler(train_labels, chosen_classes, 10,
                                       N_SHOT + N_QUERY)
                                      
                                      
    

      train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=8, pin_memory=True)

# Display image and label.
train_features, train_labels = next(iter(train_loader))
print(f"Feature batch shape: {train_features.size()}")
print(f"Labels batch shape: {train_labels.size()}")
img = train_features[3].squeeze()
label = train_labels[3]
plt.imshow(img, cmap="gray")
plt.show()
print(f"Label: {label}")
"""