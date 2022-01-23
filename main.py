import cv2
import torch
import numpy as np
from f1loss import *
import PIL
train_on_gpu = torch.cuda.is_available()
from skimage.util import random_noise
from torchvision.transforms import ToTensor
from skimage.transform import rotate, AffineTransform, warp
from trainer import *
from data import *
from f1loss import *

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

import pickle
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import os


# разные режимы датасета
DATA_MODES = ['train', 'val', 'test']
# все изображения будут масштабированы к размеру 224x224 px
RESCALE_SIZE = 224
# работаем на видеокарте
DEVICE = torch.device("cuda")

TRAIN_DIR = Path('train/simpsons_dataset/')
TEST_DIR = Path('testset/testset')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
from sklearn.model_selection import train_test_split

train_val_labels = [path.parent.name for path in train_val_files]
TRAIN_DIR = Path('train/simpsons_dataset/')
TEST_DIR = Path('testset/testset')

train_val_files = sorted(list(TRAIN_DIR.rglob('*.jpg')))
test_files = sorted(list(TEST_DIR.rglob('*.jpg')))
from sklearn.model_selection import train_test_split

train_val_labels = [path.parent.name for path in train_val_files]
train_files, val_files = train_test_split(train_val_files, test_size=0.25,shuffle=True)# \
val_dataset = SimpsonsDataset(val_files, mode='val')
label_encoder = pickle.load(open("label_encoder.pkl", 'rb'))
from torchvision import transforms, models
n_classes = len(np.unique(train_val_labels))
if val_dataset is None:
    val_dataset = SimpsonsDataset(val_files, mode='val')

train_dataset = SimpsonsDataset(train_files, mode='train')
# Создаём сеть
model_vgg16 = models.vgg16(pretrained=True)
model_vgg16.classifier[6] = nn.Linear(4096,n_classes)
model_vgg16.to(DEVICE)
for i,child in enumerate(model_vgg16.children()):
    print(i,child)
    if (i<2):
        for param in child.parameters():
            param.requires_grad = False
            print('Параметр заморожен:',child)
history = train(train_dataset, val_dataset, model=model_vgg16, epochs=2, batch_size=178)