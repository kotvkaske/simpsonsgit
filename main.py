import torch
import numpy as np
from f1loss import *
import PIL
train_on_gpu = torch.cuda.is_available()
from skimage.util import random_noise
from torchvision.transforms import ToTensor
from skimage.transform import rotate, AffineTransform, warp

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
