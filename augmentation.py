import torch
from torchvision.utils import save_image
import numpy as np
from skimage.transform import rotate, AffineTransform, warp
import pickle
import numpy as np
from skimage import io

from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path
from torchvision.transforms import ToTensor
from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

import warnings
warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import os

def rgb_rotation_picture(picture: torch.Tensor() = None, rotating_angle: float() = 0, method='wrap', ) -> torch.Tensor:
    pic_shape = picture.shape()
    assert len(pic_shape) == 3, f'Размер изображения не совпадает с RGB ({pic_shape}!= (3,X,Y))'
    picture = np.array(picture)
    rotated_1st = rotate(picture[0, :, :], angle=rotating_angle, mode=method)
    rotated_2nd = rotate(picture[1, :, :], angle=rotating_angle, mode=method)
    rotated_3rd = rotate(picture[2, :, :], angle=rotating_angle, mode=method)
    converted_pic = np.rollaxis(np.dstack((rotated_1st, rotated_2nd, rotated_3rd)), 2, 0)
    return converted_pic


def data_augmentation(list_of_labels: list(),threshold,TRAIN_DIR,TEST_DIR):
    dict_unique = dict(zip(np.unique(list_of_labels), np.zeros(len(np.unique(list_of_labels)))))
    for i in list_of_labels:
        dict_unique[i] += 1
    res_dict = {k: v for k, v in sorted(dict_unique.items(), key=lambda item: item[1])}
    classes_to_up = {k: v for k, v in sorted(dict_unique.items(), key=lambda item: item[1]) if v < threshold}
    sorted(list(TRAIN_DIR.rglob('*.jpg')))
    for i in classes_to_up.keys():
        fold_to_work = list(TRAIN_DIR.rglob(str(i) + '/' + '*.jpg'))

        for j in fold_to_work:
            file_name = j.name.split('.')[0]  # вытаскиваем имя файла без расширения
            image = Image.open(j)
            image_tensor = ToTensor()(image)
