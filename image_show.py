import torch
from matplotlib import colors, pyplot as plt
import warnings

warnings.filterwarnings(action='ignore', category=DeprecationWarning)
import os
import numpy as np


def imshow(inp, title=None, plt_ax=plt, default=False):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


def show_random_pics(dataset, pics_horizontal=3, pics_vertical=3):
    fig, ax = plt.subplots(nrows=pics_horizontal, ncols=pics_vertical, figsize=(8, 8), \
                           sharey=True, sharex=True)
    for fig_x in ax.flatten():
        random_characters = int(np.random.uniform(0, 1000))
        im_val, label = dataset[random_characters]
        img_label = " ".join(map(lambda x: x.capitalize(), \
                                 dataset.label_encoder.inverse_transform([label])[0].split('_')))
        imshow(im_val.data.cpu(), \
               title=img_label, plt_ax=fig_x)
