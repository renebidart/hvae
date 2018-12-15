
"""
??? Delete this???
"""

import os
import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

from PIL import Image
import matplotlib.pyplot as plt

#########   DISPLAYING UTILS

def read_img_to_np(img_path, bw=False):
    size = 32
    if bw:
        img = np.asarray(Image.open(img_path))
    else:
        img = np.asarray(Image.open(img_path).convert('RGB').resize((size, size), Image.ANTIALIAS))
    img = img / 255.
    return img

def torch_to_np(img):
    """image tensor -> numpy format to display (channels last, 0-1)"""
    img = img.cpu().detach().numpy()
    img = np.squeeze(img)
    if len(img.shape)>2:
        img = np.moveaxis(img, 0, 2)
    # img = np.moveaxis(img, -1, 0)
    return img

def show_imgs(imgs, labels, cols = 3):
    if not isinstance(imgs, np.ndarray):#convert torch to numpy
        imgs = torch_to_np(imgs)
        imgs = np.moveaxis(imgs, -1, 0)
        
    if not isinstance(labels, np.ndarray):#convert torch to numpy
        labels = torch_to_np(labels)
        labels = np.moveaxis(labels, 0, -1)

    n_imgs = imgs.shape[0]
    fig = plt.figure()
    for i in range(n_imgs):
        ax = fig.add_subplot(cols, np.ceil(n_imgs/float(cols)), i + 1)
        img = np.squeeze(imgs[i, :, :])
        label = np.squeeze(labels[i])
        plt.gray()
        plt.imshow(img)
        plt.axis('off')
        ax.set_title(label, fontsize=60)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_imgs)
    fig.tight_layout()
    plt.show()



# def load_image(img_path, size=32):
#     """Load jpg image into format for Foolbox"""
#     img = np.asarray(Image.open(img_path).convert('RGB').resize((size, size), Image.ANTIALIAS))
#     img = np.moveaxis(img, 2, 0) / 255.
#     return img.astype('float32')

# def show_img(image, adversarial):
#     """Ajdust foolbox format to matplotlib format and olot"""
#     image = np.moveaxis(image, 0, 2)
#     adversarial = np.moveaxis(adversarial, 0, 2)
#     difference = adversarial - image
    
#     plt.figure(figsize=(10,10))
#     plt.subplot(1, 3, 1)
#     plt.title('Original')
#     plt.imshow( image)  # division by 255 to convert [0, 255] to [0, 1]
#     plt.axis('off')

#     plt.subplot(1, 3, 2)
#     plt.title('Adversarial')
#     plt.imshow( adversarial)  # ::-1 to convert BGR to RGB
#     plt.axis('off')

#     plt.subplot(1, 3, 3)
#     plt.title('Difference')
#     plt.imshow(difference / abs(difference).max() * 0.2 + 0.5)
#     plt.axis('off')
#     plt.show()