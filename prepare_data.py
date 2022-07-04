import argparse
import os
import numpy as np
import cv2 as cv
from tqdm import tqdm

from skimage.io import imread, imshow
from skimage.transform import resize
import matplotlib.pyplot as plt
import glob

seed = 42
np.random.seed = seed

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3

BASE_DIR = "./data/isbi2015/"
MASK_TRAIN = "./data/isbi2015/masks/train/"
MASK_TEST = "./data/isbi2015/masks/test/"


def filter_black_imgs(ids, name):
    Y = np.zeros((len(ids), IMG_HEIGHT, IMG_WIDTH, 1))
    n = -1
    d = 0
    index_ = []
    for index, mask_file in enumerate(tqdm(ids)):
        mask = imread(mask_file).T
        mask = mask[30:187, 11:168]
        mask = resize(mask, (IMG_HEIGHT, IMG_WIDTH),
                      mode='constant', preserve_range=True)
        if (np.all(mask == 0)):
            d = d+1
        else:
            n = n+1
            index_.append(index)
            for i in range(0, IMG_HEIGHT-1):
                for j in range(0, IMG_WIDTH-1):
                    if mask[i, j] == 0:
                        Y[n, i, j] = 0
                    else:
                        Y[n, i, j] = 1
    print(len(Y))
    Y = Y[:n+1]
    np.save(f'data/isbi2015/{name}.npy', Y.astype(int))
    return index_


def get_related_X(ids, index_, name):
    X_train = []
    for i in tqdm(index_):
        img = imread(ids[i]).T
        img = img[30:187, 11:168]
        img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                     mode='constant', preserve_range=True)
        X_train.append(img)
    X_train = np.stack(X_train, axis=0)
    np.save(f'data/isbi2015/{name}.npy', X_train.astype(np.uint8))


def get_train_test(train_indices, test_indices, name):
    train_ids = glob.glob(BASE_DIR + name + '/train/' + "*.png")
    test_ids = glob.glob(BASE_DIR + name + '/test/' + "*.png")
    get_related_X(train_ids, train_indices, f'{name}_train')
    get_related_X(test_ids, test_indices, f'{name}_test')


if __name__ == "__main__":
    masks_ids = glob.glob(MASK_TRAIN + "*.png")
    masks_ids2 = glob.glob(MASK_TEST + "*.png")
    # show_imgs(123)
    # exit()
    train_indices = filter_black_imgs(masks_ids, 'mask_train')
    test_indices = filter_black_imgs(masks_ids2, 'mask_test')

    names = ['flair', 't2', 'pd', 'marage']
    for name in names:
        get_train_test(train_indices, test_indices, name)
