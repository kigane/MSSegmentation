import os
from pprint import pprint
import shutil
import glob


def split(folder):
    train_folder = os.path.join(folder, 'train')
    test_folder = os.path.join(folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)
    test_imgs = glob.glob(os.path.join(folder, 'training01_01*.png'))
    for img in test_imgs:
        shutil.move(img, os.path.join(test_folder, os.path.basename(img)))
    train_imgs = glob.glob(os.path.join(folder, '*.png'))
    for img in train_imgs:
        shutil.move(img, os.path.join(train_folder, os.path.basename(img)))


if __name__ == '__main__':
    dirs = os.listdir('./data/isbi2015')
    for d in dirs:
        split(os.path.join('./data/isbi2015', d))
