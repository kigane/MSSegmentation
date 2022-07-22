import glob
import os
from collections import OrderedDict

import h5py
from nbformat import read
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import torch
import torch.nn as nn

import torchvision.transforms as tf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv
from tqdm import tqdm

from util import DEVICE, METRICS, calc_metrics, load_checkpoint, calc_metrics, get_avg_dice, parse_args, tensor2im, get_model

def read_nii(path) -> np.ndarray:
    img_itk = sitk.ReadImage(path)
    # origin = img_itk.GetOrigin()
    # spacing = img_itk.GetSpacing()
    # direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)  # 
    return image

def normalize(image):
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image * 255.0

def predict_one_nii(model, nii_path, transform, args):
    imgs = read_nii(nii_path) # 181x217x181
    print(imgs.shape)
    tf_resize = tf.Resize(imgs.shape[1:])
    preds = torch.empty(imgs.shape)
    with torch.no_grad():
        for i in tqdm(range(imgs.shape[0])):
            img = transform(image=imgs[i])["image"]
            img = img.unsqueeze(0)
            pred = model(img)
            preds[i] = tf_resize(pred).squeeze()
    preds = (preds > 0.5).float()
    preds = preds.detach().cpu().numpy()
    
    out = sitk.GetImageFromArray(preds)
    sitk.WriteImage(out, './data/predict/a.nii')

def predict(args):
    model = get_model(args)
    trans = A.Compose([A.CenterCrop(157, 157), A.Resize(args.img_size, args.img_size), ToTensorV2()])
    load_checkpoint(os.path.join(
        args.checkpoints, '_'.join(args.mri_types))+f'_{args.model}.pth', model)
    model.to(DEVICE)
    model.eval()
    predict_one_nii(model, r'data\testdata_website\test01\preprocessed\test01_01_flair_pp.nii', trans, args)

if __name__ == '__main__':
    args = parse_args()
    predict(args)
