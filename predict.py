import glob
import os
from collections import OrderedDict

import albumentations as A
import cv2 as cv
import h5py
import numpy as np
import SimpleITK as sitk
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tf
from albumentations.pytorch import ToTensorV2
from matplotlib import pyplot as plt
from nbformat import read
from tqdm import tqdm

from util import (DEVICE, METRICS, calc_metrics, get_avg_dice, get_model,
                  load_checkpoint, parse_args, tensor2im)


def getVarianceMean(src, winSize):
    if src is None or winSize is None:
        print("The input parameters of getVarianceMean Function error")
        return -1
    
    if winSize % 2 == 0:
        print("The window size should be singular")
        return -1 
    
    copyBorder_map=cv.copyMakeBorder(src,winSize//2,winSize//2,winSize//2,winSize//2,cv.BORDER_REPLICATE) # padding
    shape=np.shape(src)
    
    local_mean=np.zeros_like(src)
    local_std=np.zeros_like(src)
    
    for i in range(shape[0]):
        for j in range(shape[1]):   
            temp=copyBorder_map[i:i+winSize,j:j+winSize]
            local_mean[i,j],local_std[i,j]=cv.meanStdDev(temp)
            if local_std[i,j]<=0:
                local_std[i,j]=1e-8
            
    return local_mean,local_std
    

def ACE(src, winSize, maxCg): # 有一点用
    """adaptContrastEnhancement"""
    if src is None or winSize is None or maxCg is None:
        print("The input parameters of ACE Function error")
        return -1
    shape=np.shape(src)
    meansGlobal=cv.mean(src)[0]
    localMean_map, localStd_map=getVarianceMean(src,winSize)

    for i in range(shape[0]):
        for j in range(shape[1]):
            cg = 0.2*meansGlobal/ localStd_map[i,j] # 增强系数
            if cg >maxCg:
                cg=maxCg
            elif cg<1:
                cg=1
            temp = src[i,j].astype(float)
            temp=max(0,min(localMean_map[i,j]+cg*(temp-localMean_map[i,j]),255))
            src[i,j]=temp   
    return src

def read_nii(path) -> np.ndarray:
    img_itk = sitk.ReadImage(path)
    # origin = img_itk.GetOrigin()
    # spacing = img_itk.GetSpacing()
    # direction = img_itk.GetDirection()
    image = sitk.GetArrayFromImage(img_itk)  # 
    return image

def normalize(images):
    for i in tqdm(range(images.shape[0])):
        image = images[i]
        image = (image - image.min()) / (image.max() - image.min() + 1e-8) * 255.0
        image = ACE(image, 7, 9)
        images[i] = image
    return images

def predict_one_nii(model, nii_path, transform, args):
    imgs = normalize(read_nii(nii_path)) # 181x217x181
    preds = torch.empty(imgs.shape)
    tf_resize = tf.Resize((157, 157))
    with torch.no_grad():
        for i in tqdm(range(imgs.shape[0])):
            # print(imgs[i].shape) # 217x181
            img = transform(image=imgs[i])["image"]
            # print(img.shape) # 1x224x224
            # if i == 68:
            #     plt.imshow(img[0].numpy(), cmap='gray')
            #     plt.show()
            img = img.unsqueeze(0)
            pred = model(img.to(DEVICE))
            # if i == 68:
            #     plt.imshow(pred.squeeze().cpu().detach().numpy(), cmap='gray')
            #     plt.show()
            preds[i] = F.pad(tf_resize(pred).squeeze(), [12, 12, 30, 30])
    preds = (preds > 0.5).float()
    preds = preds.detach().cpu().numpy()
    
    out = sitk.GetImageFromArray(preds)
    sitk.WriteImage(out, './data/predict/a.nii')

def predict(args):
    model = get_model(args)
    trans = A.Compose([
        A.CenterCrop(157, 157), 
        A.Resize(args.img_size, args.img_size), 
        A.Normalize(0, 1, max_pixel_value=255),
        ToTensorV2()])
    load_checkpoint(os.path.join(
        args.checkpoints, '_'.join(args.mri_types))+f'_{args.model}.pth', model)
    model.to(DEVICE)
    model.eval()
    predict_one_nii(model, r'data\training\training01\preprocessed\training01_01_flair_pp.nii', trans, args)
    # predict_one_nii(model, r'data\testdata_website\test01\preprocessed\test01_01_flair_pp.nii', trans, args)

if __name__ == '__main__':
    args = parse_args()
    predict(args)
