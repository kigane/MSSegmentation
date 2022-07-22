import glob
import os
from collections import OrderedDict

import h5py
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt
import cv2


def getVarianceMean(src, winSize):
    if src is None or winSize is None:
        print("The input parameters of getVarianceMean Function error")
        return -1
    
    if winSize % 2 == 0:
        print("The window size should be singular")
        return -1 
    
    copyBorder_map=cv2.copyMakeBorder(src,winSize//2,winSize//2,winSize//2,winSize//2,cv2.BORDER_REPLICATE) # padding
    shape=np.shape(src)
    
    local_mean=np.zeros_like(src)
    local_std=np.zeros_like(src)
    
    for i in range(shape[0]):
        for j in range(shape[1]):   
            temp=copyBorder_map[i:i+winSize,j:j+winSize]
            local_mean[i,j],local_std[i,j]=cv2.meanStdDev(temp)
            if local_std[i,j]<=0:
                local_std[i,j]=1e-8
            
    return local_mean,local_std
    

def ACE(src, winSize, maxCg): # 有一点用
    """adaptContrastEnhancement"""
    if src is None or winSize is None or maxCg is None:
        print("The input parameters of ACE Function error")
        return -1
    shape=np.shape(src)
    meansGlobal=cv2.mean(src)[0]
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


def normalize(image):
    image = (image - image.min()) / (image.max() - image.min() + 1e-8)
    return image


def process_isbi(isTrain=True):
    
    if isTrain:
        MRI_SEQ = ['flair', 't2', 'mprage', 'pd', 'mask1', 'mask2']
        nii_paths = sorted(glob.glob("./data/training/**/*.nii", recursive=True))
    else:
        MRI_SEQ = ['flair', 't2', 'mprage', 'pd']
        nii_paths = sorted(glob.glob("./data/testdata_website/**/*.nii", recursive=True))

    # k: str, v: list, 分别保存各mri类型对应的nii文件路径
    mri_seq_paths_dict = OrderedDict() 
    for t in MRI_SEQ:
        mri_seq_paths_dict[t] = [x for x in nii_paths if t in x]
        
    # 读取nii，将mri_sequence和相应的mask保存到一个h5文件中。
    # for flair, t2, mprage, pd, mask1, mask2 in zip(*mri_seq_paths_dict.values()):
    for mris in zip(*mri_seq_paths_dict.values()):
        imgs_3d = [read_nii(path) for path in mris]
        # case: 'patient0x_0x'
        case = os.path.split(mris[0])[1].split('_flair')[0].replace('training', 'patient')
        if isTrain:
            # 找到有病变的切片
            filtered_inds = []
            for i in range(imgs_3d[4].shape[0]): # mask1
                if not (np.all(imgs_3d[4][i] == 0)):
                    filtered_inds.append(i)
            filtered_imgs = [mri_3d[filtered_inds] for mri_3d in imgs_3d]
        else:
            # 去掉全黑的切片
            filtered_inds = []
            for i in range(imgs_3d[0].shape[0]):  # flair
                if not (np.all(imgs_3d[0][i] == 0)):
                    filtered_inds.append(i)
            filtered_imgs = [mri_3d[filtered_inds] for mri_3d in imgs_3d]
        
        # 将mri和相应mask存到一个h5文件中
        slice_ind = 0
        for imgs in zip(*filtered_imgs):
            # dir = 'train' if isTrain else 'test'
            f = h5py.File(f'./data/isbi2015ace/data/{case}_slice_{str(slice_ind).zfill(3)}.h5', 'w')
            imgs = [ACE(normalize(img)*255, 7, 9) for img in imgs] # [0,255]
            # imgs = [normalize(img)*255 for img in imgs] # [0,255]
            f.create_dataset('flair', data=imgs[0], compression="gzip")
            f.create_dataset('t2', data=imgs[1], compression="gzip")
            f.create_dataset('mprage', data=imgs[2], compression="gzip")
            f.create_dataset('pd', data=imgs[3], compression="gzip")
            if isTrain:
                f.create_dataset('mask1', data=imgs[4].astype(np.uint8), compression="gzip")
                f.create_dataset('mask2', data=imgs[5].astype(np.uint8), compression="gzip")
            else:
                f.create_dataset('mask1', data=np.zeros_like(imgs[0]).astype(np.uint8), compression="gzip")
                f.create_dataset('mask2', data=np.zeros_like(imgs[0]).astype(np.uint8), compression="gzip")
            f.close()
            slice_ind += 1


if __name__ == '__main__':
    process_isbi(isTrain=True)
    # imgs = read_nii(r'data\training\training01\preprocessed\training01_01_flair_pp.nii')
    # print(imgs.max())
    # print(imgs.mean())
    exit()
    f = h5py.File('./data/isbi2015raw/data/patient01_01_slice_001.h5', 'r')
    f = h5py.File('./data/isbi2015raw/data/test01_01_slice_001.h5', 'r')
    img_arrs = [
        f['flair'][:],
        f['t2'][:],
        f['mprage'][:],
        f['pd'][:],
        f['mask1'][:],
        f['mask2'][:],
    ]
    print([x.dtype for x in img_arrs])
    print([x.max() for x in img_arrs])
    print([x.mean() for x in img_arrs])
    rows, cols, grid_size = 2, 3, 4
    fig, axes = plt.subplots(rows, cols, figsize=(
        cols * grid_size, rows * grid_size))
    axes = axes.reshape(-1)
    assert len(img_arrs) == len(axes), True
    for i in range(len(img_arrs)):
        axes[i].imshow(img_arrs[i], cmap='gray')
        axes[i].set_axis_off()
    plt.subplots_adjust(wspace=0.01)
    plt.show()
