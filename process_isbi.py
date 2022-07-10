import glob
import os
from collections import OrderedDict

import h5py
import numpy as np
import SimpleITK as sitk
from matplotlib import pyplot as plt


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
            dir = 'train' if isTrain else 'test'
            f = h5py.File(f'./data/isbi2015raw/{dir}/{case}_slice_{str(slice_ind).zfill(3)}.h5', 'w')
            f.create_dataset('flair', data=normalize(imgs[0]), compression="gzip")
            f.create_dataset('t2', data=normalize(imgs[1]), compression="gzip")
            f.create_dataset('mprage', data=normalize(imgs[2]), compression="gzip")
            f.create_dataset('pd', data=normalize(imgs[3]), compression="gzip")
            if isTrain:
                f.create_dataset('mask1', data=imgs[4].astype(np.uint8), compression="gzip")
                f.create_dataset('mask2', data=imgs[5].astype(np.uint8), compression="gzip")
            else:
                f.create_dataset('mask1', data=np.zeros_like(imgs[0]).astype(np.uint8), compression="gzip")
                f.create_dataset('mask2', data=np.zeros_like(imgs[0]).astype(np.uint8), compression="gzip")
            f.close()
            slice_ind += 1


if __name__ == '__main__':
    process_isbi(isTrain=False)
    # imgs = read_nii(r'data\training\training01\preprocessed\training01_01_flair_pp.nii')
    # print(imgs.max())
    # print(imgs.mean())
    exit()
    # f = h5py.File('./data/isbi2015raw/train/patient01_01_slice_001.h5', 'r')
    # f = h5py.File('./data/isbi2015raw/test/test01_01_slice_0.h5', 'r')
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
