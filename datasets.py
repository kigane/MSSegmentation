from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as tf
import numpy as np
import torch


class MSDataset(Dataset):
    # path 为 npy 文件的位置
    def __init__(self, mri_path, mask_path, transform=None):
        self.mri = np.load(mri_path)
        self.mask = np.load(mask_path).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        if self.transform is None:
            self.transform = tf.ToTensor()
        img = self.transform(self.mri[index])
        mask = tf.ToTensor()(self.mask[index])
        return img, mask


class MS2Dataset(Dataset):
    # path 为 npy 文件的位置
    def __init__(self, mri_path1, mri_path2, mask_path, transform=None):
        self.mri1 = np.load(mri_path1)
        self.mri2 = np.load(mri_path2)
        self.mask = np.load(mask_path).astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        if self.transform is None:
            self.transform = tf.ToTensor()
        img1 = self.transform(self.mri1[index])
        img2 = self.transform(self.mri2[index])
        img = torch.cat([img1, img2], dim=0)
        mask = tf.ToTensor()(self.mask[index])
        return img, mask


def get_loader(mri_path, mask_path, batch_size=16, num_workers=0, shuffle=True, transform=None, mri_path2=None):
    if mri_path2 is None:
        dataset = MSDataset(mri_path, mask_path, transform)
    else:
        dataset = MS2Dataset(mri_path, mri_path2, mask_path, transform)
    train_len = int(len(dataset) * 0.8)
    train, val = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(train, batch_size, shuffle=shuffle,
                              num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val, batch_size, shuffle=shuffle,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


def get_test_loader(mri_path, mask_path, batch_size=16, num_workers=0, shuffle=True, transform=None, mri_path2=None):
    if mri_path2 is None:
        dataset = MSDataset(mri_path, mask_path, transform)
    else:
        dataset = MS2Dataset(mri_path, mri_path2, mask_path, transform)
    return DataLoader(dataset, batch_size, shuffle=shuffle,
                      num_workers=num_workers, pin_memory=True)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    trans = tf.Compose([
        tf.ToPILImage(),
        tf.ToTensor(),
        tf.Normalize(0.5, 0.5)
    ])

    d = MS2Dataset('./data/isbi2015/flair_train.npy',
                   './data/isbi2015/t2_train.npy',
                   './data/isbi2015/mask_train.npy',)

    t, v = get_loader('./data/isbi2015/flair_train.npy',
                      './data/isbi2015/mask_train.npy',
                      batch_size=4,
                      transform=trans)

    # it = iter(t)
    # X, Y = next(it)
    # import seaborn as sns
    # sns.histplot(X[2].reshape(-1).numpy())
    # plt.show()

    fig, axes = plt.subplots(1, 2, figsize=(8, 4))
    # axes[0].imshow(data[0][3].permute(1, 2, 0), cmap='gray')
    # axes[1].imshow(data[1][3].permute(1, 2, 0), cmap='gray')

    axes[0].imshow(d[24][0].permute(1, 2, 0)[:, :, 0], cmap='gray')
    axes[1].imshow(d[24][0].permute(1, 2, 0)[:, :, 1], cmap='gray')
    # import seaborn as sns
    # sns.histplot(d[56][0].reshape(-1).numpy())
    # sns.histplot(d[56][1].reshape(-1).numpy())
    plt.show()
