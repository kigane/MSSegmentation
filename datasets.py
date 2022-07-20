from collections import OrderedDict
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as tf
import numpy as np
import torch
import h5py


class MSH5Datasets(Dataset):
    def __init__(
        self,
        base_dir=None,
        split="train",
        mri_types=['flair'],
        transform=None,
        test_case='03_05',
        use_mask1=True
    ):
        self._base_dir = base_dir
        self.sample_list = []
        self.split = split
        self.transform = transform
        self.mri_types = mri_types
        self.use_mask1 = use_mask1

        if self.split == "train":
            with open(self._base_dir + f"/train_slices_{test_case}.list", "r") as f1:
                tmp_list = f1.readlines()
            self.sample_list = [item.replace("\n", "") for item in tmp_list]
            # with open(self._base_dir + "/unlabeled_slices.list", "r") as f1:
            #     tmp_list = f1.readlines()
            # self.sample_list += [item.replace("\n", "") for item in tmp_list]

        elif self.split == "val":
            with open(self._base_dir + f"/test_{test_case}.list", "r") as f:
                self.sample_list = f.readlines()
            self.sample_list = [item.replace("\n", "") for item in self.sample_list]
        # if num is not None and self.split == "train":
        #     self.sample_list = self.sample_list[:num]
        print("total {} samples".format(len(self.sample_list)))

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        case = self.sample_list[idx]
        h5f = h5py.File(self._base_dir + "/data/{}.h5".format(case), "r")
        # flair = h5f["flair"][:]
        # t2 = h5f["t2"][:]
        # mprage = h5f["mprage"][:]
        # pd = h5f["pd"][:]
        mask1 = h5f["mask1"][:]
        mask2 = h5f["mask2"][:]
        image = torch.stack([torch.from_numpy(h5f[t][:]) for t in self.mri_types])
        image = self.transform(image) if self.transform else image
        label = mask1 if self.use_mask1 else mask2
        label = self.transform(label.astype(np.float32)) if self.transform else tf.ToTensor()(label.astype(np.float32))
        # sample = {"image": image, "label": label}
        # sample["idx"] = idx
        return image, label


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


class MSMultiDataset(Dataset):
    # path 为 npy 文件的位置
    def __init__(self, base_dir, mri_types, is_train=True, transform=None):
        postfix = "train" if is_train else "test"
        self.mris = OrderedDict()
        for t in mri_types:
            self.mris[t] = np.load(os.path.join(base_dir, t) + f"_{postfix}.npy")
        self.mask = np.load(os.path.join(base_dir, "mask") + f"_{postfix}.npy").astype(
            np.float32
        )
        self.transform = transform

    def __len__(self):
        return len(self.mask)

    def __getitem__(self, index):
        if self.transform is None:
            self.transform = tf.ToTensor()
        imgs = []
        for mri in self.mris.values():
            imgs.append(self.transform(mri[index]))
        img = torch.cat(imgs, dim=0)
        mask = tf.ToTensor()(self.mask[index])
        return img, mask


def get_loader(
    base_dir, 
    mri_types, 
    batch_size=16, 
    num_workers=0, 
    shuffle=True, 
    transform=None,
    test_case='03_05'
):
    # dataset = MSMultiDataset(base_dir, mri_types, True, transform)
    dataset = MSH5Datasets(base_dir, "train", mri_types, transform, test_case=test_case)
    train_len = int(len(dataset) * 0.8)
    train, val = random_split(dataset, [train_len, len(dataset) - train_len])

    train_loader = DataLoader(
        train, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader


def get_test_loader(
    base_dir, 
    mri_types, 
    batch_size=16, 
    num_workers=0, 
    shuffle=True, 
    transform=None,
    test_case='03_05'
):
    # dataset = MSMultiDataset(base_dir, mri_types, False, transform)
    dataset = MSH5Datasets(base_dir, "val", mri_types, transform, test_case=test_case)
    return DataLoader(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True
    )


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    trans = tf.Compose([tf.ToPILImage(), tf.CenterCrop((157, 157)), tf.Resize((224, 224)), tf.ToTensor()])

    d = MSH5Datasets("./data/isbi2015raw", "train",  ["flair", "t2", "pd"], trans)
    # d = MSMultiDataset("./data/isbi2015", ["flair", "t2", "pd"])

    # t, v = get_loader('./data/isbi2015/flair_train.npy',
    #                   './data/isbi2015/mask_train.npy',
    #                   batch_size=4,
    #                   transform=trans)
    print(d[0][0].shape)
    print(d[0][1].max())
    print(d[0][1].min())
    print(d[0][1].mean())
    # it = iter(t)
    # X, Y = next(it)
    # import seaborn as sns
    # sns.histplot(X[2].reshape(-1).numpy())
    # plt.show()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    # axes[0].imshow(data[0][3].permute(1, 2, 0), cmap='gray')
    # axes[1].imshow(data[1][3].permute(1, 2, 0), cmap='gray')

    axes[0].imshow(d[42][0].permute(1, 2, 0)[:, :, 0], cmap="gray")
    axes[1].imshow(d[42][0].permute(1, 2, 0)[:, :, 1], cmap="gray")
    axes[2].imshow(d[42][0].permute(1, 2, 0)[:, :, 2], cmap="gray")
    # axes[3].imshow(d[24][0].permute(1, 2, 0)[:, :, 3], cmap='gray')
    # import seaborn as sns
    # sns.histplot(d[56][0].reshape(-1).numpy())
    # sns.histplot(d[56][1].reshape(-1).numpy())
    plt.show()
