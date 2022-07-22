import os
from matplotlib import pyplot as plt
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
import wandb
from datasets import get_test_loader, get_loader
import pandas as pd
from torchvision.utils import make_grid
import torchvision.transforms as tf
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2 as cv

from util import DEVICE, METRICS, calc_metrics, load_checkpoint, calc_metrics, get_avg_dice, parse_args, tensor2im, get_model


def test(args):
    model = get_model(args)
    load_checkpoint(os.path.join(
        args.checkpoints, '_'.join(args.mri_types))+f'_{args.model}.pth', model)
    model.to(DEVICE)
    model.eval()

    train_trans = A.Compose([
        A.CenterCrop(157, 157),
        A.Resize(args.img_size, args.img_size),
        A.Normalize(0, 1, max_pixel_value=255),
        ToTensorV2()
    ])

    trans = A.Compose([
        A.CenterCrop(157, 157), 
        A.Resize(args.img_size, args.img_size), 
        A.Normalize(0, 1, max_pixel_value=255),
        ToTensorV2()])

    test_loader = get_test_loader(
        args.base_dir,
        args.mri_types,
        batch_size=1,
        transform=trans
    )

    train_loader, _ = get_loader(
        args.base_dir,
        args.mri_types,
        batch_size=1,
        num_workers=0,
        transform=train_trans
    )

    metrics = []
    n = -1
    for imgs, gts in tqdm(test_loader):
        n = n+1
        imgs = imgs.to(DEVICE)
        gts = gts.to(DEVICE)
        with torch.no_grad():
            preds = model(imgs)
        preds_t = (preds > 0.5)
        dic = calc_metrics(preds_t, gts)
        metrics.append(dic.values())
        # save = make_grid(
        #     torch.cat([imgs[:, 0, :, :].unsqueeze(0), preds_t, gts], dim=0), pad_value=255, nrow=3)
        # cv.imwrite(f'./result/test/pred_{n}.png',
        #            save.permute(1, 2, 0).cpu().numpy() * 255.0)

    df_test = pd.DataFrame(metrics, columns=METRICS)
    # df_test.to_csv('result_test.csv')
    # pd.concat([df_test.mean(), df_test.std()], axis=1).T.to_csv(
    #     'result/test_mean_std.csv')
    if args.use_wandb:
        wandb.log({'Result/test': wandb.Table(dataframe=df_test.describe())})
        print(df_test.describe())
    else:
        print(df_test.describe())

    metrics = []
    for imgs, gts in tqdm(train_loader):
        imgs = imgs.to(DEVICE)
        gts = gts.to(DEVICE)
        preds = model(imgs)
        preds_t = (preds > 0.5)
        dic = calc_metrics(preds_t, gts)
        metrics.append(dic.values())

    df_train = pd.DataFrame(metrics, columns=METRICS)
    # df_train.to_csv('result_train.csv')
    # pd.concat([df_train.mean(), df_train.std()], axis=1).T.to_csv(
    #     'result/train_mean_std.csv')
    if args.use_wandb:
        wandb.log({'Result/train': wandb.Table(dataframe=df_train.describe())})
        print(df_train.describe())
    else:
        print(df_train.describe())


if __name__ == '__main__':
    args = parse_args()
    print(args)
    test(args)
    # model = UNET(1, 1, [16, 32, 64, 128], [
    #              0.1, 0.1, 0.2, 0.2, 0.3], use_bn=True)
    # load_checkpoint('checkpoints/flair_unet.pth', model)
    # model.to(DEVICE)
    # model.eval()

    # test_loader = get_test_loader(
    #     './data/isbi2015/flair_test.npy',
    #     './data/isbi2015/mask_test.npy',
    #     batch_size=1
    # )

    # train_loader, _ = get_loader(
    #     './data/isbi2015/flair_train.npy',
    #     './data/isbi2015/mask_train.npy',
    #     batch_size=1
    # )

    # metrics = []
    # n = -1
    # for imgs, gts in tqdm(test_loader):
    #     n = n+1
    #     imgs = imgs.to(DEVICE)
    #     gts = gts.to(DEVICE)
    #     with torch.no_grad():
    #         preds = model(imgs)
    #     preds_t = (preds > 0.5)
    #     dic = calc_metrics(preds_t, gts)
    #     metrics.append(dic.values())
    #     save = make_grid(
    #         torch.cat([imgs, preds_t, gts], dim=0), pad_value=255, nrow=3)
    #     cv.imwrite(f'./result/test/pred_{n}.png',
    #                save.permute(1, 2, 0).cpu().numpy() * 255.0)

    # df_test = pd.DataFrame(metrics, columns=METRICS)
    # # df_test.to_csv('result_test.csv')
    # pd.concat([df_test.mean(), df_test.std()], axis=1).T.to_csv(
    #     'result/test_mean_std.csv')
    # print(df_test.describe())

    # metrics = []
    # for imgs, gts in tqdm(train_loader):
    #     imgs = imgs.to(DEVICE)
    #     gts = gts.to(DEVICE)
    #     preds = model(imgs)
    #     preds_t = (preds > 0.5)
    #     dic = calc_metrics(preds_t, gts)
    #     metrics.append(dic.values())

    # df_train = pd.DataFrame(metrics, columns=METRICS)
    # # df_train.to_csv('result_train.csv')
    # pd.concat([df_train.mean(), df_train.std()], axis=1).T.to_csv(
    #     'result/train_mean_std.csv')
    # print(df_train.describe())
