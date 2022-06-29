from matplotlib import pyplot as plt
import numpy as np
import torch
from medpy import metric
from tqdm import tqdm
from datasets import get_test_loader, get_loader
import pandas as pd
from torchvision.utils import make_grid
import cv2 as cv

from model import UNET
from util import DEVICE, METRICS, calc_metrics, load_checkpoint, calc_metrics, get_avg_dice, tensor2im


if __name__ == '__main__':
    model = UNET(1, 1, [16, 32, 64, 128], [0.1, 0.1, 0.2, 0.2, 0.3])
    load_checkpoint('checkpoints/flair.pth', model)
    model.to(DEVICE)
    model.eval()

    test_loader = get_test_loader(
        './data/isbi2015/flair_test.npy',
        './data/isbi2015/mask_test.npy',
        batch_size=1
    )

    train_loader, _ = get_loader(
        './data/isbi2015/flair_train.npy',
        './data/isbi2015/mask_train.npy',
        batch_size=1
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
        save = make_grid(
            torch.cat([imgs, preds_t, gts], dim=0), pad_value=255, nrow=3)
        cv.imwrite(f'./result/test/pred_{n}.png',
                   save.permute(1, 2, 0).numpy() * 255.0)

    df_test = pd.DataFrame(metrics, columns=METRICS)
    # df_test.to_csv('result_test.csv')
    pd.concat([df_test.mean(), df_test.std()], axis=1).T.to_csv(
        'result/test_mean_std.csv')
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
    pd.concat([df_train.mean(), df_train.std()], axis=1).T.to_csv(
        'result/train_mean_std.csv')
    print(df_train.describe())
