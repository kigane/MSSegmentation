import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
from model import UNET
from losses import DiceBCELoss
from datasets import get_loader
from util import DEVICE, check_accuracy, get_avg_dice, parse_args, save_checkpoint, wb_mask, tensor2im


if __name__ == "__main__":
    args = parse_args()

    model = UNET(1, 1, args.features, args.dropout_ratios)
    model.to(DEVICE)
    loss_fn = DiceBCELoss(args.dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.lr_beta1, args.lr_beta2))

    mri_path = os.path.join(args.base_dir, args.mri_type + '_train.npy')
    mask_path = os.path.join(args.base_dir, 'mask_train.npy')
    train_loader, val_loader = get_loader(
        mri_path, mask_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True
    )

    for epoch in range(args.num_epochs):
        pbar = tqdm(train_loader)
        pbar.set_description(f'epoch {epoch}')
        train_losses = []
        dice_scores = []
        mask_list = []
        for batch_index, (imgs, targets) in enumerate(pbar):
            imgs = imgs.to(DEVICE)  # [NCHW]
            targets = targets.to(DEVICE)  # [N1HW]
            if epoch == 0 and batch_index == 0:
                print(f'imgs shape: {imgs.shape}')
                print(f'targets shape: {targets.shape}')
            preds = model(imgs)
            dice_scores.append(get_avg_dice(preds, targets))
            optimizer.zero_grad()
            loss = loss_fn(preds, targets)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.float())
            mask_list.append(wb_mask(imgs, preds, targets))
        wandb.log({"predictions": mask_list})
        pbar.set_postfix_str(f'loss: {sum(train_losses)/len(train_losses):2f}')
        pbar.set_postfix_str(f'dice: {sum(dice_scores)/len(dice_scores):2f}')
        wandb.log({'train/loss': sum(train_losses)/len(train_losses)})
        wandb.log({'train/dice': sum(dice_scores)/len(dice_scores)})

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }

        save_checkpoint(checkpoint, os.path.join(
            args.checkpoints, args.mri_type)+'.pth')

        # check accuracy
        acc, dice = check_accuracy(val_loader, model, device=DEVICE)

        wandb.log({
            'val/acc': acc,
            'val/dice': dice
        })
