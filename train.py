import os
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import wandb
from model import UNET, AttenUNET
from model_maunet import MAUNET, MUNET
from losses import DiceBCELoss, FocalDiceBCELoss
from datasets import get_loader
from util import DEVICE, check_accuracy, get_avg_dice, parse_args, save_checkpoint, wb_mask, tensor2im


def get_scheduler(optimizer, num_batches, args):
    if args.lr_policy == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=num_batches * args.lr_decay_freq, gamma=args.gamma)
    elif args.lr_policy == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=num_batches * args.num_epochs, eta_min=1e-6)
    else:
        scheduler = None
    return scheduler


def init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        nn.init.constant_(m.bias.data, 0)


if __name__ == "__main__":
    args = parse_args()
    if args.model == 'unet':
        model = UNET(2, 1, args.features,
                     args.dropout_ratios, use_bn=args.use_bn, act=args.activation)
    elif args.model == 'atten-unet':
        model = AttenUNET(2, 1, args.features,
                          args.dropout_ratios, use_bn=args.use_bn, act=args.activation)
    elif args.model == 'munet':
        model = MUNET(1, 1, args.features,
                      args.dropout_ratios, use_bn=args.use_bn, act=args.activation)
    elif args.model == 'maunet':
        model = MAUNET(1, 1, args.features,
                       args.dropout_ratios, use_bn=args.use_bn, act=args.activation)
    else:
        raise NotImplementedError(f'{args.model} is not implemented')

    mri_path = os.path.join(args.base_dir, args.mri_type + '_train.npy')
    mask_path = os.path.join(args.base_dir, 'mask_train.npy')
    train_loader, val_loader = get_loader(
        mri_path, mask_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        mri_path2=os.path.join(args.base_dir, args.mri_type2 + '_train.npy')
    )

    model.apply(init_weights)
    model.to(DEVICE)
    loss_fn = DiceBCELoss(args.dice_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr,
                           betas=(args.lr_beta1, args.lr_beta2))
    scheduler = get_scheduler(optimizer, len(train_loader), args)

    obar = tqdm(range(args.num_epochs))
    for epoch in obar:
        pbar = tqdm(train_loader, leave=False)
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
            if scheduler is not None:
                scheduler.step()
            train_losses.append(loss.float())
            mask_list.append(wb_mask(imgs, (preds > 0.5).float(), targets))

        wandb.log({'train/loss': sum(train_losses) /
                  len(train_losses)}, step=epoch+1)
        wandb.log({'train/dice': sum(dice_scores) /
                  len(dice_scores)}, step=epoch+1)

        if (epoch+1) % args.save_freq == 0:
            wandb.log({"predictions": mask_list[:12]}, step=epoch+1)
            # save model
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, os.path.join(
                args.checkpoints, args.mri_type)+f'_{args.model}.pth')

        # check accuracy
        acc, dice = check_accuracy(val_loader, model, device=DEVICE)

        wandb.log({
            'val/acc': acc,
            'val/dice': dice
        }, step=epoch+1)

        obar.set_description(
            f"lr: {optimizer.param_groups[0]['lr']} train/loss: {sum(train_losses)/len(train_losses):.4f}, train/dice: {sum(dice_scores)/len(dice_scores):.4f}, val/acc: {acc*100:.2f}, val/dice: {dice:.4f}")
