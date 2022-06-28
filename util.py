import argparse
from collections import OrderedDict
import os
import wandb
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchmetrics.functional import confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
METRICS = ['DSC', 'Sensitivity', 'Specificity', 'IOU',
           'ExtraFraction', 'PPV', 'NPV', 'FNR', 'ACC']


def parse_args():
    """read args from two config files specified by --basic and --advance
       default are config/basic.yml and config/config.yml
    """
    desc = "MS Segmentation"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--basic', type=str,
                        default='config/basic.yml', help='basic options')
    parser.add_argument('--advance', type=str,
                        default='config/config.yml', help='model options')
    return check_args(parser.parse_args())


def check_args(args):
    """combine arguments"""
    with open(args.basic, 'r') as f:
        basic_config = yaml.safe_load(f)
    with open(args.advance, 'r') as f:
        advance_config = yaml.safe_load(f)
    args_dict = vars(args)
    args_dict.update(basic_config)
    args_dict.update(advance_config)

    # check dirs
    check_folder(args.checkpoints)

    if args.use_wandb:
        wandb.init(
            project=args.project,
            group=args.group,
            notes=args.notes,
            config=advance_config
        )
    return args


def check_folder(log_dir):
    if not os.path.exists(log_dir):
        print(f'* {log_dir} does not exist, creating...')
        os.makedirs(log_dir)
    return log_dir


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model):
    print("=> Loading checkpoint")
    checkpoints = torch.load(checkpoint, map_location='cpu')
    model.load_state_dict(checkpoints["state_dict"])


def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        # post-processing: tranpose and scaling
        image_numpy = np.transpose(image_numpy, (1, 2, 0)) * 255.0
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)


def plt_show_imgs(img_arrs, rows=1, cols=1, grid_size=4):
    fig, axes = plt.subplots(rows, cols, figsize=(
        cols * grid_size, rows * grid_size))
    axes = axes.reshape(-1)
    assert len(img_arrs) == len(axes), True
    for i in range(len(img_arrs)):
        axes[i].imshow(tensor2im(img_arrs[i]))
        axes[i].set_axis_off()
    plt.subplots_adjust(wspace=0.01)
    plt.show()


def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x)
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()

    return num_correct/num_pixels, dice_score/len(loader)


def get_confuse_items(preds, gts, threshold=0.5):
    """tn, fp, fn, tp"""
    cm = confusion_matrix(preds, gts.long(), 2, threshold=threshold)
    tn, fp, fn, tp = cm.reshape(-1)
    return tn.item(), fp.item(), fn.item(), tp.item()


def calc_metrics(preds, gts, threshold=0.5):
    """['DSC', 'Sensitivity', 'Specificity', 'IOU', 'ExtraFraction', 'PPV', 'NPV', 'FNR', 'ACC']"""
    tn, fp, fn, tp = get_confuse_items(preds, gts, threshold)
    return OrderedDict({
        'DSC': 2*tp / (2*tp + fp + fn + 1e-8),
        'Sensitivity': tp / (tp + fn + 1e-8),
        'Specificity': tn / (tn + fp + 1e-8),
        'IOU': tp / (tp + fp + fn + 1e-8),
        'ExtraFraction': fp / (tn + fn + 1e-8),
        'PPV': tp / (tp + fp + 1e-8),
        'NPV': tn / (tn + fn + 1e-8),
        'FNR': fn / (tp + fn + 1e-8),
        'ACC': (tp + tn) / (tn + fp + fn + tp + 1e-8)
    })


def get_avg_dice(preds, gts, threshold=0.5):
    preds_t = (preds > threshold).float()
    dice_score = (2 * (preds_t * gts).sum()) / (
        (preds_t + gts).sum() + 1e-8
    )
    return dice_score


def wb_mask(bg_img, pred_mask, true_mask):
    bg_img = bg_img.detach().cpu().numpy()[0].squeeze()
    pred_mask = pred_mask.detach().cpu().numpy()[0].squeeze()
    true_mask = true_mask.detach().cpu().numpy()[0].squeeze()
    return wandb.Image(bg_img, masks={
        "prediction": {"mask_data": pred_mask, "class_labels": {0: 'background', 1: 'lesions'}},
        "ground truth": {"mask_data": true_mask, "class_labels": {0: 'background', 1: 'lesions'}}})
