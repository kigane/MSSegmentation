import argparse
import os
import wandb
import yaml
import torch

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
