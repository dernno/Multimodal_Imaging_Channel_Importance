import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
import pandas as pd 
import numpy as np
import os
import wandb
import argparse
from data.dataloader import MultiModalCancerDataset
from utils import get_partition_name
from torch.utils.data import DataLoader, WeightedRandomSampler
from utils import compute_class_weights, save_list_of_dicts_to_txt, analyze_dataset, save_random_patch_visualization, get_normalized_patient_weights
from train import train
# from utils import LayerNorm2d
from archs import network
import time
import random
from collections import defaultdict
import sys
from test import test

import torch
import pandas as pd
from captum.attr import IntegratedGradients
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gpu_name = torch.cuda.get_device_name(0) if device.type == 'cuda' else "CPU"

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

seed = 42
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

g = torch.Generator()
g.manual_seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

###############

parser = argparse.ArgumentParser()
parser.add_argument('--path_to_folds', default='/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/folds/FINAL_FOLDS/wenyi_5' )
parser.add_argument('--path_to_images', default='/mimer/NOBACKUP/groups/snic2021-7-128/nora/data')

parser.add_argument('--project_name', type=str, default='sweep', help='Choose poject name for wandb')


parser.add_argument('--crop_size', type=int, default=224) #change crop size to 224x224,shift
parser.add_argument('--epochs', type=int, default=30)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--lr', type=float, default=8e-5)
parser.add_argument('--wd', type=float, default=0.1, help='weight decay')

parser.add_argument('--model_arch_name', default='resnet50', help='[resnet50] for single model, [cafnet, embnet, fusionm4net, mmtm] for intermediate fusion')

# NOT OPTIONAL!! 
parser.add_argument('--mode', type=str, default='MM', help='[BF, FL, MM], MM means multi-modal inputs')

parser.add_argument('--channel', type=int, default=7, help='3 for BF, 4 for FL, 7 for MM')
### MM
parser.add_argument('--fusion_mode', type=str, default='E', help='[E,L,I]')

parser.add_argument('--fold', type=int, default=None, help='fold number: [0, 1, 2]')
#parser.add_argument('--mixup', type= bool, default=True, help='Whether to use the mixup augmentation')
parser.add_argument('--mixup', action='store_true', help='Use mixup augmentation')
parser.add_argument('--weighted_patients', action='store_true', help='Use weighted patients (loss + eval)')
parser.add_argument('--split_mode', type = str, default = "train_test", help= '"train_val_test" or "train_test"')
parser.add_argument('--pretrained', type=bool, default=True, help='Whether to use pretrained model')
parser.add_argument('--freeze', action='store_true', help='Whether to freeze weights')
parser.add_argument('--optimizer', type=str, default='adamw')

parser.add_argument('--server', type=str, default='alvis',help='mida1 or mida2')

parser.add_argument('--used_channels', type=str, default=None, help='Combination: [4, 5, 6] channels in total')
parser.add_argument('--note', type=str, default='default', help='Extra Info for run')

parser.add_argument('--group_partition_name', type=str, default='wenyi', help='wenyi, partition123, partition456')
parser.add_argument('--norm_config_path', type=str, default='/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/codes/input_norm/norm_config.json', help='jsonfile')
#parser.add_argument('--analyse_inputs', action='store_true', help='Analyze_inputs')
parser.add_argument('--input_norm', type=str, default = 'zscore')

parser.add_argument('--BF_aug', nargs="*", type=str, default=None, help='[posterize, solarize, gaussian, colorjitter, brightness_contrast,sharpness, autocontrast, erase]')
parser.add_argument('--FL_aug',  nargs="*", type=str, default=None, help='[colorjitter, gaussian, noise, intensity_shift, erase]')


parser.add_argument('--inference_pth', type=str, default = '/mimer/NOBACKUP/groups/snic2021-7-128/nora/logs/Baseline_P123_20_ResNet18_Early Fusion/resnet18_multimodal_early_fusion', help='Path to pth')
parser.add_argument('--not_masked_channels', type=str, default=None, help='[0123]')


def compute_integrated_gradients(model, test_dataloader, device, mode='MM'):
    """
    mode: 'BF', 'FL', or 'MM'
    model: trained torch model
    test_dataloader: DataLoader with 'image' in batch
    device: cuda or cpu
    """

    model.eval()
    print(f"\n=== Computing Integrated Gradients attributions ({mode}) ===")

    if mode == 'MM':
        # Wrapper für multimodales Modell (7 Kanäle → Split in BF + FL)
        def model_wrapper(x1, x2):
            output = model((x1, x2))
            #print(f"[DEBUG] Model output shape: {output.shape}")
            return output if output.ndim == 2 else output.view(-1, 1)

        ig = IntegratedGradients(model_wrapper)
        channel_sums = torch.zeros(7).to(device)
        channel_types = ['BF'] * 3 + ['FL'] * 4

    elif mode == 'BF':
        def model_wrapper(x):
            output = model(x)
            #print(f"[DEBUG] Model output shape: {output.shape}")
            return output if output.ndim == 2 else output.view(-1, 1)

        ig = IntegratedGradients(model_wrapper)
        channel_sums = torch.zeros(3).to(device)
        channel_types = ['BF'] * 3

    elif mode == 'FL':
        def model_wrapper(x):
            output = model(x)
            #print(f"[DEBUG] Model output shape: {output.shape}")
            return output if output.ndim == 2 else output.view(-1, 1)

        ig = IntegratedGradients(model_wrapper)
        channel_sums = torch.zeros(4).to(device)
        channel_types = ['FL'] * 4

    else:
        raise ValueError(f"Invalid mode '{mode}'. Choose from 'BF', 'FL', 'MM'.")

    channel_counts = 0

    for batch_idx, batch in enumerate(tqdm(test_dataloader)):
        images = batch['image'].to(device)
        #print(f"\n[Batch {batch_idx}] Loaded {images.shape[0]} samples — image shape: {images.shape}")

        for i in range(images.size(0)):
            try:
                if mode == 'MM':
                    input_tensor = images[i:i+1]  # [1, 7, H, W]
                    baseline_tensor = torch.zeros_like(input_tensor)

                    input_tuple = (input_tensor[:, :3], input_tensor[:, 3:])
                    baseline_tuple = (baseline_tensor[:, :3], baseline_tensor[:, 3:])

                    #print(f"Sample {i}: Input split -> BF {input_tuple[0].shape}, FL {input_tuple[1].shape}")

                    attr = ig.attribute(
                        inputs=input_tuple,
                        baselines=baseline_tuple,
                        target=None,
                        return_convergence_delta=False
                    )

                    #print(f"Sample {i}: Attributions computed -> shapes BF: {attr[0].shape}, FL: {attr[1].shape}")
                    attr_bf = attr[0].abs().mean(dim=(0, 2, 3))  # → [3]
                    attr_fl = attr[1].abs().mean(dim=(0, 2, 3))  # → [4]

                    #print(f"Sample {i}: Mean attribution BF {attr_bf}, FL {attr_fl}")
                    channel_sums[:3] += attr_bf
                    channel_sums[3:] += attr_fl

                else:
                    input_tensor = images[i:i+1]  # [1, C, H, W]
                    baseline_tensor = torch.zeros_like(input_tensor)

                    attr = ig.attribute(
                        inputs=input_tensor,
                        baselines=baseline_tensor,
                        target=None,
                        return_convergence_delta=False
                    )

                    #print(f"Sample {i}: Attribution shape: {attr.shape}")
                    attr_mean = attr.abs().mean(dim=(0, 2, 3))
                    #print(f"Sample {i}: Mean attribution per channel: {attr_mean}")
                    channel_sums += attr_mean

                channel_counts += 1

            except Exception as e:
                print(f"[ERROR] Sample {i} failed: {e}")

    # Durchschnitt berechnen
    if channel_counts > 0:
        avg_attr = (channel_sums / channel_counts).tolist()
    else:
        avg_attr = [float("nan")] * len(channel_sums)

    # df = pd.DataFrame({
    #     'Channel': list(range(len(channel_sums))),
    #     'Type': channel_types,
    #     'Attribution': avg_attr
    # }).sort_values(by='Attribution', ascending=False)

    # print("\n=== Average Attribution per Channel ===")
    # print(df.round(6))

    # return df
    # Neue DataFrame mit zwei Spalten
    df = pd.DataFrame({
        'Channel': list(range(len(channel_sums))),
        'Type': channel_types,
        'Attribution (Raw)': avg_attr
    })

    # Normierung
    total = df["Attribution (Raw)"].sum()
    df["Attribution (Norm)"] = df["Attribution (Raw)"] / total if total > 0 else float("nan")

    # Sortieren
    df = df.sort_values(by="Attribution (Norm)", ascending=False)

    print("\n=== Average Attribution per Channel ===")
    print(df.round(6))

    return df

def plot_integrated_gradients_piechart(csv_path, output_prefix="average_attribution_piechart"):
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    # Load the CSV
    df = pd.read_csv(csv_path)

    # Channel labels
    channel_labels = ["BF1", "BF2", "BF3", "FL1", "FL2", "FL3", "FL4"]

    # Colors (RGBA hex with alpha)
    channel_colors = [
        "#4d4d4d80",  # BF1
        "#49494952",  # BF2
        "#1a1a1a37",  # BF3
        "#004ea13a",  # FL1
        "#004eab4a",  # FL2
        "#0044935d",  # FL3
        "#00447b68",  # FL4
    ]

    # Plot style
    rcParams['font.family'] = 'sans-serif'
    rcParams['font.weight'] = 'bold'

    # Create pie chart
    plt.figure(figsize=(8, 8))
    plt.pie(
        df["Attribution (Norm)"],
        labels=channel_labels,
        colors=channel_colors,
        autopct='%1.1f%%',
        startangle=90,
        pctdistance=0.7,
        textprops={'fontsize': 22, 'fontweight': 'bold'}
    )
    # plt.title("BF:123 FL:14", fontsize=14, fontweight='bold')
    plt.tight_layout()

    # Save as PNG and PDF
    plt.savefig(f"{output_prefix}.png", format="png")
    plt.savefig(f"{output_prefix}.pdf", format="pdf")
    plt.close()



def run_cv_integrated_gradients(args):
    print(f"Using device: {device}")

    if args.server == 'mida1':
        log_dir = '/home/nora/data2_nora'
    elif args.server == 'alvis':
        log_dir = '/mimer/NOBACKUP/groups/snic2021-7-128/nora'
    else:
        log_dir = '/home/nora/work_nora'  # default mida2

    all_ig_results = []

    # Select folds to run
    if args.fold is not None:
        folds_to_run = [args.fold]
    else:
        folds_to_run = [0, 1, 2]  # run all folds

    for i in folds_to_run:
        print(f"\n=== Fold {i+1} ===")

        # Set mode name
        mode_name = args.mode
        if mode_name == "MM":
            if args.fusion_mode == "E":
                mode_name = "multimodal_early_fusion"
            elif args.fusion_mode == "L":
                mode_name = "multimodal_late_fusion"
        else:
            mode_name = f"{mode_name} only"

        # Extract last two components of the path
        last_two = os.path.join(
            os.path.basename(os.path.dirname(args.inference_pth)),
            os.path.basename(args.inference_pth)
        )

        model_path = os.path.join(args.inference_pth, f"fold_{i+1}", "test_best_f1_model.pth")

        os.makedirs(f'{log_dir}/logs', exist_ok=True)
        log_path = f'{log_dir}/TORTEN_integrated_gradients_logs/{last_two}/{args.note}'
        os.makedirs(log_path, exist_ok=True)

        print(f"Using model: {model_path}")
        print(f"Logging to: {log_path}")

        path_to_test_csv = f'{args.path_to_folds}/fold_{i+1}/test.csv'
        test_df = pd.read_csv(path_to_test_csv).iloc[:100]
        partition_name = get_partition_name(args.group_partition_name, i, retrain=True)

        channel = args.channel

        test_dataset = MultiModalCancerDataset(
            args.path_to_images, 
            test_df, mode=args.mode, 
            split='test', size=args.crop_size, 
            used_channels=args.used_channels, not_masked_channels=args.not_masked_channels,
            partition_name=partition_name, 
            norm_config_path=args.norm_config_path, input_norm=args.input_norm
        )

        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=args.batch_size, shuffle=False, num_workers=16
        )

        # Check first batch (debug)
        for batch in test_dataloader:
            images = batch['image']
            labels = batch['label']
            print("Shape:", images.shape)
            print("Image 0:", images[0])
            print("Image 1:", images[1])
            print("Label 0:", labels[0])
            print("Label 1:", labels[1])
            break  # only inspect the first batch

        # Load model
        if args.mode == 'MM':
            model = network.MultimodalNet(
                model_arch_name=args.model_arch_name, channel=channel, 
                fusion_mode=args.fusion_mode, pretrained=args.pretrained
            )
        else:
            model = network.SinglemodalNet(
                model_arch_name=args.model_arch_name, channel=channel, 
                pretrained=args.pretrained
            )

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        print(f"Mode: {mode_name} - #Channels: {channel}")

        df_ig = compute_integrated_gradients(model, test_dataloader, device, mode=args.mode)

        # Save results for this fold
        fold_path = os.path.join(log_path, f"integrated_gradients_fold_{i+1}.csv")
        df_ig.to_csv(fold_path, index=False)
        print(f"Saved fold result to {fold_path}")

        # Store raw attribution values for averaging
        all_ig_results.append(df_ig.set_index("Channel")[["Attribution (Raw)"]])

    # Compute average across folds
    df_avg = sum(all_ig_results) / len(all_ig_results)
    df_avg = df_avg.reset_index()
    df_avg = df_avg.rename(columns={"Attribution (Raw)": "Attribution (Raw)"})

    # Normalize
    total = df_avg["Attribution (Raw)"].sum()
    df_avg["Attribution (Norm)"] = df_avg["Attribution (Raw)"] / total if total > 0 else float("nan")

    # Add channel type
    if args.mode == 'MM':
        df_avg["Type"] = ['BF'] * 3 + ['FL'] * 4
    elif args.mode == 'BF':
        df_avg["Type"] = ['BF'] * 3
    elif args.mode == 'FL':
        df_avg["Type"] = ['FL'] * 4
    else:
        raise ValueError("Unknown mode")

    # Sort by normalized attribution
    df_avg_sorted = df_avg.sort_values(by="Attribution (Norm)", ascending=False)

    # Save
    avg_csv_path = os.path.join(log_path, "average_integrated_gradients.csv")
    df_avg_sorted.to_csv(avg_csv_path, index=False)

    # Output summary
    print("\n=== Average Attribution per Channel (Sorted) ===")
    print(df_avg_sorted.round(6))
    print(f"Saved average results to {avg_csv_path}")

    # Optionally generate pie chart
    # plot_path_prefix = os.path.join(log_path, "average_attribution_piechart")
    # plot_integrated_gradients_piechart(avg_csv_path, output_prefix=plot_path_prefix)
    # print(f"Saved pie chart as {plot_path_prefix}.png and .pdf")



if __name__ == '__main__':
    args = parser.parse_args()
    run_cv_integrated_gradients(args)


