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
parser.add_argument('--mode', type=str, default='MM', help='[BF, FL, MM], MM means multi-modal inputs') #default='MM',

parser.add_argument('--channel', type=int, default=7, help='3 for BF, 4 for FL, 7 for MM') #default=7,
### MM
parser.add_argument('--fusion_mode', type=str, default='E', help='[E,L,I]')

parser.add_argument('--fold', type=int, default=None, help='fold number: [0, 1, 2]')
#parser.add_argument('--mixup', type= bool, default=True, help='Whether to use the mixup augmentation')
parser.add_argument('--mixup', action='store_true', help='Use mixup augmentation')
parser.add_argument('--weighted_patients', action='store_true', help='Use weighted patients (loss + eval)')
#parser.add_argument('--retrain', action='store_true', help='Whether to retrain with validation set') ################# [TO DO] change approach
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

def run_cv_inference(args):
    print(f"Using device: {device}")

    all_metrics = defaultdict(list)

    if args.server == 'mida1':
        log_dir = '/home/nora/data2_nora'
    elif args.server == 'alvis':
        log_dir = '/mimer/NOBACKUP/groups/snic2021-7-128/nora'
    else:
        log_dir = '/home/nora/work_nora'  # mida2

    if args.inference_pth is None:
        print("Error: --inference_pth is required but was not provided.")
        sys.exit(1)

    if args.fold is not None:
        folds_to_run = [args.fold]  # Convert to 1-based if needed
    else:
        folds_to_run = [0, 1, 2]  # All folds

    for i in folds_to_run:#3  # Fold 1 to Fold 3
        print(f"\n======\n\n\n")
        print(f"\n=== Running inference for fold {i+1} ===")

        #########
        mode_name = args.mode
        if mode_name =="MM":
            if args.fusion_mode == "E":
                mode_name="multimodal_early_fusion"
            elif args.fusion_mode == "L":
                mode_name="multimodal_late_fusion"
        else:
            mode_name=f"{mode_name} only"
        #########

        # Get the last two components of the path
        last_two = os.path.join(
            os.path.basename(os.path.dirname(args.inference_pth)),
            os.path.basename(args.inference_pth)
        )

        #model_path = os.path.join(args.inference_pth, f'fold_{i+1}/test_best_f1_model.pth')
        model_path = os.path.join(args.inference_pth, f'fold_{i+1}/model_30.pth')
        
        os.makedirs(f'{log_dir}/logs', exist_ok=True)
        log_path = f'{log_dir}/OCCLUSION/{last_two}/{args.note}'
        os.makedirs(log_path, exist_ok=True)

        print(f"Using device: {model_path}")
        print(f'Logging in {log_path}' )

        path_to_test_csv = f'{args.path_to_folds}/fold_{i+1}/test.csv'
        

        test_df = pd.read_csv(path_to_test_csv)

        partition_name = get_partition_name(args.group_partition_name, i, retrain=True)

        

        if args.weighted_patients:
            test_patient_weights = get_normalized_patient_weights(test_df['Slide'].unique())
        else:
            test_patient_weights = None

        
        if args.used_channels != None:
            channel = len(args.used_channels)
        else:
            channel = args.channel
        
        test_dataset = MultiModalCancerDataset(
            args.path_to_images, 
            test_df, mode=args.mode, 
            split='test', size=args.crop_size, 
            used_channels=args.used_channels, not_masked_channels=args.not_masked_channels, partition_name=partition_name, 
            norm_config_path= args.norm_config_path, input_norm = args.input_norm)
        
        test_dataloader = DataLoader(test_dataset, 
                                     batch_size=args.batch_size, shuffle=False, num_workers=16)#, pin_memory= True, persistent_workers=True)
        ###
        for batch in test_dataloader:
            images = batch['image']  # shape: [B, C, H, W]
            labels = batch['label']
            print("Shape:", images.shape)

            # Nur die ersten zwei Beispiele
            print("Image 0:", images[0])
            print("Image 1:", images[1])
            print("Label 0:", labels[0])
            print("Label 1:", labels[1])
            break  # nur erstes Batch inspizieren

        #####
        if args.mode == 'MM':
            model = network.MultimodalNet(model_arch_name = args.model_arch_name, channel=channel, 
                                          fusion_mode=args.fusion_mode, pretrained=args.pretrained)
        else:
            model = network.SinglemodalNet(model_arch_name = args.model_arch_name, channel=channel, 
                                           pretrained=args.pretrained)

        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        #print(model)
        print(f"Mode {mode_name} - #Channel {channel}")


        metrics = test(model, test_dataloader, device, fold_num=i, mode=args.mode, 
                       model_arch_name=args.model_arch_name, mode_name = mode_name, 
                        log_dir = log_dir, test_patient_weights=test_patient_weights, 
                        use_weighted_patients=args.weighted_patients)

        for key, value in metrics.items():
            if value is not None:
                all_metrics[key].append(value)

    # Compute average metrics across folds
    print("\n=== Average Results Across Folds ===")
    # Compute averages
    avg_metrics = {key: sum(vals) / len(vals) for key, vals in all_metrics.items()}
    print("Averaged Results over 3 folds:")
    for k, v in avg_metrics.items():
        print(f"{k}: {v:.4f}")

    # Save averaged results to CSV
    avg_df = pd.DataFrame([avg_metrics])
    avg_csv_path = os.path.join(log_path, "model_30_inference_avg_results.csv")
    avg_df.to_csv(avg_csv_path, index=False)
    print(f"\nSaved average metrics to: {avg_csv_path}")

    # Save per-fold results to CSV
    fold_df = pd.DataFrame(all_metrics)
    fold_df.index = [f"Fold {i+1}" for i in range(fold_df.shape[0])]
    fold_csv_path = os.path.join(log_path, "model_30_inference_per_fold_results.csv")
    fold_df.to_csv(fold_csv_path)
    print(f"Saved per-fold metrics to: {fold_csv_path}")

    return avg_metrics

if __name__ == '__main__':
    args = parser.parse_args()
    run_cv_inference(args)


