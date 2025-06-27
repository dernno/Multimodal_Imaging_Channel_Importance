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

parser.add_argument('--fold', type=int, default=0, help='fold number: [0, 1, 2]')
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
parser.add_argument('--note', type=str, default='note', help='Extra Info for run')

parser.add_argument('--group_partition_name', type=str, default='wenyi', help='wenyi, partition123, partition456')
parser.add_argument('--norm_config_path', type=str, default='/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/codes/input_norm/norm_config.json', help='jsonfile')
#parser.add_argument('--analyse_inputs', action='store_true', help='Analyze_inputs')
parser.add_argument('--input_norm', type=str, default = 'zscore')

parser.add_argument('--BF_aug', nargs="*", type=str, default=None, help='[posterize, solarize, gaussian, colorjitter, brightness_contrast,sharpness, autocontrast, erase]')
parser.add_argument('--FL_aug',  nargs="*", type=str, default=None, help='[colorjitter, gaussian, noise, intensity_shift, erase]')
parser.add_argument('--not_masked_channels', type=str, default=None, help='[0123]')
#parser.add_argument('--loss_type',type=str, default='bce', help= 'bce or focal')

def main(args):
    start_time = time.time()

    if args.server == 'mida1':
        log_dir = '/home/nora/data2_nora'
    elif args.server == 'alvis':
        log_dir = '/mimer/NOBACKUP/groups/snic2021-7-128/nora'
    else:
        log_dir = '/home/nora/work_nora'  # mida2

    for i in range(3):#3
        if i != args.fold:
           continue
        ####
        # nur fold2-3   
        #if i == 0: #or i == 1:
        #    continue  # Überspringt Fold 0 und macht mit 1 weiter

        print(f"Running Fold {i+1}")  
        ####
        #i = args.fold
        mode_name = args.mode
        if mode_name =="MM":
            if args.fusion_mode == "E":
                mode_name="multimodal_early_fusion"
            elif args.fusion_mode == "L":
                mode_name="multimodal_late_fusion"
        else:
            mode_name=f"{mode_name} only"

        os.makedirs(f'{log_dir}/logs', exist_ok=True)
        log_path = f'{log_dir}/logs/{args.project_name}_{args.note}/{args.model_arch_name}_{mode_name}/fold_{i+1}'
        os.makedirs(log_path, exist_ok=True)
                
        path_to_train_csv = f'{args.path_to_folds}/fold_{i+1}/train.csv'
        path_to_val_csv = f'{args.path_to_folds}/fold_{i+1}/val.csv'
        path_to_test_csv = f'{args.path_to_folds}/fold_{i+1}/test.csv'

        train_df = pd.read_csv(path_to_train_csv)
        val_df = pd.read_csv(path_to_val_csv)
        #val_df = None
        test_df = pd.read_csv(path_to_test_csv)


        # Add Val-Data to Train-Data
        if args.split_mode=="train_test":
            #train_df = pd.concat([train_df, None], ignore_index=True)
            train_df = pd.concat([train_df, val_df], ignore_index=True)
            val_df = None
            retrain = True
        else: 
            retrain = False
        
        partition_name = get_partition_name(args.group_partition_name, i, retrain)

        ## Class-Weights
        #pos_weight, neg_weight= 0.78, 0.22
        pos_weight, neg_weight = compute_class_weights(train_df)

        ## Patient-Weights
        train_patient_weights = get_normalized_patient_weights(train_df['Slide'].unique())
        test_patient_weights = get_normalized_patient_weights(test_df['Slide'].unique())
        val_patient_weights=None
        if val_df is not None:
            val_patient_weights = get_normalized_patient_weights(val_df['Slide'].unique())
        
        print(train_patient_weights, test_patient_weights)
        
        

        #group = args.run_name
        if args.mode=='MM':
            if args.fusion_mode=='E':
                group='Early Fusion'
            elif args.fusion_mode=='L':
                group="Late Fusion"
            # elif args.fusion_mode=='I':
            #     group=args.name
        else:
            group=mode_name

        if args.used_channels != None:
            channel = len(args.used_channels)
        else:
            channel = args.channel


        wandb.run = wandb.init(reinit=True, 
                                project = args.project_name,
                                #name=f'{mode_name}_{args.note}_fold_{i+1}', #mode_name, 'w/o c=0'
                                name = f'{args.note}',
                                #name=f'lr{args.lr}_wd{args.wd}_bs{args.batch_size}_e{args.epochs}_opt{args.optimizer}',
                                group=group, #'w/o c=0'
                                job_type=args.split_mode,
                                config={
                                    "architecture": args.model_arch_name,
                                    "channel": channel,
                                    #"log_name":args.run_name,
                                    "learning_rate": args.lr,
                                    "epochs": args.epochs,
                                    "weight_decay": args.wd,
                                    "crop_size":args.crop_size,
                                    "batch_size":args.batch_size,
                                    "mixup": args.mixup,
                                    "weighted_patients": args.weighted_patients,
                                    "pretrained": args.pretrained,
                                    "gpu_name": gpu_name,
                                    "fold_name": args.path_to_folds,
                                    "ablation_combi": args.used_channels,
                                    "Label_1_weight": pos_weight,
                                    "Label_0_weight": neg_weight,
                                    "Partition_Group": args.group_partition_name,
                                    "Partition_Name": partition_name,
                                    "Note": args.note,
                                    "Input_Norm": args.input_norm,
                                },)

#######################################################
 
        
        train_dataset = MultiModalCancerDataset(
            args.path_to_images, train_df, mode=args.mode, split='train', size=args.crop_size, used_channels=args.used_channels, not_masked_channels=args.not_masked_channels, partition_name=partition_name, norm_config_path= args.norm_config_path, input_norm=args.input_norm, BF_aug=args.BF_aug, FL_aug=args.FL_aug)
        val_dataset = MultiModalCancerDataset(
            args.path_to_images, val_df, mode=args.mode, split='val', size=args.crop_size, used_channels=args.used_channels, not_masked_channels=args.not_masked_channels, partition_name=partition_name, norm_config_path= args.norm_config_path, input_norm= args.input_norm)
        test_dataset = MultiModalCancerDataset(
            args.path_to_images, test_df, mode=args.mode, split='test', size=args.crop_size, used_channels=args.used_channels, not_masked_channels=args.not_masked_channels, partition_name=partition_name, norm_config_path= args.norm_config_path, input_norm = args.input_norm)
        
        # if args.analyse_inputs:
        #     analyze_dataset(dataset=train_dataset, log_dir=log_path, dataset_name="Train")
        #     analyze_dataset(dataset=test_dataset, log_dir=log_path, dataset_name="Test")
        # else:
        #     save_random_patch_visualization(train_dataset, analysis_dir=log_path, dataset_name="Train")
        #     save_random_patch_visualization(train_dataset, analysis_dir=log_path, dataset_name="Test")

        ###
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16, worker_init_fn=seed_worker, generator=g)
        val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)
        test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=16)#, pin_memory= True, persistent_workers=True)

        #####
        if args.mode == 'MM':
            model = network.MultimodalNet(model_arch_name = args.model_arch_name, channel=channel, fusion_mode=args.fusion_mode, pretrained=args.pretrained)
        else:
            model = network.SinglemodalNet(model_arch_name = args.model_arch_name, channel=channel, pretrained=args.pretrained)

        #torch.cuda.empty_cache()
        model = model.to(device)
        print(model)
        print(f"Mode {mode_name} - #Channel {channel}")

        if args.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)
        elif args.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.wd, momentum=0.9)
        elif args.optimizer == 'adamw':
            optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        else:
            raise ValueError(f"Unknown optimizer: {args.optimizer}")

        optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-7)
        print(f"Fold {i+1}:")
        
        train_log,val_log,test_log = train(model, optimizer, scheduler,train_dataloader, val_dataloader, 
                                           test_dataloader=test_dataloader, mode=args.mode, mode_name = mode_name, 
                                           model_arch_name=args.model_arch_name, device=device, epochs=args.epochs, 
                                           fold_num=i+1, use_retrain=retrain, use_mixup=args.mixup,log_dir=log_path,
                                           pos_weight = pos_weight, neg_weight= neg_weight, 
                                           train_patient_weights = train_patient_weights, test_patient_weights=test_patient_weights, val_patient_weights=val_patient_weights, 
                                           use_weighted_patients=args.weighted_patients)#, loss_type=args.loss_type)

        save_list_of_dicts_to_txt(f'{log_path}/train_log.txt', train_log)
        save_list_of_dicts_to_txt(f'{log_path}/val_log.txt', val_log)
        save_list_of_dicts_to_txt(f'{log_path}/test_log.txt', test_log)
        

    wandb.finish()            
    end_time = time.time()
    elapsed_time = end_time - start_time
    # Convert to hours, minutes, and seconds format
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    # Format the elapsed time as a string
    elapsed_time_str = f"{int(hours)} hours, {int(minutes)} minutes, {int(seconds)} seconds"
    print(f"Total training and test time: {elapsed_time_str}")

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)

