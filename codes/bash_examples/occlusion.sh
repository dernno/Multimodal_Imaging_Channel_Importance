#!/bin/bash
#SBATCH -A projectname   
#SBATCH -t 1:00:00       
#SBATCH --gres=gpu:A40:1
#SBATCH --output=/out/Occlusion%j.out

module load Python/3.11.5-GCCcore-13.2.0
source /nora_env/bin/activate  

python /codes/inference.py \
	--path_to_folds '/folds/p123_20' \
	--path_to_images '/data'\
	--mode MM --fusion_mode E --channel 7 --mixup --split_mode "train_test" --pretrained True \
	--server alvis \
	--model_arch_name resnet18 \
	--group_partition_name 'partition123' \
	--weighted_patients \
	--inference_pth '/logs/Baseline_P123_20_ResNet18_Early Fusion/resnet18_multimodal_early_fusion' \
	--not_masked_channels 0123456 \
	--note '0123456'