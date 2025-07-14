#!/bin/bash
#SBATCH -A projectname
#SBATCH -t 1:00:00      
#SBATCH --gres=gpu:A40:1
#SBATCH --output=/out/Baseline_P123_20_ResNet18_inference%j.out

module load Python/3.11.5-GCCcore-13.2.0
source /nora_env/bin/activate 

python /codes/inference.py \
	--path_to_folds '/folds/p123_20' \
	--path_to_images '/data'\
	--mode FL --channel 4 --mixup --split_mode "train_test" --pretrained True \
	--server alvis \
	--model_arch_name resnet18 \
	--group_partition_name 'partition123' \
	--weighted_patients \
	--inference_pth '/logs/Baseline_P123_20_ResNet18_FL only/resnet18_FL only train inputnorm'\
	--note '345' \
	--not_masked_channels 123
