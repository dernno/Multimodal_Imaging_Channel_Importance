#!/bin/bash
#SBATCH -A projectname   
#SBATCH -t 3:00:00      
#SBATCH --gres=gpu:A40:1
#SBATCH --output=/out/integrated_gradients%j.out

module load Python/3.11.5-GCCcore-13.2.0
source /nora_env/bin/activate 

python /codes/integrated_gradients.py \
	--path_to_folds '/folds/p123_20' \
	--path_to_images '/data'\
	--mode MM6 --fusion_mode E --channel 6 --mixup --split_mode "train_test" --pretrained True \
	--server alvis \
	--model_arch_name resnet18 \
	--group_partition_name 'partition123' \
	--weighted_patients \
	--inference_pth '/logs/ABLATION_K6_P123_20_ResNet18_012356/resnet18_multimodal_early_fusion'\
	--note 'Test' \
	--used_channels '012356' \
