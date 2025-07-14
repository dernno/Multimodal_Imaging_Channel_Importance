#!/bin/bash
#SBATCH -A projectname  
#SBATCH -t 12:00:00    
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/out/Ablation_100%j.out

module load Python/3.11.5-GCCcore-13.2.0
source /nora_env/bin/activate  

python /codes/main.py \
	--path_to_folds '/folds/p123' \
	--path_to_images '/data'\
	--mode MM --fusion_mode E --channel 7 --mixup --split_mode "train_test" --pretrained True \
	--project_name 'Ablation_100_ResNet50' \
	--server alvis \
	--model_arch_name resnet50 \
	--note '01236' \
	--group_partition_name 'partition123' \
	--weighted_patients \
	--used_channels 01236 \
	--fold 2