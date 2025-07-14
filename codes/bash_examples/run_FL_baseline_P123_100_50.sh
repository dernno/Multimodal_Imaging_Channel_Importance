#!/bin/bash
#SBATCH -A projectname
#SBATCH -t 4-00:00:00     
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/out/Baseline_P123_100_ResNet50%j.out

module load Python/3.11.5-GCCcore-13.2.0
source /nora_env/bin/activate 

python /codes/main.py \
	--path_to_folds '/folds/p123' \
	--path_to_images '/data'\
	--mode FL --channel 4 --mixup --split_mode "train_test" --pretrained True \
	--project_name 'Baseline_P123_100_ResNet50' \
	--server alvis \
	--model_arch_name resnet50 \
	--note 'FL only 50 all' \
	--group_partition_name 'partition123' \
	--weighted_patients