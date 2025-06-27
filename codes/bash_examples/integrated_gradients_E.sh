#!/bin/bash
#SBATCH -A NAISS2024-22-1260   # Ersetze mit deinem Projekt-Konto auf Snowy
#SBATCH -t 3:00:00       # Laufzeit:  1h
#SBATCH --gres=gpu:A40:1
#SBATCH --output=/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/out/integrated_gradients%j.out

# Lade das Python-Modul, falls erforderlich
module load Python/3.11.5-GCCcore-13.2.0

# Aktiviere die virtuelle Umgebung
source /mimer/NOBACKUP/groups/snic2021-7-128/nora/nora_env/bin/activate  # Pfad zu deiner venv

python /mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/codes/integrated_gradients.py \
	--path_to_folds '/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/folds/FINAL_FOLDS/p123_20' \
	--path_to_images '/mimer/NOBACKUP/groups/snic2021-7-128/nora/data'\
	--mode MM6 --fusion_mode E --channel 6 --mixup --split_mode "train_test" --pretrained True \
	--server alvis \
	--model_arch_name resnet18 \
	--group_partition_name 'partition123' \
	--weighted_patients \
	--inference_pth '/mimer/NOBACKUP/groups/snic2021-7-128/nora/logs/ABLATION_K6_P123_20_ResNet18_012356/resnet18_multimodal_early_fusion'\
	--note 'Test' \
	--used_channels '012356' \
