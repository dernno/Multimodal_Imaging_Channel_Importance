#!/bin/bash
#SBATCH -A NAISS2024-22-1260   # Ersetze mit deinem Projekt-Konto auf Snowy
#SBATCH -t 4-00:00:00     # 4 Tage, 0 Stunden, 0 Minuten
#SBATCH --gres=gpu:A100:1
#SBATCH --output=/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/out/Baseline_P123_100_ResNet50%j.out

# Lade das Python-Modul, falls erforderlich
module load Python/3.11.5-GCCcore-13.2.0

# Aktiviere die virtuelle Umgebung
source /mimer/NOBACKUP/groups/snic2021-7-128/nora/nora_env/bin/activate  # Pfad zu deiner venv

python /mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/codes/main.py \
	--path_to_folds '/mimer/NOBACKUP/groups/snic2021-7-128/nora/letitshine/folds/FINAL_FOLDS/p123' \
	--path_to_images '/mimer/NOBACKUP/groups/snic2021-7-128/nora/data'\
	--mode BF --channel 3 --mixup --split_mode "train_test" --pretrained True \
	--project_name 'Baseline_P123_100_ResNet50' \
	--server alvis \
	--model_arch_name resnet50 \
	--note 'BF_only 50 all' \
	--group_partition_name 'partition123' \
	--weighted_patients