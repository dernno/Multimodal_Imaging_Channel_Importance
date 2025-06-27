# MA_Nora_Channel_Importance
Implementation MA Thesis: Evaluating Channel Importance using CNN for cytological image classification of oral cancer. Dataset: Perfectly aligned Brightfield and Fluorescence Image Pairs

## main.py
The main.py script orchestrates a cross-validation setup (default: 3 folds) for multimodal image classification. It supports multiple processing modes, including:
- MM (Multimodal): (E) Early Fusion: combines all modalities at the input level (7-channel input)
- MM (Multimodal): (L) Late Fusion keeps the modalities fully separated throughout the entire feature extraction stage. Each modality is passed through an independent backbone network (3 + 7-3 Channel Inputs)
- BF: BF only (3-channel input)
- FL: FL only (4-channel input)

For each fold, the script reads patient slide assignments for training, validation, and test sets from a fold.csv file. The data split can be configured via a split mode:
- train_test: merges validation data into training
- train_val_test: maintains three distinct groups

For the training data, class weights are computed to account for label imbalance between class 0 and class 1. The script then creates dataloaders (based on the MultimodalDataset implemented in data/dataloader.py) for training, validation, and test data.

Model architectures are instantiated from archs/network.py based on the chosen modality (MM(E/L), BF, FL).

Finally, the standard training (train.py) and evaluation (test.py) routines are executed for each fold.

