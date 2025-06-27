# MA_Nora_Channel_Importance
Implementation for MA Thesis: Evaluating channel importance in CNN-based classification of oral cancer cytology using perfectly aligned brightfield–fluorescence image pairs from Pap-stained whole-slide images (patch size 256 × 256).

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

### dataloader.py
This module defines the MultiModalCancerDataset:
- Loads patch-level images and corresponding metadata from a dataframe
- Reads normalization parameters (mean/std) dynamically from a JSON configuration
- Implements flexible input normalization schemes (standard z-score, clipped z-score, log transform, contrast stretching)
- Includes data augmentation for BF and FL images with configurable strategies
- Dynamic channel removal for ablation experiements
- Dynamic channel masking for occlusion experiments
- Returns dictionary containing: transformed image tensor, label (0 or 1), slide identifier (e.g. 01), patch identifier (e.g. 2044)

### train.py
The train.py module implements the core training pipeline. It supports four configurable training modes:
  1) No mixup, no patient weighting
  2) Mixup only 
  3) Patient weighting only
  4) Mixup combined with patient weighting
- Integrates mixup data augmentation to improve generalization (Mixup creates synthetic training samples by combining pairs of images and their labels using a convex linear combination, encouraging the model to learn smoother decision boundaries)
- Allows patient-level weighting to correct for potential bias in patient sampling e.g., {'05': 0.67, '07': 0.03, '09': 0.13, '15': 0.08, '16': 0.08} (Sum to 1)

The module uses standard binary cross-entropy loss with optional class balancing and patient weighting. Validation and test performance are monitored per epoch.






