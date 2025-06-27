# MA_Nora_Channel_Importance
Implementation for MA Thesis: Evaluating channel importance in CNN-based classification of oral cancer cytology using perfectly aligned brightfield–fluorescence image pairs from Pap-stained whole-slide images (patch size 256 × 256 pixel).

MIDA Group. LetItShine: Multimodal image analysis codebase, https://github.com/MIDA-group/LetItShine serves as the initial codebase and is substantially extended to incorporate methodological improvements (patient weighting, dynamic class weighting, as well as channel importance analysis tools (ablation, occlusion, integrated-gradients)).

## main.py
The main.py script orchestrates a cross-validation setup (default: 3 folds) for multimodal image classification. It supports multiple processing modes, including:
- MM (Multimodal): (E) Early Fusion: combines all modalities at the input level (7-channel input)
- MM (Multimodal): (L) Late Fusion: keeps the modalities fully separated throughout the entire feature extraction stage. Each modality is passed through an independent backbone network (3 + 7-3 channel input)
- BF: BF only (3-channel input)
- FL: FL only (4-channel input)

For each fold, the script reads patient slide assignments for training, validation, and test sets from a fold.csv file. The data split can be configured via a split mode:
- train_test: merges validation data into training
- train_val_test: maintains three distinct groups

For the training data, class weights are computed to account for label imbalance between class 0 (benign) and class 1 (malignant slide). The script then creates dataloaders (based on the MultimodalDataset implemented in data/dataloader.py) for training, validation, and test data.

Model architectures are instantiated from archs/network.py based on the chosen modality (MM(E: One 7-Input ResNet, L: Two ResNets combined before FC, BF: One 3-Input ResNet, FL: One 4-Input ResNet).

Finally, the standard training (train.py) and evaluation (test.py) routines are executed for each fold.

### dataloader.py
This module defines the MultiModalCancerDataset:
- Loads patch-level images and corresponding metadata from a dataframe
- Reads normalization parameters (mean/std) dynamically from a JSON configuration
- Implements flexible input normalization schemes (standard z-score, clipped z-score, log transform, contrast stretching)
- Includes data augmentation for BF and FL images with configurable strategies
- Dynamic channel removal for ablation experiments (e.g., self.used_channels="0123" removes channels 4,5,6)
- Dynamic channel masking for occlusion experiments (e.g., self.not_masked_channels="0123" sets channels 4,5,6 to zero)
- Returns dictionary containing: transformed image tensor, label (0 or 1), slide identifier (e.g. 01), patch identifier (e.g. 2044)

### train.py
The train.py module implements the core training pipeline. It supports four configurable training modes:
  1) No mixup, no patient weighting
  2) Mixup only 
  3) Patient weighting only
  4) Mixup combined with patient weighting
- Mixup data augmentation creates synthetic training samples by combining pairs of images and their labels using a convex linear combination, encouraging the model to learn smoother decision boundaries
- Patient-level weighting to correct for potential bias in patient sampling e.g., {'05': 0.67, '07': 0.03, '09': 0.13, '15': 0.08, '16': 0.08} (Sum to 1) based on number of images

The module uses standard binary cross-entropy loss with optional class balancing and patient weighting. Validation and test performance are monitored per epoch.

## folds
Four different fold configurations are provided:
- p123: Partitions 1–3
- p456: Partitions 4–6
- p12345: 5-fold partition across all data
- wenyi: original letitshine partition with 14 patients in the evaluation set

Each of these configurations additionally supports stratified subsampling options:

- suffix _5: uses only 5% of the data
- suffix _20: uses 20% of the data
- no suffix: uses 100% of the data

## sampling/sample_slide.ipynb
Custom data reduction strategy generates mutually exclusive e.g. 20% (dynamic) subsets using stratified random sampling per slide, preserving class balance and patient slide allocation. 


