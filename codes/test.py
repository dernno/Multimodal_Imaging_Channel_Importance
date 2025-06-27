import torch
from sklearn import metrics
from tqdm import tqdm
import pandas as pd

def test(model, dataloader, device='cuda', fold_num=1, mode="BF", model_arch_name="resnet", 
         mode_name='mode_name', log_dir='log_dir', test_patient_weights=None, use_weighted_patients=False):

    model.eval()

    labels, preds, scores, names, slides = [], [], [], [], []

    first = True
    for data in tqdm(dataloader):
        X = data['image'].to(device)
        y = data['label'].float().to(device)
        f_name = data['name']
        slide = data['slide']  # Patient ID (e.g., 05)

        if mode == 'MM':
            X = (X[:, :3, ...], X[:, 3:, ...])

        with torch.no_grad():
            y_ = model(X).squeeze(1)
            score = torch.sigmoid(y_)

        scores.extend(score.tolist())
        labels.extend(y.tolist())
        names.extend(list(f_name))
        slides.extend([f"{int(s):02d}" for s in slide])
        pred = (score > 0.5).int()
        preds.extend(pred.tolist())

    model.train()

    if use_weighted_patients:
        if first:
            print("Patient-weighted evaluation...", test_patient_weights)
            first = False

        # Retrieve patient weight for each sample
        patient_weights = []
        for pid in slides:
            if pid not in test_patient_weights:
                raise ValueError(f"[Test Patient Weighting Error] Missing patient ID {pid} in test_patient_weights.")
            patient_weights.append(test_patient_weights[pid])

        # Weighted confusion matrix components
        weighted_TP = weighted_FP = weighted_TN = weighted_FN = 0

        for label, pred, weight in zip(labels, preds, patient_weights):
            if label == 1 and pred == 1:
                weighted_TP += weight
            elif label == 0 and pred == 1:
                weighted_FP += weight
            elif label == 0 and pred == 0:
                weighted_TN += weight
            elif label == 1 and pred == 0:
                weighted_FN += weight

        # Weighted metrics
        precision = weighted_TP / (weighted_TP + weighted_FP + 1e-8)
        recall = weighted_TP / (weighted_TP + weighted_FN + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        accuracy = (weighted_TP + weighted_TN) / (weighted_TP + weighted_FP + weighted_TN + weighted_FN + 1e-8)
        auc_score = metrics.roc_auc_score(labels, scores, sample_weight=patient_weights)

        # No meaningful confusion matrix for weighted, but return dummy
        conf_matrix = None
        FPR = weighted_FP / (weighted_FP + weighted_TN + 1e-8)
        FNR = weighted_FN / (weighted_FN + weighted_TP + 1e-8)


    else:
        if first:
            print("Standard (non-weighted) evaluation...")
            first = False

        conf_matrix = metrics.confusion_matrix(labels, preds)
        TN, FP, FN, TP = conf_matrix.ravel()

        auc_score = metrics.roc_auc_score(labels, scores)
        precision = metrics.precision_score(labels, preds)
        recall = metrics.recall_score(labels, preds)
        f1 = metrics.f1_score(labels, preds)
        accuracy = metrics.accuracy_score(labels, preds)

        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        FNR = FN / (FN + TP) if (FN + TP) > 0 else 0

    results = {
        'Confusion Matrix': conf_matrix,
        'ROC AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'FPR': FPR,
        'FNR': FNR,
    }

    # df = pd.DataFrame()

    # df['name'] = names
    # df['label'] = labels
    # df['score'] = scores
    # df['pred'] = preds

    # df['TP'] = ((df['label'] == 1) & (df['pred'] == 1)).astype(int)
    # df['TN'] = ((df['label'] == 0) & (df['pred'] == 0)).astype(int)
    # df['FP'] = ((df['label'] == 0) & (df['pred'] == 1)).astype(int)
    # df['FN'] = ((df['label'] == 1) & (df['pred'] == 0)).astype(int)
    # print(results)
    # excel_path = os.path.join(log_dir, f'result_{fold_num}.xlsx')
    # #os.makedirs(f'{log_dir}/excels/{model_arch_name}_{mode_name}/', exist_ok=True)
    # df.to_excel(excel_path, index=False)

    print(results)
    return results
