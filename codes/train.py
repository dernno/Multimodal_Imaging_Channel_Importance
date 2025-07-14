import torch
import torch.nn.functional as F
import numpy as np
from sklearn import metrics
from tqdm import tqdm
import wandb
import os

from test import test

def mixup_data(x, y, alpha=1.0, device=None):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
        # lam = np.clip(np.random.beta(alpha, alpha), 0.4, 0.6)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(device)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam, w_a, w_b):
    return lam * criterion(pred, y_a, weight=w_a) + (1 - lam) * criterion(pred, y_b, weight=w_b)
    # return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train(model, optimizer, scheduler, train_dataloader, val_dataloader, test_dataloader=None,
          mode='BF', mode_name='mode_name', model_arch_name='arch_name', device='cuda',
          epochs=100, fold_num=1, use_retrain=False, use_mixup=False, log_dir='log_dir',
          pos_weight=0.78, neg_weight=0.22, 
          train_patient_weights=None, test_patient_weights=None, val_patient_weights=None,
          use_weighted_patients=False):
    
    best_acc, best_epoch, best_test_f1 = 0, 0, 0,
    his, his2, his3 = [], [], []
    k = 0

    # wandb.watch(model, log="all", log_freq=100)

    print(f'Start training: {mode_name}, name: {model_arch_name}')
    for epoch in range(epochs):
        epoch_loss_total = 0.0
        epoch_sample_count = 0

        for i, data in enumerate(train_dataloader):
            image = data['image'].to(device)
            label = data['label'].to(device).float()
            slide = data['slide']
            batch_size = label.size(0)

            if use_mixup:
                if use_weighted_patients:
                    if i == 0:
                        print("##### Weighted-Patients + MIXUP:", train_patient_weights)
                    # Step 1: Prepare slides
                    slide_ids = [f"{int(s):02d}" for s in slide]  # original order
                    index = torch.randperm(batch_size).to(device)

                    image, label_a, label_b, lam = mixup_data(image, label, alpha=0.8, device=device)

                    slide_a = slide_ids
                    slide_b = [slide_ids[i.item()] for i in index]  # new order based on mixup

                    # Step 3: Class weights
                    weight_a = torch.ones_like(label_a).float()
                    weight_a[label_a == 0] *= neg_weight
                    weight_a[label_a == 1] *= pos_weight

                    weight_b = torch.ones_like(label_b).float()
                    weight_b[label_b == 0] *= neg_weight
                    weight_b[label_b == 1] *= pos_weight

                    # Step 4: Patient weights
                    missing_a = [pid for pid in slide_a if pid not in train_patient_weights]
                    missing_b = [pid for pid in slide_b if pid not in train_patient_weights]
                    if missing_a or missing_b:
                        raise ValueError(f"[Mixup Patient Weighting Error] Missing slide IDs: {missing_a + missing_b}")

                    patient_weight_a = torch.tensor(
                        [train_patient_weights[pid] for pid in slide_a],
                        dtype=torch.float32, device=device
                    )
                    patient_weight_b = torch.tensor(
                        [train_patient_weights[pid] for pid in slide_b],
                        dtype=torch.float32, device=device
                    )

                    # Step 5: Final weights
                    weight_a = weight_a * patient_weight_a
                    weight_b = weight_b * patient_weight_b

                    weight_a = weight_a * len(weight_a)
                    weight_b = weight_b * len(weight_b)
                else:
                    if i == 0:
                        print("##### MIXUP:")
                    image, label_a, label_b, lam = mixup_data(image, label, alpha=0.8, device=device)
                    weight_a = torch.ones_like(label_a).float()
                    weight_a[label_a == 0] *= neg_weight
                    weight_a[label_a == 1] *= pos_weight

                    weight_b = torch.ones_like(label_b).float()
                    weight_b[label_b == 0] *= neg_weight
                    weight_b[label_b == 1] *= pos_weight

            if mode == 'MM':
                # Use both modalities: BF and FL
                image = (image[:, :3, ...], image[:, 3:, ...])

            pred = model(image).squeeze(1)
            # score = torch.sigmoid(pred)

            if use_mixup:
                loss = mixup_criterion(
                    F.binary_cross_entropy_with_logits, pred, label_a, label_b, lam, weight_a, weight_b)
                # loss = mixup_criterion(F.cross_entropy, pred, label_a, label_b, lam, weight_a, weight_b)
            else:
                if use_weighted_patients:
                    if i == 0:
                        print("##### Weighted-Patients without MIXUP:", train_patient_weights)
                    # Step 1: Class weights
                    class_weights = torch.where(label == 1, pos_weight, neg_weight)

                    # Step 2: Convert slide ids to two-digit strings like '05'
                    slide_ids = [f"{int(s):02d}" for s in slide]

                    # Step 3: Get patient weights for each slide
                    missing_ids = [pid for pid in slide_ids if pid not in train_patient_weights]
                    if missing_ids:
                        raise ValueError(f"[Patient Weighting Error] The following slide IDs are missing in train_patient_weights: {missing_ids}")

                    patient_weight_tensor = torch.tensor([
                        train_patient_weights[pid] for pid in slide_ids
                    ], dtype=torch.float32, device=device)

                    # Step 4: Combine weights
                    final_weights = class_weights * patient_weight_tensor
                    final_weights = final_weights * len(final_weights)

                    # Step 5: Compute loss
                    loss = F.binary_cross_entropy_with_logits(pred, label, weight=final_weights)
                    # print(final_weights)
                else:
                    # print("##### Normal Loss:")
                    weight = torch.ones_like(label).float()
                    weight[label == 0] *= neg_weight
                    weight[label == 1] *= pos_weight
                    loss = F.binary_cross_entropy_with_logits(pred, label, weight=weight)
                    # loss = F.cross_entropy(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            k += 1
            epoch_loss_total += loss.item() * batch_size
            epoch_sample_count += batch_size

            # if i % 100 == 0:
            #     wandb.log({"Train/Loss": loss})
            #     print(f'epoch: {epoch:03d}, iter: {i:04d}, loss: {loss:.6f}')
        
        scheduler.step()

        #######################################
        if use_retrain == False:
            print("Evaluating on training set...")
            results2 = val(model, train_dataloader, mode, device, pos_weight, neg_weight, use_weighted_patients=use_weighted_patients)
            his2.append(results2)
            log_data = {
                "Train/Loss2": results2['loss'],
                "Train/ROC AUC": results2['ROC AUC'],
                "Train/Precision": results2["Precision"],
                "Train/Recall": results2["Recall"],
                "Train/F1 Score": results2["F1 Score"],
                "Train/Accuracy": results2["Accuracy"],
                "Train/Learning Rate": scheduler.get_last_lr()[0]
            }
            wandb.log(log_data)

            print("Evaluating on validation set...")
            results = val(model, val_dataloader, mode, device, val_patient_weights=val_patient_weights, use_weighted_patients=use_weighted_patients)
            print("Results:")
            print(results)
            his.append(results)
            val_acc = results["Accuracy"]
            log_data = {
                "Validation/Loss": results["loss"],
                "Validation/ROC AUC": results['ROC AUC'],
                "Validation/Precision": results["Precision"],
                "Validation/Recall": results["Recall"],
                "Validation/F1 Score": results["F1 Score"],
                "Validation/Accuracy": results["Accuracy"],
                # "Validation/Example Images": example_images
            }
            wandb.log(log_data)

            if val_acc > best_acc:
                best_acc = val_acc
                best_epoch = epoch
                model_path = os.path.join(log_dir, f'val_best_acc_model.pth')
                torch.save(model.state_dict(), model_path)

            print(f'Epoch: {epoch}, val_auc: {val_acc:.4f}, best_auc: {best_acc:.4f}, best_epoch: {best_epoch}')

        if use_retrain:
            avg_loss = epoch_loss_total / epoch_sample_count
            wandb.log({"Train/Average_Loss": avg_loss}, step=epoch)
            print(f"Epoch {epoch:03d}: Weighted Average Train Loss = {avg_loss:.6f}")

        ###########################################
        print("Evaluating on test set...")
        results3 = test(model, test_dataloader, device, fold_num=fold_num, mode=mode, model_arch_name=model_arch_name, mode_name=mode_name, 
                        log_dir=log_dir, test_patient_weights=test_patient_weights, use_weighted_patients=use_weighted_patients)
        his3.append(results3)
        test_f1 = results3["F1 Score"]
        log_data = {
            "Test/ROC AUC": results3['ROC AUC'],
            "Test/Precision": results3["Precision"],
            "Test/Recall": results3["Recall"],
            "Test/F1 Score": results3["F1 Score"],
            "Test/Accuracy": results3["Accuracy"],
            "Test/FPR": results3["FPR"],
            "Test/FNR": results3["FNR"]
        }
        wandb.log(log_data)

        if epoch == epochs - 1:
            last_model_path = os.path.join(log_dir, f'model_{epoch+1}.pth')
            torch.save(model.state_dict(), last_model_path)

        if test_f1 > best_test_f1:
            best_test_f1 = test_f1
            # best_f1_epoch = epoch
            best_test_f1_model_path = os.path.join(log_dir, f'test_best_f1_model.pth')
            torch.save(model.state_dict(), best_test_f1_model_path)

    print('Training is done!')
    return his2, his, his3

def val(model, dataloader, mode='BF', device='cuda', pos_weight=0.78, neg_weight=0.22, val_patient_weights=None, use_weighted_patients=False):
    model.eval()

    labels, preds, scores, names = [], [], [], []
    for data in tqdm(dataloader):
        X = data['image'].to(device)
        y = data['label'].to(device).float()
        f_name = data['name']

        if mode == 'MM':
            X = (X[:, :3, ...], X[:, 3:, ...])  # BF and FL

        with torch.no_grad():
            y_ = model(X).squeeze(1)
            score = torch.sigmoid(y_)

        scores.extend(score.tolist())
        labels.extend(y.tolist())
        names.extend(list(f_name))
        pred = (score > 0.50).int()
        preds.extend(pred.tolist())

    conf_matrix = metrics.confusion_matrix(labels, preds)
    auc_score = metrics.roc_auc_score(labels, scores)
    precision = metrics.precision_score(labels, preds)
    recall = metrics.recall_score(labels, preds)
    f1 = metrics.f1_score(labels, preds)
    accuracy = metrics.accuracy_score(labels, preds)

    # Patient-weighted loss computation (different from test function)
    if use_weighted_patients:
        class_weights = torch.where(y == 1, pos_weight, neg_weight)
        slide = data['slide']
        slide_ids = [f"{int(s):02d}" for s in slide]

        if val_patient_weights is not None:
            missing_ids = [pid for pid in slide_ids if pid not in val_patient_weights]
            if missing_ids:
                raise ValueError(f"[Validation Patient Weighting Error] Missing slide IDs in val_patient_weights: {missing_ids}")

            patient_weight_tensor = torch.tensor([
                val_patient_weights[pid] for pid in slide_ids
            ], dtype=torch.float32, device=device)

            final_weights = class_weights * patient_weight_tensor
            final_weights = final_weights * len(final_weights)
        else:
            final_weights = class_weights

        loss = F.binary_cross_entropy_with_logits(y_, y, weight=final_weights)
    else:
        weight = torch.ones_like(y).float()
        weight[y == 0] *= neg_weight
        weight[y == 1] *= pos_weight
        loss = F.binary_cross_entropy_with_logits(y_, y, weight=weight)

    results = {
        'Confusion Matrix': conf_matrix,
        'ROC AUC': auc_score,
        'Precision': precision,
        'Recall': recall,
        'F1 Score': f1,
        'Accuracy': accuracy,
        'loss': loss
    }
    model.train()

    return results
