import os
import shutil
import torch
import yaml

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from matplotlib import cm
import random

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
    
def get_partition_name(group_partition_name, fold_index, retrain):
    if group_partition_name == 'wenyi':
        return f'wenyi_fold{fold_index + 1}'
    
    elif group_partition_name == 'partition123':
        if retrain:
            return {
                0: 'partition2_3',
                1: 'partition1_3',
                2: 'partition1_2'
            }.get(fold_index)
        else:
            return {
                0: 'partition3',
                1: 'partition1',
                2: 'partition2'
            }.get(fold_index)
        
    elif group_partition_name == 'partition456':
        if retrain:
            return {
                0: 'partition5_6',
                1: 'partition4_6',
                2: 'partition4_5'
            }.get(fold_index)
        else:
            return {
                0: 'partition6',
                1: 'partition4',
                2: 'partition5'
            }.get(fold_index)
        
    elif group_partition_name == 'partition12345':
        if retrain:
            return {
                0: 'partition2_3_4_5',
                1: 'partition1_3_4_5',
                2: 'partition1_2_4_5',
                3: 'partition1_2_3_5',
                4: 'partition1_2_3_4',
            }.get(fold_index)
        else:
            return {
                0: 'cv5_partition1',
                1: 'cv5_partition2',
                2: 'cv5_partition3',
                3: 'cv5_partition4',
                4: 'cv5_partition5',
            }.get(fold_index)

    elif group_partition_name == 'whole':
        return 'whole'
    
    elif group_partition_name == 'imagenet':
        return 'imagenet'
    
    else:
        return None
    
def save_list_of_dicts_to_txt(file_path, dict_list):
    with open(file_path, 'w') as txtfile:
        for index, my_dict in enumerate(dict_list):
            txtfile.write(f"{index}: {my_dict}\n")

def compute_class_weights(df, label_column='Diagnosis'):
    counts = df[label_column].value_counts(normalize=True)
    pos_freq = counts.get(1, 0.0)
    neg_freq = counts.get(0, 0.0)

    pos_weight = 1.0 - pos_freq
    neg_weight = 1.0 - neg_freq

    total = pos_weight + neg_weight
    if total > 0:
        pos_weight /= total
        neg_weight /= total

    return pos_weight, neg_weight

### DATA ANALYSIS

def analyze_dataset(dataset, log_dir, dataset_name="Dataset"):
    analysis_dir = os.path.join(log_dir, "Input_Data_Analysis", dataset_name)
    os.makedirs(analysis_dir, exist_ok=True)

    # ---------- 1. Create dataframe with per-channel statistics ----------
    all_stats = []
    for i in range(len(dataset)):
        sample = dataset[i]
        img_tensor = sample['image']
        data_row = dataset.df.iloc[i]
        label = data_row['Diagnosis']
        slide = data_row['Slide']
        patch = data_row['Patch']

        if isinstance(img_tensor, torch.Tensor):
            for c in range(img_tensor.size(0)):
                channel = img_tensor[c]
                all_stats.append({
                    'channel': str(c),
                    'min': channel.min().item(),
                    'max': channel.max().item(),
                    'mean': channel.mean().item(),
                    'std': channel.std().item(),
                    'label': label,
                    'slide': slide,
                    'patch': patch
                })

    df_stats = pd.DataFrame(all_stats)
    csv_path = os.path.join(analysis_dir, f"{dataset_name}_channel_stats.csv")
    df_stats.to_csv(csv_path, index=False)
    print(f"Channel statistics saved to: {csv_path}")

    # ---------- 2. Define color palette for channels ----------
    def create_channel_palette(df):
        channels = sorted(df['channel'].astype(int).unique())
        palette = {}
        grays = ["#cccccc", "#999999", "#666666"]
        blues = cm.get_cmap("Blues", len(channels) - 3 if len(channels) > 3 else 1)
        for c in channels:
            key = str(c)
            if c <= 2:
                palette[key] = grays[c]
            else:
                rgba = blues(c - 3)
                palette[key] = matplotlib.colors.rgb2hex(rgba)
        return palette

    channel_palette = create_channel_palette(df_stats)

    # Slide information for plot titles
    slide_nums = sorted(df_stats['slide'].unique())
    slide_str = ', '.join([f"{int(s):02d}" for s in slide_nums])
    full_title = f"{dataset_name} (Slides {slide_str})"

    # Color palette for diagnosis labels
    palette = {0: "#67a865", 1: "#d95f5f"}

    # ---------- 3. Save general boxplot ----------
    def save_boxplot(df, x, y, hue=None, title="", filename="plot"):
        n_channels = df['channel'].nunique()
        fig_width = max(8, n_channels * 1.5)
        plt.figure(figsize=(fig_width, 6))

        if hue:
            sns.boxplot(data=df, x=x, y=y, hue=hue,
                        palette=palette if hue == 'label' else channel_palette)
        else:
            sns.boxplot(data=df, x=x, y=y, palette=channel_palette)

        plt.title(title)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(analysis_dir, f"{filename}.{ext}"), dpi=300)
        plt.close()


    # ---------- 4. Boxplot: mean per channel ----------
    save_boxplot(df_stats, x='channel', y='mean',
                 title=f"Mean per Channel – {full_title}",
                 filename=f"{dataset_name}_boxplot_channel")

    # ---------- 5. Boxplot: one plot per slide ----------
    for slide in df_stats['slide'].unique():
        df_slide = df_stats[df_stats['slide'] == slide]
        labels = sorted(df_slide['label'].unique())
        label_str = ', '.join(str(int(l)) for l in labels)
        title = f"Slide {int(slide):02d} – Labels: {label_str}"
        fig_width = max(8, df_slide['channel'].nunique() * 1.5)

        plt.figure(figsize=(fig_width, 6))
        sns.boxplot(data=df_slide, x="channel", y="mean", palette=channel_palette)
        plt.title(title)
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            filename = f"{dataset_name}_boxplot_slide_{int(slide):02d}.{ext}"
            plt.savefig(os.path.join(analysis_dir, filename), dpi=300)
        plt.close()

    # ---------- 6. Boxplot: mean per channel by diagnosis ----------
    save_boxplot(df_stats, x='channel', y='mean', hue='label',
                 title=f"Mean per Channel by Diagnosis – {full_title}",
                 filename=f"{dataset_name}_boxplot_diagnosis")

    # ---------- 7. Histograms: for mean, std, min, max ----------
    def plot_channel_histogram(df, stat="mean"):
        df['channel'] = df['channel'].astype(str)
        channels = sorted(df['channel'].unique())
        n_channels = len(channels)
        fig, axs = plt.subplots(n_channels, 1, figsize=(8, 2.2 * n_channels))
        if n_channels == 1:
            axs = [axs]
        for i, ch in enumerate(channels):
            ax = axs[i]
            sns.histplot(df[df['channel'] == ch][stat], bins=40, kde=True, ax=ax, color=channel_palette[ch])
            ax.set_title(f"Channel {ch} – {stat}")
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            path = os.path.join(analysis_dir, f"{dataset_name}_hist_{stat}.{ext}")
            plt.savefig(path, dpi=300)
            print(f"Histogram saved to: {path}")
        plt.close()

    for stat in ["mean", "std", "min", "max"]:
        plot_channel_histogram(df_stats, stat=stat)

    # ---------- 8. PCA and t-SNE on FL channels ----------
    print("Running PCA and t-SNE on FL channels...")
    X = []
    y = []

    for i in range(len(dataset)):
        sample = dataset[i]
        img_tensor = sample['image']
        label = dataset.df.iloc[i]['Diagnosis']
        n_channels = img_tensor.size(0)

        if n_channels > 3:
            fl = img_tensor[3:]
            features = [fl[c].mean().item() for c in range(fl.size(0))]
            X.append(features)
            y.append(label)
        else:
            print(f"Sample {i} has only {n_channels} channels – FL channels missing.")

    def plot_embed(X_embedded, title, filename):
        plt.figure(figsize=(8, 6))
        sns.scatterplot(x=X_embedded[:, 0], y=X_embedded[:, 1], hue=y, palette=palette, s=50)
        plt.title(f"{title} – {dataset_name}")
        plt.xlabel("Dim 1")
        plt.ylabel("Dim 2")
        plt.legend(title="Diagnosis")
        plt.tight_layout()
        for ext in ['png', 'pdf']:
            plt.savefig(os.path.join(analysis_dir, f"{filename}.{ext}"), dpi=300)
        plt.close()

    X = np.array(X)
    y = np.array(y)

    if len(X) == 0:
        print("No FL channels found – skipping PCA and t-SNE.")
    else:
        X_pca = PCA(n_components=2).fit_transform(X)
        plot_embed(X_pca, "PCA on FL Channels", f"{dataset_name}_pca")

        if len(X) < 6:
            print(f"Only {len(X)} samples – skipping t-SNE.")
        else:
            perplexity = min(30, max(5, (len(X) - 1) // 3))
            print(f"Running t-SNE with perplexity={perplexity} on {len(X)} samples...")
            X_tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, random_state=42).fit_transform(X)
            plot_embed(X_tsne, "t-SNE on FL Channels", f"{dataset_name}_tsne")

    # ---------- 9. Save visualization of one random patch ----------
    def save_random_patch_visualization(dataset, analysis_dir, dataset_name="Dataset", index=None):
        index = index if index is not None else random.randint(0, len(dataset) - 1)
        sample = dataset[index]
        img = sample['image']
        slide = dataset.df.iloc[index]['Slide']
        patch_id = dataset.df.iloc[index]['Patch']

        n_channels = img.shape[0]
        fig, axs = plt.subplots(1, n_channels, figsize=(2.2 * n_channels, 2.2))
        if n_channels == 1:
            axs = [axs]

        for i in range(n_channels):
            ax = axs[i]
            ax.imshow(img[i].numpy(), cmap="gray")
            ax.set_title(f"C{i}", fontsize=6)
            ax.axis("off")

        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout(pad=0.2)

        filename_base = f"{dataset_name}_slide{slide:02d}_patch{patch_id}_channels"
        for ext in ["png", "pdf"]:
            path = os.path.join(analysis_dir, f"{filename_base}.{ext}")
            plt.savefig(path, dpi=300)
            print(f"Patch visualization saved to: {path}")
        plt.close()

    save_random_patch_visualization(dataset, analysis_dir=analysis_dir, dataset_name=dataset_name)

    print(f"All plots and analyses saved to: {analysis_dir}")

 # ---------- 9. Save visualization of one random patch ----------
def save_random_patch_visualization(dataset, analysis_dir, dataset_name="Dataset", index=None):
    index = index if index is not None else random.randint(0, len(dataset) - 1)
    sample = dataset[index]
    img = sample['image']
    slide = dataset.df.iloc[index]['Slide']
    patch_id = dataset.df.iloc[index]['Patch']

    n_channels = img.shape[0]
    fig, axs = plt.subplots(1, n_channels, figsize=(2.2 * n_channels, 2.2))
    if n_channels == 1:
        axs = [axs]

    for i in range(n_channels):
        ax = axs[i]
        ax.imshow(img[i].numpy(), cmap="gray")
        ax.set_title(f"C{i}", fontsize=6)
        ax.axis("off")

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.tight_layout(pad=0.2)

    filename_base = f"{dataset_name}_slide{slide:02d}_patch{patch_id}_channels"
    for ext in ["png", "pdf"]:
        path = os.path.join(analysis_dir, f"{filename_base}.{ext}")
        plt.savefig(path, dpi=300)
        print(f"Patch visualization saved to: {path}")
    plt.close()

###########
# Anzahl Patches pro Patient (Index = Patient ID - 1)
all_patch_counts = np.array([
    33707, 12686, 26196, 7057, 4302, 99320, 87290, 79518, 22980, 68251,
    19616, 30860, 70997, 73005, 34644, 35303, 16327, 42523, 1983
])

def get_normalized_patient_weights(patient_ids):
    """
    Gibt ein Dictionary mit normalisierten Gewichten für eine Teilmenge von Patienten zurück.

    Args:
        patient_ids (list[int]): Liste von Patienten-IDs (1-basiert, z. B. [1, 4, 5, 6])

    Returns:
        dict[str, float]: Mapping von Patient-ID als '02'-String → normalisiertes Gewicht
    """
    # Hole Patch-Zahlen für diese Patienten
    selected_counts = {pid: all_patch_counts[pid - 1] for pid in patient_ids}
    
    # Schritt 1: Inverse Patch-Zahlen
    inv_counts = {pid: 1.0 / count for pid, count in selected_counts.items()}

    # Schritt 2: Normalisieren auf Summe 1
    total_inv = sum(inv_counts.values())
    normalized_weights = {
        f"{pid:02d}": inv / total_inv for pid, inv in inv_counts.items()
    }

    return normalized_weights

    

