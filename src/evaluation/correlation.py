import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from scipy.spatial.distance import squareform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def compute_pairwise_correlations(features_dict, dataset_name):
    """
    Computes and prints mean absolute correlation between feature sets of different models.
    """
    print(f"\n{'='*60}")
    print(f"MEAN ABSOLUTE CORRELATIONS - {dataset_name}")
    print(f"{'='*60}")
    model_names = list(features_dict.keys())
    scaled_features = {}
    for model in model_names:
        X = StandardScaler().fit_transform(features_dict[model])
        valid = ~np.isnan(X).any(axis=0) & (np.std(X, axis=0) > 1e-10)
        scaled_features[model] = X[:, valid]

    corr_results = {}
    for i, model_i in enumerate(model_names):
        for j, model_j in enumerate(model_names):
            if j <= i:
                continue
            X_i, X_j = scaled_features[model_i], scaled_features[model_j]
            if X_i.shape[1] == 0 or X_j.shape[1] == 0:
                corr_results[(model_i, model_j)] = 0.0
                continue
            cross_corr = np.corrcoef(X_i.T, X_j.T)[:X_i.shape[1], X_i.shape[1]:]
            corr_results[(model_i, model_j)] = float(np.mean(np.abs(cross_corr)))

    fs_keys = [k for k in model_names if 'FreeSurfer' in k or 'FS_' in k or k == 'FreeSurfer']
    for fs_key in fs_keys:
        print(f"\nvs {fs_key}:")
        for model in model_names:
            if model == fs_key:
                continue
            key = (model, fs_key) if (model, fs_key) in corr_results else (fs_key, model)
            val = corr_results.get(key, 0.0)
            print(f"  {model} vs {fs_key}: {val:.4f}")

    print(f"\nAll pairwise (sorted by correlation):")
    for (m1, m2), corr in sorted(corr_results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {m1} vs {m2}: {corr:.4f}")

    return corr_results


def compute_correlation_with_cluster_labels(features_dict, fs_feature_infos, fs_keys, dataset_name):
    """
    Hierarchically clusters features within each model and forms a unified cross-model correlation matrix.
    """
    all_features, boundaries, reordered_indices, model_names_used, cluster_dfs = [], [0], {}, [], {}
    preferred_order = [
        'AnatCL_Global', 'AnatCL_Local', 'BrainIAC', 'SwinBrain',
        'CNN_0', 'CNN_42', '3D-Neuro-SimCLR'
    ] + fs_keys

    model_order = [m for m in preferred_order if m in features_dict] + [m for m in features_dict if m not in preferred_order]

    for model in model_order:
        feats = StandardScaler().fit_transform(features_dict[model])
        valid = ~np.isnan(feats).any(axis=0) & (np.std(feats, axis=0) > 1e-10)
        feats = feats[:, valid]
        if feats.shape[1] == 0:
            continue
        print(f"  {model}: {feats.shape[1]} features")

        corr = np.corrcoef(feats.T)
        corr = np.nan_to_num((corr + corr.T) / 2, nan=0.0, posinf=1.0, neginf=-1.0)
        np.fill_diagonal(corr, 1.0)

        if feats.shape[1] > 1:
            dist = np.abs(1 - np.abs(corr))
            np.fill_diagonal(dist, 0)
            condensed = np.nan_to_num(squareform((dist + dist.T) / 2, checks=False), nan=0.0, posinf=1.0, neginf=0.0)
            reorder_idx = dendrogram(linkage(condensed, method='ward'), no_plot=True)['leaves']
        else:
            reorder_idx = [0]

        if model in fs_keys and model in fs_feature_infos:
            fs_info = fs_feature_infos[model]
            thickness_indices = [i for i in reorder_idx if i < len(fs_info) and fs_info[i].get('feature_type') == 'Thickness']
            surfarea_indices  = [i for i in reorder_idx if i < len(fs_info) and fs_info[i].get('feature_type') == 'Surface_Area']
            other_indices     = [i for i in reorder_idx if i not in thickness_indices and i not in surfarea_indices]
            reorder_idx = thickness_indices + surfarea_indices + other_indices

        all_features.append(feats[:, reorder_idx])
        boundaries.append(boundaries[-1] + feats.shape[1])
        reordered_indices[model] = reorder_idx
        model_names_used.append(model)

    if not all_features:
        return np.zeros((0, 0)), boundaries, [], {}, {}

    combined = np.hstack(all_features)
    corr = np.nan_to_num((np.corrcoef(combined.T) + np.corrcoef(combined.T).T) / 2, nan=0.0, posinf=1.0, neginf=-1.0)
    np.fill_diagonal(corr, 1.0)

    for fs_key in fs_keys:
        if fs_key in reordered_indices and fs_key in fs_feature_infos:
            fs_info = fs_feature_infos[fs_key]
            cluster_df = pd.DataFrame([
                {**fs_info[idx], 'original_index': idx, 'reordered_position': pos}
                for pos, idx in enumerate(reordered_indices[fs_key]) if idx < len(fs_info)
            ])
            cluster_dfs[fs_key] = cluster_df

    return corr, boundaries, model_names_used, reordered_indices, cluster_dfs


def plot_correlation_with_labels(corr_matrix, boundaries, model_names, cluster_dfs, fs_keys, dataset_name, output_dir='.'):
    """
    Renders and saves full cross-model pairwise correlation heatmap.
    """
    if corr_matrix.size == 0:
        return

    fig, ax = plt.subplots(figsize=(16, 15))
    im = ax.imshow(corr_matrix, cmap='RdBu_r', vmin=-1, vmax=1, aspect='equal', interpolation='nearest')
    plt.colorbar(im, ax=ax, shrink=0.3)

    for b in boundaries[1:-1]:
        ax.axhline(y=b - 0.5, color='black', linewidth=2)
        ax.axvline(x=b - 0.5, color='black', linewidth=2)

    bar_width = max(2, int(0.015 * corr_matrix.shape[0]))
    colors = {'Surface_Area': '#2ECC71', 'Thickness': '#3498DB'}

    for fs_key in fs_keys:
        if fs_key not in cluster_dfs or fs_key not in model_names:
            continue
        cluster_df = cluster_dfs[fs_key]
        fs_idx = model_names.index(fs_key)
        fs_start = boundaries[fs_idx]
        for i, ft in enumerate(cluster_df.get('feature_type', []).values):
            color = colors.get(ft, '#95A5A6')
            ax.add_patch(plt.Rectangle((boundaries[-1] + 2, fs_start + i - 0.5), bar_width, 1, facecolor=color, edgecolor='none'))
            ax.add_patch(plt.Rectangle((-bar_width - 3, fs_start + i - 0.5), bar_width, 1, facecolor=color, edgecolor='none'))

    display_names = {'FS_Schaefer': 'FS: Schaefer', 'FS_aparc': 'FS: aparc', 'FreeSurfer': 'FS'}
    label_fontsize = 12

    for i in range(len(boundaries) - 1):
        pos = (boundaries[i] + boundaries[i+1]) / 2
        label = display_names.get(model_names[i], model_names[i])
        ax.text(pos, -int(corr_matrix.shape[0] * 0.04), label, ha='center', fontsize=label_fontsize, weight='bold')
        ax.text(-int(corr_matrix.shape[0] * 0.04), pos, label, ha='right', va='center', fontsize=label_fontsize, weight='bold', rotation=90)

    offset = bar_width + int(corr_matrix.shape[0] * 0.06)
    ax.set_xlim(-offset, boundaries[-1] + offset)
    ax.set_ylim(boundaries[-1] + int(corr_matrix.shape[0] * 0.04), -offset)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    ax.legend(
        handles=[
            Patch(facecolor='#2ECC71', edgecolor='black', label='Surface Area'),
            Patch(facecolor='#3498DB', edgecolor='black', label='Thickness')
        ],
        loc='upper left', fontsize=12, bbox_to_anchor=(1.02, 1.0), frameon=True
    )
    plt.tight_layout()

    out_png = os.path.join(output_dir, f"{dataset_name.lower()}_cross_model_correlation.png")
    out_pdf = os.path.join(output_dir, f"{dataset_name.lower()}_cross_model_correlation.pdf")
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_png}")
    print(f"Saved: {out_pdf}")


def export_correlation_data(corr_matrix, boundaries, model_names, cluster_dfs, dataset_name, output_dir='.'):
    """
    Saves raw correlation matrix and metadata for downstream analysis.
    """
    prefix = os.path.join(output_dir, dataset_name.lower())
    np.save(f'{prefix}_corr_matrix.npy', corr_matrix)
    meta_rows = []
    for i, model in enumerate(model_names):
        meta_rows.append({
            'dataset': dataset_name,
            'model': model,
            'boundary_start': boundaries[i],
            'boundary_end': boundaries[i+1],
            'n_features': boundaries[i+1] - boundaries[i]
        })
    pd.DataFrame(meta_rows).to_csv(f'{prefix}_corr_metadata.csv', index=False)
    print(f"Saved: {prefix}_corr_matrix.npy ({corr_matrix.shape})")
    print(f"Saved: {prefix}_corr_metadata.csv")
