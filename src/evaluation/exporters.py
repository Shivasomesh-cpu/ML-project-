import os
import numpy as np
import pandas as pd

def get_training_sizes(n_total, alphas, test_size=0.1, n_folds=5):
    n_test = int(test_size * n_total)
    n_cv = n_total - n_test
    n_train_fold = max(1, n_cv - (n_cv // n_folds))
    return [max(1, int(a * n_train_fold)) for a in alphas]


def export_cv_results_classification(results, training_sizes, dataset_name, task_name, models_list, alphas):
    rows = []
    for model in models_list:
        if model not in results:
            continue
        cv_res = results[model].get('cv_results', {})
        for i, alpha in enumerate(alphas):
            if cv_res and alpha in cv_res.get('acc', {}):
                rows.append({
                    'dataset': dataset_name,
                    'task': task_name,
                    'model': model,
                    'alpha': alpha,
                    'n_train': training_sizes[i] if i < len(training_sizes) else np.nan,
                    'val_acc_mean': cv_res['acc'].get(alpha, np.nan),
                    'val_acc_std': cv_res['acc_std'].get(alpha, np.nan),
                    'val_auc_mean': cv_res.get('auc', {}).get(alpha, np.nan),
                    'val_auc_std': cv_res.get('auc_std', {}).get(alpha, np.nan),
                    'train_acc_mean': cv_res.get('train_acc', {}).get(alpha, np.nan),
                    'train_acc_std': cv_res.get('train_acc_std', {}).get(alpha, np.nan),
                    'train_auc_mean': cv_res.get('train_auc', {}).get(alpha, np.nan),
                    'train_auc_std': cv_res.get('train_auc_std', {}).get(alpha, np.nan),
                    'test_acc_mean': results[model].get('test_acc_mean', np.nan),
                    'test_acc_std': results[model].get('test_acc_std', np.nan),
                    'test_auc_mean': results[model].get('test_auc_mean', np.nan),
                    'test_auc_std': results[model].get('test_auc_std', np.nan),
                })
    if 'DummyClassifier' in results:
        dummy = results['DummyClassifier']
        for i, alpha in enumerate(alphas):
            rows.append({
                'dataset': dataset_name,
                'task': task_name,
                'model': 'Dummy',
                'alpha': alpha,
                'n_train': training_sizes[i] if i < len(training_sizes) else np.nan,
                'val_acc_mean': dummy['cv_acc_mean'],
                'val_acc_std': dummy['cv_acc_std'],
                'val_auc_mean': dummy['cv_auc_mean'],
                'val_auc_std': dummy['cv_auc_std'],
                'train_acc_mean': np.nan,
                'train_acc_std': np.nan,
                'train_auc_mean': np.nan,
                'train_auc_std': np.nan,
                'test_acc_mean': dummy['test_acc_mean'],
                'test_acc_std': dummy['test_acc_std'],
                'test_auc_mean': dummy['test_auc_mean'],
                'test_auc_std': dummy['test_auc_std'],
            })
    return pd.DataFrame(rows)


def export_cv_results_regression(results, training_sizes, dataset_name, task_name, models_list, alphas):
    rows = []
    for model in models_list:
        if model not in results:
            continue
        cv_res = results[model].get('cv_results', {})
        for i, alpha in enumerate(alphas):
            if cv_res and alpha in cv_res.get('mae', {}):
                rows.append({
                    'dataset': dataset_name,
                    'task': task_name,
                    'model': model,
                    'alpha': alpha,
                    'n_train': training_sizes[i] if i < len(training_sizes) else np.nan,
                    'val_mae_mean': cv_res['mae'].get(alpha, np.nan),
                    'val_mae_std': cv_res['mae_std'].get(alpha, np.nan),
                    'val_r2_mean': cv_res.get('r2', {}).get(alpha, np.nan),
                    'val_r2_std': cv_res.get('r2_std', {}).get(alpha, np.nan),
                    'train_mae_mean': cv_res.get('train_mae', {}).get(alpha, np.nan),
                    'train_mae_std': cv_res.get('train_mae_std', {}).get(alpha, np.nan),
                    'train_r2_mean': cv_res.get('train_r2', {}).get(alpha, np.nan),
                    'train_r2_std': cv_res.get('train_r2_std', {}).get(alpha, np.nan),
                    'test_mae_mean': results[model].get('test_mae_mean', np.nan),
                    'test_mae_std': results[model].get('test_mae_std', np.nan),
                    'test_r2_mean': results[model].get('test_r2_mean', np.nan),
                    'test_r2_std': results[model].get('test_r2_std', np.nan),
                })
    if 'DummyRegressor' in results:
        dummy = results['DummyRegressor']
        for i, alpha in enumerate(alphas):
            rows.append({
                'dataset': dataset_name,
                'task': task_name,
                'model': 'Dummy',
                'alpha': alpha,
                'n_train': training_sizes[i] if i < len(training_sizes) else np.nan,
                'val_mae_mean': dummy['cv_mae_mean'],
                'val_mae_std': dummy['cv_mae_std'],
                'val_r2_mean': dummy['cv_r2_mean'],
                'val_r2_std': dummy['cv_r2_std'],
                'train_mae_mean': np.nan,
                'train_mae_std': np.nan,
                'train_r2_mean': np.nan,
                'train_r2_std': np.nan,
                'test_mae_mean': dummy['test_mae_mean'],
                'test_mae_std': dummy['test_mae_std'],
                'test_r2_mean': dummy['test_r2_mean'],
                'test_r2_std': dummy['test_r2_std'],
            })
    return pd.DataFrame(rows)


def build_fold_level_dataframe(dataset_tasks):
    fold_rows = []
    for ds_name, task_results_list in dataset_tasks:
        for task, res in task_results_list:
            for model, r in res.items():
                if 'Dummy' in model:
                    continue
                fr = r.get('fold_results_alpha1', {})
                if 'val_acc' in fr:
                    for i, val in enumerate(fr['val_acc']):
                        fold_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'fold': i, 'metric': 'val_acc', 'value': val
                        })
                if 'val_mae' in fr:
                    for i, val in enumerate(fr['val_mae']):
                        fold_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'fold': i, 'metric': 'val_mae', 'value': val
                        })
    return pd.DataFrame(fold_rows)


def build_heldout_dataframe(dataset_tasks, seeds):
    heldout_rows = []
    for ds_name, task_results_list in dataset_tasks:
        for task, res in task_results_list:
            for model, r in res.items():
                if 'Dummy' in model:
                    continue
                tsr = r.get('test_seed_results', {})
                if 'test_acc' in tsr:
                    for i, val in enumerate(tsr['test_acc']):
                        heldout_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'seed': seeds[i] if i < len(seeds) else i, 'metric': 'test_acc', 'value': val
                        })
                if 'test_auc' in tsr:
                    for i, val in enumerate(tsr['test_auc']):
                        heldout_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'seed': seeds[i] if i < len(seeds) else i, 'metric': 'test_auc', 'value': val
                        })
                if 'test_mae' in tsr:
                    for i, val in enumerate(tsr['test_mae']):
                        heldout_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'seed': seeds[i] if i < len(seeds) else i, 'metric': 'test_mae', 'value': val
                        })
                if 'test_r2' in tsr:
                    for i, val in enumerate(tsr['test_r2']):
                        heldout_rows.append({
                            'dataset': ds_name, 'task': task, 'model': model,
                            'seed': seeds[i] if i < len(seeds) else i, 'metric': 'test_r2', 'value': val
                        })
    return pd.DataFrame(heldout_rows)
