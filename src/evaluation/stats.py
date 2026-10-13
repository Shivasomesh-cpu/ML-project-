import numpy as np
from scipy.stats import permutation_test

def mean_statistic(sample, axis):
    return np.mean(sample, axis=axis)

def run_permutation_test_classification(results, fs_key, models_to_test, n_resamples=9999):
    """
    Bonferroni-corrected permutation test comparing foundation models vs FreeSurfer fold-level accuracy.
    Only tests models whose mean accuracy strictly exceeds FreeSurfer.
    """
    perm_results = {}
    if fs_key not in results or 'fold_results_alpha1' not in results[fs_key]:
        return perm_results

    fs_folds = np.array(results[fs_key]['fold_results_alpha1']['val_acc'])
    fs_mean = np.mean(fs_folds)

    models_beating_fs = [
        m for m in models_to_test
        if m != fs_key and m in results and 'Dummy' not in m
        and 'fold_results_alpha1' in results[m]
        and np.mean(results[m]['fold_results_alpha1']['val_acc']) > fs_mean
    ]
    n_tests = len(models_beating_fs)
    if n_tests == 0:
        return perm_results

    for model in models_beating_fs:
        model_folds = np.array(results[model]['fold_results_alpha1']['val_acc'])
        diff = model_folds - fs_folds
        res = permutation_test(
            (diff,),
            mean_statistic,
            permutation_type='samples',
            n_resamples=min(n_resamples, 1000), # configurable or safe upper bound
            vectorized=True,
            axis=0,
            alternative='greater'
        )
        p_corr = min(res.pvalue * n_tests, 1.0)
        perm_results[model] = {
            'p_raw': float(res.pvalue),
            'p_corr': float(p_corr),
            'n_tests': n_tests,
            'diff': float(np.mean(diff)),
            'significant': bool(p_corr < 0.05)
        }
        print(f"    {model}: diff={np.mean(diff):.4f}, p_raw={res.pvalue:.4f}, p_corr={p_corr:.4f}")

    return perm_results


def run_permutation_test_regression(results, fs_key, models_to_test, n_resamples=9999):
    """
    Bonferroni-corrected permutation test comparing foundation models vs FreeSurfer fold-level MAE.
    Only tests models whose mean MAE is strictly lower than FreeSurfer.
    """
    perm_results = {}
    if fs_key not in results or 'fold_results_alpha1' not in results[fs_key]:
        return perm_results

    fs_folds = np.array(results[fs_key]['fold_results_alpha1']['val_mae'])
    fs_mean = np.mean(fs_folds)

    models_beating_fs = [
        m for m in models_to_test
        if m != fs_key and m in results and 'Dummy' not in m
        and 'fold_results_alpha1' in results[m]
        and np.mean(results[m]['fold_results_alpha1']['val_mae']) < fs_mean
    ]
    n_tests = len(models_beating_fs)
    if n_tests == 0:
        return perm_results

    for model in models_beating_fs:
        model_folds = np.array(results[model]['fold_results_alpha1']['val_mae'])
        diff = model_folds - fs_folds
        res = permutation_test(
            (diff,),
            mean_statistic,
            permutation_type='samples',
            n_resamples=min(n_resamples, 1000),
            vectorized=True,
            axis=0,
            alternative='less'
        )
        p_corr = min(res.pvalue * n_tests, 1.0)
        perm_results[model] = {
            'p_raw': float(res.pvalue),
            'p_corr': float(p_corr),
            'n_tests': n_tests,
            'diff': float(np.mean(diff)),
            'significant': bool(p_corr < 0.05)
        }
        print(f"    {model}: diff={np.mean(diff):.4f}, p_raw={res.pvalue:.4f}, p_corr={p_corr:.4f}")

    return perm_results
