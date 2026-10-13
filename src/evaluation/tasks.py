import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier, DummyRegressor

DEFAULT_SEEDS = [0, 1, 2, 3, 42]
DEFAULT_ALPHAS = [0.01, 0.05, 0.1, 0.15, 0.25, 0.4, 0.6, 0.8, 1.0]

def run_classification(X_dict, y, test_size=0.1, n_splits=5, seeds=None, alphas=None, task_name="Classification"):
    """
    Evaluates representation features on classification task using Random Forest probe.
    Protocol:
      - seeds x n_splits stratified cross-validation
      - Held-out test set per seed
      - Learning curve across training fractions (alphas)
      - Metrics: Balanced Accuracy and ROC-AUC
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    print(f"\n--- {task_name} (N={len(y)}) ---")
    if len(y) < 4:
        print(f"  Warning: Only {len(y)} subjects available. Skipping task to avoid degenerate splits.")
        return {}
    results = {}

    dummy_cv_acc, dummy_cv_auc, dummy_test_acc, dummy_test_auc = [], [], [], []
    unique_cls, counts = np.unique(y, return_counts=True)
    can_stratify = len(unique_cls) > 1 and min(counts) >= 2 and len(y) >= 2 * len(unique_cls)
    test_cnt = max(len(unique_cls), int(test_size * len(y))) if can_stratify else max(1, int(test_size * len(y)))
    if test_cnt >= len(y):
        test_cnt = max(1, len(y) // 2)

    for seed in seeds:
        cv_idx, test_idx = train_test_split(
            np.arange(len(y)),
            test_size=test_cnt,
            stratify=y if can_stratify else None,
            random_state=seed
        )

        n_splits_actual = max(2, min(n_splits, len(cv_idx)))
        cv_unique, cv_counts = np.unique(y[cv_idx], return_counts=True)
        cv_can_strat = len(cv_unique) > 1 and min(cv_counts) >= n_splits_actual
        if cv_can_strat:
            skf = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
            split_gen = skf.split(cv_idx, y[cv_idx])
        else:
            kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
            split_gen = kf.split(cv_idx)

        for train_rel, val_rel in split_gen:
            train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
            dummy = DummyClassifier(strategy='stratified', random_state=seed)
            dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
            pred_val = dummy.predict(np.zeros((len(val_idx), 1)))
            dummy_cv_acc.append(balanced_accuracy_score(y[val_idx], pred_val))
            try:
                prob = dummy.predict_proba(np.zeros((len(val_idx), 1)))[:, 1]
                dummy_cv_auc.append(roc_auc_score(y[val_idx], prob))
            except Exception:
                dummy_cv_auc.append(0.5)

        dummy = DummyClassifier(strategy='stratified', random_state=seed)
        dummy.fit(np.zeros((len(cv_idx), 1)), y[cv_idx])
        pred_test = dummy.predict(np.zeros((len(test_idx), 1)))
        dummy_test_acc.append(balanced_accuracy_score(y[test_idx], pred_test))
        try:
            prob_test = dummy.predict_proba(np.zeros((len(test_idx), 1)))[:, 1]
            dummy_test_auc.append(roc_auc_score(y[test_idx], prob_test))
        except Exception:
            dummy_test_auc.append(0.5)

    results['DummyClassifier'] = {
        'best_alpha': 1.0,
        'test_acc_mean': float(np.mean(dummy_test_acc)), 'test_acc_std': float(np.std(dummy_test_acc)),
        'test_auc_mean': float(np.mean(dummy_test_auc)), 'test_auc_std': float(np.std(dummy_test_auc)),
        'cv_auc_mean': float(np.mean(dummy_cv_auc)), 'cv_auc_std': float(np.std(dummy_cv_auc)),
        'cv_acc_mean': float(np.mean(dummy_cv_acc)), 'cv_acc_std': float(np.std(dummy_cv_acc)),
        'fold_results': {'val_auc': dummy_cv_auc, 'val_acc': dummy_cv_acc},
        'test_seed_results': {'test_acc': dummy_test_acc, 'test_auc': dummy_test_auc},
    }
    print(f"DummyClassifier: AUC={np.mean(dummy_test_auc):.3f} +/- {np.std(dummy_test_auc):.3f}, Acc={np.mean(dummy_test_acc):.3f} +/- {np.std(dummy_test_acc):.3f}")

    for model, X in X_dict.items():
        all_val_auc = {a: [] for a in alphas}
        all_val_acc = {a: [] for a in alphas}
        all_train_auc = {a: [] for a in alphas}
        all_train_acc = {a: [] for a in alphas}
        all_test_acc, all_test_auc = [], []

        for seed in seeds:
            cv_idx, test_idx = train_test_split(
                np.arange(len(y)),
                test_size=test_cnt,
                stratify=y if can_stratify else None,
                random_state=seed
            )

            n_splits_actual = max(2, min(n_splits, len(cv_idx)))
            cv_unique, cv_counts = np.unique(y[cv_idx], return_counts=True)
            cv_can_strat = len(cv_unique) > 1 and min(cv_counts) >= n_splits_actual
            if cv_can_strat:
                skf = StratifiedKFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
                split_gen = skf.split(cv_idx, y[cv_idx])
            else:
                kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
                split_gen = kf.split(cv_idx)

            for train_rel, val_rel in split_gen:
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                for alpha in alphas:
                    n = max(2, int(alpha * len(train_idx)))
                    if n >= len(train_idx):
                        tr_idx = train_idx
                    else:
                        tr_unique, tr_counts = np.unique(y[train_idx], return_counts=True)
                        can_tr_strat = (
                            len(tr_unique) > 1 and
                            min(tr_counts) >= 2 and
                            n >= len(tr_unique) and
                            (len(train_idx) - n) >= len(tr_unique)
                        )
                        tr_idx, _ = train_test_split(
                            train_idx,
                            train_size=n,
                            stratify=y[train_idx] if can_tr_strat else None,
                            random_state=seed
                        )
                    if len(np.unique(y[tr_idx])) < 2:
                        continue

                    scaler = StandardScaler()
                    rf = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=6,
                        min_samples_split=max(2, min(5, len(tr_idx))),
                        random_state=seed,
                        n_jobs=1,
                        max_features='sqrt',
                        class_weight='balanced'
                    )
                    X_tr = scaler.fit_transform(X[tr_idx])
                    X_val = scaler.transform(X[val_idx])
                    rf.fit(X_tr, y[tr_idx])

                    pred_val = rf.predict(X_val)
                    all_val_acc[alpha].append(balanced_accuracy_score(y[val_idx], pred_val))
                    try:
                        prob_val = rf.predict_proba(X_val)[:, 1]
                        all_val_auc[alpha].append(roc_auc_score(y[val_idx], prob_val))
                    except Exception:
                        all_val_auc[alpha].append(0.5)

                    pred_tr = rf.predict(X_tr)
                    all_train_acc[alpha].append(balanced_accuracy_score(y[tr_idx], pred_tr))
                    try:
                        prob_tr = rf.predict_proba(X_tr)[:, 1]
                        all_train_auc[alpha].append(roc_auc_score(y[tr_idx], prob_tr))
                    except Exception:
                        all_train_auc[alpha].append(0.5)

            scaler = StandardScaler()
            rf = RandomForestClassifier(
                n_estimators=100,
                max_depth=6,
                min_samples_split=max(2, min(5, len(cv_idx))),
                random_state=seed,
                n_jobs=1,
                max_features='sqrt',
                class_weight='balanced'
            )
            X_cv = scaler.fit_transform(X[cv_idx])
            X_test = scaler.transform(X[test_idx])
            rf.fit(X_cv, y[cv_idx])

            pred_test = rf.predict(X_test)
            all_test_acc.append(balanced_accuracy_score(y[test_idx], pred_test))
            try:
                prob_test = rf.predict_proba(X_test)[:, 1]
                all_test_auc.append(roc_auc_score(y[test_idx], prob_test))
            except Exception:
                all_test_auc.append(0.5)

        avg_auc = {a: float(np.mean(all_val_auc[a])) for a in alphas if len(all_val_auc[a]) > 0}
        best_alpha = max(avg_auc, key=avg_auc.get) if avg_auc else 1.0
        train_val_gaps = {
            a: float(np.mean(all_train_auc[a]) - np.mean(all_val_auc[a]))
            for a in alphas if len(all_train_auc[a]) > 0 and len(all_val_auc[a]) > 0
        }

        results[model] = {
            'best_alpha': best_alpha,
            'test_acc_mean': float(np.mean(all_test_acc)), 'test_acc_std': float(np.std(all_test_acc)),
            'test_auc_mean': float(np.mean(all_test_auc)), 'test_auc_std': float(np.std(all_test_auc)),
            'cv_auc_mean': float(np.mean(all_val_auc[best_alpha])) if all_val_auc[best_alpha] else 0.5,
            'cv_auc_std': float(np.std(all_val_auc[best_alpha])) if all_val_auc[best_alpha] else 0.0,
            'cv_acc_mean': float(np.mean(all_val_acc[best_alpha])) if all_val_acc[best_alpha] else 0.5,
            'cv_acc_std': float(np.std(all_val_acc[best_alpha])) if all_val_acc[best_alpha] else 0.0,
            'train_val_gap': train_val_gaps.get(best_alpha, 0.0),
            'is_overfitting': train_val_gaps.get(best_alpha, 0.0) > 0.25,
            'fold_results': {
                'val_auc': all_val_auc[best_alpha],
                'val_acc': all_val_acc[best_alpha]
            },
            'fold_results_alpha1': {
                'val_acc': all_val_acc[1.0] if all_val_acc[1.0] else all_val_acc[best_alpha],
                'val_auc': all_val_auc[1.0] if all_val_auc[1.0] else all_val_auc[best_alpha]
            },
            'test_seed_results': {'test_acc': all_test_acc, 'test_auc': all_test_auc},
            'cv_results': {
                'alphas': alphas,
                'acc': {a: float(np.mean(all_val_acc[a])) for a in alphas if all_val_acc[a]},
                'auc': avg_auc,
                'acc_std': {a: float(np.std(all_val_acc[a])) for a in alphas if all_val_acc[a]},
                'auc_std': {a: float(np.std(all_val_auc[a])) for a in alphas if all_val_auc[a]},
                'train_acc': {a: float(np.mean(all_train_acc[a])) for a in alphas if all_train_acc[a]},
                'train_acc_std': {a: float(np.std(all_train_acc[a])) for a in alphas if all_train_acc[a]},
                'train_auc': {a: float(np.mean(all_train_auc[a])) for a in alphas if all_train_auc[a]},
                'train_auc_std': {a: float(np.std(all_train_auc[a])) for a in alphas if all_train_auc[a]},
            }
        }
        print(f"{model}: AUC={results[model]['test_auc_mean']:.3f} +/- {results[model]['test_auc_std']:.3f}, Acc={results[model]['test_acc_mean']:.3f} +/- {results[model]['test_acc_std']:.3f}")

    return results


def run_regression(X_dict, y, test_size=0.1, n_splits=5, seeds=None, alphas=None, task_name="Regression"):
    """
    Evaluates representation features on regression task (Age prediction) using Random Forest probe.
    Protocol:
      - seeds x n_splits KFold cross-validation
      - Held-out test set per seed
      - Learning curve across training fractions (alphas)
      - Metrics: MAE and R^2
    """
    if seeds is None:
        seeds = DEFAULT_SEEDS
    if alphas is None:
        alphas = DEFAULT_ALPHAS

    print(f"\n--- {task_name} (N={len(y)}) ---")
    if len(y) < 4:
        print(f"  Warning: Only {len(y)} subjects available. Skipping task to avoid degenerate splits.")
        return {}
    results = {}

    test_cnt = max(1, int(test_size * len(y)))
    if test_cnt >= len(y):
        test_cnt = max(1, len(y) // 2)

    dummy_cv_r2, dummy_cv_mae, dummy_test_r2, dummy_test_mae = [], [], [], []
    for seed in seeds:
        test_cnt = max(1, int(test_size * len(y)))
        cv_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_cnt, random_state=seed)

        n_splits_actual = min(n_splits, len(cv_idx))
        kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
        for train_rel, val_rel in kf.split(cv_idx):
            train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
            dummy = DummyRegressor(strategy='mean')
            dummy.fit(np.zeros((len(train_idx), 1)), y[train_idx])
            pred_val = dummy.predict(np.zeros((len(val_idx), 1)))
            dummy_cv_r2.append(r2_score(y[val_idx], pred_val))
            dummy_cv_mae.append(mean_absolute_error(y[val_idx], pred_val))

        dummy = DummyRegressor(strategy='mean')
        dummy.fit(np.zeros((len(cv_idx), 1)), y[cv_idx])
        pred_test = dummy.predict(np.zeros((len(test_idx), 1)))
        dummy_test_r2.append(r2_score(y[test_idx], pred_test))
        dummy_test_mae.append(mean_absolute_error(y[test_idx], pred_test))

    results['DummyRegressor'] = {
        'best_alpha': 1.0,
        'test_r2_mean': float(np.mean(dummy_test_r2)), 'test_r2_std': float(np.std(dummy_test_r2)),
        'test_mae_mean': float(np.mean(dummy_test_mae)), 'test_mae_std': float(np.std(dummy_test_mae)),
        'cv_r2_mean': float(np.mean(dummy_cv_r2)), 'cv_r2_std': float(np.std(dummy_cv_r2)),
        'cv_mae_mean': float(np.mean(dummy_cv_mae)), 'cv_mae_std': float(np.std(dummy_cv_mae)),
        'fold_results': {'val_r2': dummy_cv_r2, 'val_mae': dummy_cv_mae},
        'test_seed_results': {'test_mae': dummy_test_mae, 'test_r2': dummy_test_r2},
    }
    print(f"DummyRegressor: MAE={np.mean(dummy_test_mae):.2f} +/- {np.std(dummy_test_mae):.2f}, R2={np.mean(dummy_test_r2):.3f} +/- {np.std(dummy_test_r2):.3f}")

    for model, X in X_dict.items():
        all_val_r2 = {a: [] for a in alphas}
        all_val_mae = {a: [] for a in alphas}
        all_train_r2 = {a: [] for a in alphas}
        all_train_mae = {a: [] for a in alphas}
        all_test_r2, all_test_mae = [], []

        for seed in seeds:
            cv_idx, test_idx = train_test_split(np.arange(len(y)), test_size=test_cnt, random_state=seed)

            n_splits_actual = max(2, min(n_splits, len(cv_idx)))
            kf = KFold(n_splits=n_splits_actual, shuffle=True, random_state=seed)
            for train_rel, val_rel in kf.split(cv_idx):
                train_idx, val_idx = cv_idx[train_rel], cv_idx[val_rel]
                for alpha in alphas:
                    n = max(2, int(alpha * len(train_idx)))
                    tr_idx, _ = train_test_split(train_idx, train_size=n, random_state=seed) if n < len(train_idx) else (train_idx, None)
                    scaler = StandardScaler()
                    rf = RandomForestRegressor(
                        n_estimators=100,
                        max_depth=6,
                        min_samples_split=max(2, min(5, len(tr_idx))),
                        random_state=seed,
                        n_jobs=1
                    )
                    X_tr = scaler.fit_transform(X[tr_idx])
                    X_val = scaler.transform(X[val_idx])
                    rf.fit(X_tr, y[tr_idx])

                    pred_val = rf.predict(X_val)
                    pred_tr = rf.predict(X_tr)
                    all_val_r2[alpha].append(r2_score(y[val_idx], pred_val))
                    all_val_mae[alpha].append(mean_absolute_error(y[val_idx], pred_val))
                    all_train_r2[alpha].append(r2_score(y[tr_idx], pred_tr))
                    all_train_mae[alpha].append(mean_absolute_error(y[tr_idx], pred_tr))

            scaler = StandardScaler()
            rf = RandomForestRegressor(
                n_estimators=100,
                max_depth=6,
                min_samples_split=max(2, min(5, len(cv_idx))),
                random_state=seed,
                n_jobs=1
            )
            X_cv = scaler.fit_transform(X[cv_idx])
            X_test = scaler.transform(X[test_idx])
            rf.fit(X_cv, y[cv_idx])

            test_pred = rf.predict(X_test)
            all_test_r2.append(r2_score(y[test_idx], test_pred))
            all_test_mae.append(mean_absolute_error(y[test_idx], test_pred))

        avg_r2 = {a: float(np.mean(all_val_r2[a])) for a in alphas if len(all_val_r2[a]) > 0}
        best_alpha = max(avg_r2, key=avg_r2.get) if avg_r2 else 1.0
        train_val_gaps = {
            a: float(np.mean(all_train_r2[a]) - np.mean(all_val_r2[a]))
            for a in alphas if len(all_train_r2[a]) > 0 and len(all_val_r2[a]) > 0
        }

        results[model] = {
            'best_alpha': best_alpha,
            'test_r2_mean': float(np.mean(all_test_r2)), 'test_r2_std': float(np.std(all_test_r2)),
            'test_mae_mean': float(np.mean(all_test_mae)), 'test_mae_std': float(np.std(all_test_mae)),
            'cv_r2_mean': float(np.mean(all_val_r2[best_alpha])) if all_val_r2[best_alpha] else 0.0,
            'cv_r2_std': float(np.std(all_val_r2[best_alpha])) if all_val_r2[best_alpha] else 0.0,
            'cv_mae_mean': float(np.mean(all_val_mae[best_alpha])) if all_val_mae[best_alpha] else 0.0,
            'cv_mae_std': float(np.std(all_val_mae[best_alpha])) if all_val_mae[best_alpha] else 0.0,
            'train_val_gap': train_val_gaps.get(best_alpha, 0.0),
            'is_overfitting': train_val_gaps.get(best_alpha, 0.0) > 0.35,
            'fold_results': {
                'val_r2': all_val_r2[best_alpha],
                'val_mae': all_val_mae[best_alpha]
            },
            'fold_results_alpha1': {
                'val_mae': all_val_mae[1.0] if all_val_mae[1.0] else all_val_mae[best_alpha],
                'val_r2': all_val_r2[1.0] if all_val_r2[1.0] else all_val_r2[best_alpha]
            },
            'test_seed_results': {'test_mae': all_test_mae, 'test_r2': all_test_r2},
            'cv_results': {
                'alphas': alphas,
                'r2': avg_r2,
                'mae': {a: float(np.mean(all_val_mae[a])) for a in alphas if all_val_mae[a]},
                'r2_std': {a: float(np.std(all_val_r2[a])) for a in alphas if all_val_r2[a]},
                'mae_std': {a: float(np.std(all_val_mae[a])) for a in alphas if all_val_mae[a]},
                'train_r2': {a: float(np.mean(all_train_r2[a])) for a in alphas if all_train_r2[a]},
                'train_r2_std': {a: float(np.std(all_train_r2[a])) for a in alphas if all_train_r2[a]},
                'train_mae': {a: float(np.mean(all_train_mae[a])) for a in alphas if all_train_mae[a]},
                'train_mae_std': {a: float(np.std(all_train_mae[a])) for a in alphas if all_train_mae[a]},
            }
        }
        print(f"{model}: MAE={results[model]['test_mae_mean']:.2f} +/- {results[model]['test_mae_std']:.2f}, R2={results[model]['test_r2_mean']:.3f} +/- {results[model]['test_r2_std']:.3f}")

    return results
