"""
Tabular model runners: logistic (fast/halving+tuned), RF (fast/slow-tuned), XGBoost (combined/slow-only),
per-speed slow/fast analysis, top-20 slow features models.
"""

import time
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import LeaveOneGroupOut, HalvingGridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score, f1_score, precision_score, recall_score
from xgboost import XGBClassifier
from validation import tune_threshold_balacc, bootstrap_metric # <-- Import updated
logo = LeaveOneGroupOut()
imputer = SimpleImputer(strategy='median')
rng = np.random.default_rng(42)


def tune_thr_balacc(y_val, p_val):
    """Tune threshold to maximize balanced accuracy on validation set."""
    if len(np.unique(y_val)) < 2:
        return 0.5
    thr_grid = np.linspace(0.1, 0.9, 33)
    best_thr, best_bal = 0.5, -1.0
    for t in thr_grid:
        bal = balanced_accuracy_score(y_val, (p_val >= t).astype(int))
        if bal > best_bal:
            best_bal, best_thr = bal, t
    return best_thr


def tune_thr_groupkfold(Xtr, ytr, groups_tr, pipe, n_splits=5):
    gkf = GroupKFold(n_splits=min(n_splits, len(np.unique(groups_tr))))
    thrs = []
    for tr_idx, val_idx in gkf.split(Xtr, ytr, groups_tr):
        pipe.fit(Xtr[tr_idx], ytr[tr_idx])
        # Use predict_proba for tree-based models
        p_val = pipe.predict_proba(Xtr[val_idx])[:, 1]
        thr_grid = np.linspace(0.1, 0.9, 33)
        best_thr, best_bal = 0.5, -1.0
        for t in thr_grid:
            bal = balanced_accuracy_score(
                ytr[val_idx], (p_val >= t).astype(int))
            if bal > best_bal:
                best_bal, best_thr = bal, t
        thrs.append(best_thr)
    return float(np.mean(thrs)) if thrs else 0.5


def get_safe_feature_importances(model, num_features):
    """
    Robust feature importance from XGB model (gain),
    creating a full-length array (missing indices -> 0).
    """
    booster = model.get_booster()
    scores = booster.get_score(importance_type='gain')
    full = np.zeros(num_features)
    for key, score in scores.items():
        try:
            idx = int(key[1:])
            if idx < num_features:
                full[idx] = score
        except Exception:
            pass
    return full


def summarize_metrics(name, probs, y_true, thr=0.5):
    y_pred = (np.array(probs) >= thr).astype(int)
    return {
        "model": name,
        "AUROC": float(roc_auc_score(y_true, probs)),
        "AUPRC": float(average_precision_score(y_true, probs)),
        "BalancedAcc": float(balanced_accuracy_score(y_true, y_pred)),
        "MacroF1": float(f1_score(y_true, y_pred, average='macro')),
        "Precision": float(precision_score(y_true, y_pred)),
        "Recall": float(recall_score(y_true, y_pred))
    }


def run_lopo_logistic_combined(X, y, groups, k_features=40, C=0.5, verbose=True, desc="LOPO logistic (combined)"):
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler()),
        ('select', SelectKBest(f_classif, k=k_features)),
        ('clf', LogisticRegression(class_weight='balanced',
                                   C=C, solver='liblinear', max_iter=200))
    ])

    probs, y_true, y_pred, part_te = [], [], [], []
    fold_times = []
    pbar = tqdm(total=len(np.unique(groups)), desc=desc,
                leave=True) if verbose else None

    for fold, (tr, te) in enumerate(logo.split(X, y, groups)):
        thr = tune_thr_groupkfold(X[tr], y[tr], groups[tr], pipe, n_splits=5)
        t0 = time.perf_counter()
        pipe.fit(X[tr], y[tr])
        s = pipe.decision_function(X[te])
        p = 1.0 / (1.0 + np.exp(-s))  # sigmoid on raw scores
        yhat = (p >= thr).astype(int)

        probs.extend(p.tolist())
        y_true.extend(y[te].tolist())
        y_pred.extend(yhat.tolist())
        part_te.extend(groups[te].tolist())
        dt = time.perf_counter() - t0
        fold_times.append(dt)
        if pbar:
            pbar.set_postfix_str(f"fold={fold}, {dt:.1f}s")
            pbar.update(1)
    if pbar:
        pbar.close()

    probs = np.array(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if verbose:
        print(f"\n{desc}:")
        print("AUROC:", f"{roc_auc_score(y_true, probs):.3f}",
              "AUPRC:", f"{average_precision_score(y_true, probs):.3f}",
              "BalancedAcc:", f"{balanced_accuracy_score(y_true, y_pred):.3f}",
              "MacroF1:", f"{f1_score(y_true, y_pred, average='macro'):.3f}")
        print(
            f"Mean fold time: {np.mean(fold_times):.2f}s (±{np.std(fold_times):.2f}), total: {np.sum(fold_times):.1f}s")

    return probs, y_true, y_pred, np.array(part_te)


def run_lopo_rf_combined(X, y, groups, n_estimators=300, min_leaf=3, k_features=40, verbose=True):
    """
    FIXED: This function now correctly performs per-fold threshold tuning to avoid data leakage.
    For each outer fold (leaving one participant out), it creates an inner validation split
    from the training data to find an optimal decision threshold. This threshold is then
    applied only to the held-out test participant for that fold.
    """
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False)),
        ('select', SelectKBest(f_classif, k=k_features)),
        ('clf', RandomForestClassifier(
            n_estimators=n_estimators, max_features='sqrt',
            min_samples_leaf=min_leaf, class_weight='balanced_subsample',
            random_state=42, n_jobs=1))
    ])
    probs_te, y_te, yhat_te, parts_te = [], [], [], [] # <-- ADDED parts_te
    pbar = tqdm(total=len(np.unique(groups)),
                desc="LOPO RF (combined)", leave=True) if verbose else None

    for tr, te in logo.split(X, y, groups):
        # Create a stable inner validation split from this fold's training data
        grp_tr = groups[tr]
        train_parts = np.unique(grp_tr).tolist()
        rng = np.random.default_rng(
            42 + len(train_parts))  # Reproducible split
        val_count = max(1, int(0.2 * len(train_parts)))  # 20% for validation
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()

        mask_val = np.isin(grp_tr, val_parts)
        mask_core = ~mask_val

        # Fit the pipeline on the core training data (excluding the inner validation set)
        pipe.fit(X[tr][mask_core], y[tr][mask_core])

        # Tune the decision threshold on the inner validation set
        p_val = pipe.predict_proba(X[tr][mask_val])[:, 1]

        # Default to 0.5 if the validation set has only one class
        thr = tune_thr_balacc(y[tr][mask_val], p_val) if len(
            np.unique(y[tr][mask_val])) > 1 else 0.5

        # Evaluate on the final test set (the held-out participant)
        p = pipe.predict_proba(X[te])[:, 1]
        yhat = (p >= thr).astype(int)

        # Append results for this fold
        probs_te.extend(p.tolist())
        y_te.extend(y[te].tolist())
        yhat_te.extend(yhat.tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        if pbar:
            pbar.set_postfix_str(f"thr={thr:.2f}")
            pbar.update(1)

    if pbar:
        pbar.close()

    probs = np.array(probs_te)
    y_true = np.array(y_te)
    y_pred = np.array(yhat_te)

    # Print summary metrics directly
    print("\nRF LOPO (combined):")
    print("AUROC:", f"{roc_auc_score(y_true, probs):.3f}",
          "AUPRC:", f"{average_precision_score(y_true, probs):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_true, y_pred):.3f}")

    return probs, y_true, y_pred, np.array(parts_te) # <-- RETURN parts_te


def run_lopo_rf_slow_tuned(wide_df, labels_ser):
    # Build slow-only matrix
    feature_cols = [c for c in wide_df.columns if c not in (
        'participant', 'label')]
    slow_cols = [c for c in feature_cols if c.endswith('_slow')]
    X = wide_df[slow_cols].values
    y = wide_df['label'].values
    groups = wide_df['participant'].values

    logo = LeaveOneGroupOut()
    base = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False)),
        ('select', SelectKBest(f_classif, k=min(40, len(slow_cols)))),
        ('clf', RandomForestClassifier(
            n_estimators=500, max_features='sqrt',
            min_samples_leaf=3, class_weight='balanced_subsample',
            random_state=42, n_jobs=1))
    ])

    probs_te, y_te, yhat_te, parts_te = [], [], [], []
    for tr, te in logo.split(X, y, groups):
        # inner validation split per outer fold
        grp_tr = groups[tr]
        train_parts = np.unique(grp_tr).tolist()
        rng = np.random.default_rng(42 + len(train_parts))
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()
        mask_val = np.isin(grp_tr, val_parts)
        mask_core = ~mask_val

        # fit on core
        base.fit(X[tr][mask_core], y[tr][mask_core])
        # threshold tuning on val (BalancedAcc)
        p_val = base.predict_proba(X[tr][mask_val])[:, 1]
        thr_grid = np.linspace(0.1, 0.9, 33)
        best_thr, best_bal = 0.5, -1.0
        for t in thr_grid:
            bal = balanced_accuracy_score(
                y[tr][mask_val], (p_val >= t).astype(int))
            if bal > best_bal:
                best_bal, best_thr = bal, t

        # test
        p = base.predict_proba(X[te])[:, 1]
        yhat = (p >= best_thr).astype(int)
        probs_te.extend(p.tolist())
        y_te.extend(y[te].tolist())
        yhat_te.extend(yhat.tolist())
        parts_te.extend(groups[te].tolist())

    probs_te = np.array(probs_te)
    y_te = np.array(y_te)
    yhat_te = np.array(yhat_te)
    print("\nSlow-only RF (tuned):")
    print("AUROC:", f"{roc_auc_score(y_te, probs_te):.3f}",
          "AUPRC:", f"{average_precision_score(y_te, probs_te):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_te, yhat_te):.3f}",
          "MacroF1:", f"{f1_score(y_te, yhat_te, average='macro'):.3f}")
    return np.array(parts_te), probs_te, y_te, yhat_te


def run_speed(X_speed, speed_feature_names, name, y, groups):
    rng = np.random.default_rng(42)
    logo = LeaveOneGroupOut()
    imputer = SimpleImputer(strategy='median')
    probs_s, y_s, yhat_s, parts_s = [], [], [], [] # <-- ADDED parts_s
    fold_imp_dfs = []
    pbar_s = tqdm(total=len(np.unique(groups)),
                  desc=f"XGB {name}", leave=False)
    for fold, (tr, te) in enumerate(logo.split(X_speed, y, groups), 1):
        train_parts = np.unique(groups[tr]).tolist()
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()
        core_train_parts = [p for p in train_parts if p not in val_parts]
        mask_tr = np.isin(groups[tr], core_train_parts)
        mask_val = np.isin(groups[tr], val_parts)
        X_tr_raw, y_tr_raw = X_speed[tr][mask_tr], y[tr][mask_tr]
        X_val_raw, y_val_raw = X_speed[tr][mask_val], y[tr][mask_val]
        X_te_raw, y_te_raw = X_speed[te], y[te]
        imputer.fit(X_tr_raw)
        X_tr = imputer.transform(X_tr_raw)
        X_val = imputer.transform(X_val_raw)
        X_te = imputer.transform(X_te_raw)
        pos = int((y_tr_raw == 1).sum())
        neg = int((y_tr_raw == 0).sum())
        scale_pos_weight = (neg / max(pos, 1)) if pos > 0 else 1.0
        model = XGBClassifier(n_estimators=400, max_depth=3, learning_rate=0.05,  early_stopping_rounds=30, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                              reg_alpha=0.0, min_child_weight=1, objective='binary:logistic', eval_metric='auc', scale_pos_weight=scale_pos_weight, n_jobs=1, random_state=42, verbosity=0)
        model.fit(X_tr, y_tr_raw, eval_set=[(X_val, y_val_raw)], verbose=False)
        p = model.predict_proba(X_te)[:, 1]
        probs_s.extend(p.tolist())
        y_s.extend(y_te_raw.tolist())
        yhat_s.extend((p >= 0.5).astype(int).tolist())
        parts_s.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        # Use the safe function to get importances for the speed-specific model
        imp = get_safe_feature_importances(model, len(speed_feature_names))
        imp_df_fold = pd.DataFrame(
            {"feature": speed_feature_names, "importance": imp})
        fold_imp_dfs.append(imp_df_fold)
        pbar_s.update(1)
    pbar_s.close()

    print(f"\n{name}:")
    print("AUROC:", f"{roc_auc_score(y_s, probs_s):.3f}", "AUPRC:", f"{average_precision_score(y_s, probs_s):.3f}", "BalancedAcc:",
          f"{balanced_accuracy_score(y_s, np.array(yhat_s)):.3f}", "MacroF1:", f"{f1_score(y_s, np.array(yhat_s), average='macro'):.3f}")

    if fold_imp_dfs:
        full_imp_df = pd.concat(fold_imp_dfs)
        mean_imp_df = full_imp_df.groupby("feature")["importance"].mean(
        ).sort_values(ascending=False).reset_index()
        print(f"\nTop 10 features for {name}:")
        print(mean_imp_df.head(10).to_string(index=False))
    
    # <-- RETURN parts_s -->
    return np.array(probs_s), np.array(y_s), np.array(yhat_s), np.array(parts_s), mean_imp_df


def get_safe_feature_importances(model, num_features):
    """
    Reconstructs the full feature importance array, handling cases where
    XGBoost omits features with zero importance.
    """
    booster = model.get_booster()
    # Get importance scores as a dictionary {'f0': 123.4, 'f5': 56.7, ...}
    scores = booster.get_score(importance_type='gain')

    # Create a zero-filled array of the correct length
    full_importances = np.zeros(num_features)

    # Populate the array with scores, parsing the feature index from the key (e.g., 'f12' -> 12)
    for f_idx_str, score in scores.items():
        try:
            # Extract the integer index from the feature name string
            idx = int(f_idx_str[1:])
            if idx < num_features:
                full_importances[idx] = score
        except (ValueError, IndexError):
            # Handle cases where feature names might not be in the 'f#' format, though this is rare
            continue

    return full_importances

def run_xgb_combined(
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feature_names: list,
    k_features: int = 40,
    random_state: int = 42,
    desc: str = "XGBoost LOPO (Combined, improved)",
    verbose: bool = True
):
    """
    LOPO XGBoost on all features with:
      - per-fold top-k selection (mutual info),
      - early stopping on AUPRC,
      - per-fold lightweight param search,
      - validation-based threshold tuning.
    """
    logo = LeaveOneGroupOut()
    imputer = SimpleImputer(strategy='median')
    rng = np.random.default_rng(random_state)

    probs, y_true, y_pred, parts_te = [], [], [], [] # <-- ADDED parts_te
    fold_imp_dfs = []
    pbar = tqdm(total=len(np.unique(groups)), desc=desc, leave=True) if verbose else None

    param_candidates = [
        dict(n_estimators=700, max_depth=2, learning_rate=0.04, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=5, gamma=0.5, reg_lambda=4.0, reg_alpha=0.3),
        dict(n_estimators=900, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=6, gamma=1.0, reg_lambda=6.0, reg_alpha=0.5),
        dict(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=3, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0),
    ]

    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        train_parts = np.unique(groups[tr]).tolist()
        te_part = np.unique(groups[te])[0]
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(train_parts, size=val_count, replace=False).tolist()
        core_train_parts = [p for p in train_parts if p not in val_parts]
        mask_core = np.isin(groups[tr], core_train_parts)
        mask_val  = np.isin(groups[tr], val_parts)

        # top-k selection on core training
        topk_idx = _topk_by_mutual_info_fold(X[tr][mask_core], y[tr][mask_core], k=min(k_features, X.shape[1]), seed=234 + fold)
        X_tr_core_raw = X[tr][mask_core][:, topk_idx]
        X_val_raw     = X[tr][mask_val][:, topk_idx]
        X_te_raw      = X[te][:, topk_idx]

        # impute
        imputer.fit(X_tr_core_raw)
        X_tr_core = imputer.transform(X_tr_core_raw)
        X_val     = imputer.transform(X_val_raw)
        X_te      = imputer.transform(X_te_raw)

        y_tr_core = y[tr][mask_core]
        y_val     = y[tr][mask_val]
        y_te_fold = y[te]

        pos = int((y_tr_core == 1).sum()); neg = int((y_tr_core == 0).sum())
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        best_model, best_params, best_auprc = None, None, -np.inf
        for params in param_candidates:
            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',
                early_stopping_rounds=30,
                scale_pos_weight=spw,
                tree_method='hist',
                max_bin=256,
                n_jobs=1,
                random_state=random_state,
                verbosity=0,
                **params
            )
            model.fit(X_tr_core, y_tr_core, eval_set=[(X_val, y_val)], verbose=False)
            p_val = model.predict_proba(X_val)[:, 1]
            auprc = average_precision_score(y_val, p_val)
            if auprc > best_auprc:
                best_auprc = auprc
                best_model = model
                best_params = params

        # threshold tuning on validation
        p_val = best_model.predict_proba(X_val)[:, 1]
        thr = tune_thr_balacc(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.5

        # test
        p_te = best_model.predict_proba(X_te)[:, 1]
        yhat = (p_te >= thr).astype(int)

        probs.extend(p_te.tolist()); y_true.extend(y_te_fold.tolist()); y_pred.extend(yhat.tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        imp = get_safe_feature_importances(best_model, len(topk_idx))
        imp_df = pd.DataFrame({"feature": [feature_names[i] for i in topk_idx], "importance": imp})
        fold_imp_dfs.append(imp_df)

        if pbar:
            pbar.set_postfix_str(f"fold={fold}, te={te_part}, best_iter={best_model.best_iteration}, thr={thr:.2f}, valAUPRC={best_auprc:.3f}")
            pbar.update(1)
    if pbar:
        pbar.close()

    probs = np.array(probs); y_true = np.array(y_true); y_pred = np.array(y_pred)
    print(f"\n{desc}:")
    print("AUROC:", f"{roc_auc_score(y_true, probs):.3f}",
          "AUPRC:", f"{average_precision_score(y_true, probs):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_true, y_pred):.3f}",
          "MacroF1:", f"{f1_score(y_true, y_pred, average='macro'):.3f}")

    if fold_imp_dfs:
        full_imp_df = pd.concat(fold_imp_dfs)
        mean_imp_df = full_imp_df.groupby("feature")["importance"].mean().sort_values(ascending=False).reset_index()
        mean_imp_df.to_csv("xgb_combined_top_features_improved.csv", index=False)
    else:
        mean_imp_df = None

    return probs, y_true, y_pred, np.array(parts_te), mean_imp_df # <-- RETURN parts_te


def _topk_by_mutual_info_fold(X_tr, y_tr, k=20, seed=42):
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_tr)
    mi = mutual_info_classif(X_tr_imp, y_tr, random_state=seed)
    idx = np.argsort(mi)[::-1][:min(k, len(mi))]
    return idx

def run_xgb_slow_lopo(
    X_slow: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    feat_names: list,
    k_features: int = 20,
    random_state: int = 42,
    verbose: bool = True
):
    """
    LOPO XGBoost (slow-only) with:
      - per-fold top-k feature selection (mutual info on training participants only),
      - early stopping on AUPRC (imbalance-aware),
      - small per-fold hyperparameter search on the validation set,
      - validation-based threshold tuning (balanced accuracy).
    Returns metrics dict and mean feature importance DataFrame.
    """
    logo = LeaveOneGroupOut()
    imputer = SimpleImputer(strategy='median')
    rng = np.random.default_rng(random_state)

    probs_te, y_te, yhat_te, parts_te = [], [], [], [] # <-- ADDED parts_te
    fold_imp_dfs = []
    pbar = tqdm(total=len(np.unique(groups)),
                desc="XGBoost LOPO (slow-only, improved)", leave=True) if verbose else None

    # lightweight param candidates (regularized + shallow trees)
    param_candidates = [
        dict(n_estimators=600, max_depth=2, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=5, gamma=1.0, reg_lambda=5.0, reg_alpha=0.5),
        dict(n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=5, gamma=0.5, reg_lambda=4.0, reg_alpha=0.3),
        dict(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=3, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0),
    ]

    for fold, (tr, te) in enumerate(logo.split(X_slow, y, groups), 1):
        # Participants for this fold
        train_parts = np.unique(groups[tr]).tolist()
        te_part = np.unique(groups[te])[0]
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()
        core_train_parts = [p for p in train_parts if p not in val_parts]
        mask_core = np.isin(groups[tr], core_train_parts)
        mask_val = np.isin(groups[tr], val_parts)

        # per-fold top-k selection from core training only
        topk_idx = _topk_by_mutual_info_fold(X_slow[tr][mask_core], y[tr][mask_core], k=min(
            k_features, X_slow.shape[1]), seed=123 + fold)
        X_tr_core_raw = X_slow[tr][mask_core][:, topk_idx]
        X_val_raw = X_slow[tr][mask_val][:, topk_idx]
        X_te_raw = X_slow[te][:, topk_idx]

        # impute
        imputer.fit(X_tr_core_raw)
        X_tr_core = imputer.transform(X_tr_core_raw)
        X_val = imputer.transform(X_val_raw)
        X_te = imputer.transform(X_te_raw)

        y_tr_core = y[tr][mask_core]
        y_val = y[tr][mask_val]
        y_te_fold = y[te]

        pos = int((y_tr_core == 1).sum())
        neg = int((y_tr_core == 0).sum())
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        # per-fold param search (choose by best validation AUPRC)
        best_model, best_params, best_auprc = None, None, -np.inf
        for params in param_candidates:
            model = XGBClassifier(
                objective='binary:logistic',
                eval_metric='aucpr',          # PR-AUC for imbalance
                early_stopping_rounds=30,
                scale_pos_weight=spw,
                tree_method='hist',
                max_bin=256,
                n_jobs=1,
                random_state=random_state,
                verbosity=0,
                **params
            )
            model.fit(X_tr_core, y_tr_core, eval_set=[
                      (X_val, y_val)], verbose=False)
            # Evaluate val AUPRC directly from eval history
            # XGBoost doesn't return metric history via sklearn API ⇒ compute explicitly
            p_val = model.predict_proba(X_val)[:, 1]
            auprc = average_precision_score(y_val, p_val)
            if auprc > best_auprc:
                best_auprc = auprc
                best_model = model
                best_params = params

        # Threshold tuning on validation
        p_val = best_model.predict_proba(X_val)[:, 1]
        thr = tune_thr_balacc(y_val, p_val) if len(
            np.unique(y_val)) > 1 else 0.5

        # Test
        p_te = best_model.predict_proba(X_te)[:, 1]
        yhat = (p_te >= thr).astype(int)

        probs_te.extend(p_te.tolist())
        y_te.extend(y_te_fold.tolist())
        yhat_te.extend(yhat.tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        # importances in the reduced feature space
        imp = get_safe_feature_importances(best_model, len(topk_idx))
        imp_df = pd.DataFrame(
            {"feature": [feat_names[i] for i in topk_idx], "importance": imp})
        fold_imp_dfs.append(imp_df)

        if pbar:
            pbar.set_postfix_str(
                f"fold={fold}, te={te_part}, best_iter={best_model.best_iteration}, thr={thr:.2f}, valAUPRC={best_auprc:.3f}")
            pbar.update(1)
    if pbar:
        pbar.close()

    probs_te = np.array(probs_te)
    y_te = np.array(y_te)
    yhat_te = np.array(yhat_te)
    print("\nXGBoost LOPO (slow-only, improved):",
          "AUROC:", f"{roc_auc_score(y_te, probs_te):.3f}",
          "AUPRC:", f"{average_precision_score(y_te, probs_te):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_te, yhat_te):.3f}",
          "Precision:", f"{precision_score(y_te, yhat_te):.3f}",
          "Recall:", f"{recall_score(y_te, yhat_te):.3f}",
          "MacroF1:", f"{f1_score(y_te, yhat_te, average='macro'):.3f}")

    # Aggregate importances
    full_imp = pd.concat(fold_imp_dfs)
    mean_imp = full_imp.groupby("feature")["importance"].mean(
    ).sort_values(ascending=False).reset_index()
    mean_imp.to_csv(
        "slow_xgb_feature_importance_mean_improved.csv", index=False)
    mean_imp.head(20).to_csv("slow_xgb_top_features_improved.csv", index=False)
    
    # <-- RETURN all results for metrics_dict
    return probs_te, y_te, yhat_te, np.array(parts_te), mean_imp


def run_rf_slow_lopo(X_slow, y, groups, k_features=None):
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='median')),
        ('scale', StandardScaler(with_mean=False)),
        ('select', SelectKBest(f_classif, k=min(
            k_features or 40, X_slow.shape[1]))),
        ('clf', RandomForestClassifier(
            n_estimators=600, max_features='sqrt', min_samples_leaf=3,
            class_weight='balanced_subsample', random_state=42, n_jobs=1))
    ])

    probs, y_true, y_pred, parts_te = [], [], [], [] # <-- ADDED parts_te
    pbar = tqdm(total=len(np.unique(groups)),
                desc="RF LOPO (slow-only)", leave=True)
    for fold, (tr, te) in enumerate(logo.split(X_slow, y, groups), 1):

        grp_tr = groups[tr]
        train_parts = np.unique(grp_tr).tolist()
        rng = np.random.default_rng(42 + fold)
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()
        mask_val = np.isin(grp_tr, val_parts)
        mask_core = ~mask_val

        # fit on core train
        pipe.fit(X_slow[tr][mask_core], y[tr][mask_core])

        # threshold tuning on validation
        p_val = pipe.predict_proba(X_slow[tr][mask_val])[:, 1]
        thr = tune_thr_balacc(y[tr][mask_val], p_val)

        p = pipe.predict_proba(X_slow[te])[:, 1]
        probs.extend(p.tolist())
        y_true.extend(y[te].tolist())
        y_pred.extend((p >= thr).astype(int).tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        pbar.set_postfix_str(f"fold={fold}, thr={thr:.2f}")
        pbar.update(1)
    pbar.close()

    print("\nRF LOPO (slow-only):",
          "AUROC:", f"{roc_auc_score(y_true, probs):.3f}",
          "AUPRC:", f"{average_precision_score(y_true, probs):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_true, y_pred):.3f}",
          "Precision:", f"{precision_score(y_true, y_pred):.3f}",
          "Recall:", f"{recall_score(y_true, y_pred):.3f}",
          "MacroF1:", f"{f1_score(y_true, y_pred, average='macro'):.3f}")

    # <-- RETURN all results for metrics_dict
    return np.array(probs), np.array(y_true), np.array(y_pred), np.array(parts_te)


def _topk_by_mutual_info(X_tr, y_tr, k=20, seed=42):
    imp = SimpleImputer(strategy='median')
    X_tr_imp = imp.fit_transform(X_tr)
    mi = mutual_info_classif(X_tr_imp, y_tr, random_state=seed)
    return np.argsort(mi)[::-1][:min(k, len(mi))]


def run_simple_top20_slow(wide_df, labels_ser, k=20):
    """
    FIX: Fold-wise top-k selection from training participants only. No global file read.
    Train logistic on selected slow features per fold; evaluate on held-out participant.
    """
    cols = [c for c in wide_df.columns if c not in ('participant', 'label')]
    slow_cols = [c for c in cols if c.endswith('_slow')]
    X = wide_df[slow_cols].values
    y = wide_df['label'].values
    groups = wide_df['participant'].values
    feat_names = slow_cols

    probs, y_true, y_hat, parts_te = [], [], [], [] # <-- ADDED parts_te
    pbar = tqdm(total=len(np.unique(groups)),
                desc="Simple Top-20 (slow) LOPO", leave=True)
    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        grp_tr = groups[tr]
        # fold-wise top-k selection on training participants only
        topk_idx = _topk_by_mutual_info(
            X[tr], y[tr], k=min(k, X.shape[1]), seed=123 + fold)
        Xtr, Xte = X[tr][:, topk_idx], X[te][:, topk_idx]

        # small validation (threshold tuning)
        train_parts = np.unique(grp_tr).tolist()
        rng = np.random.default_rng(123 + fold)
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(
            train_parts, size=val_count, replace=False).tolist()
        mask_val = np.isin(grp_tr, val_parts)
        mask_core = ~mask_val

        pipe = Pipeline([
            ('impute', SimpleImputer(strategy='median')),
            ('scale', StandardScaler()),
            ('clf', LogisticRegression(class_weight='balanced',
             C=0.5, solver='liblinear', max_iter=300))
        ])
        pipe.fit(Xtr[mask_core], y[tr][mask_core])

        # tune threshold on validation
        s_val = pipe.decision_function(Xtr[mask_val])
        p_val = 1.0/(1.0+np.exp(-s_val))
        thr_grid = np.linspace(0.1, 0.9, 33)
        best_thr, best_bal = 0.5, -1.0
        for t in thr_grid:
            bal = balanced_accuracy_score(
                y[tr][mask_val], (p_val >= t).astype(int))
            if bal > best_bal:
                best_bal, best_thr = bal, t

        # test
        s_te = pipe.decision_function(Xte)
        p_te = 1.0/(1.0+np.exp(-s_te))
        probs.extend(p_te.tolist())
        y_true.extend(y[te].tolist())
        y_hat.extend(((p_te >= best_thr).astype(int)).tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID
        
        pbar.set_postfix_str(f"fold={fold}, thr={best_thr:.2f}")
        pbar.update(1)
    pbar.close()

    probs = np.array(probs)
    y_true = np.array(y_true)
    y_hat = np.array(y_hat)
    print("\nSimple Top-20 Logistic (slow):",
          "AUROC:", f"{roc_auc_score(y_true, probs):.3f}",
          "AUPRC:", f"{average_precision_score(y_true, probs):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_true, y_hat):.3f}",
          "Precision:", f"{precision_score(y_true, y_hat):.3f}",
          "Recall:", f"{recall_score(y_true, y_hat):.3f}",
          "MacroF1:", f"{f1_score(y_true, y_hat, average='macro'):.3f}")
          
    # <-- RETURN all results for metrics_dict
    return np.array(probs), np.array(y_true), np.array(y_hat), np.array(parts_te)


