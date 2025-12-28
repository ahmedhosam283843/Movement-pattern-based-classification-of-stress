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



    """
    LOPO XGBoost on Top-20 slow features:
      - fold-wise top-k selection from training participants only (mutual info),
      - early stopping on AUPRC (imbalance-aware),
      - validation-based threshold tuning (balanced accuracy),
      - aggregates and saves mean feature importances across folds.
    Returns metrics dict and mean-importance DataFrame.
    """
    # Build slow-only matrix
    cols = [c for c in wide_df.columns if c not in ('participant', 'label')]
    slow_cols = [c for c in cols if c.endswith('_slow')]
    X = wide_df[slow_cols].values
    y = wide_df['label'].values
    groups = wide_df['participant'].values
    feat_names = slow_cols

    probs_te, y_te, yhat_te, parts_te = [], [], [], [] # <-- ADDED parts_te
    fold_imp_dfs = []

    rng = np.random.default_rng(random_state)
    pbar = tqdm(total=len(np.unique(groups)), desc="XGB Top-20 (slow) LOPO", leave=True) if verbose else None

    # Lightweight, regularized parameter candidates
    param_candidates = [
        dict(n_estimators=600, max_depth=2, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9,
             min_child_weight=5, gamma=1.0, reg_lambda=5.0, reg_alpha=0.5),
        dict(n_estimators=800, max_depth=2, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=5, gamma=0.5, reg_lambda=4.0, reg_alpha=0.3),
        dict(n_estimators=500, max_depth=3, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
             min_child_weight=3, gamma=0.0, reg_lambda=1.0, reg_alpha=0.0),
    ]

    for fold, (tr, te) in enumerate(logo.split(X, y, groups), 1):
        grp_tr = groups[tr]
        train_parts = np.unique(grp_tr).tolist()
        val_count = max(1, int(0.2 * len(train_parts)))
        val_parts = rng.choice(train_parts, size=val_count, replace=False).tolist()
        mask_val = np.isin(grp_tr, val_parts)
        mask_core = ~mask_val

        # Fold-wise Top-k on training participants only
        topk_idx = _topk_by_mutual_info(X[tr][mask_core], y[tr][mask_core], k=min(k, X.shape[1]), seed=123 + fold)
        X_tr_core_raw = X[tr][mask_core][:, topk_idx]
        X_val_raw     = X[tr][mask_val][:, topk_idx]
        X_te_raw      = X[te][:, topk_idx]

        # Impute
        imputer.fit(X_tr_core_raw)
        X_tr_core = imputer.transform(X_tr_core_raw)
        X_val     = imputer.transform(X_val_raw)
        X_te      = imputer.transform(X_te_raw)

        y_tr_core = y[tr][mask_core]
        y_val     = y[tr][mask_val]
        y_te_fold = y[te]

        pos = int((y_tr_core == 1).sum()); neg = int((y_tr_core == 0).sum())
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0

        # Per-fold param search (choose by best validation AUPRC)
        best_model, best_auprc = None, -np.inf
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

        # Threshold tuning on validation (BalancedAcc)
        p_val = best_model.predict_proba(X_val)[:, 1]
        thr = tune_thr_balacc(y_val, p_val) if len(np.unique(y_val)) > 1 else 0.5

        # Test
        p_te = best_model.predict_proba(X_te)[:, 1]
        yhat = (p_te >= thr).astype(int)

        probs_te.extend(p_te.tolist())
        y_te.extend(y_te_fold.tolist())
        yhat_te.extend(yhat.tolist())
        parts_te.extend(groups[te].tolist()) # <-- STORE PARTICIPANT ID

        # Importances in reduced feature space
        imp = get_safe_feature_importances(best_model, len(topk_idx))
        imp_df = pd.DataFrame({"feature": [feat_names[i] for i in topk_idx], "importance": imp})
        fold_imp_dfs.append(imp_df)

        if pbar:
            pbar.set_postfix_str(f"fold={fold}, best_iter={best_model.best_iteration}, thr={thr:.2f}, valAUPRC={best_auprc:.3f}")
            pbar.update(1)

    if pbar:
        pbar.close()

    probs_te = np.array(probs_te)
    y_te = np.array(y_te)
    yhat_te = np.array(yhat_te)
    print("\nXGB Top-20 (slow) LOPO:",
          "AUROC:", f"{roc_auc_score(y_te, probs_te):.3f}",
          "AUPRC:", f"{average_precision_score(y_te, probs_te):.3f}",
          "BalancedAcc:", f"{balanced_accuracy_score(y_te, yhat_te):.3f}",
          "Precision:", f"{precision_score(y_te, yhat_te):.3f}",
          "Recall:", f"{recall_score(y_te, yhat_te):.3f}",
          "MacroF1:", f"{f1_score(y_te, yhat_te, average='macro'):.3f}")

    # Aggregate importances
    full_imp = pd.concat(fold_imp_dfs) if fold_imp_dfs else pd.DataFrame(columns=["feature","importance"])
    mean_imp = (full_imp.groupby("feature")["importance"].mean()
                .sort_values(ascending=False).reset_index())
    mean_imp.to_csv("xgb_top20_slow_feature_importance_mean.csv", index=False)
    mean_imp.head(20).to_csv("xgb_top20_slow_top_features.csv", index=False)

    # <-- RETURN all results for metrics_dict
    return probs_te, y_te, yhat_te, np.array(parts_te), mean_imp