from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score
from sklearn.model_selection import LeaveOneGroupOut, StratifiedShuffleSplit

import run_all_trial_fixed_window as fixed
import run_clean_model_sweep as models
import run_glove_guided_dicing as base


FS = 1200.0
FROZEN_CONFIG = {
    "label_source": "CH62 cue labels only",
    "trial_policy": "all 90 gesture trials retained",
    "glove_usage": "not used for labels, alignment, filtering, ranking, rejection, training, or scoring",
    "dice_s": 1.49,
    "bp_window_s": 0.09,
    "mode": "cue_offset",
    "offset_s": 0.0,
    "top_k": 3,
    "local_stride_s": 0.05,
    "bands": [(50, 90), (90, 130), (130, 200), (200, 300)],
    "feature_set": "summary",
    "model": "selectk_800_lda",
    "group_aggregation": "sum predict_proba across windows from the same trial",
}


def config_hash(config: dict[str, object]) -> str:
    payload = json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_frozen_features(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[fixed.Trial]]:
    ds = base.load_dataset(mat_path)
    trials = fixed.gesture_trials(ds.labels)
    if len(trials) != 90:
        raise ValueError(f"Expected 90 gesture trials, found {len(trials)}")

    cleaned = models.car_notch_whiten(ds.ecog, FS, whiten=True)
    filtered_bands = [
        models.bandpass(cleaned, FS, low, high) for low, high in FROZEN_CONFIG["bands"]  # type: ignore[index]
    ]
    X, y, groups = fixed.build_all_trial_examples(
        filtered_bands=filtered_bands,
        trials=trials,
        fs=FS,
        dice_s=float(FROZEN_CONFIG["dice_s"]),
        bp_window_s=float(FROZEN_CONFIG["bp_window_s"]),
        mode=str(FROZEN_CONFIG["mode"]),
        offset_s=float(FROZEN_CONFIG["offset_s"]),
        top_k=int(FROZEN_CONFIG["top_k"]),
        local_stride_s=float(FROZEN_CONFIG["local_stride_s"]),
    )
    features = models.feature_matrix(str(FROZEN_CONFIG["feature_set"]), X)
    return features, y, groups, trials


def group_labels(y: np.ndarray, groups: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    uniq = np.unique(groups)
    labels = []
    for group in uniq:
        labels.append(int(np.bincount(y[groups == group]).argmax()))
    return uniq.astype(int), np.asarray(labels, dtype=int)


def evaluate_split(
    features: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    train_groups: np.ndarray,
    test_groups: np.ndarray,
    seed: int,
) -> dict[str, object]:
    train_mask = np.isin(groups, train_groups)
    test_mask = np.isin(groups, test_groups)
    train_idx = np.flatnonzero(train_mask)
    test_idx = np.flatnonzero(test_mask)

    model = fixed.make_model(str(FROZEN_CONFIG["model"]), seed)
    model.fit(features[train_idx], y[train_idx])

    pred = model.predict(features[test_idx])
    proba = model.predict_proba(features[test_idx])
    classes = np.asarray(model.classes_)
    labels = np.sort(np.unique(y))

    sample_records = []
    for idx, p, pr in zip(test_idx, pred, proba):
        sample_records.append(
            {
                "row_index": int(idx),
                "trial_index": int(groups[idx]),
                "window_position": int(idx % int(FROZEN_CONFIG["top_k"])),
                "true_label": int(y[idx]),
                "pred_label": int(p),
                "proba": {str(int(c)): float(pr[i]) for i, c in enumerate(classes)},
            }
        )

    group_records = []
    group_true = []
    group_pred = []
    for group in np.unique(groups[test_idx]):
        mask = groups[test_idx] == group
        true_label = int(np.bincount(y[test_idx][mask]).argmax())
        summed = proba[mask].sum(axis=0)
        pred_label = int(classes[np.argmax(summed)])
        group_true.append(true_label)
        group_pred.append(pred_label)
        group_records.append(
            {
                "trial_index": int(group),
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": bool(true_label == pred_label),
                "window_pred_labels": [int(v) for v in pred[mask]],
                "summed_proba": {str(int(c)): float(summed[i]) for i, c in enumerate(classes)},
            }
        )

    group_true_arr = np.asarray(group_true)
    group_pred_arr = np.asarray(group_pred)
    return {
        "n_train_trials": int(len(train_groups)),
        "n_test_trials": int(len(test_groups)),
        "train_trials": [int(v) for v in train_groups],
        "test_trials": [int(v) for v in test_groups],
        "sample_accuracy": float(accuracy_score(y[test_idx], pred)),
        "group_accuracy": float(accuracy_score(group_true_arr, group_pred_arr)),
        "sample_confusion_matrix": confusion_matrix(y[test_idx], pred, labels=labels).tolist(),
        "group_confusion_matrix": confusion_matrix(group_true_arr, group_pred_arr, labels=labels).tolist(),
        "sample_per_class_recall": recall_score(y[test_idx], pred, labels=labels, average=None, zero_division=0).tolist(),
        "group_per_class_recall": recall_score(
            group_true_arr, group_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "wrong_groups": [record for record in group_records if not record["correct"]],
        "group_predictions": group_records,
        "sample_predictions": sample_records,
    }


def stratified_holdout_groups(
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    uniq, labels = group_labels(y, groups)
    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_i, test_i = next(splitter.split(uniq.reshape(-1, 1), labels))
    return np.sort(uniq[train_i]), np.sort(uniq[test_i])


def evaluate_leave_one_trial_out(features: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int) -> dict[str, object]:
    logo = LeaveOneGroupOut()
    labels = np.sort(np.unique(y))
    sample_true = []
    sample_pred = []
    group_true = []
    group_pred = []
    group_records = []

    for train_idx, test_idx in logo.split(features, y, groups):
        test_group = int(np.unique(groups[test_idx])[0])
        model = fixed.make_model(str(FROZEN_CONFIG["model"]), seed)
        model.fit(features[train_idx], y[train_idx])
        pred = model.predict(features[test_idx])
        proba = model.predict_proba(features[test_idx])
        classes = np.asarray(model.classes_)
        true_label = int(np.bincount(y[test_idx]).argmax())
        summed = proba.sum(axis=0)
        pred_label = int(classes[np.argmax(summed)])

        sample_true.extend(y[test_idx].tolist())
        sample_pred.extend(pred.tolist())
        group_true.append(true_label)
        group_pred.append(pred_label)
        group_records.append(
            {
                "trial_index": test_group,
                "true_label": true_label,
                "pred_label": pred_label,
                "correct": bool(true_label == pred_label),
                "window_pred_labels": [int(v) for v in pred],
                "summed_proba": {str(int(c)): float(summed[i]) for i, c in enumerate(classes)},
            }
        )

    sample_true_arr = np.asarray(sample_true)
    sample_pred_arr = np.asarray(sample_pred)
    group_true_arr = np.asarray(group_true)
    group_pred_arr = np.asarray(group_pred)
    return {
        "n_folds": int(len(group_records)),
        "sample_accuracy": float(accuracy_score(sample_true_arr, sample_pred_arr)),
        "group_accuracy": float(accuracy_score(group_true_arr, group_pred_arr)),
        "sample_confusion_matrix": confusion_matrix(sample_true_arr, sample_pred_arr, labels=labels).tolist(),
        "group_confusion_matrix": confusion_matrix(group_true_arr, group_pred_arr, labels=labels).tolist(),
        "sample_per_class_recall": recall_score(
            sample_true_arr, sample_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "group_per_class_recall": recall_score(
            group_true_arr, group_pred_arr, labels=labels, average=None, zero_division=0
        ).tolist(),
        "wrong_groups": [record for record in group_records if not record["correct"]],
        "group_predictions": group_records,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Frozen all-trial evaluation; no hyperparameter sweep.")
    parser.add_argument("--mat", type=Path, default=Path("ECoG_Handpose.mat"))
    parser.add_argument("--seed", type=int, default=20260426)
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--report", type=Path, default=Path("frozen_publication_eval_report.json"))
    args = parser.parse_args()

    features, y, groups, trials = build_frozen_features(args.mat)
    train_groups, test_groups = stratified_holdout_groups(y, groups, args.test_size, args.seed)

    holdout = evaluate_split(features, y, groups, train_groups, test_groups, args.seed)
    loto = evaluate_leave_one_trial_out(features, y, groups, args.seed)

    _, trial_labels = group_labels(y, groups)
    report = {
        "status": "frozen_evaluation_no_sweep",
        "caveat": (
            "The pipeline is frozen before this script runs, but the freeze was chosen after previous exploration "
            "on this same dataset. Treat this as an auditable locked evaluation, not an untouched external-test estimate."
        ),
        "config_hash_sha256": config_hash(FROZEN_CONFIG),
        "frozen_config": FROZEN_CONFIG,
        "preprocessing_scope": (
            "CAR, fixed notch filters, fixed whitening filter, and fixed bandpass filters are applied to the full "
            "continuous recording before splitting. No supervised transform is fit before splitting; scaler, "
            "SelectKBest, and LDA are fit only on training data inside each evaluation split."
        ),
        "dataset": {
            "mat": str(args.mat),
            "n_trials": int(np.unique(groups).size),
            "n_examples": int(len(y)),
            "trial_class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(trial_labels, return_counts=True))},
            "example_class_counts": {str(int(k)): int(v) for k, v in zip(*np.unique(y, return_counts=True))},
            "trial_index_to_samples": [
                {
                    "trial_index": int(trial.group),
                    "label": int(trial.label),
                    "start_sample": int(trial.start),
                    "end_sample": int(trial.end),
                    "duration_s": float((trial.end - trial.start) / FS),
                }
                for trial in trials
            ],
        },
        "holdout_once": holdout,
        "leave_one_trial_out": loto,
    }
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Wrote {args.report}")
    print(
        f"Holdout once: sample={holdout['sample_accuracy'] * 100:.2f}% "
        f"group={holdout['group_accuracy'] * 100:.2f}% wrong={holdout['wrong_groups']}"
    )
    print(
        f"Leave-one-trial-out: sample={loto['sample_accuracy'] * 100:.2f}% "
        f"group={loto['group_accuracy'] * 100:.2f}% wrong={loto['wrong_groups']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
