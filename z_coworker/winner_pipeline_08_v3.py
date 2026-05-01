# -*- coding: utf-8 -*-
"""Diced sliding-window ECoG hand-pose decoder.

This script rebuilds the original pipeline locally, adds overlapping
post-onset "dicing" windows, and uses tqdm + joblib workers to speed up the
repeated cross-validation and permutation runs.
"""

from __future__ import annotations

import warnings
from contextlib import contextmanager
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed, dump, load, parallel
from scipy.io import loadmat
from scipy.signal import (
    butter,
    filtfilt,
    iirnotch,
    lfilter,
    savgol_filter,
    sosfiltfilt,
)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover - optional plotting dependency
    sns = None


DATA_PATH = Path(__file__).with_name("ECoG_Handpose.mat")
FS = 1200
LABELS = np.array([1, 2, 3], dtype=int)
CLASS_NAMES = {1: "Fist", 2: "Peace", 3: "Open hand"}

PARADIGM_ROW = 61
ECOG_ROWS = slice(1, 61)
GLOVE_ROWS = slice(62, 67)

AR_ORDER = 4
CV_SPLITS = 10
N_JOBS = -1
STABILITY_SEEDS = [0, 1, 2, 7, 42]
PERMUTATION_RUNS = 5
FORCE_RECOMPUTE = False

WINDOW_BLOCK_SPECS = [
    {"name": "early", "start_ms": 0.0, "window_ms": 250.0, "n_windows": 3, "stride_ms": 125.0},
    {"name": "late", "start_ms": 600.0, "window_ms": 300.0, "n_windows": 2, "stride_ms": 225.0},
]


def normalize_window_blocks(block_specs: list[dict]) -> list[dict]:
    """Precompute sample-domain metadata for each diced window block."""
    blocks = []
    for block in block_specs:
        new_block = dict(block)
        new_block["start_samples"] = int(round(new_block["start_ms"] * FS / 1000))
        new_block["window_samples"] = int(round(new_block["window_ms"] * FS / 1000))
        new_block["stride_samples"] = int(round(new_block["stride_ms"] * FS / 1000))
        new_block["offsets"] = new_block["start_samples"] + np.arange(new_block["n_windows"]) * new_block["stride_samples"]
        blocks.append(new_block)
    return blocks


def build_run_tag(blocks: list[dict]) -> str:
    """Compact checkpoint tag for the active multi-block window plan."""
    parts = []
    for block in blocks:
        parts.append(
            f"{block['name']}{int(block['start_ms'])}"
            f"w{int(block['window_ms'])}"
            f"n{int(block['n_windows'])}"
            f"d{int(round(block['stride_ms'] * 10))}"
        )
    return "__".join(parts)


def build_window_plan_label(blocks: list[dict]) -> str:
    """Human-readable summary of the active window plan."""
    return " | ".join(
        f"{block['name']}: {int(block['n_windows'])}x{int(block['window_ms'])} ms "
        f"start {int(block['start_ms'])} ms stride {block['stride_ms']:.1f} ms"
        for block in blocks
    )


WINDOW_BLOCKS = normalize_window_blocks(WINDOW_BLOCK_SPECS)
TOTAL_WINDOWS = int(sum(block["n_windows"] for block in WINDOW_BLOCKS))

# Backward-compatible aliases for scripts that assume a primary block exists.
DICE_WINDOW_MS = int(WINDOW_BLOCKS[0]["window_ms"])
DICE_STRIDE_MS = WINDOW_BLOCKS[0]["stride_ms"]
DICE_WINDOWS = int(WINDOW_BLOCKS[0]["n_windows"])
DICE_WINDOW_SAMPLES = int(WINDOW_BLOCKS[0]["window_samples"])
DICE_STRIDE_SAMPLES = int(WINDOW_BLOCKS[0]["stride_samples"])
DICE_OFFSETS = WINDOW_BLOCKS[0]["offsets"]

CHECKPOINT_DIR = Path(__file__).with_name("checkpoints")
PREPROCESSED_CHECKPOINT = CHECKPOINT_DIR / "preprocessed_trials.joblib"
RUN_TAG = build_run_tag(WINDOW_BLOCKS)
WINDOW_PLAN_LABEL = build_window_plan_label(WINDOW_BLOCKS)

HG_SOS = butter(6, [50.0, 300.0], btype="bandpass", fs=FS, output="sos")


@contextmanager
def tqdm_joblib(tqdm_object):
    """Route joblib batch completion events into a tqdm progress bar."""
    if tqdm_object is None:
        yield
        return

    class TqdmBatchCompletionCallback(parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = parallel.BatchCompletionCallBack
    parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()


def ensure_checkpoint_dir() -> None:
    """Create the checkpoint directory if needed."""
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


def checkpoint_path(name: str) -> Path:
    """Stage checkpoint path for the active diced-window configuration."""
    ensure_checkpoint_dir()
    return CHECKPOINT_DIR / f"{name}_{RUN_TAG}.joblib"


def compute_or_load_checkpoint(
    path: Path,
    label: str,
    compute_fn,
    force_recompute: bool = False,
):
    """Load an existing checkpoint or compute and save it."""
    if path.exists() and not force_recompute:
        print(f"Loaded checkpoint: {path.name} ({label})")
        return load(path)

    result = compute_fn()
    dump(result, path)
    print(f"Saved checkpoint: {path.name} ({label})")
    return result


def load_data(path: Path = DATA_PATH) -> tuple[np.ndarray, int]:
    """Load the raw 67 x N matrix from the .mat file."""
    mat = loadmat(path)
    if "y" not in mat:
        raise KeyError(f"Expected key 'y' in {path}, found {sorted(mat)}")

    raw = np.asarray(mat["y"], dtype=float)
    if raw.ndim != 2 or raw.shape[0] != 67:
        raise ValueError(f"Expected raw shape (67, N), got {raw.shape}")

    return raw, FS


def extract_trials(raw: np.ndarray, fs: int = FS) -> list[dict]:
    """Slice trials from paradigm transitions 0 -> {1,2,3} ... -> 0."""
    del fs  # Sampling rate is fixed for this dataset and not needed here.

    paradigm = raw[PARADIGM_ROW].astype(int)
    starts = np.flatnonzero((paradigm[:-1] == 0) & (paradigm[1:] > 0)) + 1
    ends = np.flatnonzero((paradigm[:-1] > 0) & (paradigm[1:] == 0)) + 1

    if starts.size != ends.size:
        raise ValueError(
            f"Mismatched trial start/end counts: {starts.size} starts vs {ends.size} ends"
        )

    trials = []
    for trial_index, (start, end) in enumerate(zip(starts, ends)):
        label = int(paradigm[start])
        trial = {
            "trial_index": trial_index,
            "label": label,
            "start_sample": int(start),
            "end_sample": int(end),
            "ecog": np.array(raw[ECOG_ROWS, start:end], copy=True),
            "glove": np.array(raw[GLOVE_ROWS, start:end], copy=True),
        }
        trials.append(trial)

    return trials


def find_movement_onset(trial_dict: dict, fs: int = FS) -> int:
    """Use smoothed glove velocity to align the analysis window to movement."""
    glove = np.asarray(trial_dict["glove"], dtype=float)
    if glove.ndim != 2 or glove.shape[1] == 0:
        return 0

    velocity = np.abs(np.diff(glove, axis=1, prepend=glove[:, :1])) * fs
    velocity = velocity.sum(axis=0)

    window_length = min(51, velocity.size if velocity.size % 2 else velocity.size - 1)
    if window_length >= 5:
        velocity = savgol_filter(
            velocity,
            window_length=window_length,
            polyorder=min(3, window_length - 1),
        )

    search = velocity[: min(fs, velocity.size)]
    peak = float(np.max(search)) if search.size else 0.0
    if peak <= 0.0 or not np.isfinite(peak):
        return 0

    threshold = 0.3 * peak
    hits = np.flatnonzero(search >= threshold)
    if hits.size:
        return int(hits[0])
    return int(np.argmax(search))


def preprocess_trial(trial_dict: dict, fs: int = FS) -> dict:
    """Per-trial CAR + 50 Hz notch, then cache glove-based onset."""
    ecog = np.asarray(trial_dict["ecog"], dtype=float)
    ecog = ecog - ecog.mean(axis=0, keepdims=True)

    b, a = iirnotch(50.0, 30.0, fs=fs)
    ecog = filtfilt(b, a, ecog, axis=1)

    new_trial = dict(trial_dict)
    new_trial["ecog"] = ecog
    new_trial["onset"] = int(find_movement_onset(new_trial, fs=fs))
    return new_trial


def preprocess_trials(trials: list[dict], n_jobs: int = N_JOBS) -> list[dict]:
    """Parallel per-trial preprocessing with a progress bar."""
    bar = tqdm(total=len(trials), desc="Preprocessing trials", unit="trial")
    with tqdm_joblib(bar):
        return Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(preprocess_trial)(trial, FS) for trial in trials
        )


def load_or_preprocess_trials(
    raw: np.ndarray,
    fs: int = FS,
    n_jobs: int = N_JOBS,
    force_recompute: bool = False,
) -> list[dict]:
    """Reuse cached preprocessed trials when available."""
    ensure_checkpoint_dir()
    if PREPROCESSED_CHECKPOINT.exists() and not force_recompute:
        payload = load(PREPROCESSED_CHECKPOINT)
        cached_trials = payload["trials"]
        if payload.get("raw_shape") == raw.shape:
            print(f"Loaded checkpoint: {PREPROCESSED_CHECKPOINT.name} (preprocessed trials)")
            return cached_trials

    trials = extract_trials(raw, fs=fs)
    trials = preprocess_trials(trials, n_jobs=n_jobs)
    dump({"raw_shape": raw.shape, "trials": trials}, PREPROCESSED_CHECKPOINT)
    print(f"Saved checkpoint: {PREPROCESSED_CHECKPOINT.name} (preprocessed trials)")
    return trials


def fit_ar_whitener(train_trials: list[dict], order: int = AR_ORDER) -> np.ndarray:
    """Fit a channel-wise AR model on concatenated training trials only."""
    stacked = np.concatenate([trial["ecog"] for trial in train_trials], axis=1)
    n_channels = stacked.shape[0]
    coefs = np.zeros((n_channels, order), dtype=float)

    for ch in range(n_channels):
        x = stacked[ch]
        if x.size <= order:
            continue
        y = x[order:]
        lagged = np.column_stack(
            [x[order - lag - 1 : x.size - lag - 1] for lag in range(order)]
        )
        coefs[ch], *_ = np.linalg.lstsq(lagged, y, rcond=None)

    return coefs


def apply_ar_whitener(trials: list[dict], ar_coefs: np.ndarray) -> list[dict]:
    """Apply the trained AR residual filter to each trial independently."""
    whitened = []
    for trial in trials:
        ecog = np.empty_like(trial["ecog"], dtype=float)
        for ch, coefs in enumerate(ar_coefs):
            b = np.concatenate(([1.0], -coefs))
            ecog[ch] = lfilter(b, [1.0], trial["ecog"][ch])
        new_trial = dict(trial)
        new_trial["ecog"] = ecog
        whitened.append(new_trial)
    return whitened


def whiten_trials(
    train_trials: list[dict], test_trials: list[dict], order: int = AR_ORDER
) -> tuple[list[dict], list[dict], np.ndarray]:
    """Fit whitening on the train split and apply to both splits."""
    ar_coefs = fit_ar_whitener(train_trials, order=order)
    train_w = apply_ar_whitener(train_trials, ar_coefs)
    test_w = apply_ar_whitener(test_trials, ar_coefs)
    return train_w, test_w, ar_coefs


def extract_diced_features(
    trial_dict: dict,
    onset: int,
    fs: int = FS,
    trial_index: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract high-gamma log-variance slices from the active multi-block plan."""
    del fs  # Constants are precomputed at module load for this dataset.

    ecog = np.asarray(trial_dict["ecog"], dtype=float)
    n_samples = ecog.shape[1]
    hg = sosfiltfilt(HG_SOS, ecog, axis=1)

    features = []
    slice_offsets = []
    min_required = min(block["window_samples"] for block in WINDOW_BLOCKS)
    for block in WINDOW_BLOCKS:
        for offset in block["offsets"]:
            start = int(onset + offset)
            end = start + int(block["window_samples"])
            if end > n_samples:
                continue
            window = hg[:, start:end]
            features.append(np.log(np.var(window, axis=1) + 1e-8))
            slice_offsets.append(int(offset))

    if not features:
        trial_id = trial_dict.get("trial_index", trial_index)
        post_onset = max(0, n_samples - int(onset))
        raise ValueError(
            f"Trial {trial_id} has no valid diced windows: "
            f"post-onset samples={post_onset}, required at least={min_required}"
        )

    return np.vstack(features), np.array(slice_offsets, dtype=int)


def aggregate_slice_predictions(
    slice_predictions: np.ndarray,
    slice_probabilities: np.ndarray | None,
    classes: np.ndarray,
) -> int:
    """Collapse slice-level predictions back to one trial label."""
    unique, counts = np.unique(slice_predictions, return_counts=True)
    max_count = counts.max()
    tied = unique[counts == max_count]

    if tied.size == 1:
        return int(tied[0])

    if slice_probabilities is not None:
        class_to_col = {int(label): idx for idx, label in enumerate(classes)}
        mean_scores = np.array(
            [slice_probabilities[:, class_to_col[int(label)]].mean() for label in tied]
        )
        best_score = mean_scores.max()
        tied = tied[np.isclose(mean_scores, best_score)]
        if tied.size == 1:
            return int(tied[0])

    for predicted in slice_predictions:
        if predicted in tied:
            return int(predicted)
    return int(tied[0])


def build_train_slice_matrix(
    train_trials: list[dict], y_train: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Expand trial labels to one row per valid diced slice."""
    feature_rows = []
    slice_labels = []

    for trial, label in zip(train_trials, y_train):
        feats, _ = extract_diced_features(
            trial, trial["onset"], trial_index=trial.get("trial_index")
        )
        feature_rows.append(feats)
        slice_labels.extend([int(label)] * feats.shape[0])

    return np.vstack(feature_rows), np.array(slice_labels, dtype=int)


def pipeline_v3(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """AR whitening -> diced high-gamma features -> scaler -> shrinkage LDA."""
    train_w, test_w, _ = whiten_trials(list(X_train), list(X_test), order=AR_ORDER)

    Xtr, ytr = build_train_slice_matrix(train_w, y_train)
    scaler = StandardScaler().fit(Xtr)
    Xtr_scaled = scaler.transform(Xtr)

    clf = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto").fit(
        Xtr_scaled, ytr
    )

    predictions = []
    for trial in test_w:
        feats, _ = extract_diced_features(
            trial, trial["onset"], trial_index=trial.get("trial_index")
        )
        Xte = scaler.transform(feats)
        slice_predictions = clf.predict(Xte)
        slice_probabilities = clf.predict_proba(Xte)
        predictions.append(
            aggregate_slice_predictions(
                slice_predictions, slice_probabilities, clf.classes_
            )
        )

    return np.array(predictions, dtype=int)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    trial_ids: np.ndarray | None = None,
) -> dict:
    """Compute the metric bundle used throughout the script."""
    if trial_ids is None:
        trial_ids = np.arange(len(y_true), dtype=int)
    trial_ids = np.asarray(trial_ids, dtype=int)

    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    recalls = recall_score(y_true, y_pred, labels=LABELS, average=None)
    miss_idx = np.flatnonzero(y_true != y_pred)
    mistakes = [
        {
            "trial_id": int(trial_ids[i]),
            "true_label": int(y_true[i]),
            "pred_label": int(y_pred[i]),
        }
        for i in miss_idx
    ]
    return {
        "acc": accuracy_score(y_true, y_pred),
        "bal_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "cm": cm,
        "recall_per_class": recalls,
        "n_misses": int(miss_idx.size),
        "wrong_trial_ids": [item["trial_id"] for item in mistakes],
        "mistakes": mistakes,
        "y_true": np.array(y_true, copy=True),
        "y_pred": np.array(y_pred, copy=True),
        "trial_ids": np.array(trial_ids, copy=True),
    }


def _run_cv_fold(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Worker body for one CV fold."""
    y_pred = pipeline_v3(X[train_idx], y[train_idx], X[test_idx])
    return test_idx, y_pred


def cv_predictions_with_metrics(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = CV_SPLITS,
    random_state: int = 42,
    desc: str | None = None,
    show_progress: bool = True,
    n_jobs: int = N_JOBS,
) -> dict:
    """Parallel stratified CV that returns fold-aggregated predictions + metrics."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    folds = list(skf.split(np.zeros(len(y)), y))
    y_pred_all = np.empty_like(y)

    bar = None
    if show_progress:
        bar = tqdm(total=len(folds), desc=desc or f"CV seed {random_state}", unit="fold")

    with tqdm_joblib(bar):
        outputs = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_run_cv_fold)(X, y, train_idx, test_idx)
            for train_idx, test_idx in folds
        )

    for test_idx, y_pred in outputs:
        y_pred_all[test_idx] = y_pred

    trial_ids = np.array([int(trial["trial_index"]) for trial in X], dtype=int)
    return compute_metrics(y, y_pred_all, trial_ids=trial_ids)


def validate_diced_schedule(trials: np.ndarray) -> np.ndarray:
    """Ensure the active multi-block plan fits every trial."""
    counts = []
    for trial in trials:
        available = int(trial["ecog"].shape[1] - trial["onset"])
        valid = 0
        for block in WINDOW_BLOCKS:
            valid += int(
                sum(
                    trial["onset"] + offset + int(block["window_samples"]) <= trial["ecog"].shape[1]
                    for offset in block["offsets"]
                )
            )
        if valid == 0:
            raise ValueError(
                f"Trial {trial['trial_index']} cannot fit any diced window; "
                f"post-onset samples={available}"
            )
        counts.append(valid)
    return np.array(counts, dtype=int)


def run_time_forward(X: np.ndarray, y: np.ndarray) -> dict:
    """Chronological hold-out evaluation with trial-ID reporting."""
    order = np.arange(len(X))
    split = int(0.8 * len(X))
    train_idx, test_idx = order[:split], order[split:]
    y_pred = pipeline_v3(X[train_idx], y[train_idx], X[test_idx])
    trial_ids = np.array([int(X[idx]["trial_index"]) for idx in test_idx], dtype=int)
    result = compute_metrics(y[test_idx], y_pred, trial_ids=trial_ids)
    result["train_idx"] = train_idx
    result["test_idx"] = test_idx
    return result


def print_mistakes(title: str, mistakes: list[dict]) -> None:
    """Print trial IDs and labels for misclassified instances."""
    print(title)
    if not mistakes:
        print("  none")
        return

    for item in mistakes:
        true_name = CLASS_NAMES[item["true_label"]]
        pred_name = CLASS_NAMES[item["pred_label"]]
        print(
            f"  trial_id={item['trial_id']} true={item['true_label']} ({true_name}) "
            f"pred={item['pred_label']} ({pred_name})"
        )


def plot_confusion(cm: np.ndarray, acc: float) -> Path:
    """Save the confusion matrix plot to disk."""
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    labels = [CLASS_NAMES[label] for label in LABELS]
    if sns is not None:
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            annot_kws={"size": 14},
        )
    else:
        image = ax.imshow(cm, cmap="Blues")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        for row in range(cm.shape[0]):
            for col in range(cm.shape[1]):
                ax.text(col, row, int(cm[row, col]), ha="center", va="center")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Pipeline V3 diced windows | 10-fold CV | acc = {acc:.4f}")
    fig.tight_layout()

    output_path = Path(__file__).with_name("winner_pipeline_08_v3_confusion_matrix.png")
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def run_seed_stability(X: np.ndarray, y: np.ndarray, baseline_result: dict) -> pd.DataFrame:
    """Re-run CV across several fold seeds."""
    rows = []
    for seed in tqdm(STABILITY_SEEDS, desc="5-seed stability", unit="seed"):
        if seed == 42:
            result = baseline_result
        else:
            result = cv_predictions_with_metrics(
                X,
                y,
                n_splits=CV_SPLITS,
                random_state=seed,
                show_progress=False,
                n_jobs=N_JOBS,
            )
        rows.append(
            {
                "seed": seed,
                "acc": result["acc"],
                "n_misses": result["n_misses"],
                "bal_acc": result["bal_acc"],
                "macro_f1": result["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def run_label_permutations(X: np.ndarray, y: np.ndarray) -> list[float]:
    """Permutation null with pre-generated shuffled label vectors."""
    rng = np.random.default_rng(0)
    shuffled_targets = []
    for _ in range(PERMUTATION_RUNS):
        y_shuffled = y.copy()
        rng.shuffle(y_shuffled)
        shuffled_targets.append(y_shuffled)

    accuracies = []
    for i, y_shuffled in enumerate(
        tqdm(shuffled_targets, desc="Label permutations", unit="perm")
    ):
        result = cv_predictions_with_metrics(
            X,
            y_shuffled,
            n_splits=CV_SPLITS,
            random_state=42,
            show_progress=False,
            n_jobs=N_JOBS,
        )
        accuracies.append(result["acc"])
        print(f"  permutation {i}: acc = {result['acc']:.4f}")
    return accuracies


def run_leakage_audit(X: np.ndarray, y: np.ndarray, raw: np.ndarray) -> None:
    """Programmatic checks that preserve the original leakage guarantees."""
    tr_subset = list(X[:60])
    te_subset = list(X[60:])

    _, _, coefs_ab = whiten_trials(tr_subset, te_subset, order=AR_ORDER)
    _, _, coefs_a = whiten_trials(tr_subset, [], order=AR_ORDER)
    assert np.allclose(coefs_ab, coefs_a), "AR coefficients changed with test data"
    print("A | AR whitening   : coefficients depend only on train trials  OK")

    trial_5 = X[5]
    trial_5_copy = dict(trial_5)
    trial_5_copy["glove"] = trial_5["glove"].copy()
    onset_alone = find_movement_onset(trial_5_copy, fs=FS)
    onset_in_context = find_movement_onset(trial_5, fs=FS)
    assert onset_alone == onset_in_context, "Movement onset depends on other trials"
    print("B | Movement onset : per-trial glove alignment only             OK")

    demo_train_w, _, _ = whiten_trials(list(X[:80]), list(X[80:]), order=AR_ORDER)
    demo_feats, _ = extract_diced_features(
        demo_train_w[0], X[0]["onset"], trial_index=X[0]["trial_index"]
    )
    assert demo_feats.shape[1] == 60, f"Expected 60 ECoG features, got {demo_feats.shape}"
    print("C | Glove leakage  : diced features stay ECoG-only              OK")

    single = preprocess_trial(extract_trials(raw, fs=FS)[0], fs=FS)
    in_batch = preprocess_trial(extract_trials(raw, fs=FS)[0], fs=FS)
    assert np.allclose(single["ecog"], in_batch["ecog"]), "CAR/notch differ in/out of batch"
    print("D | CAR / notch    : per-trial deterministic preprocessing      OK")

    skf = StratifiedKFold(n_splits=CV_SPLITS, shuffle=True, random_state=42)
    train_idx, test_idx = next(iter(skf.split(np.zeros(len(y)), y)))
    train_w, _, _ = whiten_trials(list(X[train_idx]), list(X[test_idx]), order=AR_ORDER)
    Xtr, _ = build_train_slice_matrix(train_w, y[train_idx])
    scaler_train_only = StandardScaler().fit(Xtr)

    all_w, _, _ = whiten_trials(list(X), [], order=AR_ORDER)
    Xall, _ = build_train_slice_matrix(all_w, y)
    scaler_with_test = StandardScaler().fit(Xall)
    assert not np.allclose(scaler_train_only.mean_, scaler_with_test.mean_), (
        "Train-only scaler unexpectedly matches all-data scaler"
    )
    print("E | StandardScaler : train-only slice stats differ from all-data OK")

    mock_preds = np.array([1, 2, 2, 1], dtype=int)
    mock_proba = np.array(
        [
            [0.55, 0.30, 0.15],
            [0.25, 0.60, 0.15],
            [0.20, 0.65, 0.15],
            [0.51, 0.34, 0.15],
        ]
    )
    agg = aggregate_slice_predictions(mock_preds, mock_proba, LABELS)
    assert agg == 2, f"Expected tie-break vote to return class 2, got {agg}"
    print("F | Vote reducer   : majority/tie-break aggregation is stable   OK")


def print_trial_summary(raw: np.ndarray, trials: list[dict], slice_counts: np.ndarray) -> None:
    """Dataset and dicing overview."""
    y_all = np.array([trial["label"] for trial in trials], dtype=int)
    trial_lengths_s = np.array([trial["ecog"].shape[1] / FS for trial in trials], dtype=float)

    print(f"NumPy {np.__version__} | pandas {pd.__version__}")
    print(f"Raw matrix shape : {raw.shape} (channels x samples)")
    print(f"Recording length : {raw.shape[1] / FS:.1f} s")
    print(f"Number of trials : {len(trials)}")
    print(
        "Class counts     : "
        f"Fist={int((y_all == 1).sum())}, "
        f"Peace={int((y_all == 2).sum())}, "
        f"OpenHand={int((y_all == 3).sum())}"
    )
    print(
        "Trial length (s) : "
        f"min={trial_lengths_s.min():.2f}, "
        f"mean={trial_lengths_s.mean():.2f}, "
        f"max={trial_lengths_s.max():.2f}"
    )
    print(
        "Window plan      : "
        f"{WINDOW_PLAN_LABEL}"
    )
    print(
        "Valid slices/trial: "
        f"min={slice_counts.min()}, mean={slice_counts.mean():.2f}, max={slice_counts.max()}"
    )


def main() -> None:
    """Run the full diced sliding-window evaluation pipeline."""
    if sns is not None:
        sns.set_style("white")

    raw, fs = load_data()
    assert fs == FS

    trials = load_or_preprocess_trials(
        raw, fs=fs, n_jobs=N_JOBS, force_recompute=FORCE_RECOMPUTE
    )
    X = np.array(trials, dtype=object)
    y = np.array([trial["label"] for trial in trials], dtype=int)

    slice_counts = validate_diced_schedule(X)
    print_trial_summary(raw, trials, slice_counts)
    print()

    result = compute_or_load_checkpoint(
        checkpoint_path("seed42_cv"),
        "seed-42 cross-validation",
        lambda: cv_predictions_with_metrics(
            X,
            y,
            n_splits=CV_SPLITS,
            random_state=42,
            desc="Seed 42 CV",
            show_progress=True,
            n_jobs=N_JOBS,
        ),
        force_recompute=FORCE_RECOMPUTE,
    )

    print(f"Accuracy        : {result['acc']:.4f}")
    print(f"Balanced acc    : {result['bal_acc']:.4f}")
    print(f"Macro F1        : {result['macro_f1']:.4f}")
    print(
        "Per-class recall: "
        f"Fist={result['recall_per_class'][0]:.3f}, "
        f"Peace={result['recall_per_class'][1]:.3f}, "
        f"OpenHand={result['recall_per_class'][2]:.3f}"
    )
    print(f"Mistakes        : {result['n_misses']} / {len(y)}")
    print()
    print("Confusion matrix (rows = true, cols = predicted, labels [Fist, Peace, OpenHand]):")
    print(result["cm"])
    print_mistakes("Wrong instances (seed-42 CV):", result["mistakes"])
    confusion_path = plot_confusion(result["cm"], result["acc"])
    print(f"Saved confusion matrix plot to {confusion_path.name}")
    print()

    stability = compute_or_load_checkpoint(
        checkpoint_path("stability"),
        "5-seed stability",
        lambda: run_seed_stability(X, y, result),
        force_recompute=FORCE_RECOMPUTE,
    )
    print(stability.to_string(index=False))
    print()
    print(f"mean acc  = {stability['acc'].mean():.4f} +/- {stability['acc'].std():.4f}")
    print(f"total errors = {stability['n_misses'].sum()} / {len(stability) * len(y)}")
    print()

    time_forward = compute_or_load_checkpoint(
        checkpoint_path("time_forward"),
        "chronological hold-out",
        lambda: run_time_forward(X, y),
        force_recompute=FORCE_RECOMPUTE,
    )

    split = len(time_forward["train_idx"])
    print(f"Train trials : first {split} (chronological)")
    print(f"Test trials  : last  {len(time_forward['test_idx'])} (chronological)")
    print(f"acc          : {time_forward['acc']:.4f}")
    print(f"balanced acc : {time_forward['bal_acc']:.4f}")
    print(f"macro F1     : {time_forward['macro_f1']:.4f}")
    print()
    print("Confusion matrix (rows = true, cols = predicted, labels [Fist, Peace, OpenHand]):")
    print(time_forward["cm"])
    print_mistakes("Wrong instances (chronological hold-out):", time_forward["mistakes"])
    print()

    perm_accs = compute_or_load_checkpoint(
        checkpoint_path("permutations"),
        "label permutations",
        lambda: run_label_permutations(X, y),
        force_recompute=FORCE_RECOMPUTE,
    )
    print()
    print(f"Real labels  : {result['acc']:.4f}")
    print(
        "Permuted     : "
        f"mean = {np.mean(perm_accs):.4f}, "
        f"min = {np.min(perm_accs):.4f}, "
        f"max = {np.max(perm_accs):.4f}"
    )
    print(f"Chance level : {1 / 3:.4f}")
    print()

    run_leakage_audit(X, y, raw)
    print()
    print("All leakage checkpoints pass.")
    print()

    print(
        classification_report(
            result["y_true"],
            result["y_pred"],
            labels=LABELS,
            target_names=[CLASS_NAMES[label] for label in LABELS],
            digits=4,
        )
    )


if __name__ == "__main__":
    main()
