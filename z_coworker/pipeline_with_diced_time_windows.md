# Frozen ECoG Hand-Pose Classification Pipeline

## Summary

This document describes the frozen ECoG hand-pose classification pipeline implemented in
`run_frozen_publication_eval.py` and evaluated in `frozen_publication_eval_report.json`.

The purpose of this pipeline is to provide an auditable, non-sweeping evaluation of a fixed
classification method after exploratory development. The classifier uses only ECoG signals and
CH62 cue labels. The glove channels are not used for labels, alignment, filtering, window selection,
trial rejection, training, or scoring in this frozen benchmark.

The most defensible locked evaluation is leave-one-trial-out cross-validation over all 90 gesture
trials:

| Evaluation | Trial accuracy | Window accuracy | Failed trials |
|---|---:|---:|---|
| Once-only stratified holdout | 100.00% | 100.00% | none |
| Leave-one-trial-out | 98.89% | 98.89% | trial 0 only |

The once-only holdout split did not include trial 0 in the test set, so it should not be used as
the primary performance claim. Leave-one-trial-out evaluates every trial exactly once and is the
preferred frozen estimate for this dataset.

## Files

The frozen evaluation is implemented by:

```text
run_frozen_publication_eval.py
```

The evaluation output is:

```text
frozen_publication_eval_report.json
```

The frozen configuration hash recorded in the report is:

```text
1aa1c2fe617c4bb039a36f664dfb3d493c1e7be4000920242b252763769a2f04
```

This hash is computed from the frozen pipeline parameters, not from the full source code or dataset.

## Dataset

The input file is:

```text
ECoG_Handpose.mat
```

The MATLAB variable `y` contains 67 channels sampled at 1200 Hz. The relevant channels are:

| Channel range | Meaning |
|---|---|
| CH1 | Time |
| CH2-CH61 | ECoG channels |
| CH62 | Cue label |
| CH63-CH67 | Glove channels |

Only CH2-CH61 and CH62 are used in the frozen classifier.

The classifier keeps only gesture trials with CH62 labels 1, 2, and 3:

| Label | Class |
|---:|---|
| 1 | Fist / rock |
| 2 | Peace / scissors |
| 3 | Open / paper |

All 90 gesture trials are retained:

| Class | Trials | Diced windows |
|---:|---:|---:|
| 1 | 30 | 90 |
| 2 | 30 | 90 |
| 3 | 30 | 90 |
| Total | 90 | 270 |

Each trial contributes exactly three ECoG windows, so the final sample count is 270.

## Frozen Parameters

No parameter sweep is performed by the frozen evaluator. The following parameters are fixed:

| Component | Value |
|---|---|
| Label source | CH62 cue labels only |
| Trial policy | Keep all 90 gesture trials |
| Glove usage | None |
| Sampling rate | 1200 Hz |
| ECoG channels | CH2-CH61 |
| Window length | 1.49 s |
| Window mode | Cue offset |
| Main cue offset | 0.00 s |
| Number of windows per trial | 3 |
| Local stride | 0.05 s |
| Window starts | cue - 0.05 s, cue + 0.00 s, cue + 0.05 s |
| Bandpower window | 0.09 s |
| Frequency bands | 50-90, 90-130, 130-200, 200-300 Hz |
| Feature set | Summary features |
| Classifier | SelectKBest(800) + shrinkage LDA |
| Trial aggregation | Sum predicted probabilities over the 3 windows |

## Preprocessing

Preprocessing is applied to the continuous ECoG recording before window extraction:

1. Common average reference across ECoG channels.
2. Notch filtering at 50, 100, 150, 200, 250, and 300 Hz.
3. Fixed whitening filter.
4. Bandpass filtering into four high-gamma bands:
   - 50-90 Hz
   - 90-130 Hz
   - 130-200 Hz
   - 200-300 Hz

The preprocessing is unsupervised and does not use labels or glove channels.

Important methodological caveat: filtering is performed once on the full continuous recording.
Because zero-phase filtering uses neighboring time samples, this is not a fully isolated per-fold
signal preprocessing scheme. It is not supervised train/test label leakage, but it should be
reported as continuous-recording preprocessing. A stricter implementation could filter each
training and test fold independently with adequate margins, but that was not the frozen pipeline
used for the reported numbers.

## Trial Windowing

Gesture trials are identified as contiguous nonzero CH62 segments with labels 1, 2, or 3.

For each trial, three windows are extracted:

| Window position | Start time relative to cue onset | End time relative to cue onset |
|---:|---:|---:|
| 0 | -0.05 s | 1.44 s |
| 1 | 0.00 s | 1.49 s |
| 2 | +0.05 s | 1.54 s |

All three windows inherit the CH62 cue label of the parent trial.

All three windows from a trial are assigned the same group identifier. This ensures that grouped
evaluation never places windows from the same trial in both train and test sets.

## Feature Extraction

For each frequency band and ECoG channel, the pipeline computes bandpower over the 1.49 s window
using a 0.09 s bandpower window. The four band-specific bandpower tensors are concatenated.

The summary feature transform then computes the following statistics over time for each channel-band
trace:

1. Mean.
2. Standard deviation.
3. Minimum.
4. Maximum.
5. 25th percentile.
6. 75th percentile.
7. Linear slope.
8. Mean of the first temporal third.
9. Mean of the middle temporal third.
10. Mean of the final temporal third.

These features are deterministic and contain no learned parameters.

## Classifier

The classifier is a scikit-learn pipeline:

```text
StandardScaler()
SelectKBest(score_func=f_classif, k=800)
LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
```

`StandardScaler`, `SelectKBest`, and `LinearDiscriminantAnalysis` are fit only on training samples
inside each evaluation split. This avoids supervised feature-selection leakage.

The classifier produces a probability vector for each diced window. Trial-level prediction is
computed by summing the three probability vectors belonging to the same trial and selecting the
class with the largest summed probability.

## Evaluation Protocols

Two frozen evaluations are reported.

### Once-Only Stratified Holdout

A single stratified group holdout split is generated using seed `20260426` and test size `0.20`.
The split is stratified at the trial level.

Training trials:

```text
0, 3, 4, 5, 6, 8, 9, 10, 12, 14, 15, 16, 17, 19, 20, 21, 23, 24, 25, 26,
27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39, 40, 41, 42, 44, 45, 46,
47, 48, 49, 50, 51, 52, 53, 54, 55, 57, 58, 59, 60, 61, 63, 64, 65, 66,
69, 70, 71, 73, 74, 75, 76, 78, 80, 81, 82, 84, 85, 86, 87, 89
```

Test trials:

```text
1, 2, 7, 11, 13, 18, 22, 38, 43, 56, 62, 67, 68, 72, 77, 79, 83, 88
```

This holdout split contains six test trials per class.

### Leave-One-Trial-Out

Leave-one-trial-out evaluation trains on 89 trials and tests on the remaining trial. This is repeated
once for each of the 90 trials. This protocol evaluates every trial exactly once and is therefore
more informative for this small dataset than the single holdout split.

## Results

### Once-Only Stratified Holdout

| Metric | Value |
|---|---:|
| Training trials | 72 |
| Test trials | 18 |
| Test windows | 54 |
| Trial accuracy | 100.00% |
| Window accuracy | 100.00% |

Window-level confusion matrix, labels `[1, 2, 3]`:

```text
[[18,  0,  0],
 [ 0, 18,  0],
 [ 0,  0, 18]]
```

Trial-level confusion matrix, labels `[1, 2, 3]`:

```text
[[6, 0, 0],
 [0, 6, 0],
 [0, 0, 6]]
```

This result should be interpreted cautiously because trial 0, the known systematically failed trial,
is in the training set rather than the test set.

### Leave-One-Trial-Out

| Metric | Value |
|---|---:|
| Folds | 90 |
| Test trials per fold | 1 |
| Trial accuracy | 98.89% |
| Window accuracy | 98.89% |
| Wrong trials | 1 |

Window-level confusion matrix, labels `[1, 2, 3]`:

```text
[[90,  0,  0],
 [ 0, 90,  0],
 [ 0,  3, 87]]
```

Trial-level confusion matrix, labels `[1, 2, 3]`:

```text
[[30,  0,  0],
 [ 0, 30,  0],
 [ 0,  1, 29]]
```

The only failed trial is trial 0:

| Trial | CH62 true label | Predicted label | Window predictions |
|---:|---:|---:|---|
| 0 | 3 | 2 | `[2, 2, 2]` |

The summed trial probabilities for trial 0 are:

```text
class 1: 3.578e-37
class 2: 3.000
class 3: 3.178e-209
```

This indicates that trial 0 is not a borderline model decision in the frozen pipeline. It is a
high-confidence class-2 prediction despite the CH62 label being class 3.

## Trial 0 Diagnostic Context

Separate diagnostic analyses using glove channels showed that trial 0 is likely label-inconsistent
or physiologically inconsistent with its CH62 cue label. These glove diagnostics are not part of the
frozen classifier and are not used in training or scoring.

For trial 0, the CH62 label is 3, but the glove-derived state inside the CH62 cue interval contains:

| Glove state | Duration |
|---:|---:|
| 0 | 1.155 s |
| 1 | 0.000 s |
| 2 | 0.845 s |
| 3 | 0.000 s |

The glove trace never matched class 3 during the cue-labeled trial. This is consistent with the
ECoG classifier predicting class 2 for all three windows.

The frozen benchmark nevertheless keeps trial 0 and scores it against the CH62 label, because the
benchmark goal is to evaluate the CH62 cue-label task without glove-based filtering or correction.

## Leakage and Bias Assessment

### Controlled Sources

The following potential leakage sources are controlled in the frozen evaluation:

| Risk | Mitigation |
|---|---|
| Same-trial windows split between train/test | Grouped split by original trial index |
| Supervised feature selection before CV | `SelectKBest` is inside the sklearn pipeline |
| Scaling before CV | `StandardScaler` is inside the sklearn pipeline |
| LDA fit outside split | LDA is fit only inside each split |
| Glove-derived labels or rejection | Glove is not used |
| Trial rejection | No gesture trial is rejected |

### Remaining Caveats

The following caveats remain:

1. The pipeline was frozen after exploratory analysis on this dataset. Therefore, the result is an
   auditable locked evaluation, not a fully untouched external validation estimate.
2. Continuous-recording filtering is applied before splitting. This is label-free but not strictly
   isolated per fold due to zero-phase filtering.
3. The dataset contains one subject/session. The result does not demonstrate generalization across
   subjects, sessions, electrode layouts, or recording days.
4. Trial 0 appears inconsistent with its CH62 cue label. The reported accuracy keeps this trial and
   therefore reflects performance against the recorded cue labels, not corrected behavioral labels.

## Reproducibility

Run the frozen evaluation with:

```powershell
python run_frozen_publication_eval.py --report frozen_publication_eval_report.json
```

The script writes:

```text
frozen_publication_eval_report.json
```

The report includes:

1. Frozen parameter hash.
2. Dataset counts.
3. Trial start/end samples.
4. Holdout split membership.
5. Holdout predictions.
6. Leave-one-trial-out predictions.
7. Wrong-trial records.
8. Window-level and trial-level confusion matrices.

## Recommended Performance Claim

The most appropriate concise claim is:

> Using CH62 cue labels and all 90 gesture trials without glove-based filtering, the frozen ECoG-only
> pipeline achieved 98.89% leave-one-trial-out trial accuracy. The sole error was trial 0, which was
> consistently predicted as class 2 despite a CH62 label of class 3. Independent glove diagnostics
> suggest that this trial is inconsistent with its cue label, but it was retained in the benchmark.

Avoid claiming that the method achieved 100% accuracy on the full dataset. The 100% holdout result is
split-dependent and excludes trial 0 from the test set.
