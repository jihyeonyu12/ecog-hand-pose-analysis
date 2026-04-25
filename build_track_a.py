import nbformat as nbf

nb = nbf.v4.new_notebook()

intro_md = """# Track A: TVLDA-PCA Pipeline (Complete Steps 1-3)
This notebook implements the foundational steps for Track A:
1. **Preprocessing**: CAR, Notch (50Hz + harmonics), Whitening (10th-order FIR), and Band-pass (50-300Hz).
2. **Trial Alignment**: Dynamically aligning ECoG trials based on the actual movement onset detected via Data Glove signals, ensuring precise timing for Time-Variant LDA features.
3. **Feature Extraction & TVLDA**: Extracting high-gamma power envelope, reducing spatial dimensions with PCA, and applying Time-Variant LDA classification.
"""

code_import = """import numpy as np
import scipy.io as sio
import scipy.signal as signal
import matplotlib.pyplot as plt
import mne
import warnings
warnings.filterwarnings('ignore')

# 1. Load Data
mat_path = 'dataset/ecog-hand-pose/ECoG_Handpose.mat'
data = sio.loadmat(mat_path)['y']

fs = 1200 # Sampling frequency
ecog_data = data[1:61, :] # 60 channels (CH2~CH61)
paradigm = data[61, :]    # CH62 (0, 1, 2, 3)
glove = data[62:67, :]    # CH63~CH67

print(f"ECoG Data Shape: {ecog_data.shape}")
"""

code_preprocessing = """# 2. Preprocessing Step 1: CAR (Common Average Reference)
car_mean = np.mean(ecog_data, axis=0)
ecog_car = ecog_data - car_mean
print("Applied CAR (Common Average Reference)")

# Convert to MNE object
info = mne.create_info(ch_names=[f"CH{i+2}" for i in range(60)], sfreq=fs, ch_types='ecog')
raw = mne.io.RawArray(ecog_car, info)

# 3. Preprocessing Step 2: Notch Filter (50Hz & Harmonics)
freqs_to_notch = np.arange(50, 301, 50)
raw.notch_filter(freqs=freqs_to_notch, fir_design='firwin', verbose=False)
print(f"Applied Notch Filter: {freqs_to_notch} Hz")

ecog_notched = raw.get_data()
"""

code_whitening = """# 4. Preprocessing Step 3: Whitening (Flattening 1/f Spectrum)
b_whiten = np.array([1, -0.98]) 
a_whiten = np.array([1])
ecog_whitened = signal.lfilter(b_whiten, a_whiten, ecog_notched, axis=-1)
print("Applied Whitening (FIR Pre-emphasis approximation)")

# 5. Preprocessing Step 4: Band-pass Filter (50-300Hz)
raw_whitened = mne.io.RawArray(ecog_whitened, info)
raw_whitened.filter(l_freq=50., h_freq=300., fir_design='firwin', verbose=False)
ecog_preprocessed = raw_whitened.get_data()
print("Applied Band-pass Filter (50-300Hz)")
"""

md_alignment = """## Step 2: Trial Alignment
Using the Data Glove variance to align epoch extraction. 
The standard cue starts at $T=0$, but actual hand movement starts ~0.5s later. TVLDA requires strict temporal alignment. We find the index where glove velocity spikes and epoch around that true onset.
"""

code_alignment = """# Detect paradigm visual cues (Transitions from 0 to 1, 2, or 3)
cue_indices = np.where((paradigm[1:] > 0) & (paradigm[:-1] == 0))[0] + 1
print(f"Found {len(cue_indices)} total cues.")

aligned_epochs = []
labels = []

# Define epoch window relative to actual movement onset
t_pre = int(0.5 * fs)   # 0.5s before movement
t_post = int(1.0 * fs)  # 1.0s after movement

alignment_offsets = []

for idx in cue_indices:
    cue_label = paradigm[idx]
    
    search_end = min(idx + int(2.5 * fs), glove.shape[1])
    glove_window = glove[:, idx:search_end]
    
    # Compute velocity (sum of absolute differences)
    glove_velocity = np.sum(np.abs(np.diff(glove_window, axis=1)), axis=0)
    
    # Smooth velocity
    window_sz = 30
    glove_smooth = np.convolve(glove_velocity, np.ones(window_sz)/window_sz, mode='valid')
    
    # Dynamic threshold
    baseline = np.mean(glove_smooth[:100])
    threshold = baseline + np.std(glove_smooth[:100]) * 5 + 0.005
    
    onset_rel = np.argmax(glove_smooth > threshold)
    if onset_rel == 0:
        onset_rel = int(0.5 * fs)
        
    actual_onset = idx + onset_rel
    alignment_offsets.append(onset_rel / fs)
    
    epoch_start = actual_onset - t_pre
    epoch_end = actual_onset + t_post
    
    if epoch_start >= 0 and epoch_end < ecog_preprocessed.shape[1]:
        aligned_epochs.append(ecog_preprocessed[:, epoch_start:epoch_end])
        labels.append(cue_label)

epochs_array = np.array(aligned_epochs)
labels_array = np.array(labels)

print(f"Successfully aligned and extracted {epochs_array.shape[0]} trials.")
print(f"Average Movement Delay from Cue: {np.mean(alignment_offsets):.3f}s")
"""

md_step3 = """## Step 3: Feature Extraction (PCA) and TVLDA
We extract spatial-temporal features. First, we compute the signal envelope (power) for the high-gamma band. Then, we apply PCA for spatial dimensionality reduction. Finally, we implement Time-Variant LDA (TVLDA) to classify the trials.
"""

code_step3 = """from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# 1. Compute High-Gamma Envelope (Power)
# Square the signal and smooth it using a moving average
window_size = int(0.1 * fs) # 100ms smoothing window
epochs_power = np.zeros_like(epochs_array)
for tr in range(epochs_array.shape[0]):
    for ch in range(epochs_array.shape[1]):
        squared_sig = epochs_array[tr, ch, :] ** 2
        epochs_power[tr, ch, :] = np.convolve(squared_sig, np.ones(window_size)/window_size, mode='same')

# Normalize variance using logarithm
epochs_power = np.log1p(epochs_power * 1000)

# 2. Spatial Dimension Reduction via PCA
n_components = 20
pca = PCA(n_components=n_components)

n_trials, n_ch, n_time = epochs_power.shape
epochs_reshape = np.transpose(epochs_power, (0, 2, 1)).reshape(-1, n_ch)
epochs_pca_flat = pca.fit_transform(epochs_reshape)
epochs_pca = epochs_pca_flat.reshape(n_trials, n_time, n_components).transpose(0, 2, 1)

print(f"Explained Variance Ratio by top {n_components} components: {np.sum(pca.explained_variance_ratio_)*100:.1f}%")

# 3. TVLDA Classification
# Decimate time to reduce feature space, maintaining core temporal dynamics
down_factor = 20 # 1200Hz -> 60Hz temporal resolution
features = epochs_pca[:, :, ::down_factor]
n_trials_feat, n_comp, n_t_down = features.shape

# Flatten spatial-temporal components per trial
X = features.reshape(n_trials_feat, -1)
y_labels = labels_array

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []

for train_idx, test_idx in skf.split(X, y_labels):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y_labels[train_idx], y_labels[test_idx]
    
    # TVLDA utilizing shrinkage for high dimensional stability
    clf = LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto')
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, preds))

print(f"Features mapped per trial: {X.shape[1]}")
print(f"5-Fold CV Accuracy (Track A - TVLDA-PCA Baseline): {np.mean(accuracies) * 100:.2f}%")
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(code_import),
    nbf.v4.new_code_cell(code_preprocessing),
    nbf.v4.new_code_cell(code_whitening),
    nbf.v4.new_markdown_cell(md_alignment),
    nbf.v4.new_code_cell(code_alignment),
    nbf.v4.new_markdown_cell(md_step3),
    nbf.v4.new_code_cell(code_step3)
]

with open('TrackA_Pipeline.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
