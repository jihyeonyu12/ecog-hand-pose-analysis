import nbformat as nbf

nb = nbf.v4.new_notebook()

intro_md = """# ECoG Hand Pose - Initial Analysis
본 노트북은 `unicorn-bi/Hackathon` 에 있는 ECoG Hand Pose 데이터를 분석하기 위한 기본 템플릿입니다.

**데이터 설명 (data_description.pdf 참고):**
- **데이터 크기:** 67 Channels x 507,025 Samples
- **스펙:**
  - CH1: time 샘플
  - CH2~CH61: ECoG 신호 (Right sensorimotor cortex)
  - CH62: 패러다임 정보 (0: Relax, 1: Rock/Fist, 2: Peace/Scissors, 3: Open Hand/Paper)
  - CH63~CH67: Data glove (Thumb, Index, Middle, Ring, Little)
- **샘플링 속도 (Fs):** 1200 Hz
"""

code_load = """import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 데이터 로드
mat_path = 'dataset/ecog-hand-pose/ECoG_Handpose.mat'
data = scipy.io.loadmat(mat_path)
y = data['y']

print(f"Data shape: {y.shape} (Channels x Time)")

# 기본 정보
fs = 1200 # 1200Hz
time_vec = y[0, :]
ecog_data = y[1:61, :]
paradigm = y[61, :]
glove_data = y[62:67, :]

print(f"Total duration: {time_vec[-1]} seconds")
"""

code_plot_glove = """# Data Glove 및 패러다임 정보 시각화 (앞부분 일부 확인)
start_sec = 0
end_sec = 30
start_idx = int(start_sec * fs)
end_idx = int(end_sec * fs)

t = time_vec[start_idx:end_idx]

plt.figure(figsize=(15, 6))

# 패러다임 (어떤 동작을 지시했는지)
plt.plot(t, paradigm[start_idx:end_idx], label="Paradigm (Cue)", color='black', linewidth=2, linestyle='--')

# Data Glove 출력
glove_labels = ['Thumb', 'Index', 'Middle', 'Ring', 'Little']
for i in range(5):
    # 값 스케일이 다르므로 보기 좋게 약간 스케일링하거나 offset을 줄 수 있습니다.
    scaled_glove = glove_data[i, start_idx:end_idx] / np.max(glove_data) * 3 
    plt.plot(t, scaled_glove, label=f"Glove: {glove_labels[i]}", alpha=0.8)

plt.title("Paradigm Cue & Data Glove (0 to 30s)")
plt.xlabel("Time (s)")
plt.ylabel("Signal Value")
plt.yticks([0, 1, 2, 3], ['Relax (0)', 'Fist/Rock (1)', 'Peace (2)', 'Open Hand (3)'])
plt.legend(loc='upper right')
plt.show()
"""

code_plot_ecog = """# 특정 시간의 ECoG 신호 확인 (CH2 ~ CH5)
plt.figure(figsize=(15, 6))

for i in range(4): # CH2 ~ CH5
    plt.plot(t, ecog_data[i, start_idx:end_idx] + i * 200, label=f"CH{i+2}", alpha=0.8)

plt.title("ECoG Signals - Channels 2-5 (0 to 30s)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (Offset added for visualization)")
plt.legend(loc='upper right')
plt.show()
"""

nb['cells'] = [
    nbf.v4.new_markdown_cell(intro_md),
    nbf.v4.new_code_cell(code_load),
    nbf.v4.new_code_cell(code_plot_glove),
    nbf.v4.new_code_cell(code_plot_ecog)
]

with open('ECoG_Analysis.ipynb', 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
