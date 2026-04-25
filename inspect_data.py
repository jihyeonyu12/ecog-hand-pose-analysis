import scipy.io as sio
import numpy as np

mat_path = 'dataset/ecog-hand-pose/ECoG_Handpose.mat'
print(f"Loading {mat_path}...")
mat = sio.loadmat(mat_path)

print("\n--- Keys and Shapes ---")
for key, value in mat.items():
    if not key.startswith('__'):
        print(f"\nKey: {key}")
        if hasattr(value, 'shape'):
            print(f"  Shape: {value.shape}")
            print(f"  Type:  {type(value)}")
        if isinstance(value, np.ndarray) and value.size > 0 and value.size < 100:
            print(f"  Value: {value}")
