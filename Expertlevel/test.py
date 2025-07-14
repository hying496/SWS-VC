import numpy as np
data = np.load("/landmarks_mp_train\\npy\\angry\\Training_3908.npy")
print(data.shape)
landmarks = np.load("/landmarks_mp_train\\npy\\angry\\Training_3908.npy")
print(landmarks[:10])