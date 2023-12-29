import numpy as np

k_values = np.array([1, 5, 10, 20, 50, 100, 200, 500]).astype(int)
# Open trained k values
val = np.load('kNN_validation.npy')
# train = np.load('kNN_training.npy')
# test = np.load('kNN_test.npy')

for i in range(8):
    print(f"validation accuracy for k value {k_values[i]} = {val[i] / 10000}")
    
