import numpy as np

M = 49
K = 160
sparsity_percentage = 0.8

total_elements_lhs = M * K
num_zeros = int(total_elements_lhs * sparsity_percentage)


lhs_flat = np.random.randint(low=-128, high=128, size=total_elements_lhs, dtype=np.int8)
zero_indices = np.random.choice(total_elements_lhs, num_zeros, replace=False)
lhs_flat[zero_indices] = 0
lhs_flat = lhs_flat.reshape((M, K))

s = ''
for i in range(lhs_flat.shape[0]):
    for j in range(lhs_flat.shape[1]):
        if i == lhs_flat.shape[0] - 1 and j == lhs_flat.shape[1] - 1:
            s += str(lhs_flat[i, j]) 
        elif j == lhs_flat.shape[1] - 1:
            s += str(lhs_flat[i, j]) + ',\n'
        else:
            s += str(lhs_flat[i, j]) + ', '

adj_path = '../../adj_data/SpMM/adj_mx.txt'
f = open(adj_path, "w+")
f.write(s)
f.close()