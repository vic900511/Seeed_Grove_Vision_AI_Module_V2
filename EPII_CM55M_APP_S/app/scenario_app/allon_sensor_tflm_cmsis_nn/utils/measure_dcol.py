import numpy as np

weight = []
path = '../adj_data/Sparse_FIR'
LHS_H = 256
LHS_W = 64

with open('{}/adj_mx.txt'.format(path), 'r', encoding='utf-8') as f:
    for line in f:
        parts = [x.strip() for x in line.strip().split(',') if x.strip()]
        ints = [int(x) for x in parts]
        weight.extend(ints)


print(len(weight))
matrix = np.array(weight).reshape(LHS_H, LHS_W)

nnz = 0
nnzchunk = 0
for i in range(0, LHS_H, 4):
    chunk = matrix[i:i+4]
    nnz += np.count_nonzero(chunk)
    for j in range(LHS_W):
        if np.any(chunk[:, j]):
            nnzchunk += 1

print(nnz)
print(nnzchunk)
print(nnz / nnzchunk)

