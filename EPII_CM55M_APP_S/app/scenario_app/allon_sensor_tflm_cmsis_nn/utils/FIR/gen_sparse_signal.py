import numpy as np

sample_cnt = 256
coeff_cnt = 64
filter_cnt = 1000
sparsity = 0.8

nonzero_count = int((1 - sparsity) * sample_cnt)

# 生成稀疏 int8 数组
signal = np.zeros(sample_cnt, dtype=np.int8)
indices = np.random.choice(sample_cnt, size=nonzero_count, replace=False)
values = np.random.randint(1, 128, size=nonzero_count, dtype=np.int8)
signs = np.random.choice([-1, 1], size=nonzero_count)
signal[indices] = (values * signs).astype(np.int8)

cnt = 0
for i in range(sample_cnt):
    if signal[i] == 0:
        cnt += 1
print(cnt / sample_cnt)

Toeplitz = np.zeros((sample_cnt, coeff_cnt), dtype=np.int8)
for i in range(sample_cnt):
    sig = i
    for j in range(coeff_cnt):
        if sig < 0:
            break
        Toeplitz[i, j] = signal[sig]
        sig -= 1


s = ''
for i in range(Toeplitz.shape[0]):
    for j in range(Toeplitz.shape[1]):
        if i == Toeplitz.shape[0] - 1 and j == Toeplitz.shape[1] - 1:
            s += str(Toeplitz[i, j]) 
        elif j == Toeplitz.shape[1] - 1:
            s += str(Toeplitz[i, j]) + ',\n'
        else:
            s += str(Toeplitz[i, j]) + ', '
adj_path = './adj_mx.txt'
f = open(adj_path, "w+")
f.write(s)
f.close()


coeff = np.random.randint(-128, 128, size=(coeff_cnt, filter_cnt), dtype=np.int8)
s = ''
for i in range(coeff.shape[0]):
    for j in range(coeff.shape[1]):
        if i == coeff.shape[0] - 1 and j == coeff.shape[1] - 1:
            s += str(coeff[i, j]) 
        elif j == coeff.shape[1] - 1:
            s += str(coeff[i, j]) + ',\n'
        else:
            s += str(coeff[i, j]) + ', '
adj_path = './input.txt'
f = open(adj_path, "w+")
f.write(s)
f.close()
