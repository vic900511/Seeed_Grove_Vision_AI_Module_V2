import numpy as np

W = 28
H = 28

out_W = W // 2
out_H = H // 2
imageNum = 250

D = np.zeros((out_W * out_H, W * H), dtype=np.int8)

r = 0
for oy in range(out_H):
    for ox in range(out_W):
        iy, ix = 2 * oy, 2 * ox
        base = iy * W + ix
        for off in (0, 1, W, W + 1):
            D[r, base + off] = 1
        r += 1
print(D.shape)

s = ''
for i in range(D.shape[0]):
    for j in range(D.shape[1]):
        if i == D.shape[0] - 1 and j == D.shape[1] - 1:
            s += str(D[i, j]) 
        elif j == D.shape[1] - 1:
            s += str(D[i, j]) + ',\n'
        else:
            s += str(D[i, j]) + ', '
adj_path = './adj_mx.txt'
f = open(adj_path, "w+")
f.write(s)
f.close()

img = np.random.randint(-128, 128, size=(W * H, imageNum), dtype=np.int8)
print(img.shape)
s = ''
for i in range(img.shape[0]):
    for j in range(img.shape[1]):
        if i == img.shape[0] - 1 and j == img.shape[1] - 1:
            s += str(img[i, j]) 
        elif j == img.shape[1] - 1:
            s += str(img[i, j]) + ',\n'
        else:
            s += str(img[i, j]) + ', '
adj_path = './input.txt'
f = open(adj_path, "w+")
f.write(s)
f.close()