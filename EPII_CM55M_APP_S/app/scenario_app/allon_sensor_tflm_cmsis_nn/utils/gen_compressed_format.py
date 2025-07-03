import numpy as np
import os
import random
import math
import time

def jaccard_sim(a, b):
    s1, s2 = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    if union == 0:
        return -1
    return intersection / union

def cos_sim(a, b):
    a, b = np.where(a > 0, 1, 0), np.where(b > 0, 1, 0)
    return np.dot(a, b) / math.sqrt(np.count_nonzero(a) * np.count_nonzero(b))


def cos_sim2(a, b):
    a, b = np.where(a > 0, 0, 1), np.where(b > 0, 0, 1)
    return np.dot(a, b) / math.sqrt(np.sum(a) * np.sum(b))


def new_sim(a, b, alpha=1.2):
    a, b = np.where(a > 0, 0, 1), np.where(b > 0, 0, 1)
    s1, s2 = set(np.nonzero(a)[0]), set(np.nonzero(b)[0])
    intersection = len(s1 & s2)
    union = len(s1 | s2)
    return max(0, intersection - alpha * (union-intersection))

def union(a, b):
    return np.where((a+b) != 0, 1, 0)

def reordering_by_sim(adj_mat, simfunc='Jaccard'):
    MAX = 2
    mapping_list = []
    num_group = int(np.ceil(adj_mat.shape[0] / MAX))
    ungrouped_idx = set(range(adj_mat.shape[0]))

    random.seed(1)
    for _ in range(num_group):
        choose_idx = random.sample(ungrouped_idx, 1)[0]
        mapping_list.append(choose_idx)
        ungrouped_idx.remove(choose_idx)
        choose = np.where(adj_mat[choose_idx] != 0, 1, 0)

        if _ == num_group - 1:
            mapping_list += list(ungrouped_idx)
            break

        for _ in range(MAX-1):
            M, M_idx = -1, -1
            for idx in ungrouped_idx:
                if simfunc == 'Jaccard':
                    score = jaccard_sim(choose, adj_mat[idx])
                elif simfunc == 'cosine2':
                    score = cos_sim2(choose, adj_mat[idx])
                elif simfunc == 'new_sim':
                    score = new_sim(choose, adj_mat[idx])
                else:
                    score = cos_sim(choose, adj_mat[idx])
                if score > M:
                    M, M_idx = score, idx

            mapping_list.append(M_idx)
            ungrouped_idx.remove(M_idx)
            choose = union(choose,  adj_mat[M_idx])
    
    
    new_adj_mat = np.zeros(adj_mat.shape, dtype=adj_mat.dtype)
    index_to = {k: v for v, k in enumerate(mapping_list)}

    for i in range(adj_mat.shape[0]):
        for j in range(adj_mat.shape[1]):
            if adj_mat[i][j] != 0:
                new_adj_mat[index_to[i]][j] = adj_mat[i][j]

    return new_adj_mat, mapping_list

def rosko_packing(adj_mat, path):
    M, K = adj_mat.shape
    A_p = ''
    loc_m = ''
    col_ind = ''
    nnz = ''

    for k in range(K):
        cols = 0
        for m in range(M):
            if adj_mat[m][k] != 0:
                A_p += str(adj_mat[m][k]) + ', '
                loc_m += str(m) + ', '
                cols += 1
        if cols > 0:
            col_ind += str(k) + ', '
            nnz += str(cols) + ', '

    nnz += '0'

    if not os.path.exists('{}/Rosko'.format(path)):
        try:
            os.makedirs('{}/Rosko'.format(path))
        except Exception as e:
            print(f"創建資料夾時發生錯誤: {e}")
    
    A_p_path = '{}/Rosko/A_p.txt'.format(path)
    f = open(A_p_path, "w+")
    f.write(A_p)
    f.close()

    loc_m_path = '{}/Rosko/loc_m.txt'.format(path)
    f = open(loc_m_path, "w+")
    f.write(loc_m)
    f.close()

    col_ind_path = '{}/Rosko/col_ind.txt'.format(path)
    f = open(col_ind_path, "w+")
    f.write(col_ind)
    f.close()

    nnz_path = '{}/Rosko/nnz.txt'.format(path)
    f = open(nnz_path, "w+")
    f.write(nnz)
    f.close()

def fourrows_packing(matrix, path):
    mapping = {(1, 1, 1, 1): [], (1, 1, 1, 0): [], (1, 1, 0, 1): [], (1, 0, 1, 1): [], (0, 1, 1, 1): [], (1, 1, 0, 0): [], (1, 0, 1, 0): [], (1, 0, 0, 1): [], (0, 1, 1, 0): [], (0, 1, 0, 1): [], (0, 0, 1, 1): [], (1, 0, 0, 0): [], (0, 1, 0, 0): [], (0, 0, 1, 0): [], (0, 0, 0, 1): []}
    pattern_matrix = np.where(matrix != 0, 1, 0)
    if matrix.shape[0] % 4:
        padding_row = np.zeros((4 - matrix.shape[0] % 4, pattern_matrix.shape[1]), dtype=int)
        pattern_matrix = np.vstack([pattern_matrix, padding_row])
        matrix = np.vstack([matrix, padding_row])

    nzv = ''
    col = ''
    start = ''
    start_idx_cnt = 0

    for i in range(int(matrix.shape[0] / 4)):
        sub = pattern_matrix[4 * i : 4 * (i + 1)]
        for j in range(matrix.shape[1]):
            key = tuple(sub[:, j])
            if key != (0, 0, 0, 0):
                mapping[key].append(j)
            
        nz_val = ''
        col_idx = ''
        start_idx = str(start_idx_cnt) + ', '

        for k, v in mapping.items():
            non_zero_indices = [i for i, val in enumerate(k) if val != 0]
            for idx in v:
                col_idx += str(idx) + ', '
                col_val = matrix[4 * i : 4 * (i + 1), idx]
                for nz in col_val[non_zero_indices]:
                    start_idx_cnt += 1
                    nz_val += str(nz) + ', '

            start_idx += str(start_idx_cnt) + ', '
            v.clear()

        nzv += nz_val + '\n'
        col += col_idx + '\n'
        start += start_idx + '\n'

    if not os.path.exists('{}/fourrows'.format(path)):
        try:
            os.makedirs('{}/fourrows'.format(path))
        except Exception as e:
            print(f"創建資料夾時發生錯誤: {e}")

    data_path = '{}/fourrows/nz_val.txt'.format(path)
    f = open(data_path, "w+")
    f.write(nzv)
    f.close()

    idx_path = '{}/fourrows/col_idx.txt'.format(path)
    f = open(idx_path, "w+")
    f.write(col)
    f.close()

    ptr_path = '{}/fourrows/start_idx.txt'.format(path)
    f = open(ptr_path, "w+")
    f.write(start)
    f.close()

def csr_packing(matrix, path):
    csr_idx = ''
    csr_data = ''
    csr_rowptr = '0,'
    s = ''
    cnt = 0

    for i in range(matrix.shape[0]):
        filter = matrix[i].flatten()
        for j in range(filter.shape[0]):
            if i == matrix.shape[0] - 1 and j == filter.shape[0] - 1:
                s += str(filter[j]) 
            elif j == filter.shape[0] - 1:
                s += str(filter[j]) + ',\n'
            else:
                s += str(filter[j]) + ', '

            if filter[j] != 0:
                cnt += 1
                csr_idx += str(j) + ', '
                csr_data += str(filter[j]) + ', '

        csr_rowptr += str(cnt) + ', '

    # adj_path = 'model/{}/adj_mx.txt'.format(sparsity)
    # f = open(adj_path, "w+")
    # f.write(s)
    # f.close()

    if not os.path.exists('{}/csr'.format(path)):
        try:
            os.makedirs('{}/csr'.format(path))
        except Exception as e:
            print(f"創建資料夾時發生錯誤: {e}")

    csr_idx_path = '{}/csr/csr_idx_idx.txt'.format(path)
    f = open(csr_idx_path, "w+")
    f.write(csr_idx)
    f.close()

    csr_data_path = '{}/csr/csr_data.txt'.format(path)
    f = open(csr_data_path, "w+")
    f.write(csr_data)
    f.close()

    csr_rowptr_path = '{}/csr/csr_rowptr.txt'.format(path)
    f = open(csr_rowptr_path, "w+")
    f.write(csr_rowptr)
    f.close()

def mcsr_packing(matrix, sparsity):
    Mcsr_idx = ''
    Mcsr_data = ''
    Mcsr_rowptr = '0, '
    s = ''
    cnt = 0

    for i in range(0, matrix.shape[0], 2):
        if i == matrix.shape[0] - 1:
            row1filter = matrix[i].flatten()
            for j in range(row1filter.shape[0]):
                if row1filter[j]:
                    cnt += 1
                    Mcsr_idx += str(j) + ', '
                    Mcsr_data += str(row1filter[j]) + ', '

            Mcsr_rowptr += str(cnt)

        else:
            row1filter = matrix[i].flatten()
            row2filter = matrix[i + 1].flatten()

            for j in range(row1filter.shape[0]):
                if row1filter[j] != 0 or row2filter[j] != 0:
                    cnt += 1
                    Mcsr_idx += str(j) + ', '
                    Mcsr_data += str(row1filter[j]) + ', '
                    Mcsr_data += str(row2filter[j]) + ', '
            
            Mcsr_rowptr += str(cnt) + ', '
            
    # Mcsr_idx_path = 'model/{}/MCSR_GR/mcsrindices.txt'.format(sparsity)
    # f = open(Mcsr_idx_path, "w+")
    # f.write(Mcsr_idx)
    # f.close()

    # Mcsr_data_path = 'model/{}/MCSR_GR/mcsrdata.txt'.format(sparsity)
    # f = open(Mcsr_data_path, "w+")
    # f.write(Mcsr_data)
    # f.close()

    # Mcsr_rowptr_path = 'model/{}/MCSR_GR/mcsrindptr.txt'.format(sparsity)
    # f = open(Mcsr_rowptr_path, "w+")
    # f.write(Mcsr_rowptr)
    # f.close()

def gen_random_input(row, col, path):
    rng = np.random.default_rng()
    mat = rng.integers(-128, 128, size=(row, col)).astype(np.int8)

    s = ''
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if i == mat.shape[0] - 1 and j == mat.shape[1] - 1:
                s += str(mat[i, j]) 
            elif j == mat.shape[1] - 1:
                s += str(mat[i, j]) + ',\n'
            else:
                s += str(mat[i, j]) + ', '

    adj_path = '{}/input.txt'.format(path)
    f = open(adj_path, "w+")
    f.write(s)
    f.close()

if __name__ == '__main__':
    weight = []
    path = '../adj_data/profile'
    LHS_H = 12
    LHS_W = 9

    with open('{}/adj_mx.txt'.format(path), 'r', encoding='utf-8') as f:
        for line in f:
            parts = [x.strip() for x in line.strip().split(',') if x.strip()]
            ints = [int(x) for x in parts]
            weight.extend(ints)
    print(len(weight))
    matrix = np.array(weight).reshape(LHS_H, LHS_W)
    
    print("current sparsity: ", end='')
    print(1 - np.count_nonzero(matrix) / (matrix.shape[0] * matrix.shape[1]))
    print("shape: ", matrix.shape)

    # fourrows_packing(matrix, path)
    # csr_packing(matrix, path)
    rosko_packing(matrix, path)
    # gen_random_input(148, 100, path)
