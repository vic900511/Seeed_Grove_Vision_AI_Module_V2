import os

path = "../adj_data/wav2letter_op5"
os.chdir(path)

csr_count = 0
with open('./csr/csr_data.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    csr_count += count

with open('./csr/csr_idx_idx.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    csr_count += 4 * count

with open('./csr/csr_rowptr.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    csr_count += 4 * count

fourrows_count = 0
with open('./fourrows/nz_val.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    fourrows_count += count

with open('./fourrows/col_idx.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    fourrows_count += 4 * count

with open('./fourrows/start_idx.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    fourrows_count += 4 * count

rosko_count = 0
with open('./Rosko/A_p.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    rosko_count += count

with open('./Rosko/col_ind.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    rosko_count += 4 * count

with open('./Rosko/loc_m.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    rosko_count += 4 * count

with open('./Rosko/nnz.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    rosko_count += 4 * count

with open('./adj_mx.txt', 'r', encoding='utf-8') as f:
    content = f.read()
    numbers_list = content.strip().split(',')
    count = len(numbers_list)
    print("FC: ", count)

print("csr: ", csr_count)
print("Rosko: ", rosko_count)
print("fourrows: ", fourrows_count)




