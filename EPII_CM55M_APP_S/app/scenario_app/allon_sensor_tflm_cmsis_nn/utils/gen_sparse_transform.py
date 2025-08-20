import tensorflow as tf
import numpy as np
from datasets import load_dataset

np.set_printoptions(threshold=np.inf)

dataset_stream = load_dataset("HuggingFaceM4/COCO", streaming=True)
ds_validation_stream = dataset_stream['validation']

skipped_ds = ds_validation_stream.skip(2)
specific_image_ds = skipped_ds.take(1)

specific_example = next(iter(specific_image_ds))
img = specific_example['image']
# img_ycbcr = img.convert('YCbCr')
# Y_channel_pil, Cb_channel_pil, Cr_channel_pil = img_ycbcr.split()

# Y_channel = np.array(Y_channel_pil)


# kvz_g_dct_8_s16_1D = [64, 64, 64, 64, 64, 64, 64, 64, 89, 75, 50, 18, -18, -50, -75, -89, 83, 36, -36, -83, -83, -36, 36, 83, 75, -18, -89, -50, 50, 89, 18, -75, 64, -64, -64, 64, 64, -64, -64, 64, 50, -89, 18, 75, -75, -18, 89, -50, 36, -83, 83, -36, -36, 83, -83, 36, 18, -50, 75, -89, 89, -75, 50, -18]
# block_size = 8
# M = np.array(kvz_g_dct_8_s16_1D).reshape(block_size, -1).astype(np.int8)
# M_transpose = M.T

# Tmp_result = np.zeros_like(Y_channel, dtype=np.int8)

# for y in range(0, Y_channel.shape[0], block_size):
#     for x in range(0, Y_channel.shape[1], block_size):
#         input_block = Y_channel[y:y+block_size, x:x+block_size].astype(np.int8)
#         tmp_block = np.matmul(input_block, M_transpose)
#         Tmp_result[y:y+block_size, x:x+block_size] = tmp_block


# LHS = np.zeros((Y_channel.shape[0], Y_channel.shape[0])).astype(np.int8)

# for y in range(0, Y_channel.shape[0], block_size):
#     LHS[y:y+block_size, y:y+block_size] = M


# s = ''
# for i in range(LHS.shape[0]):
#     for j in range(LHS.shape[1]):
#         if i == LHS.shape[0] - 1 and j == LHS.shape[1] - 1:
#             s += str(LHS[i, j]) 
#         elif j == LHS.shape[1] - 1:
#             s += str(LHS[i, j]) + ',\n'
#         else:
#             s += str(LHS[i, j]) + ', '

# adj_path = '../../adj_data/DCT/adj_mx.txt'
# f = open(adj_path, "w+")
# f.write(s)
# f.close()

# s = ''
# for i in range(Tmp_result.shape[0]):
#     for j in range(Tmp_result.shape[1]):
#         if i == Tmp_result.shape[0] - 1 and j == Tmp_result.shape[1] - 1:
#             s += str(Tmp_result[i, j]) 
#         elif j == Tmp_result.shape[1] - 1:
#             s += str(Tmp_result[i, j]) + ',\n'
#         else:
#             s += str(Tmp_result[i, j]) + ', '

# adj_path = '../../adj_data/DCT/input.txt'
# f = open(adj_path, "w+")
# f.write(s)
# f.close()