import tensorflow as tf
import numpy as np
from datasets import load_dataset

dataset_stream = load_dataset("benjamin-paine/imagenet-1k-256x256", streaming=True)
ds_validation_stream = dataset_stream['validation']

skipped_ds = ds_validation_stream.skip(2)
specific_image_ds = skipped_ds.take(1)

specific_example = next(iter(specific_image_ds))
img = specific_example['image']
img_ycbcr = img.convert('YCbCr')
Y_channel_pil, Cb_channel_pil, Cr_channel_pil = img_ycbcr.split()

Y_channel_np = np.array(Y_channel_pil)

kvz_g_dct_8_s16_1D = {64, 64, 64, 64, 64, 64, 64, 64, 89, 75, 50, 18, -18, -50, -75, -89, 83, 36, -36, -83, -83, -36, 36, 83, 75, -18, -89, -50, 50, 89, 18, -75, 64, -64, -64, 64, 64, -64, -64, 64, 50, -89, 18, 75, -75, -18, 89, -50, 36, -83, 83, -36, -36, 83, -83, 36, 18, -50, 75, -89, 89, -75, 50, -18}



