import tensorflow as tf

# --- 1. 定義輸入參數 (根據您先前的情境) ---


# 將維度轉換為 TensorFlow 預設的 NHWC 格式
# Batch, Height, Width, Channels
input_shape_nhwc = (1, 1024, 34, 4)

# TensorFlow 的濾波器格式為 (H_k, W_k, C_in, K)
# Kernel_Height, Kernel_Width, Input_Channels, Output_Channels(K)
filter_shape_hwck = (1, 32, 4, 4)

# --- 2. 建立輸入和濾波器的張量 (Tensor) ---

# 我們不需要真實數據，只需建立正確形狀的「佔位」張量即可
# 這會是一個隨機數值組成的張量，但形狀是我們需要的
input_tensor = tf.random.normal(input_shape_nhwc)
filter_tensor = tf.random.normal(filter_shape_hwck)

print(f"輸入張量形狀 (NHWC): {input_tensor.shape}")
print(f"濾波器形狀 (HWCK): {filter_tensor.shape}")
print("-" * 30)

# --- 3. 執行 2D 卷積運算 ---
# model = keras.layers.Conv2D(filters=250, kernel_size=(1, 48), input_shape=(1, 296, 39), strides=(1, 2), dilation_rate=(1, 1), padding='same', kernel_initializer=k, use_bias=False)

# 使用 tf.nn.conv2d 函式
# padding='SAME' 會讓 TensorFlow 自動計算填充，以符合我們之前的維度計算規則
output_tensor = tf.nn.conv2d(
    input=input_tensor,
    filters=filter_tensor,
    strides=(1, 1),
    padding='VALID'  
)

# --- 4. 取得並印出輸出維度 ---

output_shape = output_tensor.shape

print(f"卷積輸出的張量形狀 (NHWC): {output_shape}")
# 卷積輸出的張量形狀 (NHWC): (1, 1, 142, 100)