import tensorflow as tf

input_shape_nhwc = (1, 1024, 34, 4)


filter_shape_hwck = (1, 32, 4, 4)


input_tensor = tf.random.normal(input_shape_nhwc)
filter_tensor = tf.random.normal(filter_shape_hwck)

print(f"輸入張量形狀 (NHWC): {input_tensor.shape}")
print(f"濾波器形狀 (HWCK): {filter_tensor.shape}")
print("-" * 30)

# model = keras.layers.Conv2D(filters=250, kernel_size=(1, 48), input_shape=(1, 296, 39), strides=(1, 2), dilation_rate=(1, 1), padding='same', kernel_initializer=k, use_bias=False)

output_tensor = tf.nn.conv2d(
    input=input_tensor,
    filters=filter_tensor,
    strides=(1, 1),
    padding='VALID'  
)


output_shape = output_tensor.shape

