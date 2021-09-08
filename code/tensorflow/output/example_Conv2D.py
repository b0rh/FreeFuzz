import tensorflow as tf
import numpy as np
filters = 4
kernel_size = 2
cls = tf.keras.layers.Conv2D(filters=filters,kernel_size=kernel_size,)
ins_0 = tf.random.uniform([4, 4, 4, 4], dtype=tf.float64)
ins = [ins_0,]
res = cls(*ins)
