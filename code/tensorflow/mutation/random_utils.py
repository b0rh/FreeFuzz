import random
import numpy as np
import tensorflow as tf


def random_tensor(shape, dtype=tf.float32):
    minv = 0
    maxv = 1
    value = None
    if isinstance(dtype, str):
        if not "tf." in dtype:
            dtype = "tf." + dtype
        dtype = dtype.replace("_ref", "")
        try:
            dtype = eval(dtype)
        except:
            dtype = tf.float32

    if len(shape) > 0:
        for i in range(len(shape)):
            if shape[i] is None or shape[i] < 0:
                shape[i] = random.randint(1,3)

    if dtype.is_floating or dtype.is_complex or dtype == tf.string or dtype == tf.bool:
        pass
    elif "uint" in dtype.name or ("int" in dtype.name and dtype.name != "int64"):
        try:
            minv = dtype.min if random.random() < 0.3 else 0
            maxv = dtype.max if random.random() < 0.3 else 1
            if maxv == tf.uint64.max:
                maxv = tf.int64.max - 1
        except Exception as e:
            assert (0)
    else:
        try:
            minv = dtype.min
            maxv = dtype.max
        except Exception as e:
            minv, maxv = 0, 1
    return value, minv, maxv

def random_keras_tensor(shape, dtype):
    return random_tensor(shape, dtype)

def random_variable(shape, dtype):
    value, minv, maxv = random_tensor(shape, dtype)
    return None, minv, maxv

def generate_ndarray(shape, dtype):
    if isinstance(shape, list):
        shape = [1 if x is None else x for x in shape]
    if 'int' in dtype:
        return np.random.uniform(low=0.0, high=2.0, size=shape).astype(dtype)
    return np.random.rand(*shape).astype(dtype)