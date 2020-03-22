import keras.backend as K
import tensorflow as tf

def expand_and_tile(x,outsize):
    x = K.expand_dims(x, axis=1)
    x = K.expand_dims(x, axis=1)
    x = K.tile(x, [1, outsize, outsize, 1])
    return x
def expand_and_tile_1(x,outchannels):
    x = K.expand_dims(x, axis=-1)
    x = K.tile(x, [1,1,outchannels])
    return x
def normalize_by_dim(x,dim=1024.):
    d=tf.convert_to_tensor(dim)
    return x/K.sqrt(d)
def split_dim_concat_batch(x,n):
    return tf.concat(tf.split(x, n, axis=-1), axis=0)
def split_batch_concat_dim(x,n):
    return tf.concat(tf.split(x, n, axis=0), axis=-1 )
def normalize(x):
    #cof=tf.convert_to_tensor(1.)
    x=(x+1.)/2.
    return K.clip(x,1e-6,1.-1e-6)
def l2_normalize(x):
    return  tf.nn.l2_normalize(x, axis=-1,epsilon=1e-6)
def softmax(x):
    return K.softmax(x-tf.reduce_max(x),-1)
