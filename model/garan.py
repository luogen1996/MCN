from keras.layers import Conv2D, Add, ZeroPadding2D, ReLU,UpSampling2D,Flatten, Concatenate, MaxPooling2D,Multiply,Input,Lambda,Dense,Dropout,Dot,Reshape,Activation,GlobalAveragePooling2D,AveragePooling2D
import keras.backend as K
import tensorflow as tf
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from model.utils import *
def global_attention_block(F_v,f_q):
    """
    :param F_v: N,w,h,d/k
    :param f_q: N,1,d/k
    :return: F_att: N,w,h,d/k
    """
    fv_shape=K.int_shape(F_v)
    fq_shape=K.int_shape(f_q)

    #reshape
    F_v_=Reshape([fv_shape[1]*fv_shape[2],fv_shape[3]])(F_v) #(N,w*h,d/k)
    #linear project
    F_v_proj_0 = Dense(fv_shape[-1])(F_v_)  #(N,w*h,d/k)
    F_v_proj_1 = Dense(fv_shape[-1])(F_v_)  ##(N,w*h,d/k)
    F_v_proj_2 = Dense(fv_shape[-1])(F_v_)  #(N,w*h,d/k)
    f_q_proj = Dense(fv_shape[-1])(f_q) #(N,1,d/k)
    #transpose
    f_q_proj = Lambda(tf.transpose, arguments={'perm': [0, 2, 1]})(f_q_proj)#(N,d/k,1)

    #scaled dot for collection
    E_col=Dot([-1,1])([F_v_proj_0,f_q_proj]) #(N,w*h,1)
    E_col=Lambda(normalize_by_dim,arguments={'dim':float(fq_shape[-1])})(E_col) #(N,w*h,1)
    A_col = Lambda(tf.transpose, arguments={'perm': [0, 2, 1]})(E_col) #(N,1,w*h)
    A_col = Activation('softmax')(A_col)  # softmax throgh axis=-1 #(N,1,w*h) #f_collect

    # scaled dot for distribution
    E_dis=Dot([-1,1])([F_v_proj_2,f_q_proj]) #(N,w*h,1)
    E_dis=Lambda(normalize_by_dim,arguments={'dim':float(fq_shape[-1])})(E_dis) #(N,w*h,1)
    A_dis= Activation('sigmoid')(E_dis)  #sigmoid for distribution (N,w*h,1)
    #collect
    f_att=Dot([-1,1])([A_col,F_v_proj_1]) #(N,d/h)
    # f_att=Reshape(fq_shape[1:])(f_att)#((N,1,d/h)
    #distribution
    F_att=Dot([-1,1])([A_dis,f_att]) #(N,w*h,1) dot (N,1,d/k)-->(N,w*h,d/k) distribution
   # F_att=Lambda(multipy,arguments={'num':fv_shape[1]*fv_shape[2]})(F_att)
    F_att=Reshape(fv_shape[1:])(F_att)#(N,w,h,d/k)
    #attention map
    E_col=Reshape([fv_shape[1],fv_shape[2],1])(E_col)#(N,w,h,1)
    return F_att,E_col
def global_attentive_reason_unit(F_v,f_q,k=2):

    """
    F_v:N,w,h,d
    f_q:N,q_d
    k: split_num, default=2

    """
    f_q=Lambda(K.expand_dims, arguments={'axis': 1})(f_q)
    #split dim concat batch
    F_v_s=Lambda(split_dim_concat_batch,arguments={'n':k})(F_v)
    f_q_s=Lambda(split_dim_concat_batch,arguments={'n':k})(f_q)

    #attention
    F_att,E=global_attention_block(F_v_s,f_q_s)

    #split batch concat dim
    F_att=Lambda(split_batch_concat_dim,arguments={'n':k})(F_att)
    E=Lambda(split_batch_concat_dim,arguments={'n': k})(E) #(1024)
    #mean through dims
    E=Lambda(K.mean,arguments={'axis':-1,'keepdims':True})(E)
    #residual connect
    F_v_=Add()([F_att,F_v]) #(N,1,1024)
    # normalization
    F_v_ = LeakyReLU(alpha=0.1)(BatchNormalization()(F_v_))

    return F_v_,F_att
