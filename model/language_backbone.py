import keras as K
import tensorflow as tf
import spacy
from keras.layers.recurrent import GRU,LSTM
from keras.layers import Bidirectional,Input,Dense,BatchNormalization,LeakyReLU,Lambda,Reshape,Add,Masking,Dropout,Multiply
from keras import  Model
import re
import  math
import numpy as np
import keras.backend as K
def gru_rnn_module_a(word_embs,rnn_dim,dropout,return_seq):
    with tf.variable_scope('gru_module'):
        if dropout>0.:
            lstm_cell = Bidirectional(GRU(rnn_dim,return_sequences=return_seq,dropout=dropout),merge_mode="sum")(word_embs)
        else:
            lstm_cell=Bidirectional(GRU(rnn_dim,return_sequences=return_seq),merge_mode="sum")(word_embs)
        return lstm_cell
def gru_rnn_module_s(word_embs,rnn_dim,dropout,return_seq):
    with tf.variable_scope('gru_module'):
        if dropout>0.:
            lstm_cell = GRU(rnn_dim,dropout=dropout,return_sequences=return_seq)(word_embs)
        else:
            lstm_cell=GRU(rnn_dim,return_sequences=return_seq)(word_embs)
        return lstm_cell

def build_nlp_model(q_input,rnn_dim,bidirection,dropout,lang_att):
    if not lang_att:
        q_input = Masking()(q_input)
    if bidirection:
        bi_rnn=gru_rnn_module_a(q_input,rnn_dim,dropout,lang_att)
    else:
        bi_rnn = gru_rnn_module_s(q_input, rnn_dim, dropout,lang_att)
    if lang_att:
        bi_rnn_weights = Dense(K.int_shape(bi_rnn)[-1], activation='tanh')(bi_rnn)
        bi_rnn_weights = Dropout(0.1)(bi_rnn_weights)
        bi_rnn_weights = Lambda(K.softmax, arguments={'axis': 1})(bi_rnn_weights)
        bi_rnn = Multiply()([bi_rnn, bi_rnn_weights])
        bi_rnn = Lambda(K.sum, arguments={'axis': 1})(bi_rnn)
    return bi_rnn
