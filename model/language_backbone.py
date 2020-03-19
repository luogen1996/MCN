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
def seq_to_list(s):
    t_str = s.lower()
    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;', r'\n']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

def qlist_to_vec(max_length, q_list):   # use for what??

    nlp=spacy.load('en_vectors_web_lg')
    glove_matrix = []
    glove_dict = {}
    q_len = len(q_list)
    if q_len > max_length:
        q_len = max_length
    for i in range(max_length):
        if i < q_len:
            w=q_list[i]
            if w not in glove_dict:
                glove_dict[w]=nlp(u'%s'%w).vector
            glove_matrix.append(glove_dict[w])
        else:
            glove_matrix.append(np.zeros(300,dtype=float))
    return glove_matrix, q_len

def get_last_state(x):
    return x[:,-1,:]
def get_mask(x):
    #x (N,T,embed)
    x=K.sum(x,-1, keepdims=True)
    boolean_mask = K.any(K.not_equal(x, 0.),
                         axis=-1, keepdims=True)
    return K.cast(boolean_mask, K.dtype(x))
def exp_(x):
    return K.exp(x)
def norm(x):
    return x / (K.sum(x, axis=-1, keepdims=True) + K.epsilon())
def gelu(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))

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