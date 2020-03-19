"""Miscellaneous utility functions."""

from functools import reduce

from PIL import Image
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
import spacy
import re
import cv2
import time
from keras_bert.tokenizer import Tokenizer
from keras_bert.loader import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert import extract_embeddings
import os
def compose(*funcs):
    """Compose arbitrarily many functions, evaluated left to right.

    Reference: https://mathieularose.com/function-composition-in-python/
    """
    # return lambda x: reduce(lambda v, f: f(v), funcs, x)
    if funcs:
        return reduce(lambda f, g: lambda *a, **kw: g(f(*a, **kw)), funcs)
    else:
        raise ValueError('Composition of empty sequence not supported.')

def letterbox_image(image, size):
    '''resize image with unchanged aspect ratio using padding'''
    iw, ih = image.size
    w, h = size
    scale = min(w/iw, h/ih)
    nw = int(iw*scale)
    nh = int(ih*scale)

    image = image.resize((nw,nh), Image.BICUBIC)
    new_image = Image.new('RGB', size, (128,128,128))
    new_image.paste(image, ((w-nw)//2, (h-nh)//2))
    return new_image

def rand(a=0, b=1):
    return np.random.rand()*(b-a) + a



def get_bert_input(text,vocabs,max_len=512):
    tokenizer = Tokenizer(vocabs, cased=False)
    token=[]
    segment=[]
    token, segment = tokenizer.encode(text, max_len=max_len)
    token.append(token)
    segment.append(segment)
    token.extend([0] * (max_len - len(token)))
    segment.extend([0] * (max_len - len(token)))
    return [token,segment]


def seq_to_list(s):
    '''
    note: 2018.10.3
    use for process sentences
    '''
    t_str = s.lower()
    for i in [r'\?', r'\!', r'\'', r'\"', r'\$', r'\:', r'\@', r'\(', r'\)', r'\,', r'\.', r'\;', r'\n']:
        t_str = re.sub(i, '', t_str)
    for i in [r'\-', r'\/']:
        t_str = re.sub(i, ' ', t_str)
    q_list = re.sub(r'\?', '', t_str.lower()).split(' ')
    q_list = list(filter(lambda x: len(x) > 0, q_list))
    return q_list

def qlist_to_vec(max_length, q_list,embed):
    '''
    note: 2018.10.3
    use for process sentences
    '''
    glove_matrix = []
    glove_dict = {}
    q_len = len(q_list)
    if q_len > max_length:
        q_len = max_length
    for i in range(max_length):
        if i < q_len:
            w=q_list[i]
            if w not in glove_dict:
                glove_dict[w]=embed(u'%s'%w).vector
            glove_matrix.append(glove_dict[w])
        else:
            glove_matrix.append(np.zeros(300,dtype=float))
    return np.array(glove_matrix)
def get_random_data(annotation_line, input_shape,embed,config, train_mode=True, max_boxes=1):
    '''random preprocessing for real-time data augmentation'''
    SEG_DIR=config['seg_gt_path']
    line = annotation_line.split()
    h, w = input_shape
    stop=len(line)
    for i in range(1,len(line)):
        if (line[i]=='~'):
            stop=i
            break
    # print(line[1:stop])
    box_ = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:stop]])
    box=np.zeros([1,5])
    seg_id=box_[0][-1]
    box[0]=box_[0][:-1]
    seg_map=np.load(os.path.join(SEG_DIR,str(seg_id)+'.npy'))
    seg_map_ori=np.array(seg_map).astype(np.float32)
    seg_map=Image.fromarray(seg_map_ori)
    # print(np.shape(box))
    # print(box)
    #####################################
    #sentence process maxlength set to 20  and  random choose one for train
    sentences=[]
    sent_stop=stop+1
    for i in range(stop+1,len(line)):
        if line[i]=='~':
            sentences.append(line[sent_stop:i])
            sent_stop=i+1
    sentences.append(line[sent_stop:len(line)])
    choose_index=np.random.choice(len(sentences))
    sentence=sentences[choose_index]
    # print(qlist)
    if config['use_bert']:
        vocabs = load_vocabulary(config['bert_path']+'/vocab.txt')
        word_vec=get_bert_input(sentence,vocabs,512)
    else:
        word_vec=qlist_to_vec(config['word_len'], sentence,embed)
    # print(word_vec)
    # print(np.shape(word_vec))
    #######################################
    image = Image.open(os.path.join(config['image_path'],line[0]))
    iw, ih = image.size

    scale = min(w / iw, h / ih)
    nw = int(iw * scale)
    nh = int(ih * scale)
    dx = (w - nw) // 2
    dy = (h - nh) // 2
    ori_image = image


    image = image.resize((nw, nh), Image.BICUBIC)
    new_image = Image.new('RGB', (w, h), (128, 128, 128))
    new_image.paste(image, (dx, dy))
    image_data = np.array(new_image) / 255.

    seg_map = seg_map.resize((nw, nh))
    new_map = Image.new('L', (w, h), (0))
    new_map.paste(seg_map, (dx, dy))
    seg_map_data = np.array(new_map)
    seg_map_data = cv2.resize(seg_map_data, (
    seg_map_data.shape[0] // config['seg_out_stride'], seg_map_data.shape[0] // config['seg_out_stride']),interpolation=cv2.INTER_NEAREST)
    seg_map_data = np.reshape(seg_map_data, [np.shape(seg_map_data)[0], np.shape(seg_map_data)[1], 1])
        # print(new_image.size)

    # correct boxes
    box_data = np.zeros((max_boxes, 5))
    if len(box) > 0:
        if len(box) > max_boxes: box = box[:max_boxes]
        box[:, [0, 2]] = box[:, [0, 2]] * scale + dx
        box[:, [1, 3]] = box[:, [1, 3]] * scale + dy
        box_data[:len(box)] = box
    box_data = box_data[:, 0:4]  #delete classfy

    if not train_mode:
        word_vec=[qlist_to_vec(config['word_len'], sent,embed) for sent in sentences]
        return image_data, box_data,word_vec,ori_image,sentences,np.expand_dims(seg_map_ori ,-1)
    return image_data, box_data,word_vec,seg_map_data


def lr_step_decay(lr_start=0.001, steps=[30, 40]):
    def get_lr(epoch):
        decay_rate = len(steps)
        for i, e in enumerate(steps):
            if epoch < e:
                decay_rate = i
                break
        lr = lr_start / (10 ** (decay_rate))
        return lr
    return get_lr
#powre decay
def lr_power_decay(lr_start=2.5e-4,lr_power=0.9,  warm_up_lr=0.,step_all=45*1414,warm_up_step=1000):
    # step_per_epoch=3286
    def warm_up(base_lr, lr, cur_step, end_step):
        return base_lr + (lr - base_lr) * cur_step / end_step
    def get_learningrate(epoch):

        if epoch<warm_up_step:
            lr = warm_up(warm_up_lr, lr_start, epoch, warm_up_step)
        else:
            lr = lr_start * ((1 - float(epoch-warm_up_step) / (step_all-warm_up_step)) ** lr_power)
        return lr
        # print("learning rate is", lr)
    return get_learningrate