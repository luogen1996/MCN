import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from keras.optimizers import Adam
from model.mcn_model import  yolo_body, yolo_loss
import tensorflow as tf
from utils.parse_config import  config
from loader.loader import Generator
from callbacks.eval import Evaluate
from keras.callbacks import TensorBoard, ModelCheckpoint
from callbacks.learning_scheduler import LearningRateScheduler
from callbacks.common import RedirectModel
from utils.utils import lr_step_decay


np.random.seed(config['seed'])
tf.set_random_seed(config['seed'])

MODELS_PATH = os.path.join(config['log_path'], 'models')
if not os.path.exists(MODELS_PATH):
    os.makedirs(MODELS_PATH)

class Learner(object):
    def __init__(self):
        with open(config['train_set']) as f:
            train_lines = f.readlines()
        print('train len',len(train_lines))
        with open(config['evaluate_set']) as f:
            val_lines = f.readlines()
        self.val_set_num=len(val_lines)
        print('val len', self.val_set_num)
        np.random.shuffle(train_lines)
        self.train_data=train_lines
        # Validation set path
        self.val_data=val_lines
        # Detecter setting
        self.anchors_path = config['anchors_file']
        self.anchors = self.get_anchors(self.anchors_path)
        self.input_shape = (config['input_size'], config['input_size'], 3)# multiple of 32, hw
        self.word_len=config['word_len']
        self.embed_dim = config['embed_dim']
        self.seg_out_stride=config['seg_out_stride']
        # training batch size
        self.start_epoch=config['start_epoch']

        self.n_freeze=185+12
        if config['backbone']=='vgg':
            self.n_freeze = 34 + 12
        self.yolo_model, self.yolo_body = self.create_model(yolo_weights_path=config['pretrained_weights'],freeze_body=config['free_body'])


        #data init
        self.train_generator = Generator(self.train_data, config, self.anchors)

        #call_back_init
        call_backs = []
        logging = TensorBoard(log_dir=config['log_path'])
        call_backs.append(logging)
        ap_evaluate = Evaluate(self.val_data,self.anchors,config, tensorboard=logging)
        call_backs.append(RedirectModel(ap_evaluate,self.yolo_body))
        checkpoint_map = ModelCheckpoint(config['log_path'] + '/models/best_map.h5',
                                         verbose=1,
                                         save_best_only=True,
                                         save_weights_only=True,
                                         monitor="det_acc",
                                         mode='max')
        call_backs.append(checkpoint_map)
        lr_schedue = LearningRateScheduler(lr_step_decay(config['lr'],config['steps']), logging,verbose=1,init_epoch=config['start_epoch'])
        call_backs.append(lr_schedue)
        self.callbacks=call_backs
    def create_model(self, load_pretrained=True, freeze_body=1,
                     yolo_weights_path='/home/luogen/weights/coco/yolo_weights.h5'):
        K.clear_session()  # get a new session
        image_input = Input(shape=(self.input_shape))
        q_input = Input(shape=[self.word_len, self.embed_dim], name='q_input')
        h, w,_ = self.input_shape
        num_anchors = len(self.anchors)
########12.17  change label size to be suitable with scales
        det_gt = [Input(shape=(h // {0: 32, 1: 16, 2: 8}[l], w // {0: 32, 1: 16, 2: 8}[l],  3, 5)) for l
                  in range(1)]
        seg_gt=Input(shape=(h//self.seg_out_stride,w//self.seg_out_stride,1))

        model_body = yolo_body(image_input, q_input, num_anchors,config)  ######    place

        if load_pretrained:
            model_body.load_weights(yolo_weights_path, by_name=True, skip_mismatch=True)
            print('Load weights {}.'.format(yolo_weights_path))
            if freeze_body in [1, 2]:
                # Freeze darknet53 body or freeze all but 3 output layers.
                num = (self.n_freeze, len(model_body.layers) - 3)[freeze_body - 1]
                # print(num)
                for i in range(num): model_body.layers[i].trainable = False
                for i in range(num,len(model_body.layers)): print(model_body.layers[i].name)
                print('Freeze the first {} layers of total {} layers.'.format(num, len(model_body.layers)))

        model_loss = Lambda(yolo_loss, output_shape=(1,), name='yolo_loss',
                            arguments={'anchors': self.anchors,  'ignore_thresh': 0.5,'seg_loss_weight':config['seg_loss_weight']})(
            [*model_body.output, *det_gt,seg_gt])
        model = Model([model_body.input[0], model_body.input[1], *det_gt,seg_gt], model_loss)
        model.summary()

        return model, model_body

    def train(self):

        # Yolo Compile
        self.yolo_model.compile(loss={'yolo_loss': lambda y_true, y_pred: y_pred}, optimizer=Adam(lr=config['lr']))
        if config['workers']>0:
            use_multiprocessing=True
        else:
            use_multiprocessing=False
        self.yolo_model.fit_generator(self.train_generator,
                                      callbacks=self.callbacks,
                                      epochs=config['epoches'],
                                      initial_epoch=config['start_epoch'],
                                      verbose = 1,
                                      workers = config['workers'],
                                      use_multiprocessing = use_multiprocessing,
                                      max_queue_size = config['max_queue_size']
        )
    @staticmethod
    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


if __name__ == "__main__":
    learner=Learner()
    learner.train()
