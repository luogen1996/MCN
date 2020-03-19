import os
import numpy as np
import keras.backend as K
from keras.layers import Input, Lambda
from keras.models import Model
from model.mcn_model import  yolo_body, yolo_loss
from utils.parse_config import  config
from callbacks.eval import Evaluate
from callbacks.common import RedirectModel
import shutil

class Evaluator(object):
    def __init__(self):
        with open(config['evaluate_set']) as f:
            val_lines = f.readlines()
        self.val_set_num=len(val_lines)
        print('val len', self.val_set_num)
        # Validation set path
        self.val_data=val_lines
        # Detecter setting
        self.anchors_path = config['anchors_file']
        self.anchors = self.get_anchors(self.anchors_path)
        self.input_shape = (config['input_size'], config['input_size'], 3)# multiple of 32, hw
        self.word_len=config['word_len']
        self.embed_dim = config['embed_dim']
        self.seg_out_stride=config['seg_out_stride']

        self.yolo_model, self.yolo_body = self.create_model(yolo_weights_path=config['evaluate_model'],freeze_body=-1)

        #evaluator init
        self.evaluator = RedirectModel(Evaluate(self.val_data,self.anchors,config, tensorboard=None),self.yolo_body)
        self.evaluator.on_train_begin()

    def create_model(self, load_pretrained=True, freeze_body=1,
                     yolo_weights_path='/home/luogen/weights/coco/yolo_weights.h5'):
        K.clear_session()  # get a new session
        image_input = Input(shape=(self.input_shape))
        q_input = Input(shape=[self.word_len, self.embed_dim], name='q_input')
        h, w,_ = self.input_shape
        num_anchors = len(self.anchors)
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
        return model, model_body

    def eval(self):
        results=dict()
        self.evaluator.on_epoch_end(-1,results)
        det_acc=results['det_acc']
        seg_iou=results['seg_iou']
        seg_prec=results['seg_prec']
        ie_score=results['ie_score']
        #dump the  result to .txt
        if os.path.exists('result/'):
            shutil.rmtree('result/')
        os.mkdir('result/')
        with open('result/result.txt', 'w') as f_w:
            f_w.write('segmentation result:' + '\n')
            f_w.write('iou: %.4f\n'%(seg_iou))
            for item in seg_prec:
                f_w.write('iou@%.2f: %.4f'%(item,seg_prec[item] )+'\n')
            f_w.write('\n')
            f_w.write('detection result:' + '\n')
            f_w.write('Acc@.5: %.4f\n' % (det_acc))
            f_w.write('\n')
            f_w.write('IE score : %.4f' % (ie_score) + '% \n')

    @staticmethod
    def get_anchors(anchors_path):
        """loads the anchors from a file"""
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)


if __name__ == "__main__":
    evaluator=Evaluator()
    evaluator.eval()
