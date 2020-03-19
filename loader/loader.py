import numpy as np
import random
import warnings
import spacy
import keras
from utils.utils import get_random_data
from model.mcn_model import preprocess_true_boxes
class Generator(keras.utils.Sequence):
    """ Abstract generator class.
    """
    def __init__(
        self,
        data,
        config,
        anchors,
        shuffle=True,
        train_mode=True,
    ):
        self.shuffle=shuffle
        self.data=data
        self.config=config
        self.train_mode=train_mode
        self.batch_size=config['batch_size']
        self.embed=spacy.load(config['word_embed'])
        self.input_shape = (config['input_size'], config['input_size'])
        self.anchors=anchors
        self.on_epoch_end()
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.data)
            self.group()
    def size(self):
        return len(self.data)
    def group(self):
        """ Order the images according to self.order and makes groups of self.batch_size.
        """
        # divide into groups, one group = one batch
        self.groups = [[self.data[x % len(self.data)] for x in range(i, i + self.batch_size)] for i in range(0, len(self.data), self.batch_size)]

    def __len__(self):
        """
        Number of batches for generator.
        """

        return len(self.groups)
    def get_batch(self,datas):
        image_data = []
        box_data = []
        word_data = []
        seg_data = []
        for data in datas:
            image, box, word_vec, seg_map = get_random_data(data, self.input_shape, self.embed, self.config,
                                                            train_mode=self.train_mode)
            word_data.append(word_vec)
            image_data.append(image)
            box_data.append(box)
            seg_data.append(seg_map)
        image_data = np.array(image_data)
        box_data = np.array(box_data)
        word_data = np.array(word_data)
        seg_data = np.array(seg_data)
        det_data = preprocess_true_boxes(box_data, self.input_shape, self.anchors)
        return image_data, word_data, det_data, seg_data
    def __getitem__(self, index):
        """
        Keras sequence method for generating batches.
        """

        group = self.groups[index]
        image_data, word_data, det_data, seg_data=self.get_batch(group)
        # print(np.shape(inputs))
        return [image_data, word_data, *det_data, seg_data],np.zeros(self.batch_size)