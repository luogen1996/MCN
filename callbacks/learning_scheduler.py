from __future__ import absolute_import
from __future__ import print_function

import os

import numpy as np
from utils.tensorboard_logging import log_scalar

from keras.callbacks import Callback
import keras.backend as K


class LearningRateScheduler(Callback):
    """Learning rate scheduler.

    # Arguments
        schedule: a function that takes an epoch index as input
            (integer, indexed from 0) and current learning rate
            and returns a new learning rate as output (float).
        verbose: int. 0: quiet, 1: update messages.
    """

    def __init__(self, schedule, tensorboard, init_epoch=0,verbose=0):
        super(LearningRateScheduler, self).__init__()
        self.schedule = schedule
        self.verbose = verbose
        self.step=0
        self.epoch=init_epoch
        self.tensorboard=tensorboard
        self.lr=0.
    def on_epoch_begin(self, epoch, logs=None):
        self.epoch+=1
        self.lr = self.schedule(self.epoch)
        K.set_value(self.model.optimizer.lr, self.lr)
        if self.verbose > 0:
            print('\nEpoch %05d: LearningRateScheduler setting learning '
                  'rate to %.4f' % (self.epoch, self.lr))
    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)

    def on_batch_begin(self, batch, logs=None):
        self.step+=1
        log_scalar(self.tensorboard, 'learning rate', self.lr, self.step)

