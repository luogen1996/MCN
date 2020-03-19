from __future__ import unicode_literals
from keras.layers import  Lambda
import keras
from keras_bert.layers import Extract, MaskedGlobalMaxPool1D
from keras_bert.loader import load_trained_model_from_checkpoint, load_vocabulary
from keras_bert.tokenizer import Tokenizer
from collections import namedtuple
import keras.backend as K
import os
import numpy as np
__all__ = [
    'POOL_NSP', 'POOL_MAX', 'POOL_AVE',
    'get_checkpoint_paths', 'build_bert',
]


POOL_NSP = 'POOL_NSP'
POOL_MAX = 'POOL_MAX'
POOL_AVE = 'POOL_AVE'


def get_checkpoint_paths(model_path):
    CheckpointPaths = namedtuple('CheckpointPaths', ['config', 'checkpoint', 'vocab'])
    config_path = os.path.join(model_path, 'bert_config.json')
    checkpoint_path = os.path.join(model_path, 'bert_model.ckpt')
    vocab_path = os.path.join(model_path, 'vocab.txt')
    return CheckpointPaths(config_path, checkpoint_path, vocab_path)
def bert_output_sum(o):
    return o[:768]+o[768:768*2]+o[768*2:768*3]+o[768*3:]
def build_bert( model,
                poolings=None,
                output_layer_num=1):
    """Extract embeddings from texts.

    :param model: Path to the checkpoint or built model without MLM and NSP.
    :param texts: Iterable texts.
    :param poolings: Pooling methods. Word embeddings will be returned if it is None.
                     Otherwise concatenated pooled embeddings will be returned.
    :param vocabs: A dict should be provided if model is built.
    :param cased: Whether it is cased for tokenizer.
    :param batch_size: Batch size.
    :param cut_embed: The computed embeddings will be cut based on their input lengths.
    :param output_layer_num: The number of layers whose outputs will be concatenated as a single output.
                             Only available when `model` is a path to checkpoint.
    :return: A list of numpy arrays representing the embeddings.
    """
    if isinstance(model, (str, type(u''))):
        paths = get_checkpoint_paths(model)
        model = load_trained_model_from_checkpoint(
            config_file=paths.config,
            checkpoint_file=paths.checkpoint,
            output_layer_num=output_layer_num,
        )

    outputs = []

    if poolings is not None:
        if isinstance(poolings, (str, type(u''))):
            poolings = [poolings]
        # outputs = []
        for pooling in poolings:
            if pooling == POOL_NSP:
                outputs.append(Extract(index=0, name='Pool-NSP')(model.outputs[0]))
            elif pooling == POOL_MAX:
                outputs.append(MaskedGlobalMaxPool1D(name='Pool-Max')(model.outputs[0]))
            elif pooling == POOL_AVE:
                outputs.append(keras.layers.GlobalAvgPool1D(name='Pool-Ave')(model.outputs[0]))
            else:
                raise ValueError('Unknown pooling method: {}'.format(pooling))
        # print(outputs)
        if len(outputs) == 1:
            outputs = outputs[0]
        else:
            outputs = keras.layers.Concatenate(name='Concatenate')(outputs)
        outputs=Lambda(bert_output_sum)(outputs)
        # model = keras.models.Model(inputs=model.inputs, outputs=outputs)
    return model.inputs,outputs