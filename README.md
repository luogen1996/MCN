# Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation

[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/luogen1996/MCN/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/keras-%237732a8)

[《Multi-task Collaborative Network for Joint Referring Expression Comprehension and Segmentation》](https://arxiv.org/abs/2003.08813)

by Gen Luo, Yiyi Zhou, Xiaoshuai Sun, Liujuan Cao, Chenglin Wu, Cheng Deng and Rongrong Ji.

IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020, Oral

## Introduction

This repository is keras implementation of MCN.  The principle of MCN is a multimodal and multitask collaborative learning framework. In MCN, RES can help REC to achieve better language-vision alignment, while REC can help RES to better locate the referent. In addition, we address a key challenge in this multi-task setup, i.e., the prediction conflict, with two innovative designs namely, Consistency Energy Maximization (CEM) and Adaptive Soft Non-Located Suppression (ASNLS).  The network structure is illustrated as following:

<p align="center">
  <img src="https://github.com/luogen1996/MCN/blob/master/fig1.png" width="90%"/>
</p>

## Citation

    @inproceedings{luo2020multi,
      title={Multi-task Collaborative  Network for Joint  Referring Expression Comprehension and Segmentation},
      author={Luo, Gen and Zhou, Yiyi and Sun, Xiaoshuai and Cao, Liujuan and Wu, Chenglin and
      Deng, Cheng and Ji Rongrong},
      booktitle={CVPR},
      year={2020}
    }
## Prerequisites

- Python 3.6

- tensorflow-1.9.0 for cuda 9 or tensorflow-1.14.0 for cuda10

- keras-2.2.4

- spacy (you should download the glove embeddings by running `spacy download en_vectors_web_lg` )

- Others (progressbar2, opencv, etc. see [requirement.txt](https://github.com/luogen1996/MCN/blob/master/requirement.txt))

## Data preparation

-  Follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/MCN/blob/master/data/README.md) to generate training data and testing data of RefCOCO, RefCOCO+ and RefCOCOg.

-  Download the pretrained weights of backbone (vgg and darknet). We provide pretrained weights of keras  version for this repo and another  darknet version for  facilitating  the researches based on pytorch or other frameworks.  All pretrained backbones are trained  on COCO 2014 *train+val*  set while removing the images appeared in the *val+test* sets of RefCOCO, RefCOCO+ and RefCOCOg (nearly 6500 images).  Please follow the instructions of  [DATA_PRE_README.md](https://github.com/luogen1996/MCN/blob/master/data/README.md) to download them.

## Training 

1. Preparing your settings. To train a model, you should  modify ``./config/config.json``  to adjust the settings  you want. The default settings are used for RefCOCO, which are easy to achieve 80.0 and 62.0  accuracy for REC and RES respectively on the *val* set. We also provide  example configs for reproducing our results on [RefCOCO+](https://github.com/luogen1996/MCN/blob/master/config/config.Example_Refcoco%2B.json) and [RefCOCOg](https://github.com/luogen1996/MCN/blob/master/config/config.Example_Refcocog.json).
2. Training the model. run ` train.py`  under the main folder to start training:
```
python train.py
```
3. Testing the model.  You should modify  the setting json to check the model path ``evaluate_model`` and dataset ``evaluate_set`` using for evaluation.  Then, you can run ` test.py`  by
```
python test.py
```
​	After finishing the evaluation,  a result file will be generated  in ``./result`` folder.

4. Training log.  Logs are stored in ``./log`` directory, which records the detailed training curve and accuracy per epoch. If you want to log the visualizations, please  set  ``log_images`` to ``1`` in ``config.json``.   By using tensorboard you can see the training details like below：
  <p align="center">
  <img src="https://github.com/luogen1996/MCN/blob/master/fig2.png" width="90%"/>
  </p>
  
**Notably, running this codes can achieve  better performance than the results of our paper. (Nearly 1~4\% improvements on each dataset.) This is because we have done many optimizations  lately,  such as carefully adjusting some training hyperparameters, optimizing the training codes and  selecting a better checkpoint of pre-trained  backbone, etc. In addition, it's fine that the losses do not decline when you use vgg16 as backbone. It may be a display problem and doesn't influence the performance.**

## Pre-trained Models and Logs

Following the steps of Data preparation and Training, you can reproduce and  get   better results in our paper. We provide the pre-trained models and training logs  for RefCOCO, RefCOCO+, RefCOCOg and Referit. 

1) RefCOCO:  [Darknet (312M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiRL_WITB7kfqX0St?e=JlWTBX), [vgg16(214M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiScJ7zFZNZOXE6VI?e=b1iCYa).
<table>
<tr><th> Detection/Segmentation (Darknet) </th><th> Detection/Segmentation (vgg16)</th></tr>
<tr><td>

| val               | test A            | test B            |
| ----------------- | ----------------- | ----------------- |
| 80.61\%/63.12\% | 83.38\%/65.05\% | 75.51\%/60.99\% |
</td><td>

| val  | test A | test B |
| ---- | ------ | ------ |
| 79.68\%/61.51\% | 81.49\%/63.25\% | 75.30\%/60.46\% |
</td></tr> </table>

2) RefCOCO+:  [Darknet (312M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiROAVl3RuIllJLAC?e=qPMity), [vgg16(214M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiShEnn5tmeI0bM_q?e=x7BOcs).
<table>
<tr><th> Detection/Segmentation (Darknet) </th><th> Detection/Segmentation (vgg16)</th></tr>
<tr><td>

| val               | test A            | test B            |
| ----------------- | ----------------- | ----------------- |
| 69.10\%/53.00\% | 74.17\%/57.00\% | 59.75\%/46.96\% |
</td><td>

| val  | test A | test B |
| ---- | ------ | ------ |
| 64.67\%/49.04\% | 69.25\%/51.94\% | 57.01\%/44.31\% |
</td></tr> </table>

3) RefCOCOg:  [Darknet (312M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiRRafGe8qzDDuLci?e=qUBLBT), [vgg16(214M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiSkOuTZ0g1LIRBVl?e=YitGvV).
<table>
<tr><th> Detection/Segmentation (Darknet) </th><th> Detection/Segmentation (vgg16)</th></tr>
<tr><td>

| val               | test              |
| ----------------- | ----------------- |
| 68.95\% / 50.65\% | 67.88\% / 50.62\% |
</td><td>

| val  | test |
| ---- | ---- |
| 63.50\% / 47.81\% | 63.32\% / 47.94\% |
</td></tr> </table>

4) Referit:  [Darknet (312M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiRUQCWpBx1D5cm8_?e=MDjO2I), [vgg16(214M)](https://1drv.ms/u/s!AmrFUyZ_lDVGiSouSmtCBg5zhlB_?e=1ONzTK).
<table>
<tr><th> Detection/Segmentation (Darknet) </th><th> Detection/Segmentation (vgg16)</th></tr>
<tr><td>

| val               | test              |
| ----------------- | ----------------- |
| 69.29\% / 57.00\% | 67.65\% / 55.42\% |

</td><td>

| val               | test              |
| ----------------- | ----------------- |
| 68.28\% / 56.19\% | 65.49\% / 53.68\% |

</td></tr> </table>

## Acknowledgement

 Thanks for a lot of codes from [keras-yolo3](https://github.com/qqwweee/keras-yolo3) , [keras-retinanet](https://github.com/fizyr/keras-retinanet)  and the framework of  [darknet](https://github.com/AlexeyAB/darknet) using for backbone pretraining.

