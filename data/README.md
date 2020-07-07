# Dataset Preparation

Note that the following steps are required only if you want to prepare the annotations from parent repository.

Here I have outlined the steps to prepare the datasets of RefCOCO, RefCOCO+ and RefCOCOg and pretrained weights of visual backbone:

The project directory is $ROOT

## Annotations
Current directory is located at   $DATA_PREP=\$ROOT/data to generate annotations.

1. Download the cleaned data and extract them into "data" folder
   - 1) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
   - 2) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip 
   - 3) http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip 
1. Prepare images from COCO train2014, and unzip the annotations. At this point the directory should look like:
```
$DATA_PREP
|-- refcoco
    |-- instances.json
    |-- refs(google).p
    |-- refs(unc).p
|-- refcoco+
    |-- instances.json
    |-- refs(unc).p
|-- refcocog
    |-- instances.json
    |-- refs(google).p
    |-- refs(umd).p
|-- images
	|-- train2014
```
3. After that, you should run $DATA_PREP/data_process.py to generate the annotations. For example, to generate the annotations for RefCOCO,  you can run the code:

```
cd $ROOT/data_prep
python data_process.py --data_root $DATA_PREP --output_dir $DATA_PREP --dataset refcoco --split unc --generate_mask
```
4. At this point the directory  $DATA_PREP should look like: 
```
   $ROOT/data
   |-- refcoco
       |-- instances.json
       |-- refs(google).p
       |-- refs(unc).p
   |-- refcoco+
       |-- instances.json
       |-- refs(unc).p
   |-- refcocog
       |-- instances.json
       |-- refs(google).p
       |-- refs(umd).p
   |-- anns
       |-- refcoco
       |-- refcoco+
       |-- refcocog
   |-- masks
       |-- refcoco
       |-- refcoco+
       |-- refcocog
   |-- images
       |-- train2014
   |-- weights
       |-- pretrained_weights
```
## Pretrained Weights of Visual Backbone

We provide the pretrained weights of vgg and darknet backbone, which are trained by [darknet-yolov3](https://github.com/AlexeyAB/darknet). We remove all images appearing in the *val+test* splits of RefCOCO, RefCOCO+ and RefCOCOg. You should download the following weights into $DATA_PREP/weights.

| Pretrained Weights of Backbone                 |                        keras version                         |                       darknet version                        |
| ---------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| DarkNet53-yolov3                               | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGiHOaFzCO5FL5paqn?e=qwGxvZ), [Baidu Cloud](https://pan.baidu.com/s/1LpC1W8jR1XhvqvGeCMyNYg) (password:xvue) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGiHRJTf6TIdUMQsmP?e=0xso8d), [Baidu Cloud](https://pan.baidu.com/s/1nUJqqgTFY5lBmTkpNVJRiA) (password:az2j) |
| Vgg16-yolov3                                  | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkMPh_zKPyocbMSG?e=ehYFNJ), [Baidu Cloud](https://pan.baidu.com/s/118wzf1ncC5W31jgxZML6yg)(password:wdb8) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkTyADdOCMXJs8lH?e=n103v6), [Baidu Cloud](https://pan.baidu.com/s/13WjFBO3ZVOmZOR8DDfGXiw)(password:4tml) |

The weights of darknet pretrained on the whole *train* set of MS-COCO are also released as following, which will boost the  performance of REC around 2~3% in practice.   Notbly, we release it for training on other datasets like referit, but do not advise to use it for RefCOCO, RefCOCO+ and RefCOCOg.  **(it is incorrect to use the COCO pre-trained backbone on RefCOCO, RefCOCO+, and RefCOCOg datasets)** .  

| Pretrained Weights of Backbone    |                        keras version                         |                       darknet version                       |
| --------------------------------- | :----------------------------------------------------------: | :---------------------------------------------------------: |
| DarkNet-yolov3 (COCO pre-trained) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkfaOxziFJr01WNy?e=kwl3h1), [Baidu Cloud](https://pan.baidu.com/s/1rt2ml0oOLSAbHahVksDg1w) (password:l9q6) | [Link](https://pjreddie.com/media/files/yolov3.weights) |

Tips: In our  practice, different checkpoints  partly  varies the performence of model. if you require other checkpoints, please contact [us](luogen@stu.xmu.edu.cn).
