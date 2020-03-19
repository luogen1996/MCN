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
## Pretrained Weights

We provide the pretrained weights of vgg and darknet backbone, which are trained by yolov3 detection model. We remove all images appearing in the *val+test* splits of RefCOCO, RefCOCO+ and RefCOCOg. You should download the following weights into $DATA_PREP/weights.

| Pretrained Weights of Backbone                 |                        keras version                         |                       darknet version                        |
| ---------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| DarkNet53-yolov3                               | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkUrWqMATdsWzr6P?e=6EbVlR) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkZjXaSyU3Q4w3HO?e=9wUH62) |
| DarkNet53-yolov3 (finetuned on RefCOCO(train)) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkUrWqMATdsWzr6P?e=6EbVlR) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkZjXaSyU3Q4w3HO?e=9wUH62) |
| Vgg16-yolov3                                   | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkMPh_zKPyocbMSG?e=ehYFNJ) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkTyADdOCMXJs8lH?e=n103v6) |

Notably, the weights `DarkNet53-yolov3 (finetuned on RefCOCO(train))`  are further finetuned on the *train* set of RefCOCO, which will achieve better results for RefCOCO and RefCOCO+.  (it's not suitable for RefCOCOg, whose images from the *val+test* set  also appears in the *train* set).  Meanwhile, the weights of darknet pretrained on the whole *train* set of MS-COCO are also released as following, which will boost the  performance of REC around 3% in practice.  However, we do not advise to use it   **since it is incorrect to use the COCO pre-trained backbone on RefCOCO, RefCOCO+, and RefCOCOg datasets** .  

| Pretrained Weights of Backbone    |                        keras version                         |                       darknet version                       |
| --------------------------------- | :----------------------------------------------------------: | :---------------------------------------------------------: |
| DarkNet-yolov3 (COCO pre-trained) | [OneDrive](https://1drv.ms/u/s!AmrFUyZ_lDVGgkfaOxziFJr01WNy?e=kwl3h1) | [OneDrive](https://pjreddie.com/media/files/yolov3.weights) |

Tips: In our  practice, the best pretrained checkpoint  do not  perform best on REC and RES. if you require other checkpoints, please contact [us](luogen@stu.xmu.edu.cn).