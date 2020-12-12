# Machine Learning Final Project

#### **[ Models ] & [ Path ] to be downloaded from [Google Drive](https://drive.google.com/drive/folders/1FCbOwFw94OynbKCR_3kjpEV4_5si5VkP?usp=sharing)**
 - Dataset & a pickle dump of the preprocessed data (Please download all the files in this path and place them in the project root path)
    *   ./dataset/
 - Pretrained weights on COCO dataset
    *   ./yolo/data/yolov4_coco_pretrained_weights.pth
 - Weights of our best YOLO model (mAP 63.47%)
    *   ./yolo/data/yolov4_test.pth
 - vgg 16 model (for OpenCV demo) 
    *   ./vgg.pth
    
#### **Transfer learning notes**
 - Before you run the Transfer_learning.ipynb, please make sure to download the dataset folder from the google drive and put it in the project root path
 
#### **Opencv_demo notes** 
 - You can either use the vgg model from the google drive to run the demo, or run the Transfer_learning.ipynb and save a model (Resnet 50 or vgg 16). 


#### **YOLO method notes**
 - ./yolo/
    * Main structure derives from reference project [1]
 - ./yolo/annotation_preprocess.ipynb
    *  Make annotation files for training YOLO models.
 - ./yolo/train.ipynb
    *  Train a YOLO model.
 - ./yolo/predict.ipynb
    *  Detect images or videos (based on yolo.py in the folder).
 - ./yolo/eval.ipynb
    *  Evaluate a YOLO model (generate ground-truth files and detection-results files).
 - ./yolo/yolo.py
    * Use the given YOLO model to make predictions.
    * You may want to change the model weights path in this file to try out different models.
 - ./yolo/data
    * Store train / val / test dataset annotations.
    * Store classes and anchors data.
    * Store font file for detection results labeling.
 - ./yolo/loss_data
    * Store training losses
 - ./yolo/map
    * Code for computing mAP (please run main.py after running eval.ipynb) [2]
 - ./yolo/nets/
    * YOLO V4 implementation [1]
 - ./yolo/utils/
    * Utility code

#### **References**

1. [yolo4-pytorch](https://github.com/bubbliiiing/yolov4-pytorch)
2. [mAP](https://github.com/Cartucho/mAP)



