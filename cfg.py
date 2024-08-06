# -*- coding: utf-8 -*-
'''
@Time          : 2024/07/05 16:27
@Author        : Larry
@File          : Cfg.py
@Noice         :
@Modificattion :
    @Author    :
    @Time      :
    @Detail    :

'''
import os
from easydict import EasyDict


_BASE_DIR = os.path.dirname(os.path.abspath(__file__))

Cfg = EasyDict()

Cfg.UnFreeze_Epoch      = 300	#300 #2001
Cfg.Unfreeze_batch_size = 32	#32	#2

#Cfg.model_path   = 'weight/yolov4_tiny_weights_coco.pth'
Cfg.model_path  = 'weight/yolov4-tiny.pth'
#Cfg.classes_path = 'model_data/coco_classes.txt'
Cfg.classes_path = 'data/obj.names'

Cfg.predict_model_path =  'checkpoints/last_epoch_weights.pth'  #last_epoch_weights.pth 或是 best_epoch_weights.pth


Cfg.anchors_path = 'model_data/yolo_anchors.txt'
Cfg.anchors_mask = [[3, 4, 5], [1, 2, 3]]


Cfg.train_label = os.path.join(_BASE_DIR, 'data', 'train.txt')
Cfg.val_label   = os.path.join(_BASE_DIR, 'data' ,'valid.txt')


Cfg.checkpoints = os.path.join(_BASE_DIR, 'checkpoints')
Cfg.TRAIN_TENSORBOARD_DIR = os.path.join(_BASE_DIR, 'log')

Cfg.keep_checkpoint_max = 10
Cfg.conf_thresh =  0.5 #v4 0.4    #tiny 0.5
Cfg.nms_thresh =  0.3  #v4 0.6    #tiny 0.3
