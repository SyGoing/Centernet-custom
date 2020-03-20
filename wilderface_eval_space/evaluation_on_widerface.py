#!/usr/bin/ python3
# -*- coding: utf-8 -*-
# @Time    : 2019-10-17
# @Author  : vealocia
# @FileName: evaluation_on_widerface.py


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import math
import os
import sys
import _init_paths
import numpy as np

import cv2
from opts import opts
from detectors.detector_factory import detector_factory

image_ext = ['jpg', 'jpeg', 'png', 'webp']
video_ext = ['mp4', 'mov', 'avi', 'mkv']
time_stats = ['tot', 'load', 'pre', 'net', 'dec', 'post', 'merge']

candidate_size = 640
threshold = 0.1

val_image_root = "./WIDER_val/images"  # path to widerface valuation image root
val_result_txt_save_root = "./widerface_evaluation/"  # result directory

opt = opts().init()
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpus_str
opt.debug = max(opt.debug, 1)
Detector = detector_factory[opt.task]


detector = Detector(opt)

opt.debug=-1
print(opt.debug)

counter = 0
for parent, dir_names, file_names in os.walk(val_image_root):
    for file_name in file_names:
        if not file_name.lower().endswith('jpg'):
            continue
        im = cv2.imread(os.path.join(parent, file_name), cv2.IMREAD_COLOR)
        #im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        ret = detector.run(im)
        results=ret["results"][1]
        ids=np.where(results[:,4]>0.4)

        boxes_info=[results[ids[id],:] for id in range(len(ids))]
        boxes_info=np.array(boxes_info)
        boxes_info.resize([boxes_info.shape[1],boxes_info.shape[2]])

        event_name = parent.split('/')[-1]
        if not os.path.exists(os.path.join(val_result_txt_save_root, event_name)):
            os.makedirs(os.path.join(val_result_txt_save_root, event_name))
        fout = open(os.path.join(val_result_txt_save_root, event_name, file_name.split('.')[0] + '.txt'), 'w')
        fout.write(file_name.split('.')[0] + '\n')


        fout.write(str(boxes_info.shape[0]) + '\n')
        for i in range( boxes_info.shape[0]):
            bbox =  boxes_info[i, :]
            fout.write('%d %d %d %d %.03f' % (math.floor(bbox[0]), math.floor(bbox[1]), math.ceil(bbox[2] - bbox[0]), math.ceil(bbox[3] - bbox[1]), bbox[4]  if bbox[4] <= 1 else 1) + '\n')
        fout.close()
        counter += 1
        print('[%d] %s is processed.' % (counter, file_name))

# note: with score_threshold = 0.11 and hard_nms, MAP of 320-input model on widerface val set is: 0.785/0.695/0.431