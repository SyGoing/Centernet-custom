from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.utils.data as data
import numpy as np
import torch
import json
import cv2
import os
from utils.image import flip, color_aug
from utils.image import get_affine_transform, affine_transform
from utils.image import gaussian_radius, draw_umich_gaussian, draw_msra_gaussian
from utils.image import draw_dense_reg
import math

import xml.etree.ElementTree as ET

class CTDetFaceVOCDataset(data.Dataset):
    num_classes = 1
    default_resolution = [640,640]
    mean = np.array([0.5, 0.5, 0.5],
                    dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.5, 0.5, 0.5],
                   dtype=np.float32).reshape(1, 1, 3)

    def __init__(self, opt, split,add_lm=False):
        super(CTDetFaceVOCDataset, self).__init__()
        self.data_dir = os.path.join(opt.data_dir, 'wilderface')
        self.img_dir = os.path.join(self.data_dir, 'JPEGImages')  # images

        self.class_names = ('__background__', 'face')
        self.max_objs = 500
        self._valid_ids = np.arange(1, 2, dtype=np.int32)
        self.has_landmark=add_lm

        self.cat_ids = {v: i for i, v in enumerate(self._valid_ids)}

        self._data_rng = np.random.RandomState(123)
        self._eig_val = np.array([0.2141788, 0.01817699, 0.00341571],
                                 dtype=np.float32)
        self._eig_vec = np.array([
            [-0.58752847, -0.69563484, 0.41340352],
            [-0.5832747, 0.00994535, -0.81221408],
            [-0.56089297, 0.71832671, 0.41158938]
        ], dtype=np.float32)
        self.split = split
        self.opt = opt

        if self.split == 'val':
            image_sets_file = self.data_dir + "/ImageSets/Main/test.txt"
        elif self.split == 'train':
            image_sets_file = self.data_dir + "/ImageSets/Main/trainval.txt"

        self.images = CTDetFaceVOCDataset._read_image_ids(image_sets_file)
        self.num_samples = len(self.images)

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

        print('Loaded {} {} samples'.format(split, self.num_samples))

    def _to_float(self, x):
        return float("{:.2f}".format(x))

    def convert_eval_format(self, all_bboxes):
        detections = [[[] for __ in range(self.num_samples)] \
                      for _ in range(self.num_classes + 1)]
        for i in range(self.num_samples):
            img_id = self.images[i]
            for j in range(1, self.num_classes + 1):
                if isinstance(all_bboxes[img_id][j], np.ndarray):
                    detections[j][i] = all_bboxes[img_id][j].tolist()
                else:
                    detections[j][i] = all_bboxes[img_id][j]
        return detections

    def __len__(self):
        return self.num_samples

    def save_results(self, results, save_dir):
        json.dump(self.convert_eval_format(results),
                  open('{}/results.json'.format(save_dir), 'w'))

    def run_eval(self, results, save_dir):
        # result_json = os.path.join(save_dir, "results.json")
        # detections  = self.convert_eval_format(results)
        # json.dump(detections, open(result_json, "w"))
        self.save_results(results, save_dir)
        os.system('python tools/reval.py ' + \
                  '{}/results.json'.format(save_dir))

    def _read_image_ids(image_sets_file):
        ids = []
        with open(image_sets_file) as f:
            for line in f:
                ids.append(line.rstrip())
        return ids


    def _get_border(self, border, size):
        i = 1
        while size - border // i <= border // i:
            i *= 2
        return border // i

    def _get_annotation(self, image_id):
        annotation_file = self.data_dir + f"/Annotations/{image_id}.xml"
        objects = ET.parse(annotation_file).findall("object")
        items=[]
        for object in objects:
            class_name = object.find('name').text.lower().strip()
            # we're only concerned with clases in our list
            if class_name in self.class_names:
                bbox = object.find('bndbox')

                # VOC dataset format follows Matlab, in which indexes start from 0
                x1 = float(bbox.find('xmin').text) - 1
                y1 = float(bbox.find('ymin').text) - 1
                x2 = float(bbox.find('xmax').text) - 1
                y2 = float(bbox.find('ymax').text) - 1
                box=[x1, y1, x2, y2]
                label=self.class_dict[class_name]

                item=(box,label)
                items.append(item)
        return items


    def __getitem__(self, index):
        img_id = self.images[index]
        img_path=self.data_dir +f"/JPEGImages/{img_id}.jpg"

        annotationFile = self.data_dir + f"/Annotations/{img_id}.xml"

        anns = []
        if os.path.exists(annotationFile):
            anns = self._get_annotation(img_id)

        num_objs = min(len(anns), self.max_objs)
        img = cv2.imread(img_path)

        height, width = img.shape[0], img.shape[1]
        c = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
        if self.opt.keep_res:
            input_h = (height | self.opt.pad) + 1
            input_w = (width | self.opt.pad) + 1
            s = np.array([input_w, input_h], dtype=np.float32)
        else:
            s = max(img.shape[0], img.shape[1]) * 1.0
            input_h, input_w = self.opt.input_h, self.opt.input_w

        flipped = False
        if self.split == 'train':
            if not self.opt.not_rand_crop:
                s = s * np.random.choice(np.arange(0.3, 1.4, 0.1))
                w_border = self._get_border(128, img.shape[1])
                h_border = self._get_border(128, img.shape[0])
                c[0] = np.random.randint(low=w_border, high=img.shape[1] - w_border)
                c[1] = np.random.randint(low=h_border, high=img.shape[0] - h_border)
            else:
                sf = self.opt.scale
                cf = self.opt.shift
                c[0] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                c[1] += s * np.clip(np.random.randn() * cf, -2 * cf, 2 * cf)
                s = s * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)

            if np.random.random() < self.opt.flip:
                flipped = True
                img = img[:, ::-1, :]
                c[0] = width - c[0] - 1

        trans_input = get_affine_transform(
            c, s, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input,
                             (input_w, input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp.astype(np.float32) / 255.)
        if self.split == 'train' and not self.opt.no_color_aug:
            color_aug(self._data_rng, inp, self._eig_val, self._eig_vec)
        inp = (inp - self.mean) *self.std
        inp = inp.transpose(2, 0, 1)

        output_h = input_h // self.opt.down_ratio
        output_w = input_w // self.opt.down_ratio
        num_classes = self.num_classes
        trans_output = get_affine_transform(c, s, 0, [output_w, output_h])

        hm = np.zeros((num_classes, output_h, output_w), dtype=np.float32)
        hm_posweight=np.zeros((num_classes, output_h, output_w), dtype=np.float32)

        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, output_h, output_w), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, num_classes * 2), dtype=np.uint8)

        draw_gaussian = draw_msra_gaussian if self.opt.mse_loss else \
            draw_umich_gaussian

        gt_det = []
        for k in range(num_objs):
            box,label= anns[k]
            bbox=np.array(box,dtype=np.float32)
            cls_id=int(self.cat_ids[label])
            if flipped:
                bbox[[0, 2]] = width - bbox[[2, 0]] - 1
            bbox[:2] = affine_transform(bbox[:2], trans_output)
            bbox[2:] = affine_transform(bbox[2:], trans_output)
            bbox[[0, 2]] = np.clip(bbox[[0, 2]], 0, output_w - 1)
            bbox[[1, 3]] = np.clip(bbox[[1, 3]], 0, output_h - 1)
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                radius = self.opt.hm_gauss if self.opt.mse_loss else radius
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_gaussian(hm[cls_id], ct_int, radius)

                """添加正样本的权重弱监督，给大目标更大的权重，提高目标整体的质置信度"""
                mxface = 300
                miface = 25
                mxline = max(w, h)
                gamma = (mxline - miface) / (mxface - miface) * 10
                gamma = min(max(0, gamma), 10) + 1
                draw_gaussian(hm_posweight[cls_id],ct_int,2,gamma)
                """"""   """"""

                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * output_w + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt.dense_wh:
                    draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)
                gt_det.append([ct[0] - w / 2, ct[1] - h / 2,
                               ct[0] + w / 2, ct[1] + h / 2, 1, cls_id])

        ret = {'input': inp, 'hm': hm, 'hm_weights':hm_posweight,'reg_mask': reg_mask, 'ind': ind, 'wh': wh}
        if self.opt.dense_wh:
            hm_a = hm.max(axis=0, keepdims=True)
            dense_wh_mask = np.concatenate([hm_a, hm_a], axis=0)
            ret.update({'dense_wh': dense_wh, 'dense_wh_mask': dense_wh_mask})
            del ret['wh']
        elif self.opt.cat_spec_wh:
            ret.update({'cat_spec_wh': cat_spec_wh, 'cat_spec_mask': cat_spec_mask})
            del ret['wh']
        if self.opt.reg_offset:
            ret.update({'reg': reg})
        if self.opt.debug > 0 or not self.split == 'train':
            gt_det = np.array(gt_det, dtype=np.float32) if len(gt_det) > 0 else \
                np.zeros((1, 6), dtype=np.float32)
            meta = {'c': c, 's': s, 'gt_det': gt_det, 'img_id': img_id}
            ret['meta'] = meta
        return ret


