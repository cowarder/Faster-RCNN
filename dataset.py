import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from utils import get_config, read_image, rescale_image, rescale_box, random_flip_image, flip_box
import os


VOC_BBOX_LABEL_NAMES = (
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor')


class Transform(object):

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_info):
        img, bbox, label = img_info
        _, org_H, org_W = img.shape
        img =  rescale_image(img, self.min_size, self.max_size)
        _, H, W = img.shape
        scale = min(H / org_H, W / org_W)
        bbox = rescale_box(bbox, (org_H, org_H), (H, W))

        img, y_flip, x_flip = random_flip_image(img)
        bbox = flip_box(bbox, (H,W), y_flip, x_flip)
        return img, bbox, label, scale


class Dataset(Dataset):
    def __init__(self, split='trainval'):
        self.opt = get_config('config.cfg')
        self.data_dir = self.opt.voc_data_dir
        id_list_file = os.path.join(
            self.data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [i.strip() for i in open(id_list_file)]
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.trans = Transform(self.opt.min_size, self.opt.max_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        id = self.ids[idx]
        annos = ET.parse(os.path.join(self.data_dir, 'Annotations', id + '.xml'))
        bbox = []
        label = []

        for obj in annos.findall('object'):
            pos = obj.find('bndbox')
            bbox.append([int(pos.find(tag).text) for tag in ['ymin', 'xmin', 'ymax', 'xmax']])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.asarray(bbox).astype(np.float32)
        label = np.asarray(label).astype(np.int32)

        img_file = os.path.join(self.data_dir, 'JPEGImages', idx + '.jpg')
        img = read_image(img_file)
        img, bboxs, labels, scale = self.trans((img, bbox, label))