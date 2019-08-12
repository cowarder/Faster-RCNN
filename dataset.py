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
    """
    Trasform an image and bounding boxes, which includes rescale and random-flip.
    """

    def __init__(self, min_size=600, max_size=1000):
        self.min_size = min_size
        self.max_size = max_size

    def __call__(self, img_info):
        img, bbox, label = img_info
        _, org_h, org_w = img.shape
        img = rescale_image(img, self.min_size, self.max_size)
        _, h, w= img.shape
        scale = min(h / org_h, w / org_w)
        bbox = rescale_box(bbox, (org_h, org_h), (h, w))

        img, y_flip, x_flip = random_flip_image(img)
        bbox = flip_box(bbox, (h,w), y_flip, x_flip)
        return img, bbox, label, scale


class Dataset(Dataset):
    def __init__(self, split='trainval'):
        self.opt = get_config('config.cfg')
        self.data_dir = self.opt.voc_data_dir
        id_list_file = os.path.join(
            self.data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        with open(id_list_file) as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.label_names = VOC_BBOX_LABEL_NAMES
        self.trans = Transform(self.opt.min_size, self.opt.max_size)

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        annos = ET.parse(os.path.join(self.data_dir, 'Annotations', image_id + '.xml'))
        bbox = []
        label = []

        for obj in annos.findall('object'):
            pos = obj.find('bndbox')
            bbox.append([int(pos.find(tag).text) for tag in ['ymin', 'xmin', 'ymax', 'xmax']])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.asarray(bbox).astype(np.float32)
        label = np.asarray(label).astype(np.int32)

        img_file = os.path.join(self.data_dir, 'JPEGImages', image_id + '.jpg')
        img = read_image(img_file)
        img, bbox, label, scale = self.trans((img, bbox, label))
        return img, bbox, label, scale


class TestDataset(Dataset):
    def __init__(self, split='test'):
        self.opt = get_config('config.cfg')
        self.data_dir = self.opt.test_data_dir
        id_list_file = os.path.join(
            self.data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        with open(id_list_file) as f:
            self.ids = [line.strip() for line in f.readlines()]
        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        image_id = self.ids[index]
        annos = ET.parse(os.path.join(self.data_dir, 'Annotations', image_id + '.xml'))
        bbox = []
        label = []

        for obj in annos.findall('object'):
            pos = obj.find('bndbox')
            bbox.append([int(pos.find(tag).text) for tag in ['ymin', 'xmin', 'ymax', 'xmax']])
            name = obj.find('name').text.lower().strip()
            label.append(VOC_BBOX_LABEL_NAMES.index(name))

        bbox = np.asarray(bbox).astype(np.float32)
        label = np.asarray(label).astype(np.int32)

        img_file = os.path.join(self.data_dir, 'JPEGImages', image_id + '.jpg')
        img = read_image(img_file)

        _, org_h, org_w = img.shape
        img = rescale_image(img, int(self.opt.min_size), int(self.opt.max_size))
        _, h, w = img.shape
        scale = min(h / org_h, w / org_w)
        bbox = rescale_box(bbox, (org_h, org_h), (h, w))
        return img, bbox, label, (org_h,org_w)


def main():
    # for test
    id_list_file = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/ImageSets/Main/trainval.txt'
    data_dir = 'data/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    with open(id_list_file) as f:
        ids = [line.strip() for line in f.readlines()]

    image_id = ids[9]
    print(image_id)
    annos = ET.parse(os.path.join(data_dir, 'Annotations', image_id + '.xml'))
    bbox = []
    label = []

    for obj in annos.findall('object'):
        pos = obj.find('bndbox')
        bbox.append([int(pos.find(tag).text) for tag in ['ymin', 'xmin', 'ymax', 'xmax']])
        name = obj.find('name').text.lower().strip()
        label.append(VOC_BBOX_LABEL_NAMES.index(name))

    bbox = np.asarray(bbox).astype(np.float32)
    label = np.asarray(label).astype(np.int32)
    img_file = os.path.join(data_dir, 'JPEGImages', image_id + '.jpg')
    img = read_image(img_file)
    trans = Transform(600, 1000)
    img, bboxs, labels, scale = trans((img, bbox, label))
    print(len(img), len(bboxs), labels, scale)


if __name__=='__main__':
    main()