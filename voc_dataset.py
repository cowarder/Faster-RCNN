import os
import xml.etree.ElementTree as ET
import numpy as np
from torch.utils.data import Dataset
from utils import read_image

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

class VOCBBoxDataset(Dataset):

    def __init__(self, data_dir, split='trainval'):
        id_list_file = os.path.join(
            data_dir, 'ImageSets/Main/{0}.txt'.format(split))

        self.ids = [i.strip() for i in open(id_list_file)]
        self.data_dir = data_dir

        self.label_names = VOC_BBOX_LABEL_NAMES

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        """
        returns the idx-image, its format is (C, H, W), RGB
        :param idx: the idx_th image.
        :return:
        """

        id = self.ids[idx]
        annos = ET.parse(os.path.join(self.data_dir, 'Annotations', id + '.xml'))
        bboxs = []
        labels = []

        for obj in annos.findall('object'):
            pos = obj.find('bndbox')
            bboxs.append([int(pos.find(tag).text) for tag in ['ymin', 'xmin', 'ymax', 'xmax']])
            name = obj.find('name').text.lower().strip()
            labels.append(VOC_BBOX_LABEL_NAMES.index(name))

        bboxs = np.asarray(bboxs).astype(np.float32)
        labels = np.stack(labels).astype(np.int32)

        img_file = os.path.join(self.data_dir, 'JPEGImages', idx + '.jpg')
        img = read_image(img_file)

        return img, bboxs, labels