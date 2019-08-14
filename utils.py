from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import skimage
import random


def get_config(cfg_file='config.cfg'):
    """
    Get configuration.
    :param cfg_file: configuretion file
    :return: configuration dict file
    """

    with open(cfg_file, 'r') as f:
        lines = f.readlines()

    paras = {}
    for line in lines:
        line = line.strip()
        if len(line) == 0 or line[0] == '#':
            continue
        key, value = line.split('=')
        paras[key.strip()] = value.strip()
    return paras


def read_image(img_file, dtype=np.float32):
    """
    Read an image and transform it into ndarray format.

    :param img_file: image file
    :return: Image object
    """

    img = Image.open(img_file).convert('RGB')
    img = np.asarray(img, dtype=dtype)
    # (H, W, C) ->(C, H, W)
    return img.transpose((2, 0, 1))


def noarmalize(img):
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    return norm(torch.from_numpy(img)).numpy()


def rescale_image(img, min_size=600, max_size=1000):
    """
    Rescale image, keep w and h in range [min_size, max_size], while keep aspect ratio
    :param img: ndarray image
    :param min_size: min size
    :param max_size: max size
    :return: rescaled iamge
    """
    C, H, W = img.shape
    scale1 = min_size / min(H, W)
    scale2 = max_size / max(H, W)
    # get the min scale and rescale image by its value
    scale = min(scale1, scale2)
    img = skimage.transform.resize(img, (C, H*scale, W*scale), mode='reflect', anti_aliasing=False)
    return noarmalize(img)


def rescale_box(bbox, org_hw, hw):
    """
    Rescale box as image
    :param bbox: bounding box (n by 4)
    :param org_hw: original image H and W
    :param hw: rescale image H and W
    :return: rescaled box
    """
    org_h, org_w = org_hw
    h, w = hw
    y_scale = h / org_h
    x_scale = w / org_w
    bbox[:, 0] = bbox[:, 0] * y_scale
    bbox[:, 1] = bbox[:, 1] * x_scale
    bbox[:, 2] = bbox[:, 2] * y_scale
    bbox[:, 3] = bbox[:, 3] * x_scale
    return bbox


def random_flip_image(img):
    """
    random flip image, x-axis or y-axis, or both
    :param img: ndarray format image
    :return: image, if x or t axis flipped
    """
    x_flip = random.randint(1, 10000) % 2
    y_flip = random.randint(1, 10000) % 2

    if y_flip:
        img = img[:, ::-1, :]
    if x_flip:
        img = img[:, :, ::-1]

    return img, y_flip, x_flip


def flip_box(bbox, size, y_flip=False, x_flip=False):
    """
    Flip boxes as image flipped
    :param bbox: bounding box
    :param size: image size
    :param y_flip: if y-axis flipped
    :param x_flip: if x-axis flipped
    :return: flipped bounding box
    """
    H, W = size

    if y_flip:
        y_max = H - bbox[:, 0]
        y_min = H - bbox[:, 2]
        bbox[:, 0] = y_min
        bbox[:, 2] = y_max

    if x_flip:
        x_max = W - bbox[:, 1]
        x_min = W - bbox[:, 3]
        bbox[:, 1] = x_min
        bbox[:, 3] = x_max

    return bbox


def loc2box(bbox, loc):
    """
    Decode bounding boxes from bounding box offsets and scales

    :param bbox: bounding box  (R, 4)
    :param loc: predicted bounding box offset and scale (R, 4)
    :return:decoted bounding box (R, 4) (ymin, xmin, ymax, xmax)
    """

    box_h = bbox[]
