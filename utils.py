from PIL import Image
import numpy as np
from torchvision import transforms
import torch
import skimage
import random


def get_config(cfg_file='config.cfg'):
    """
    Get configuration.

    Args:
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
    Read an image and transform it into ndarray format

    Args:
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

    Args:
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

    Args:
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
    Randomly flip image, x-axis or y-axis, or both

    Args:
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

    Args:
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
    bounding box regression

    Args:
    :param bbox: bounding box  (R, 4) (min_y, min_x, max_y, max_y)
    :param loc: predicted bounding box offset and scale (R, 4) (y, x, h w)
    :return:decoted bounding box (R, 4) (ymin, xmin, ymax, xmax)
    """

    ph = bbox[:, 2] - bbox[:, 0]
    pw = bbox[:, 3] - bbox[:, 1]
    px = bbox[:, 0] + pw / 2.0
    py = bbox[:, 1] + ph / 2.0

    ty = loc[:, 0::4]
    tx = loc[:, 1::4]
    th = loc[:, 2::4]
    tw = loc[:, 3::4]

    center_y = ty * ph[:, np.newaxis] + py[:, np.newaxis]
    center_x = tx * pw[:, np.newaxis] + px[:, np.newaxis]
    h = np.exp(th) * ph[:, np.newaxis]
    w = np.exp(tw) * pw[:, np.newaxis]

    bboxes = np.zeros(loc.shape, dtype=loc.dtype)
    bboxes[:, 0::4] = center_y - h / 2
    bboxes[:, 1::4] = center_x - w / 2
    bboxes[:, 2::4] = center_y + h / 2
    bboxes[:, 3::4] = center_x + w / 2

    return bboxes


def cal_iou(box1, box2):
    """
    Calculate iou value between two boxes

    Args:
    :param box1: array([y_min, x_min, y_max, x_max])
    :param box2: array([y_min, x_min, y_max, x_max])
    :return:
    """
    y_min = min(box1[0], box2[0])
    x_min = min(box1[1], box2[1])
    y_max = max(box1[2], box2[2])
    x_max = max(box1[3], box2[3])

    w1 = box1[3] - box1[1]
    h1 = box1[2] - box1[0]
    w2 = box2[3] - box2[1]
    h2 = box2[2] - box2[0]

    all_w = x_max - x_min
    all_h = y_max - y_min

    inter_w = w1 + w2 - all_w
    inter_h = h1 + h2 - all_h

    if inter_w <= 0 or inter_h <= 0:
        return 0.0

    inter = inter_w * inter_h
    union = w1 * h1 + w2 * h2 - inter
    return 1.0 * inter / union


def nms(bbox, thresh):
    """
    Non-maximum suppression

    Args:
    :param bbox: bounding box shape:(R, 4), they are sorted by score by default (ndarray)
    :param thresh: nms threshold
    :return: bounding box, dtype=ndarray
    """

    keep = np.ones(bbox.shape[0])

    for i in range(0, bbox.shape[0]):
        box = bbox[i]
        if keep[i] == 0:
            continue
        for j in range(i+1, bbox.shape[0]):
            if keep[j] == 0:
                continue
            iou = cal_iou(bbox[i], bbox[j])
            if iou >= thresh:
                keep[j] = 0
    out_boxes=[bbox[index] for index in range(0, len(keep)) if keep[index] == 1]
    return np.array(out_boxes)

