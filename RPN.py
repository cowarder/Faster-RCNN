import numpy as np
import torch.nn as nn
import torch
import six


class RPN(nn.Module):

    def __init__(self):
        super(RPN, self).__init__()

    def generate_anchor(self, base_size=16, ratios=[0.5, 1.0, 2.0], scales=[128, 256, 512]):
        """
        Generate original anchors(9 anchors).
        :param base_size: scale ratio in feature extraction progress, related to network structure(VGG=16).
        :param ratios: ratios of width to height.
        :param scales: rescale width and height of original image pixels.
        :return: anchors.
        """
        y = base_size / 2.0
        x = base_size / 2.0
        anchors = np.zeros(((len(ratios)*len(scales)), 4), dtype=np.float32)

        for i in range(len(ratios)):
            for j in range(len(scales)):
                ratio = ratios[i]
                scale = scales[j]
                h = base_size*np.sqrt(ratio[i])*scales[j]
                w = base_size*np.sqrt(1.0 / ratio[i])*scales[j]

                index = i * len(scales) + j
                anchors[index, 0] = y - h / 2
                anchors[index, 1] = x - w / 2
                anchors[index, 2] = y + h / 2
                anchors[index, 3] = x + w / 2

        return anchors
