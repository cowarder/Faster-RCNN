import numpy as np
import torch.nn as nn
import torch
import six


class ProposalCreator:
    """
    Create region proposals
    """

    def __init__(self, feature_model, nms_thresh=0.7, proposals_train_before=12000, proposals_train_after=2000,
                 proposals_test_before=6000, proposals_test_after=300, min_size=16):

        """

        :param feature_model:
        :param nms_thresh: non-maximum algorithm threshold
        :param proposals_train_before: proposal number selected for nms(train progress)
        :param proposals_train_after: proposal number preserved after nms(train progress)
        :param proposals_test_before: proposal number selected for nms(test progress)
        :param proposals_test_after: proposal number preserved after nms(test progress)
        :param min_size: size threshold to filter proposals
        """
        self.model = feature_model
        self.nms_thresh = nms_thresh
        self.proposals_train_before = proposals_train_before
        self.proposals_train_after = proposals_train_after
        self.proposals_test_before = proposals_test_before
        self.proposals_test_after = proposals_test_after
        self.min_size = min_size

    def __call__(self, loc, score, anchor, img_size, scale=1.0):
        """
        Input should be ndarray, R: H*W*9

        Args:
        :param loc: predicted offset and scaling to anchors (R,4)
        :param score: foreground probability
        :param anchor: coordinates of anchors(R, 4)
        :param img_size: (H, W) after scaling
        :param scale: scale ratio
        :return: array of coordinates of proposal boxes, shape:(S, 4), S depends on predicted bounding boxes and number
        of bounding boxes discarded by nms
        """

        if self.model.training:
            proposal_before = self.proposals_train_before
            proposal_after = self.proposals_train_after
        else:
            proposal_before = self.proposals_test_before
            proposal_after = self.proposals_test_after






class RPN(nn.Module):

    def __init__(self, int_channels=512, mid_channel=512, ratios=[0.5, 1, 2],
                 scales=[8, 16, 32], base_size=16):
        """
        :param int_channels: RPN network input channel
        :param mid_channel: RPN network intermediate channel
        :param ratios: ratios of width to height
        :param scales: rescale width and height of original image pixels
        :param base_size: scale ratio in feature extraction progress, related to network structure(VGG=16)
        """
        super(RPN, self).__init__()
        self.anchors = self.generate_anchor(base_size, ratios, scales)
        self.base_size = base_size
        nn.Conv2d

    def generate_anchor(self, base_size=16, ratios=[0.5, 1.0, 2.0], scales=[8, 16, 32]):
        """
        Generate original anchors(9 anchors)
        :param base_size: scale ratio in feature extraction progress, related to network structure(VGG=16)
        :param ratios: ratios of width to height
        :param scales: rescale width and height of original image pixels
        :return: anchors
        """
        y = base_size / 2.0
        x = base_size / 2.0
        anchors = np.zeros(((len(ratios)*len(scales)), 4), dtype=np.float32)

        for i in range(len(ratios)):
            for j in range(len(scales)):
                h = base_size*np.sqrt(ratios[i])*scales[j]
                w = base_size*np.sqrt(1.0 / ratios[i])*scales[j]

                index = i * len(scales) + j
                anchors[index, 0] = y - h / 2
                anchors[index, 1] = x - w / 2
                anchors[index, 2] = y + h / 2
                anchors[index, 3] = x + w / 2

        return anchors


a = RPN()
print(a.generate_anchor())