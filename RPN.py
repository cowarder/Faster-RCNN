import numpy as np
import torch.nn as nn
import torch
from torch.nn import functional as F
from utils import loc2box, nms


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
        :param score: foreground probability  (R, )
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

        # bounding box regression
        roi = loc2box(anchor, loc)

        # clip roi
        # y_min y_max in  range [0, image_h]
        # x_min x_max in range [0, image_w]
        roi[:, 0::2] = np.clip(roi[:, 0::2], 0, img_size[0])
        roi[:, 1::2] = np.clip(roi[:, 1::2], 0, img_size[1])

        # remove roi which w or h less than self.min_size * scale
        thresh = self.min_size * scale
        h = roi[:, 2] - roi[:, 0]
        w = roi[:, 3] - roi[:, 1]
        keep = np.where((h >= thresh) & (w >= thresh))[0]
        roi = roi[keep, :]
        score = score[keep]

        # sort score of proposals and select proposal_before proposals
        order = score.ravel().argsort()[::-1]
        if proposal_before > 0:
            order = order[:proposal_before]
        roi = roi[order, :]

        roi = nms(roi, thresh)
        if proposal_after > 0:
            roi = roi[:proposal_after]
        return roi


class RPN(nn.Module):

    def __init__(self, int_channels=512, mid_channel=512, ratios=[0.5, 1, 2],
                 scales=[8, 16, 32], feat_size=16, proposal_creator_params={}):
        """
        :param int_channels: RPN network input channel
        :param mid_channel: RPN network intermediate channel
        :param ratios: ratios of width to height
        :param scales: rescale width and height of original image pixels
        :param base_size: scale ratio in feature extraction progress, related to network structure(VGG=16)
        """
        super(RPN, self).__init__()
        self.anchors = self.generate_anchor(feat_size, ratios, scales)
        self.feat_size = feat_size
        n_anchor = self.anchors.shape[0]
        self.conv1 = nn.Conv2d(int_channels, mid_channel, 3, 1, 1)
        self.score = nn.Conv2d(mid_channel, n_anchor*2, 1, 1, 0)
        self.loc = nn.Conv2d(mid_channel, n_anchor*4, 1, 1, 0)
        self.proposal_layer = ProposalCreator(self, **proposal_creator_params)

    def forward(self, x, img_size, scale=1.0):
        """
        RPN forward process

        Args:
        H and W: height and width of extracted feature
        N: batch size
        C: channel
        A: anchors assigned to each pixel, default value is 9

        :param x: feature map estracted by conv layer. Its size is (N, C, H, W)
        :param img_size: tuple(h, w), which contains hwight and width of scaled image
        :param scale: scale ratio
        :return:
        """

        n, _, hh, ww = x.shape
        anchor = shift_anchor(self.anchors, self.feat_size, hh, ww)
        n_anchor = anchor.shape[0] // (hh*ww)
        inter_layer = F.relu(self.conv1(x))

        rpn_loc = self.loc(inter_layer)
        # rpn_loss: (N, W*H, 4)
        rpn_loss = rpn_loc.permute(0, 2, 3, 1).contiguous().view(n, -1, 4)
        rpn_score = self.score(inter_layer)
        rpn_score = rpn_score.permute(0, 2, 3, 1).contiguous()
        rpn_softmax_score = F.softmax(rpn_score.view(n, hh, ww, n_anchor, 2), dim=4)
        rpn_fg_score = rpn_softmax_score[:, :, :, :, 1].contiguous()
        rpn_fg_score = rpn_fg_score.view(n, -1)
        rpn_score = rpn_score.view(n, -1, 2)

        rois = list()
        roi_indices = list()
        for i in range(n):
            roi = self.proposal_layer(
                rpn_loc[i].cpu().data.numpy(),
                rpn_fg_score[i].cpu().data.numpy(),
                anchor, img_size,scale
            )
            batch_index = i * np.ones((len(roi),), dtype=np.int32)
            rois.append(roi)
            roi_indices.append(batch_index)
        rois = np.concatenate(rois, axis=0)
        return rpn_loc, rpn_score, rois, roi_indices, anchor


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


def shift_anchor(anchor, feat_stride, height, width):
    """
    shift anchor, we get K cells, and for every cell we assign A anchors
    :param anchor: anchors generated by func: generate_anchor shape:(9, 4)
    :param feat_stride: feature rescale compared with original image
    :param height: feature map height
    :param width: feature map width
    :return: shifted anchors shape:(k*A, 4)
    """

    shift_y = np.arange(0, feat_stride*height, feat_stride)
    shift_x = np.arange(0, feat_stride*width, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shift = np.stack((shift_y.ravel(), shift_x.ravel(), shift_y.ravel(), shift_x.ravel()), axis=1)
    A = anchor.shape[0]
    K = shift.shape[0]

    anchor = anchor.reshape((1, A, 4)) + shift.reshape((1, K, 4)).transpose((1, 0, 2))
    anchor = anchor.reshape((K * A, 4)).astype(np.float32)
    return anchor


a = RPN()
print(a.generate_anchor())