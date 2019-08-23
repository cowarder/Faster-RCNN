from torchvision.ops import RoIPool
import torch
import torch.nn as nn


class FasterRCNN(nn.Module):
    """
    Faster RCNN

    Faster RCNN includes three parts:
    1.feature extractor: input is image, and output is feature map
    2.region proposal network: input is feature map, and output is ROIs
    3.localization and classification head: input is ROIs, and output is their location and class probability

    the three parts correspond to (extractor, rpn, head) in init function
    """

    def __init__(self, extractor, rpn, head):
        super(FasterRCNN, self).__init__()
        self.extractor = extractor
        self.rpn = rpn
        self.head = head

    @property
    def n_class(self):
        return self.head.n_class

    def forward(self, x, scale=1.0):
        """
        forward pass of Faster RCNN
        :param x: 4-dim image Tensor, (N, C, H, W)
        :param scale: rescale rate of image
        :return: Tensor
        """
        img_size = x.shape[2:]
        x = self.extractor(x)

        rpn_loc, rpn_score, rois, roi_indices, anchor = self.rpn(x, img_size, scale)

        # roi_cls_locs:(R, L*4)  roi_scores:(R, L), L is the number of classes excluding background
        # R is the number of proposals of images in one batch
        roi_cls_locs, roi_scores = self.head(x, rois, roi_indices)

        # shape  (R, L*4)      (R, L)   (R, 4)   (R, 1)
        return roi_cls_locs, roi_scores, rois, roi_indices
