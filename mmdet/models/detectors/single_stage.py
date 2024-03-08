# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import torch.nn as nn

from mmdet.core import bbox2result
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector

from torch.nn import functional as F


class SRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 num_channels,
                 num_feats,
                 num_blocks,
                 upscale) -> None:
        super(SRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(num_channels, num_feats, 3, padding=1)
        )

        body = []
        for i in range(num_blocks):
            body.append(nn.Conv2d(num_feats, num_feats, 3, padding=1))
            body.append(nn.ReLU(True))

        self.body = nn.Sequential(*body)

        self.upsample = nn.Sequential(
            nn.Conv2d(num_feats, 3 * (upscale ** 2), 3, padding=1),
            nn.PixelShuffle(upscale)
        )

    def forward(self, x):
        res = self.head(x)
        out = self.body(res)
        out = self.upsample(res + out)
        return out


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution + batch norm + relu"""
    return torch.nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.ReLU()
    )


def resize(input,
           size=None,
           scale_factor=None,
           mode='nearest',
           align_corners=None,
           warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
            if output_h > input_h or output_w > input_w:
                if ((output_h > 1 and output_w > 1 and input_h > 1
                     and input_w > 1) and (output_h - 1) % (input_h - 1)
                        and (output_w - 1) % (input_w - 1)):
                    warnings.warn(
                        f'When align_corners={align_corners}, '
                        'the output would more aligned if '
                        f'input size {(input_h, input_w)} is `x+1` and '
                        f'out size {(output_h, output_w)} is `nx+1`')
    return F.interpolate(input, size, scale_factor, mode, align_corners)


class BasicIRNet(nn.Module):
    """
    Implementation based on methods from the AIM 2022 Challenge on
    Efficient and Accurate Quantized Image Super-Resolution on Mobile NPUs
    https://arxiv.org/pdf/2211.05910.pdf
    """

    def __init__(self,
                 in_plane,
                 upscale) -> None:
        super(BasicIRNet, self).__init__()

        self.head = nn.Sequential(
            nn.Conv2d(in_plane, in_plane, 3, padding=1)
        )

        self.body = nn.ModuleList()
        self.num_upsample = 2 if upscale is 4 else 3
        for i in range(self.num_upsample):
            self.body.append(conv3x3(int(in_plane/2**i), int(in_plane / 2**(i+1))))

        self.end = nn.Conv2d(int(in_plane / 2**(self.num_upsample)), 2, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False)

    def forward(self, x):

        x = self.head(x)
        for i in range(self.num_upsample):
            x = resize(self.body[i](x), scale_factor=(2, 2), mode='bilinear')
        out = self.end(x)
        return out


@DETECTORS.register_module()
class SingleStageDetector(BaseDetector):
    """Base class for single-stage detectors.

    Single-stage detectors directly and densely predict bounding boxes on the
    output features of the backbone+neck.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 add_esg=False,
                 weight_esg=10):
        super(SingleStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)
        if neck is not None:
            self.neck = build_neck(neck)
        bbox_head.update(train_cfg=train_cfg)
        bbox_head.update(test_cfg=test_cfg)
        self.bbox_head = build_head(bbox_head)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        # add by ldy
        if add_esg:
            self.add_esg = True
            # self.branch_ir = self.build_ir(num_channels=256, num_feats=48, upscale=bbox_head['strides'][0], num_blocks=1)

            if bbox_head['type'] == 'RepPointsHead':
                self.stride = bbox_head['point_strides'][0]
                self.branch_esg = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['point_strides'][0])
            elif bbox_head['type'] == 'FCOSHead':
                self.stride = bbox_head['strides'][0]
                self.branch_esg = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['strides'][0])
            elif bbox_head['type'] == 'ATSSHead':
                self.stride = bbox_head['anchor_generator']['strides'][0]
                self.branch_esg = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['anchor_generator']['strides'][0])
            else:
                self.branch_esg = BasicIRNet(in_plane=neck['out_channels'], upscale=bbox_head['strides'][0])

            self.loss_esg = nn.CrossEntropyLoss()
            self.weight_esg = weight_esg

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        # 实现特征图的调制 x[0] = x[0] * weight
        if hasattr(self, 'add_esg'):
            x0 = x[0]
            x_ = x[1:]
            # esg的输出计算损失
            esg_out = self.branch_esg(x0)       #[b,2,h,w]

            # print(self.train())
            if self.training:
                self.loss_esg_ = dict()
                loss_esg_ = self.weight_esg * self.loss_esg(esg_out, self.esg_gt)
                self.loss_esg_['loss_esg'] = loss_esg_
            # 计算权重weight -> pooling
            weight = esg_out[:, 1, :, :]       #[b,1,h,w]
            weight = F.max_pool2d(weight, kernel_size=self.stride).unsqueeze(1)  #[b,1,h/s,w/s]
            # x[0] = x[0] * weight
            x0 = weight * x0
            x = (x0,) + x_
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None):
        """
        Args:
            img (Tensor): Input images of shape (N, C, H, W).
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): A List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                :class:`mmdet.datasets.pipelines.Collect`.
            gt_bboxes (list[Tensor]): Each item are the truth boxes for each
                image in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): Class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): Specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        super(SingleStageDetector, self).forward_train(img, img_metas)
        if hasattr(self, 'add_esg'):
            self.esg_gt = self.build_target_esg(gt_bboxes, img_metas)

        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_ignore)

        if hasattr(self, 'add_esg'):
            losses.update(self.loss_esg_)
        return losses

    # 构建目标mask
    def build_target_esg(self, gt_bboxes, img_metas):
        # build object map
        list_object_maps = []
        for i, gt_bbox in enumerate(gt_bboxes):
            object_map = torch.zeros(img_metas[0]["batch_input_shape"], device=gt_bboxes[0].device)
            for index in range(gt_bbox.shape[0]):
                gt = gt_bbox[index]
                # 宽和高都小于64为条件
                if (int(gt[2])-int(gt[0])) <= 64 and (int(gt[3]) - int(gt[1])) <= 64:
                    object_map[int(gt[1]):(int(gt[3])+1), int(gt[0]):(int(gt[2])+1)] = 1

            list_object_maps.append(object_map[None])

        object_maps = torch.cat(list_object_maps, dim=0)
        return object_maps.long()

    def simple_test(self, img, img_metas, rescale=False):
        """Test function without test-time augmentation.

        Args:
            img (torch.Tensor): Images with shape (N, C, H, W).
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        feat = self.extract_feat(img)
        results_list = self.bbox_head.simple_test(
            feat, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            imgs (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        assert hasattr(self.bbox_head, 'aug_test'), \
            f'{self.bbox_head.__class__.__name__}' \
            ' does not support test-time augmentation'

        feats = self.extract_feats(imgs)
        results_list = self.bbox_head.aug_test(
            feats, img_metas, rescale=rescale)
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in results_list
        ]
        return bbox_results

    def onnx_export(self, img, img_metas, with_nms=True):
        """Test function without test time augmentation.

        Args:
            img (torch.Tensor): input images.
            img_metas (list[dict]): List of image information.

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        # get origin input shape to support onnx dynamic shape

        # get shape as tensor
        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        # get pad input shape to support onnx dynamic shape for exporting
        # `CornerNet` and `CentripetalNet`, which 'pad_shape' is used
        # for inference
        img_metas[0]['pad_shape_for_onnx'] = img_shape

        if len(outs) == 2:
            # add dummy score_factor
            outs = (*outs, None)
        # TODO Can we change to `get_bboxes` when `onnx_export` fail
        det_bboxes, det_labels = self.bbox_head.onnx_export(
            *outs, img_metas, with_nms=with_nms)

        return det_bboxes, det_labels
