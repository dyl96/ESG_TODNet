# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .single_stage import SingleStageDetector
from torch.nn import functional as F

@DETECTORS.register_module()
class FCOS(SingleStageDetector):
    """Implementation of `FCOS <https://arxiv.org/abs/1904.01355>`_"""

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 add_esg=False,
                 weight_esg=10):
        super(FCOS, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg, add_esg=add_esg, weight_esg=weight_esg)

    # 获取X weight X'
    def get_features(self, img):
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)

        # 实现特征图的调制 x[0] = x[0] * weight
        if hasattr(self, 'add_esg'):
            x0 = x[0]
            x_ = x[1:]
            # esg的输出计算损失
            esg_out = self.branch_esg(x0)  # [b,2,h,w]

            # print(self.train())
            if self.training:
                self.loss_esg_ = dict()
                loss_esg_ = self.weight_esg * self.loss_esg(esg_out, self.esg_gt)
                self.loss_esg_['loss_esg'] = loss_esg_
            # 计算权重weight -> pooling
            weight = esg_out[:, 1, :, :]  # [b,1,h,w]
            weight = F.max_pool2d(weight, kernel_size=self.stride).unsqueeze(1)  # [b,1,h/s,w/s]
            # x[0] = x[0] * weight
            x1 = weight * x0

        return x0, x1, weight