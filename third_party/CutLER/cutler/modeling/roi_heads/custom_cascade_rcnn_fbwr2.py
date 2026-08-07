# Copyright (c) Meta Platforms, Inc. and affiliates.
# Modified by XuDong Wang from https://github.com/facebookresearch/detectron2/blob/main/detectron2/modeling/roi_heads/cascade_rcnn.py

from typing import List
import torch
from torch import nn
from torch.autograd.function import Function
import numpy as np
import math

from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.structures import Boxes, pairwise_iou
from structures import pairwise_iou_max_scores
from detectron2.structures import Instances
from detectron2.utils.events import get_event_storage

from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.box_head import build_box_head
from .fast_rcnn import FastRCNNOutputLayers, fast_rcnn_inference
from .roi_heads import ROI_HEADS_REGISTRY, CustomStandardROIHeads

import torch.nn.functional as F

class _ScaleGradient(Function):
    @staticmethod
    def forward(ctx, input, scale):
        ctx.scale = scale
        return input

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.scale, None


@ROI_HEADS_REGISTRY.register()
class CustomCascadeROIHeads(CustomStandardROIHeads):
    """
    The ROI heads that implement :paper:`Cascade R-CNN`.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_heads: List[nn.Module],
        box_predictors: List[nn.Module],
        proposal_matchers: List[Matcher],
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_pooler (ROIPooler): pooler that extracts region features from given boxes
            box_heads (list[nn.Module]): box head for each cascade stage
            box_predictors (list[nn.Module]): box predictor for each cascade stage
            proposal_matchers (list[Matcher]): matcher with different IoU thresholds to
                match boxes with ground truth for each stage. The first matcher matches
                RPN proposals with ground truth, the other matchers use boxes predicted
                by the previous stage as proposals and match them with ground truth.
        """
        assert "proposal_matcher" not in kwargs, (
            "CustomCascadeROIHeads takes 'proposal_matchers=' for each stage instead "
            "of one 'proposal_matcher='."
        )
        # The first matcher matches RPN proposals with ground truth, done in the base class
        kwargs["proposal_matcher"] = proposal_matchers[0]
        num_stages = self.num_cascade_stages = len(box_heads)
        box_heads = nn.ModuleList(box_heads)
        box_predictors = nn.ModuleList(box_predictors)
        assert len(box_predictors) == num_stages, f"{len(box_predictors)} != {num_stages}!"
        assert len(proposal_matchers) == num_stages, f"{len(proposal_matchers)} != {num_stages}!"
        super().__init__(
            box_in_features=box_in_features,
            box_pooler=box_pooler,
            box_head=box_heads,
            box_predictor=box_predictors,
            **kwargs,
        )
        self.proposal_matchers = proposal_matchers

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret.pop("proposal_matcher")
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features              = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution        = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales            = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio           = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type              = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        cascade_bbox_reg_weights = cfg.MODEL.ROI_BOX_CASCADE_HEAD.BBOX_REG_WEIGHTS
        cascade_ious             = cfg.MODEL.ROI_BOX_CASCADE_HEAD.IOUS
        assert len(cascade_bbox_reg_weights) == len(cascade_ious)
        assert cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,  \
            "CustomCascadeROIHeads only support class-agnostic regression now!"
        assert cascade_ious[0] == cfg.MODEL.ROI_HEADS.IOU_THRESHOLDS[0]
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        pooled_shape = ShapeSpec(
            channels=in_channels, width=pooler_resolution, height=pooler_resolution
        )

        box_heads, box_predictors, proposal_matchers = [], [], []
        for match_iou, bbox_reg_weights in zip(cascade_ious, cascade_bbox_reg_weights):
            box_head = build_box_head(cfg, pooled_shape)
            box_heads.append(box_head)
            box_predictors.append(
                FastRCNNOutputLayers(
                    cfg,
                    box_head.output_shape,
                    box2box_transform=Box2BoxTransform(weights=bbox_reg_weights),
                )
            )
            proposal_matchers.append(Matcher([match_iou], [0, 1], allow_low_quality_matches=False))
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_heads": box_heads,
            "box_predictors": box_predictors,
            "proposal_matchers": proposal_matchers,
        }

    def forward(self, images, features, proposals, targets=None):
        del images
        if self.training:
            proposals = self.label_and_sample_proposals(proposals, targets)

        if self.training:
            # Need targets to box head
            losses = self._forward_box(features, proposals, targets)
            losses.update(self._forward_mask(features, proposals))
            losses.update(self._forward_keypoint(features, proposals))
            return proposals, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    # def _forward_box(self, features, proposals, targets=None):
    #     features_list = [features[f] for f in self.box_in_features]
    #     head_outputs = []
    #     prev_pred_boxes = None
    #     image_sizes = [x.image_size for x in proposals]

    #     for k in range(self.num_cascade_stages):
    #         if k > 0:
    #             proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
    #             if self.training:
    #                 proposals = self._match_and_label_boxes(proposals, k, targets)
    #         predictions = self._run_stage(features_list, proposals, k)
    #         prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
    #         head_outputs.append((self.box_predictor[k], predictions, proposals))

    #     if self.training:
    #         losses = {}
    #         storage = get_event_storage()

    #         for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
    #             with storage.name_scope("stage{}".format(stage)):

    #                 if self.use_droploss:
    #                     try:
    #                         grayscale_batch = features["grayscale"]  # [B, 1, H, W]
    #                     except KeyError:
    #                         raise ValueError("Grayscale tensor missing in features dict. Ensure DatasetMapper includes it.")

    #                     crater_threshold = getattr(self, "crater_threshold", 0.01) # define threshold
    #                     iou_threshold = getattr(self, "droploss_iou_thresh", 0.3)

    #                     weights_all = []

    #                     for i, props in enumerate(proposals):
    #                         gray_tensor = grayscale_batch[i, 0]  # [H, W]
    #                         boxes = props.proposal_boxes.tensor  # [N, 4]
    #                         gt_boxes = props.gt_boxes.tensor     # [M, 4]

    #                         if len(boxes) == 0:
    #                             weights_all.append(torch.tensor([], device=boxes.device))
    #                             continue

    #                         if len(gt_boxes) == 0:
    #                             # No GT: keep all proposals to avoid dropping everything
    #                             weights_all.append(torch.ones(len(boxes), device=boxes.device))
    #                             continue

    #                         # Compute crater score for each box
    #                         crater_scores = torch.tensor(
    #                             [crater_score_from_tensor(gray_tensor, box.tolist(), crater_threshold)
    #                             for box in boxes],
    #                             dtype=torch.float32,
    #                             device=boxes.device
    #                         )

    #                         # Compute max IoU to GT for each box
    #                         iou_matrix = pairwise_iou(Boxes(boxes), Boxes(gt_boxes))  # [N, M]
    #                         iou_max = iou_matrix.max(dim=1).values  # [N]

    #                         # Drop if crater_score is high AND IoU is low
    #                         drop_mask = (crater_scores >= crater_threshold) & (iou_max <= iou_threshold)
    #                         weights = (~drop_mask).float()  # keep = 1.0, drop = 0.0

    #                         weights_all.append(weights)

    #                     weights = torch.cat(weights_all).detach()
    #                     stage_losses = predictor.losses(predictions, proposals, weights=weights)
    #                 else:
    #                     stage_losses = predictor.losses(predictions, proposals)

    #             losses.update({k + "_stage{}".format(stage): v for k, v in stage_losses.items()})

    #         return losses

    #     else:
    #         # Inference path
    #         scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
    #         scores = [
    #             sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
    #             for scores_per_image in zip(*scores_per_stage)
    #         ]
    #         predictor, predictions, proposals = head_outputs[-1]
    #         boxes = predictor.predict_boxes(predictions, proposals)
    #         pred_instances, _ = fast_rcnn_inference(
    #             boxes,
    #             scores,
    #             image_sizes,
    #             predictor.test_score_thresh,
    #             predictor.test_nms_thresh,
    #             predictor.test_topk_per_image,
    #         )
    #         return pred_instances

    def _forward_box(self, features, proposals, targets=None):
        features_list = [features[f] for f in self.box_in_features]
        head_outputs = []
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(features_list, proposals, k)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(predictions, proposals)
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            storage = get_event_storage()

            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                with storage.name_scope(f"stage{stage}"):

                    if self.use_droploss:
                        crater_threshold = getattr(self, "crater_threshold", 0.03) # 0.03 ≃ 2000/(255*255)
                        iou_threshold = getattr(self, "droploss_iou_thresh", 0.3)

                        try:
                            grayscale_batch = features["grayscale"]  # [B, 1, H, W]
                        except KeyError:
                            raise ValueError("Grayscale tensor missing in features dict.")

                        all_boxes = []
                        img_indices = []
                        weights_all = []

                        for i, props in enumerate(proposals):
                            boxes = props.proposal_boxes.tensor  # [N, 4]
                            if len(boxes) == 0:
                                continue
                            all_boxes.append(boxes)
                            img_indices.append(torch.full((len(boxes),), i, dtype=torch.long, device=boxes.device))

                        if not all_boxes:
                            weights = torch.tensor([], device=grayscale_batch.device)
                            stage_losses = predictor.losses(predictions, proposals, weights=weights)
                            losses.update({k + f"_stage{stage}": v for k, v in stage_losses.items()})
                            return losses

                        all_boxes = torch.cat(all_boxes, dim=0)
                        img_indices = torch.cat(img_indices, dim=0)

                        crater_flags = crater_score_from_tensor_fast(
                            grayscale_batch, all_boxes, img_indices, threshold=crater_threshold
                        )

                        start = 0
                        for i, props in enumerate(proposals):
                            boxes = props.proposal_boxes.tensor
                            n = len(boxes)
                            if n == 0:
                                weights_all.append(torch.tensor([], device=boxes.device))
                                continue

                            gt_boxes = props.gt_boxes.tensor
                            crater_mask = crater_flags[start:start+n]

                            if len(gt_boxes) == 0:
                                weights_all.append(torch.ones(n, device=boxes.device))
                            else:
                                iou_matrix = pairwise_iou(Boxes(boxes), Boxes(gt_boxes))
                                iou_max = iou_matrix.max(dim=1).values
                                drop_mask = (crater_mask >= 1.0) & (iou_max <= iou_threshold)
                                weights_all.append((~drop_mask).float())

                            start += n

                        weights = torch.cat(weights_all).detach()
                        stage_losses = predictor.losses(predictions, proposals, weights=weights)
                    else:
                        stage_losses = predictor.losses(predictions, proposals)

                    losses.update({k + f"_stage{stage}": v for k, v in stage_losses.items()})

            return losses

        else:
            scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
            scores = [
                sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
                for scores_per_image in zip(*scores_per_stage)
            ]
            predictor, predictions, proposals = head_outputs[-1]
            boxes = predictor.predict_boxes(predictions, proposals)
            pred_instances, _ = fast_rcnn_inference(
                boxes,
                scores,
                image_sizes,
                predictor.test_score_thresh,
                predictor.test_nms_thresh,
                predictor.test_topk_per_image,
            )
            return pred_instances

    @torch.no_grad()
    def _match_and_label_boxes(self, proposals, stage, targets):
        """
        Match proposals with groundtruth using the matcher at the given stage.
        Label the proposals as foreground or background based on the match.

        Args:
            proposals (list[Instances]): One Instances for each image, with
                the field "proposal_boxes".
            stage (int): the current stage
            targets (list[Instances]): the ground truth instances

        Returns:
            list[Instances]: the same proposals, but with fields "gt_classes" and "gt_boxes"
        """
        num_fg_samples, num_bg_samples = [], []
        for proposals_per_image, targets_per_image in zip(proposals, targets):
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            # proposal_labels are 0 or 1
            matched_idxs, proposal_labels = self.proposal_matchers[stage](match_quality_matrix)
            if len(targets_per_image) > 0:
                gt_classes = targets_per_image.gt_classes[matched_idxs]
                # Label unmatched proposals (0 label from matcher) as background (label=num_classes)
                gt_classes[proposal_labels == 0] = self.num_classes
                gt_boxes = targets_per_image.gt_boxes[matched_idxs]
            else:
                gt_classes = torch.zeros_like(matched_idxs) + self.num_classes
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(proposals_per_image), 4))
                )
            proposals_per_image.gt_classes = gt_classes
            proposals_per_image.gt_boxes = gt_boxes

            num_fg_samples.append((proposal_labels == 1).sum().item())
            num_bg_samples.append(proposal_labels.numel() - num_fg_samples[-1])

        # Log the number of fg/bg samples in each stage
        storage = get_event_storage()
        storage.put_scalar(
            "stage{}/roi_head/num_fg_samples".format(stage),
            sum(num_fg_samples) / len(num_fg_samples),
        )
        storage.put_scalar(
            "stage{}/roi_head/num_bg_samples".format(stage),
            sum(num_bg_samples) / len(num_bg_samples),
        )
        return proposals

    def _run_stage(self, features, proposals, stage):
        """
        Args:
            features (list[Tensor]): #lvl input features to ROIHeads
            proposals (list[Instances]): #image Instances, with the field "proposal_boxes"
            stage (int): the current stage

        Returns:
            Same output as `FastRCNNOutputLayers.forward()`.
        """
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        # The original implementation averages the losses among heads,
        # but scale up the parameter gradients of the heads.
        # This is equivalent to adding the losses among heads,
        # but scale down the gradients on features.
        if self.training:
            box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](box_features)

    def _create_proposals_from_boxes(self, boxes, image_sizes):
        """
        Args:
            boxes (list[Tensor]): per-image predicted boxes, each of shape Ri x 4
            image_sizes (list[tuple]): list of image shapes in (h, w)

        Returns:
            list[Instances]: per-image proposals with the given boxes.
        """
        # Just like RPN, the proposals should not have gradients
        boxes = [Boxes(b.detach()) for b in boxes]
        proposals = []
        for boxes_per_image, image_size in zip(boxes, image_sizes):
            boxes_per_image.clip(image_size)
            if self.training:
                # do not filter empty boxes at inference time,
                # because the scores from each stage need to be aligned and added later
                boxes_per_image = boxes_per_image[boxes_per_image.nonempty()]
            prop = Instances(image_size)
            prop.proposal_boxes = boxes_per_image
            proposals.append(prop)
        return proposals

# Funciona - Não apagar
# from torchvision.ops import roi_align

# def crater_score_from_tensor_fast(gray_batch, boxes, image_indices, threshold=0.03):
#     """
#     Efficient crater score calculation using torch operations (on GPU).
#     gray_batch: Tensor [B, 1, H, W], values in [0, 1]
#     boxes: Tensor [N, 4] in (x1, y1, x2, y2)
#     image_indices: LongTensor [N], each value in 0..B-1
#     threshold: float
#     Returns: FloatTensor [N] with 1.0 if crater_score > threshold else 0.0
#     """
#     device = gray_batch.device
#     boxes = boxes.clone()

#     # Convert to [0, 1] relative coordinates
#     B, _, H, W = gray_batch.shape
#     boxes[:, [0, 2]] /= W  # x1, x2
#     boxes[:, [1, 3]] /= H  # y1, y2

#     # Create roi_align-style format: [batch_index, x1, y1, x2, y2]
#     roi_boxes = torch.cat([
#         image_indices.unsqueeze(1).float(),
#         boxes
#     ], dim=1)  # [N, 5]

#     # Use ROIAlign-style crop and resize
#     gray_batch = gray_batch.to(boxes.device)  # Ensure it's on CUDA
#     crops = roi_align(gray_batch, roi_boxes, output_size=(32, 32), aligned=True)  # [N, 1, 32, 32]

#     # Compute gradients (Sobel filter approximation)
#     sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=torch.float32, device=device).view(1,1,3,3)
#     sobel_y = torch.tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=torch.float32, device=device).view(1,1,3,3)

#     # Ensure Sobel kernels are on the same device and dtype
#     sobel_x = sobel_x.to(dtype=crops.dtype, device=crops.device)
#     sobel_y = sobel_y.to(dtype=crops.dtype, device=crops.device)

#     grad_x = F.conv2d(crops, sobel_x, padding=1)
#     grad_y = F.conv2d(crops, sobel_y, padding=1)

#     grad_mag = (grad_x**2 + grad_y**2).sqrt()  # [N, 1, 32, 32]
#     crater_score = grad_mag.mean(dim=[1,2,3])  # [N]

#     return (crater_score > threshold).float()

import torch
import torch.nn.functional as F
from torchvision.ops import roi_align

# Predefine Sobel filters once (CPU, float32)
_SOBEL_X = torch.as_tensor(
    [[1, 0, -1],
     [2, 0, -2],
     [1, 0, -1]], dtype=torch.float32
).view(1, 1, 3, 3)

_SOBEL_Y = torch.as_tensor(
    [[1, 2, 1],
     [0, 0, 0],
     [-1, -2, -1]], dtype=torch.float32
).view(1, 1, 3, 3)


def crater_score_from_tensor_fast(gray_batch, boxes, image_indices, threshold=0.03):
    """
    Efficient crater score calculation using torch operations (on GPU).
    gray_batch: Tensor [B, 1, H, W], values in [0, 1]
    boxes: Tensor [N, 4] in (x1, y1, x2, y2)
    image_indices: LongTensor [N], each value in 0..B-1
    threshold: float
    Returns: FloatTensor [N] with 1.0 if crater_score > threshold else 0.0
    """
    device, dtype = gray_batch.device, gray_batch.dtype
    boxes = boxes.clone()

    # Convert to [0, 1] relative coordinates
    B, _, H, W = gray_batch.shape
    boxes[:, [0, 2]] /= W  # x1, x2
    boxes[:, [1, 3]] /= H  # y1, y2

    # ROIAlign expects [batch_index, x1, y1, x2, y2]
    roi_boxes = torch.cat(
        [image_indices.unsqueeze(1).float(), boxes], dim=1
    )  # [N, 5]

    # Crop + resize
    crops = roi_align(gray_batch, roi_boxes, output_size=(32, 32), aligned=True)  # [N, 1, 32, 32]

    # Move Sobel kernels once per call
    sobel_x = _SOBEL_X.to(device=device, dtype=dtype)
    sobel_y = _SOBEL_Y.to(device=device, dtype=dtype)

    with torch.no_grad():  # we don’t need grads for this
        grad_x = F.conv2d(crops, sobel_x, padding=1)
        grad_y = F.conv2d(crops, sobel_y, padding=1)
        grad_mag = (grad_x**2 + grad_y**2).sqrt()  # [N, 1, 32, 32]

    crater_score = grad_mag.mean(dim=[1, 2, 3])  # [N]
    return (crater_score > threshold).float()
