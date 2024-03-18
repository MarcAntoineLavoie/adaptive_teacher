import torch
from typing import Dict, List, Optional, Tuple, Union
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.utils.events import get_event_storage
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.layers import ShapeSpec
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    StandardROIHeads,
)
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from adapteacher.modeling.roi_heads.fast_rcnn import FastRCNNFocaltLossOutputLayers

import numpy as np
from detectron2.modeling.poolers import ROIPooler


from torch.nn import functional as F
from torch import nn, distributions
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.layers import Linear, cat#, Conv2d, get_norm
from detectron2.config import configurable
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from fvcore.nn import smooth_l1_loss
from adapteacher.modeling.modeling_utils import get_probabilistic_loss_weight, covariance_output_to_cholesky, clamp_log_variance
from torch.autograd.function import Function
from detectron2.modeling.matcher import Matcher

from detectron2.layers import ShapeSpec, batched_nms, cat, cross_entropy, nonzero_tuple

from adapteacher.modeling.meta_arch.rcnn import FCDiscriminator_inst, grad_reverse

import random
import geomloss

@ROI_HEADS_REGISTRY.register()
class ProbROIHeadsPseudoLab(StandardROIHeads):
    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
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
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )

        # box_predictor = ProbabilisticFastRCNNOutputLayers(cfg, box_head.output_shape)
        
        cls_var_loss = None
        compute_cls_var = None
        cls_var_num_samples = None

        bbox_cov_loss = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME
        compute_bbox_cov = bbox_cov_loss != 'none'
        bbox_cov_num_samples = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES

        bbox_cov_type = cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE
        if bbox_cov_type == 'diagonal':
            # Diagonal covariance matrix has N elements
            bbox_cov_dims = 4
        else:
            # Number of elements required to describe an NxN covariance matrix is
            # computed as:  (N * (N + 1)) / 2
            bbox_cov_dims = 10

        # select_iou = cfg.MODEL.PROBABILISTIC_MODELING.SELECT_IOU2

        box_predictor = ProbabilisticFastRCNNOutputLayers(
            cfg,
            box_head.output_shape,
            compute_cls_var,
            cls_var_loss,
            cls_var_num_samples,
            compute_bbox_cov,
            bbox_cov_loss,
            bbox_cov_type,
            bbox_cov_dims,
            bbox_cov_num_samples,
            )

        # if cfg.MODEL.ROI_HEADS.LOSS == "CrossEntropy":
        #     box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        # elif cfg.MODEL.ROI_HEADS.LOSS == "FocalLoss":
        #     box_predictor = FastRCNNFocaltLossOutputLayers(cfg, box_head.output_shape)
        # else:
        #     raise ValueError("Unknown ROI head loss.")

        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
        }

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        compute_loss=True,
        branch="",
        compute_val_loss=False,
        unsup=False,
        current_step=0,
        prob_iou=False,
        select_iou=False,
        targets_gt=None,
        use_gt_only=False,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:

        del images
        if self.training and compute_loss:  # apply if training loss
            assert targets
            # 1000 --> 512

            if targets_gt is not None:
                proposals_gt = self.label_and_sample_proposals(proposals, targets_gt, branch=branch, use_gt_only=use_gt_only)
            else:
                proposals_gt = None

            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )


        elif compute_val_loss:  # apply if val loss
            assert targets
            # 1000 --> 512
            temp_proposal_append_gt = self.proposal_append_gt
            self.proposal_append_gt = False
            proposals = self.label_and_sample_proposals(
                proposals, targets, branch=branch
            )  # do not apply target on proposals
            self.proposal_append_gt = temp_proposal_append_gt
        del targets

        if (self.training and compute_loss) or compute_val_loss:
            losses, _ = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch, unsup=unsup, current_step=current_step, proposals_gt=proposals_gt,
            )
            n = len(proposals[0])
            for i in range(len(proposals)):
                proposals[i].class_logits = torch.nn.functional.softmax(_[0][i*n:(i+1)*n], dim=1)
            return proposals, losses
        else:
            pred_instances, predictions = self._forward_box(
                features, proposals, compute_loss, compute_val_loss, branch, unsup=unsup,
            )

            return pred_instances, predictions

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        compute_loss: bool = True,
        compute_val_loss: bool = False,
        branch: str = "",
        unsup: bool = False,
        current_step = 0,
        proposals_gt = None,
        # prob_iou: bool = False,
        # select_iou: bool = False,
    ) -> Union[Dict[str, torch.Tensor], List[Instances]]:
        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        if self.box_predictor.align_proposals and 'supervised' == branch and self.align_gt_proposals:
        # elif self.box_predictor.align_proposals and 'supervised' in branch and self.align_gt_proposals:
            box_features_gt = self.box_pooler(features, [x.proposal_boxes for x in proposals_gt])
            self.process_proposals(box_features_gt, proposals_gt, branch, use_bg=self.use_bg, points_per_proposals=self.points_per_proposals, subsampling=self.sampling)
        elif self.box_predictor.align_proposals and 'supervised' in branch:
            self.process_proposals(box_features, proposals, branch, use_bg=self.use_bg, points_per_proposals=self.points_per_proposals, subsampling=self.sampling)
            # self.keep_proposals[branch] = [predictions[0], cat([p.gt_classes for p in proposals], dim=0)]
        box_features = self.box_head(box_features)
        if self.box_predictor.compute_bbox_cov and branch == 'supervised':
            predictions = self.box_predictor(box_features, proposals=proposals)
        else:
            predictions = self.box_predictor(box_features)

        
        # if self.keep_stats:
        #     pass



        if (
            self.training and compute_loss
        ) or compute_val_loss:  # apply if training loss or val loss
            del box_features
            losses = self.box_predictor.losses(predictions, proposals, unsup=unsup, current_step=current_step, branch=branch)

            if self.train_on_pred_boxes:
                with torch.no_grad():
                    pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                        predictions, proposals
                    )
                    for proposals_per_image, pred_boxes_per_image in zip(
                        proposals, pred_boxes
                    ):
                        proposals_per_image.proposal_boxes = Boxes(pred_boxes_per_image)
            return losses, predictions
        else:
            pred_instances, ids_ = self.box_predictor.inference(predictions, proposals, unsup=unsup)
            for instance, proposal, ids in zip(pred_instances, proposals, ids_):
                instance.rpn_score = proposal.objectness_logits[ids]
                # if self.keep_stats:
                #     instance.logits = box_features[ids,:]
            del box_features
            return pred_instances, predictions

    @torch.no_grad()
    def label_and_sample_proposals(
        self, proposals: List[Instances], targets: List[Instances], branch: str = "", use_gt_only=False,
    ) -> List[Instances]:
        gt_boxes = [x.gt_boxes for x in targets]
        is_gt = [torch.zeros(len(x)) for x in proposals]
        if self.proposal_append_gt:
            n_preds = [len(x) for x in proposals]
            proposals = add_ground_truth_to_proposals(gt_boxes, proposals)
            is_gt = [torch.cat((y,torch.ones(len(x)-len(y)))) for x, y in zip(proposals, is_gt)]
            if use_gt_only:
                proposals = [prop[n:] for prop, n in zip(proposals, n_preds)]

        proposals_with_gt = []

        num_fg_samples = []
        num_bg_samples = []
        for proposals_per_image, targets_per_image, from_gt in zip(proposals, targets, is_gt):
            has_gt = len(targets_per_image) > 0
            match_quality_matrix = pairwise_iou(
                targets_per_image.gt_boxes, proposals_per_image.proposal_boxes
            )
            matched_idxs, matched_labels = self.proposal_matcher(match_quality_matrix)
            sampled_idxs, gt_classes = self._sample_proposals(
                matched_idxs, matched_labels, targets_per_image.gt_classes
            )

            proposals_per_image = proposals_per_image[sampled_idxs]
            proposals_per_image.gt_classes = gt_classes

            if has_gt:
                sampled_targets = matched_idxs[sampled_idxs]
                proposal_type = from_gt.to(device=proposals_per_image.gt_classes.device)[sampled_idxs] + (proposals_per_image.gt_classes < max(gt_classes)).long()
                proposals_per_image.proposal_type = proposal_type
                proposals_per_image.orig_box = sampled_idxs
                for (trg_name, trg_value) in targets_per_image.get_fields().items():
                    if trg_name.startswith("gt_") and not proposals_per_image.has(
                        trg_name
                    ):
                        if trg_name in ["gt_iou", "gt_scores"]:
                            ids_bg = torch.where(gt_classes == 8)[0]
                            vals = trg_value[sampled_targets]
                            vals[ids_bg] = 1.0
                            proposals_per_image.set(trg_name, vals)
                        else:
                            proposals_per_image.set(trg_name, trg_value[sampled_targets])

            else:
                gt_boxes = Boxes(
                    targets_per_image.gt_boxes.tensor.new_zeros((len(sampled_idxs), 4))
                )
                proposals_per_image.gt_boxes = gt_boxes
                proposals_per_image.proposal_type = torch.zeros(len(sampled_idxs)).to(device=proposals_per_image.gt_boxes.device)
                proposals_per_image.orig_box = -1*torch.ones(len(sampled_idxs)).to(device=proposals_per_image.gt_boxes.device)

            num_bg_samples.append((gt_classes == self.num_classes).sum().item())
            num_fg_samples.append(gt_classes.numel() - num_bg_samples[-1])
            if self.use_gt_proposals_only:
                ids = torch.where(proposals_per_image.proposal_type == 2)[0]
                proposals_per_image = proposals_per_image[ids]
            proposals_with_gt.append(proposals_per_image)

        if "supervised" in branch and not use_gt_only:
            storage = get_event_storage()
            storage.put_scalar(
                "roi_head/num_target_fg_samples_" + branch, np.mean(num_fg_samples)
            )
            storage.put_scalar(
                "roi_head/num_target_bg_samples_" + branch, np.mean(num_bg_samples)
            )

        return proposals_with_gt
    
    def process_proposals(self, box_features, proposals, branch, use_bg=False, points_per_proposals=10, subsampling='random'):
        use_bg = use_bg
        n_samples = points_per_proposals
        max_samples = 200
        device = box_features.device

        feat_shape = box_features.shape
        m = feat_shape[2]*feat_shape[3]
        labels = cat([p.gt_classes for p in proposals], dim=0)
        # n1 = labels.max()
        n1 = 8
        if use_bg:
            n1 = n1 + 1
        vals = []
        nvals = []
        for label in range(n1):
            ids1 = labels == label
            if ids1.any():
                if subsampling == 'random':
                    features = box_features[ids1,:,:,:].reshape(-1,512,m)
                    n2 = features.shape[0]
                    features = features.transpose(1,2).reshape(-1,512)
                    ids2 = sum([random.sample(list(range(i*m,(i+1)*m)), n_samples) for i in range(n2)], [])
                    vals.append(features[ids2,:])
                    nvals.append(len(ids2))
                elif subsampling == 'centre':
                    vals.append(box_features.transpose(0,1)[:,ids1,2:5,2:5].flatten(1,3).transpose(0,1))
                    nvals.append(ids1.sum()*9)
            else:
                vals.append([])
                nvals.append(0) 

        samples = self.sample_queue(branch, vals, nvals, max_samples=max_samples, device=device)
        self.update_queue(branch, vals)

        self.current_proposals[branch] = samples

    def build_queues(self, n_classes, n_samples, feat_dim, base_count=0):
        self.register_buffer("queue_source", torch.randn(n_classes, n_samples, feat_dim))
        self.register_buffer("queue_target", torch.randn(n_classes, n_samples, feat_dim))
        self.register_buffer('source_counts', torch.ones(n_classes)*base_count)
        self.register_buffer('target_counts', torch.ones(n_classes)*base_count)
        self.l_queue = n_samples

    def build_prototypes(self, n_classes, feat_dim):
        self.register_buffer("queue_source", torch.randn(n_classes, feat_dim))

    def sample_queue(self, branch, vals, nvals, max_samples=200, device=None):
        if branch == 'supervised':
            queue = self.queue_source
        elif branch == 'supervised_target':
            queue = self.queue_target
        else:
            raise ValueError("Unknown branch")

        s1 = list(range(self.l_queue))
        outputs = []
        labels = []
        for label in range(len(vals)):
            nval = nvals[label]
            if not nval:
                n = max_samples
                random.shuffle(s1)
                ids = s1[:n]
                new_vals = queue[label,ids,:].to(device=device)
                outputs.append(new_vals)
            elif nval < max_samples:
                n = max_samples - nval
                random.shuffle(s1)
                ids = s1[:n]
                new_vals = queue[label,ids,:].to(device=device)
                outputs.append(cat((vals[label], new_vals),dim=0))
            else:
                s2 = list(range(nval))
                random.shuffle(s2)
                ids2 = s2[:max_samples]
                outputs.append(vals[label][ids2,:])
            labels.append(torch.ones(max_samples)*label)

        return (labels, outputs)

    def update_queue(self, branch, vals):
        if branch == 'supervised':
            queue = self.queue_source
            counts = self.source_counts
        elif branch == 'supervised_target':
            queue = self.queue_target
            counts = self.target_counts
        else:
            raise ValueError("Unknown branch")
        max_samples = queue.shape[1]
        for label in range(len(vals)):
            n = len(vals[label])
            counts[label] = counts[label] + n
            if n >= max_samples:
                ids = list(range(n))
                random.shuffle(ids)
                ids = ids[:max_samples]
                queue[label,:,:] = vals[label][ids,:].detach().cpu()
            elif not n:
                pass
            else:
                queue[label,:,:] = cat((vals[label].detach().cpu(),queue[label,n:,:]))

    def update_prototypes(self, vals):
        for label in range(len(vals)):
            # for proposal in vals[label]
            pass
    

class ProbabilisticFastRCNNOutputLayers(nn.Module):
    """
    Four linear layers for predicting Fast R-CNN outputs:
      (1) proposal-to-detection box regression deltas
      (2) classification scores
      (3) box regression deltas covariance parameters (if needed)
      (4) classification logits variance (if needed)
    """

    @configurable
    def __init__(
        self,
        input_shape,
        *,
        box2box_transform,
        num_classes,
        cls_agnostic_bbox_reg=False,
        smooth_l1_beta=0.0,
        test_score_thresh=0.0,
        test_nms_thresh=0.5,
        test_topk_per_image=100,
        compute_cls_var=False,
        compute_bbox_cov=False,
        bbox_cov_dims=4,
        cls_var_loss='none',
        cls_var_num_samples=10,
        bbox_cov_loss='none',
        bbox_cov_type='diagonal',
        dropout_rate=0.0,
        annealing_step=0,
        bbox_cov_num_samples=1000,
        smooth_l1_beta_prob=0.0, 
        prob_iou=False,
        use_scale=False,
        scale_expo_score=0.0,
        scale_expo_iou=0.0,
        domain_invariant_inst=0.0,
        align_proposals=False,
        normed_proj=True,
        burnup_steps=20000,
        gt_inject_cov_weight=1.0,
        select_iou=False,
        align_use_proj=True,
        box_norm_class=False,
        suppress_box=1.0
    ):
        """
        NOTE: this interface is experimental.

        Args:
            input_shape (ShapeSpec): shape of the input feature to this module
            box2box_transform (Box2BoxTransform or Box2BoxTransformRotated):
            num_classes (int): number of foreground classes
            cls_agnostic_bbox_reg (bool): whether to use class agnostic for bbox regression
            smooth_l1_beta (float): transition point from L1 to L2 loss.
            test_score_thresh (float): threshold to filter predictions results.
            test_nms_thresh (float): NMS threshold for prediction results.
            test_topk_per_image (int): number of top predictions to produce per image.
            compute_cls_var (bool): compute classification variance
            compute_bbox_cov (bool): compute box covariance regression parameters.
            bbox_cov_dims (int): 4 for diagonal covariance, 10 for full covariance.
            cls_var_loss (str): name of classification variance loss.
            cls_var_num_samples (int): number of samples to be used for loss computation. Usually between 10-100.
            bbox_cov_loss (str): name of box covariance loss.
            bbox_cov_type (str): 'diagonal' or 'full'. This is used to train with loss functions that accept both types.
            dropout_rate (float): 0-1, probability of drop.
            annealing_step (int): step used for KL-divergence in evidential loss to fully be functional.
        """
        super().__init__()
        if isinstance(input_shape, int):  # some backward compatibility
            input_shape = ShapeSpec(channels=input_shape)
        input_size = input_shape.channels * \
            (input_shape.width or 1) * (input_shape.height or 1)

        self.compute_cls_var = False
        self.compute_bbox_cov = compute_bbox_cov

        self.bbox_cov_dims = bbox_cov_dims
        self.bbox_cov_num_samples = bbox_cov_num_samples

        # self.dropout_rate = dropout_rate
        # self.use_dropout = self.dropout_rate != 0.0

        self.cls_var_loss = cls_var_loss
        self.cls_var_num_samples = cls_var_num_samples

        self.annealing_step = annealing_step

        self.bbox_cov_loss = bbox_cov_loss
        self.bbox_cov_type = bbox_cov_type

        # The prediction layer for num_classes foreground classes and one background class
        # (hence + 1)
        if box_norm_class:
            self.cls_score = NormedLinear(input_size, num_classes + 1)
        else:
            self.cls_score = Linear(input_size, num_classes + 1)
        num_bbox_reg_classes = 1.0 if cls_agnostic_bbox_reg else num_classes
        box_dim = len(box2box_transform.weights)
        self.bbox_pred = Linear(input_size, num_bbox_reg_classes * box_dim)

        if box_norm_class:
            nn.init.normal_(self.cls_score.linear.weight, std=0.01)
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            for l in [self.cls_score.linear, self.bbox_pred]:
                nn.init.constant_(l.bias, 0)
        else:
            nn.init.normal_(self.cls_score.weight, std=0.01)
            nn.init.normal_(self.bbox_pred.weight, std=0.001)
            for l in [self.cls_score, self.bbox_pred]:
                nn.init.constant_(l.bias, 0)

        self.select_iou = select_iou

        # if self.compute_cls_var:
        #     self.cls_var = Linear(input_size, num_classes + 1)
        #     nn.init.normal_(self.cls_var.weight, std=0.0001)
        #     nn.init.constant_(self.cls_var.bias, 0)

        if self.compute_bbox_cov and self.bbox_cov_loss != 'energy_loss2':
            self.bbox_cov = Linear(
                input_size,
                num_bbox_reg_classes *
                bbox_cov_dims)
            nn.init.normal_(self.bbox_cov.weight, std=0.0001)
            nn.init.constant_(self.bbox_cov.bias, 0)
            if self.select_iou:
                self.bbox_iou_pred = Cov2IoUHead(in_dim=8, feat_dim=50, depth=5)

        elif self.compute_bbox_cov and self.bbox_cov_loss == 'energy_loss2':
            self.bbox_prob_gen = Linear(
                input_size + bbox_cov_dims,
                bbox_cov_num_samples * bbox_cov_dims)
            nn.init.normal_(self.bbox_prob_gen.weight, std=0.0001)
            nn.init.constant_(self.bbox_prob_gen.bias, 0)
            self.cov_loss = geomloss.SamplesLoss(loss='energy')

        self.box2box_transform = box2box_transform
        self.smooth_l1_beta = smooth_l1_beta
        self.test_score_thresh = test_score_thresh
        self.test_nms_thresh = test_nms_thresh
        self.test_topk_per_image = test_topk_per_image

        self.smooth_l1_beta_prob = smooth_l1_beta_prob
        self.prob_iou = prob_iou
        self.use_scale = use_scale
        self.scale_expo_score = scale_expo_score
        self.scale_expo_iou = scale_expo_iou

        self.domain_invariant_inst = domain_invariant_inst
        if self.domain_invariant_inst:
            self.DA_layer = FCDiscriminator_inst(input_shape.channels)
            self.DA_scores = []
        self.align_proposals = align_proposals
        self.burnup_steps = burnup_steps

        self.gt_inject_cov_weight = gt_inject_cov_weight

        if self.align_proposals:
            feat_dim = 512
            self.use_proj = align_use_proj
            self.proj_head = ProjectionHead(feat_dim=feat_dim, use_proj=self.use_proj, normed=normed_proj)

        self.select_iou = select_iou
        self.suppress_box = suppress_box

    @classmethod
    def from_config(cls,
                    cfg,
                    input_shape,
                    compute_cls_var,
                    cls_var_loss,
                    cls_var_num_samples,
                    compute_bbox_cov,
                    bbox_cov_loss,
                    bbox_cov_type,
                    bbox_cov_dims,
                    bbox_cov_num_samples):
        return {
            "input_shape": input_shape,
            "box2box_transform": Box2BoxTransform(weights=cfg.MODEL.ROI_BOX_HEAD.BBOX_REG_WEIGHTS),
            # fmt: off
            "num_classes": cfg.MODEL.ROI_HEADS.NUM_CLASSES,
            "cls_agnostic_bbox_reg": cfg.MODEL.ROI_BOX_HEAD.CLS_AGNOSTIC_BBOX_REG,
            "smooth_l1_beta": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA,
            "test_score_thresh": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
            "test_nms_thresh": cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
            "test_topk_per_image": cfg.TEST.DETECTIONS_PER_IMAGE,
            "compute_cls_var": compute_cls_var,
            "cls_var_loss": cls_var_loss,
            "cls_var_num_samples": cls_var_num_samples,
            "compute_bbox_cov": compute_bbox_cov,
            "bbox_cov_dims": bbox_cov_dims,
            "bbox_cov_loss": bbox_cov_loss,
            "bbox_cov_type": bbox_cov_type,
            # "dropout_rate": cfg.MODEL.PROBABILISTIC_MODELING.DROPOUT_RATE
            "dropout_rate": None,
            # "annealing_step": cfg.SOLVER.STEPS[1],
            "annealing_step": cfg.MODEL.PROBABILISTIC_MODELING.ANNEALING_STEP,
            "bbox_cov_num_samples": bbox_cov_num_samples,
            # fmt: on,
            "smooth_l1_beta_prob": cfg.MODEL.ROI_BOX_HEAD.SMOOTH_L1_BETA_PROB,
            "prob_iou": cfg.MODEL.PROBABILISTIC_MODELING.PROB_IOU,
            "use_scale": cfg.MODEL.PROBABILISTIC_MODELING.SCALE_LOSS,
            "scale_expo_score": cfg.MODEL.PROBABILISTIC_MODELING.SCALE_EXPO_SCORE,
            "scale_expo_iou": cfg.MODEL.PROBABILISTIC_MODELING.SCALE_EXPO_IOU,
            "domain_invariant_inst": cfg.SEMISUPNET.DOMAIN_ADV_INST,
            "align_proposals": cfg.SEMISUPNET.ALIGN_PROPOSALS,
            "normed_proj": cfg.SEMISUPNET.ALIGN_NORMED,
            "burnup_steps": cfg.SEMISUPNET.BURN_UP_STEP,
            "gt_inject_cov_weight": cfg.MODEL.PROBABILISTIC_MODELING.GT_INJECT_COV_WEIGHT,
            "select_iou": cfg.MODEL.PROBABILISTIC_MODELING.SELECT_IOU2,
            "align_use_proj": cfg.SEMISUPNET.ALIGN_USE_PROJ,
            "box_norm_class": cfg.SEMISUPNET.BOX_NORM_CLASS,
            "suppress_box": cfg.SEMISUPNET.SUPPRESS_BBOX_CORR,
        }

    def forward(self, x, proposals=None):
        """
        Returns:
            Tensor: Nx(K+1) logits for each box
            Tensor: Nx4 or Nx(Kx4) bounding box regression deltas.
            Tensor: Nx(K+1) logits variance for each box.
            Tensor: Nx4(10) or Nx(Kx4(10)) covariance matrix parameters. 4 if diagonal, 10 if full.
        """
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        scores = self.cls_score(x)
        proposal_deltas = self.bbox_pred(x)

        # Compute logits variance if needed
        if self.compute_cls_var:
            score_vars = self.cls_var(x)
        else:
            score_vars = None

        # Compute box covariance if needed
        if self.compute_bbox_cov and self.bbox_cov_loss != "energy_loss2":
            proposal_covs = self.bbox_cov(x)
            if self.bbox_cov_loss == 'samplenet':
                proposal_covs = proposal_covs.reshape(proposal_covs.shape[0], -1, 320)
                proposal_deltas = proposal_covs.mean(axis=1)
        elif self.compute_bbox_cov and self.bbox_cov_loss == "energy_loss2":
            proposal_types = cat([p.proposal_type for p in proposals], dim=0)
            fg_idxs = torch.nonzero(proposal_types > 0, as_tuple=True)[0]
            proposal_gt = cat([p.gt_classes for p in proposals], dim=0)
            fg_gt_classes = proposal_gt[fg_idxs]
            box_cols = self.bbox_cov_dims * fg_gt_classes[:, None] + torch.arange(self.bbox_cov_dims).to(device=fg_gt_classes[:, None].device)
            props = proposal_deltas[fg_idxs[:, None],box_cols]
            x2 = cat((x[fg_idxs,:], props), dim=1)
            proposal_covs = self.bbox_prob_gen(x2)
            
        else:
            proposal_covs = None

        if self.domain_invariant_inst:
            x_rev = grad_reverse(x)
            self.DA_scores = self.DA_layer(x_rev)

        return scores, proposal_deltas, score_vars, proposal_covs

    def losses(self, predictions, proposals, current_step=0, unsup=False, branch="supervised"):
        """
        Args:
            predictions: return values of :meth:`forward()`.
            proposals (list[Instances]): proposals that match the features
                that were used to compute predictions.
            current_step: current optimizer step. Used for losses with an annealing component.
        """
        global device

        pred_class_logits, pred_proposal_deltas, pred_class_logits_var, pred_proposal_covs = predictions

        if len(proposals):
            box_type = type(proposals[0].proposal_boxes)
            # cat(..., dim=0) concatenates over all images in the batch
            proposals_boxes = box_type.cat(
                [p.proposal_boxes for p in proposals])
            assert (
                not proposals_boxes.tensor.requires_grad), "Proposals should not require gradients!"

            # The following fields should exist only when training.
            if proposals[0].has("gt_boxes"):
                gt_boxes = box_type.cat([p.gt_boxes for p in proposals])
                assert proposals[0].has("gt_classes")
                gt_classes = cat([p.gt_classes for p in proposals], dim=0)
                proposal_type = cat([p.proposal_type for p in proposals], dim=0)
        else:
            proposals_boxes = Boxes(
                torch.zeros(
                    0, 4, device=pred_proposal_deltas.device))

        no_instances = len(proposals) == 0  # no instances found

        # Compute Classification Loss
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            loss_cls = 0.0 * F.cross_entropy(
                pred_class_logits,
                torch.zeros(
                    0,
                    dtype=torch.long,
                    device=pred_class_logits.device),
                reduction="sum",)
        elif unsup and self.use_scale and all(["gt_scores" in props._fields.keys() for props in proposals]):
            loss_temp = F.cross_entropy(
                pred_class_logits, gt_classes, reduction="none")
            gt_scores = cat([p.gt_scores for p in proposals], dim=0) ** self.scale_expo_score
            weights_score = gt_scores.shape[0] / gt_scores.sum() * gt_scores
            loss_cls = (loss_temp*weights_score).mean()

        else:
            loss_cls = F.cross_entropy(
                pred_class_logits, gt_classes, reduction="mean")

        # Compute regression loss:
        if no_instances:
            # TODO 0.0 * pred.sum() is enough since PT1.6
            loss_box_reg = 0.0 * smooth_l1_loss(
                pred_proposal_deltas,
                torch.zeros_like(pred_proposal_deltas),
                0.0,
                reduction="sum",
            )
        else:
            gt_proposal_deltas = self.box2box_transform.get_deltas(
                proposals_boxes.tensor, gt_boxes.tensor
            )
            box_dim = gt_proposal_deltas.size(1)  # 4 or 5
            cls_agnostic_bbox_reg = pred_proposal_deltas.size(1) == box_dim
            device = pred_proposal_deltas.device

            bg_class_ind = pred_class_logits.shape[1] - 1

            # Box delta loss is only computed between the prediction for the gt class k
            # (if 0 <= k < bg_class_ind) and the target; there is no loss defined on predictions
            # for non-gt classes and background.
            # Empty fg_inds produces a valid loss of zero as long as the size_average
            # arg to smooth_l1_loss is False (otherwise it uses torch.mean internally
            # and would produce a nan loss).
            fg_inds = torch.nonzero(
                (gt_classes >= 0) & (gt_classes < bg_class_ind), as_tuple=True
            )[0]
            if cls_agnostic_bbox_reg:
                # pred_proposal_deltas only corresponds to foreground class for
                # agnostic
                gt_class_cols = torch.arange(box_dim, device=device)
            else:
                fg_gt_classes = gt_classes[fg_inds]
                # pred_proposal_deltas for class k are located in columns [b * k : b * k + b],
                # where b is the dimension of box representation (4 or 5)
                # Note that compared to Detectron1,
                # we do not perform bounding box regression for background
                # classes.
                gt_class_cols = box_dim * \
                    fg_gt_classes[:, None] + torch.arange(box_dim, device=device)
                gt_covar_class_cols = self.bbox_cov_dims * \
                    fg_gt_classes[:, None] + torch.arange(self.bbox_cov_dims, device=device)
                gt_proposal_types = proposal_type[fg_inds]

            loss_reg_normalizer = gt_classes.numel()

            pred_proposal_deltas = pred_proposal_deltas[fg_inds[:,
                                                                None], gt_class_cols]
            gt_proposals_delta = gt_proposal_deltas[fg_inds]

            if self.compute_bbox_cov and len(fg_inds) and not unsup and "2" not in self.bbox_cov_loss:
                pred_proposal_covs = pred_proposal_covs[fg_inds[:,
                                                                None], gt_covar_class_cols]
                pred_proposal_covs = clamp_log_variance(pred_proposal_covs)

                if self.bbox_cov_loss == 'negative_log_likelihood':
                    if self.bbox_cov_type == 'diagonal':
                        # Compute regression negative log likelihood loss according to:
                        # "What Uncertainties Do We Need in Bayesian Deep Learning for Computer Vision?", NIPS 2017
                        loss_box_reg = 0.5 * torch.exp(-pred_proposal_covs) * smooth_l1_loss(
                            pred_proposal_deltas, gt_proposals_delta, beta=self.smooth_l1_beta_prob)
                        loss_covariance_regularize = 0.5 * pred_proposal_covs
                        loss_box_reg += loss_covariance_regularize

                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer
                    else:
                        # Multivariate Gaussian Negative Log Likelihood loss using pytorch
                        # distributions.multivariate_normal.log_prob()
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs)

                        multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                            pred_proposal_deltas, scale_tril=forecaster_cholesky)

                        loss_box_reg = - \
                            multivariate_normal_dists.log_prob(gt_proposals_delta)
                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer

                elif self.bbox_cov_loss == 'second_moment_matching':
                    # Compute regression covariance using second moment
                    # matching.
                    loss_box_reg = smooth_l1_loss(pred_proposal_deltas,
                                                gt_proposals_delta,
                                                self.smooth_l1_beta_prob)
                    errors = (pred_proposal_deltas - gt_proposals_delta)
                    if self.bbox_cov_type == 'diagonal':
                        # Handel diagonal case
                        second_moment_matching_term = smooth_l1_loss(
                            torch.exp(pred_proposal_covs), errors ** 2, beta=self.smooth_l1_beta_prob)
                        loss_box_reg += second_moment_matching_term
                        loss_box_reg = torch.sum(
                            loss_box_reg) / loss_reg_normalizer
                    else:
                        # Handel full covariance case
                        errors = torch.unsqueeze(errors, 2)
                        gt_error_covar = torch.matmul(
                            errors, torch.transpose(errors, 2, 1))

                        # This is the cholesky decomposition of the covariance matrix.
                        # We reconstruct it from 10 estimated parameters as a
                        # lower triangular matrix.
                        forecaster_cholesky = covariance_output_to_cholesky(
                            pred_proposal_covs)

                        predicted_covar = torch.matmul(
                            forecaster_cholesky, torch.transpose(
                                forecaster_cholesky, 2, 1))

                        second_moment_matching_term = smooth_l1_loss(
                            predicted_covar, gt_error_covar, beta=self.smooth_l1_beta_prob, reduction='sum')
                        loss_box_reg = (
                            torch.sum(loss_box_reg) + second_moment_matching_term) / loss_reg_normalizer

                elif self.bbox_cov_loss == 'energy_loss':
                    forecaster_cholesky = covariance_output_to_cholesky(
                        pred_proposal_covs)

                    # Define per-anchor Distributions
                    multivariate_normal_dists = distributions.multivariate_normal.MultivariateNormal(
                        pred_proposal_deltas, scale_tril=forecaster_cholesky)
                    # Define Monte-Carlo Samples
                    distributions_samples = multivariate_normal_dists.rsample(
                        (self.bbox_cov_num_samples + 1,))

                    distributions_samples_1 = distributions_samples[0:self.bbox_cov_num_samples, :, :]
                    distributions_samples_2 = distributions_samples[1:self.bbox_cov_num_samples + 1, :, :]

                    # Compute energy score
                    weight_mat = torch.where(gt_proposal_types==2,self.gt_inject_cov_weight,1.0).unsqueeze(0).unsqueeze(-1)

                    loss_covariance_regularize = - (weight_mat * smooth_l1_loss(
                        distributions_samples_1,
                        distributions_samples_2,
                        beta=self.smooth_l1_beta_prob,
                        reduction="none")).sum() / self.bbox_cov_num_samples   # Second term

                    gt_proposals_delta_samples = torch.repeat_interleave(
                        gt_proposals_delta.unsqueeze(0), self.bbox_cov_num_samples, dim=0)

                    loss_first_moment_match = (weight_mat * 2.0 * smooth_l1_loss(
                        distributions_samples_1,
                        gt_proposals_delta_samples,
                        beta=self.smooth_l1_beta_prob,
                        reduction="none")).sum() / self.bbox_cov_num_samples  # First term

                    # Final Loss
                    loss_box_reg = (
                        loss_first_moment_match + loss_covariance_regularize) / loss_reg_normalizer
                    
                    # if self.bbox_iou_pred:
                    if 0:
                        pass
                        # pred_boxes = self.box2box_transform.a (pred_proposal_deltas
                        # iou = intersect_self(result.pred_boxes.tensor, box_covs[keep])
                        # gt_proposal_deltas = self.box2box_transform.get_deltas(proposals_boxes.tensor, gt_boxes.tensor)

                else:
                    raise ValueError(
                        'Invalid regression loss name {}.'.format(
                            self.bbox_cov_loss))

                # Perform loss annealing. Not really essential in Generalized-RCNN case, but good practice for more
                # elaborate regression variance losses.
                standard_regression_loss = smooth_l1_loss(pred_proposal_deltas,
                                                          gt_proposals_delta,
                                                          self.smooth_l1_beta,
                                                          reduction="sum",)
                standard_regression_loss = standard_regression_loss / loss_reg_normalizer

                # print(current_step)
                probabilistic_loss_weight = get_probabilistic_loss_weight(
                    current_step, self.annealing_step)

                loss_box_reg = (1.0 - probabilistic_loss_weight) * \
                    standard_regression_loss + probabilistic_loss_weight * loss_box_reg

            elif self.compute_bbox_cov and len(fg_inds) and not unsup and "2" in self.bbox_cov_loss:
                prob_boxes = cat((pred_proposal_deltas, pred_proposal_covs), dim=1)
                
                loss_box_reg = self.cov_loss(prob_boxes.reshape(fg_inds.shape[0],-1,4), gt_proposals_delta.unsqueeze(1)).mean() * 0.1

                # Perform loss annealing. Not really essential in Generalized-RCNN case, but good practice for more
                # elaborate regression variance losses.
                standard_regression_loss = smooth_l1_loss(pred_proposal_deltas,
                                                          gt_proposals_delta,
                                                          self.smooth_l1_beta,
                                                          reduction="sum",)
                standard_regression_loss = standard_regression_loss / loss_reg_normalizer

                # print(current_step)
                probabilistic_loss_weight = get_probabilistic_loss_weight(
                    current_step, self.annealing_step)

                loss_box_reg = (1.0 - probabilistic_loss_weight) * \
                    standard_regression_loss + probabilistic_loss_weight * loss_box_reg


            elif unsup and self.use_scale and all(["gt_iou" in props._fields.keys() for props in proposals]):
                loss_box_temp = smooth_l1_loss(pred_proposal_deltas,
                                              gt_proposals_delta,
                                              self.smooth_l1_beta,
                                              reduction="none",).sum(dim=1)
                gt_ious = cat([p.gt_iou for p in proposals], dim=0)[fg_inds] ** self.scale_expo_iou
                weights_iou = gt_ious.shape[0] / gt_ious.sum() * gt_ious
                loss_box_reg = (loss_box_temp*weights_iou).sum() / loss_reg_normalizer
            
            else:
                loss_box_reg = smooth_l1_loss(pred_proposal_deltas,
                                              gt_proposals_delta,
                                              self.smooth_l1_beta,
                                              reduction="sum",)
                loss_box_reg = loss_box_reg / loss_reg_normalizer

        if branch == 'supervised_target':
            loss_box_reg *= self.suppress_box

        if self.domain_invariant_inst:
            if current_step >= self.burnup_steps:
                scale = 1.0
            else:
                scale = 1e-10
            use_only_fg = False
            if use_only_fg:
                n = pred_class_logits.shape[1] -1
                fg_idx = gt_classes < n
                # logit_thresh = pred_class_logits.gather(1, gt_classes.unsqueeze(1)) > 0.7
                self.DA_scores = self.DA_scores[fg_idx]
            if branch == 'supervised':
                DA_label = 0
            elif branch == 'supervised_target':
                DA_label = 1
            loss_DA_inst = scale * self.domain_invariant_inst * F.binary_cross_entropy_with_logits(self.DA_scores, torch.FloatTensor(self.DA_scores.data.size()).fill_(DA_label).to(self.DA_scores.device))

            losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg, "loss_DA_inst": loss_DA_inst}
        else:
            losses = {"loss_cls": loss_cls, "loss_box_reg": loss_box_reg}
        return losses 

    def inference(self, predictions, proposals, unsup=0):
        """
        Returns:
            list[Instances]: same as `fast_rcnn_inference`.
            list[Tensor]: same as `fast_rcnn_inference`.
        """
        
        scores, box_deltas, score_covs, box_covs = predictions
        n = scores.shape[0]
        if score_covs is None:
            score_covs = -torch.ones(n)
        if box_covs is None:
            box_covs = -torch.ones(n)
        else:
            box_covs = torch.exp(box_covs)

        num_inst_per_image = [len(p) for p in proposals]
        score_covs = score_covs.split(num_inst_per_image, dim=0)
        box_covs = box_covs.split(num_inst_per_image, dim=0)

        boxes = self.predict_boxes(predictions, proposals)
        scores = self.predict_probs(predictions, proposals)
        image_shapes = [x.image_size for x in proposals]

        return prob_fast_rcnn_inference(
            boxes,
            scores,
            box_covs,
            score_covs,
            image_shapes,
            self.test_score_thresh,
            self.test_nms_thresh,
            self.test_topk_per_image,
            self.prob_iou*unsup,
            self.select_iou,
        )

    def predict_boxes_for_gt_classes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted boxes for GT classes in case of
                class-specific box head. Element i of the list has shape (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        scores, proposal_deltas = predictions
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        N, B = proposal_boxes.shape
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)

        K = predict_boxes.shape[1] // B
        if K > 1:
            gt_classes = torch.cat([p.gt_classes for p in proposals], dim=0)
            # Some proposals are ignored or have a background class. Their gt_classes
            # cannot be used as index.
            gt_classes = gt_classes.clamp_(0, K - 1)

            predict_boxes = predict_boxes.view(N, K, B)[torch.arange(
                N, dtype=torch.long, device=predict_boxes.device), gt_classes]
        num_prop_per_image = [len(p) for p in proposals]
        return predict_boxes.split(num_prop_per_image)

    def predict_boxes(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class-specific or class-agnostic boxes
                for each image. Element i has shape (Ri, K * B) or (Ri, B), where Ri is
                the number of predicted objects for image i and B is the box dimension (4 or 5)
        """
        if not len(proposals):
            return []
        _, proposal_deltas, _, _ = predictions
        num_prop_per_image = [len(p) for p in proposals]
        proposal_boxes = [p.proposal_boxes for p in proposals]
        proposal_boxes = proposal_boxes[0].cat(proposal_boxes).tensor
        predict_boxes = self.box2box_transform.apply_deltas(
            proposal_deltas, proposal_boxes
        )  # Nx(KxB)
        return predict_boxes.split(num_prop_per_image)

    def predict_probs(self, predictions, proposals):
        """
        Returns:
            list[Tensor]: A list of Tensors of predicted class probabilities for each image.
                Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
                for image i.
        """
        scores, _, _, _ = predictions
        num_inst_per_image = [len(p) for p in proposals]
        if self.cls_var_loss == "evidential":
            alphas = get_dir_alphas(scores)
            dirichlet_s = alphas.sum(1).unsqueeze(1)
            # Compute probabilities
            probs = alphas / dirichlet_s
        else:
            probs = F.softmax(scores, dim=-1)
        return probs.split(num_inst_per_image, dim=0)

def prob_fast_rcnn_inference(
    boxes: List[torch.Tensor],
    scores: List[torch.Tensor],
    box_covs: List[torch.Tensor],
    score_covs: List[torch.Tensor],
    image_shapes: List[Tuple[int, int]],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    prob_iou,
    select_iou,
):
    """
    Call `fast_rcnn_inference_single_image` for all images.

    Args:
        boxes (list[Tensor]): A list of Tensors of predicted class-specific or class-agnostic
            boxes for each image. Element i has shape (Ri, K * 4) if doing
            class-specific regression, or (Ri, 4) if doing class-agnostic
            regression, where Ri is the number of predicted objects for image i.
            This is compatible with the output of :meth:`FastRCNNOutputLayers.predict_boxes`.
        scores (list[Tensor]): A list of Tensors of predicted class scores for each image.
            Element i has shape (Ri, K + 1), where Ri is the number of predicted objects
            for image i. Compatible with the output of :meth:`FastRCNNOutputLayers.predict_probs`.
        image_shapes (list[tuple]): A list of (width, height) tuples for each image in the batch.
        score_thresh (float): Only return detections with a confidence score exceeding this
            threshold.
        nms_thresh (float):  The threshold to use for box non-maximum suppression. Value in [0, 1].
        topk_per_image (int): The number of top scoring detections to return. Set < 0 to return
            all detections.

    Returns:
        instances: (list[Instances]): A list of N instances, one for each image in the batch,
            that stores the topk most confidence detections.
        kept_indices: (list[Tensor]): A list of 1D tensor of length of N, each element indicates
            the corresponding boxes/scores index in [0, Ri) from the input, for image i.
    """
    result_per_image = [
        prob_fast_rcnn_inference_single_image(
            scores_per_image, boxes_per_image, score_cov_img, box_cov_img, image_shape, score_thresh, nms_thresh, topk_per_image, prob_iou=prob_iou, select_iou=select_iou
        )
        for scores_per_image, boxes_per_image, score_cov_img, box_cov_img, image_shape in zip(scores, boxes, score_covs, box_covs, image_shapes)
    ]
    return [x[0] for x in result_per_image], [x[1] for x in result_per_image]


def prob_fast_rcnn_inference_single_image(
    scores,
    boxes,
    score_covs,
    box_covs,
    image_shape: Tuple[int, int],
    score_thresh: float,
    nms_thresh: float,
    topk_per_image: int,
    prob_iou,
    select_iou,
):
    """
    Single-image inference. Return bounding-box detection results by thresholding
    on scores and applying non-maximum suppression (NMS).

    Args:
        Same as `fast_rcnn_inference`, but with boxes, scores, and image shapes
        per image.

    Returns:
        Same as `fast_rcnn_inference`, but for only one image.
    """
    valid_mask = torch.isfinite(boxes).all(dim=1) & torch.isfinite(scores).all(dim=1)
    if not valid_mask.all():
        boxes = boxes[valid_mask]
        scores = scores[valid_mask]
        box_covs = box_covs[valid_mask]
        score_covs = score_covs[valid_mask]

    scores = scores[:, :-1]
    num_bbox_reg_classes = boxes.shape[1] // 4
    # Convert to Boxes to use the `clip` function ...
    boxes = Boxes(boxes.reshape(-1, 4))
    boxes.clip(image_shape)
    boxes = boxes.tensor.view(-1, num_bbox_reg_classes, 4)  # R x C x 4

    # 1. Filter results based on detection scores. It can make NMS more efficient
    #    by filtering out low-confidence detections.
    filter_mask = scores > score_thresh  # R x K
    # R' x 2. First column contains indices of the R predictions;
    # Second column contains indices of classes.
    filter_inds = filter_mask.nonzero()
    if num_bbox_reg_classes == 1:
        boxes = boxes[filter_inds[:, 0], 0]
    else:
        boxes = boxes[filter_mask]
    scores = scores[filter_mask]
    # self_iou = intersect_self(boxes, box_covs[keep])



    # 2. Apply NMS for each class independently.
    keep = batched_nms(boxes, scores, filter_inds[:, 1], nms_thresh)
    if topk_per_image >= 0:
        keep = keep[:topk_per_image]
    boxes, scores, filter_inds = boxes[keep], scores[keep], filter_inds[keep]

    result = Instances(image_shape)
    result.pred_boxes = Boxes(boxes)
    result.scores = scores
    result.pred_classes = filter_inds[:, 1]

    if not (box_covs < 0).any():
        box_covs = box_covs.view(-1,num_bbox_reg_classes, 4)
        box_covs = box_covs[filter_mask]
        result.pred_box_covs = box_covs[keep]
        if prob_iou and keep.shape[0]:
            result.iou = intersect_self(result.pred_boxes.tensor, box_covs[keep])
        else:
            result.iou = -torch.ones_like(scores)
    else:
        result.pred_box_covs = -torch.ones_like(boxes)
    if not (score_covs < 0).any() and score_covs.dim() > 1:
        score_covs = score_covs[filter_mask]
    else:
        result.pred_score_covs = -torch.ones_like(scores)

    return result, filter_inds[:, 0]

def intersect_self(dt_bbox, bbox_cov):
    bbox_dif = Box2BoxTransform((10.0, 10.0, 5.0, 5.0))

    with torch.no_grad():
        n = 100
        mean_IoU = torch.zeros(dt_bbox.shape[0]).cuda()
        bbox_dist = torch.distributions.normal.Normal(torch.zeros(4).cuda(), bbox_cov**0.5)
        sample_set = bbox_dist.sample((n,)).transpose(0,1).reshape(dt_bbox.shape[0],-1)
        prob_bboxes = bbox_dif.apply_deltas(sample_set, dt_bbox)
        for i in range(dt_bbox.shape[0]):
            mean_IoU[i] = pairwise_iou(Boxes(prob_bboxes[i,:].reshape(n,4)), Boxes(dt_bbox[i,:].unsqueeze(0))).mean()

    return mean_IoU

class ProjectionHead(nn.Module):
    def __init__(self, feat_dim=1024, use_proj=True, normed=True):
        super(ProjectionHead, self).__init__()
        self.normed = normed
        self.use_proj = use_proj
        if use_proj:
            self.head = nn.Sequential(nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), nn.Linear(feat_dim, feat_dim))
    
    def forward(self, x):
        if self.use_proj:
            out = self.head(x)
        else:
            out = x
        if self.normed:
            normed_out = nn.functional.normalize(out)
            return normed_out
        else:
            return out
        
class Cov2IoUHead(nn.Module):
    def __init__(self, in_dim=8, feat_dim=50, depth=5):
        super(ProjectionHead, self).__init__()
        inner_stack = [nn.Linear(feat_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU()]*(depth-2)
        self.head = nn.Sequential(nn.Linear(in_dim, feat_dim), nn.BatchNorm1d(feat_dim), nn.ReLU(), *inner_stack, nn.Linear(feat_dim, 1))
    
    def forward(self, x):
        out = nn.Sigmoid(self.head(x))
        return out

class NormedLinear(nn.Module):
    def __init__(self, in_dim=1024, out_dim=9):
        super(NormedLinear, self).__init__()
        self.linear = Linear(in_dim, out_dim)
    
    def forward(self, x):
        out = self.linear(x)
        return nn.functional.normalize(out)
    

# import matplotlib.pyplot as plt

# pool_features_gt = self.box_pooler(features, [x.proposal_boxes for x in proposals_gt[0]])
# box_features_gt = self.box_head(pool_features_gt)
# if self.box_predictor.compute_bbox_cov and branch == 'supervised':
#     predictions = self.box_predictor(box_features_gt, proposals=proposals_gt)
# else:
#     predictions = self.box_predictor(box_features_gt)




# v2 = 666/20
# v1 = 1333/41
# new_boxes = torch.zeros(41,20,4)
# for i in range(41):
#     for j in range(20):
#         new_boxes[i,j,:] = torch.tensor([i*v1, j*v2, (i+1)*v1, (j+1)*v2])

# box_features2 = self.box_pooler([features[0]], [Boxes(new_boxes.reshape(820,4).cuda())])
# box_features3 = self.box_head(box_features2)
# predictions2 = self.box_predictor(box_features3)
# predictions3 = torch.softmax(predictions2[0][:,:8], dim=1)
# im_cls = predictions3.reshape(41,20,8).transpose(0,2).transpose(1,2).unsqueeze(0).cpu()
# im_big = F.interpolate(im_cls, size=(1333, 667), mode='bicubic', align_corners=False).squeeze().transpose(0,2)

# im_test = im_big[:,:,2]
# plt.figure();plt.imshow(im_test);plt.show()


# import numpy as np
# import matplotlib.pyplot as plt

# # f1 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/props_v2_nom.csv'
# # f2 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/props_v2_nom_adv010.csv'

# # val1 = np.genfromtxt(f1, delimiter=',')
# # val2 = np.genfromtxt(f2, delimiter=',')

# # plt.figure();plt.plot(val1[:,1],val1[:,2]);plt.plot(val2[:,1],val2[:,2])#;plt.plot(val3[:,1],val3[:,2]);
# # plt.xlabel('Epoch');plt.ylabel('N pos anchors');plt.legend(['Baseline','Inst. Adv.']);plt.tight_layout()

# f1 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/ap50_v2_nom.csv'
# # f2 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/ap50_v2_nom_adv010_teacher.csv'
# # f3 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/ap50_v2_nom_adv010_student.csv'
# f2 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/ap50_v2_iou70_min30.csv'
# f3 = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/ap50_v2_iou70_scale1.csv'

# val1 = np.genfromtxt(f1, delimiter=',')
# val2 = np.genfromtxt(f2, delimiter=',')
# val3 = np.genfromtxt(f3, delimiter=',')

# plt.figure();plt.plot(val1[:,1],val1[:,2]);plt.plot(val2[:,1],val2[:,2]);plt.plot(val3[:,1],val3[:,2]);
# plt.xlabel('Epoch');plt.ylabel('AP50');plt.legend(['Baseline','IoU','IoU + Scalet']);plt.tight_layout()

# plt.show()