# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
from detectron2.config import configurable
# from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY
# from detectron2.modeling.meta_arch.rcnn import GeneralizedRCNN
import logging
from typing import Dict, Tuple, List, Optional
from collections import OrderedDict
from detectron2.modeling.proposal_generator import build_proposal_generator
from detectron2.modeling.backbone import build_backbone, Backbone
from detectron2.modeling.roi_heads import build_roi_heads
from detectron2.utils.events import get_event_storage
from detectron2.structures import ImageList, Instances
# from adapteacher.modeling.prob_rcnn import ProbabilisticFastRCNNOutputLayers

############### Image discriminator ##############
class FCDiscriminator_img(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_img, self).__init__()

        self.conv1 = nn.Conv2d(num_classes, ndf1, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(ndf1, ndf2, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(ndf2, ndf2, kernel_size=3, padding=1)
        self.classifier = nn.Conv2d(ndf2, 1, kernel_size=3, padding=1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

############### Instance discriminator ##############
class FCDiscriminator_inst(nn.Module):
    def __init__(self, num_classes, ndf1=256, ndf2=128):
        super(FCDiscriminator_inst, self).__init__()

        self.lin1 = nn.Linear(num_classes, ndf1)
        self.lin2 = nn.Linear(ndf1, ndf2)
        self.classifier = nn.Linear(ndf2, 1)

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x = self.lin1(x)
        x = self.leaky_relu(x)
        x = self.lin2(x)
        x = self.leaky_relu(x)
        x = self.classifier(x)
        return x
#################################

################ Gradient reverse function
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()

def grad_reverse(x):
    return GradReverse.apply(x)

#######################

@META_ARCH_REGISTRY.register()
class DAobjTwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        align_proposals: bool,
        # dis_loss_weight: float = 0,
        proj_type: str,
        dino_out_dim: int,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        if proj_type == 'integrated':
            self.integrated_proj = True
            in_dims = self.backbone.output_shape()[dis_type].channels
            self.proj_layer = nn.Sequential(nn.ReLU(), nn.Conv2d(in_dims, dino_out_dim, 1, 1))
        else:
            self.integrated_proj = False

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        if self.integrated_proj:
            self.D_img = FCDiscriminator_img(dino_out_dim)
        else:
            self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()


        if align_proposals:
            self.roi_heads.align_proposals = True
        else:
            self.roi_heads.align_proposals = False



    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        dino_out_dim_dict = {'dinov2_vits14':384,'dinov2_vitb14':768,'dinov2_vitl14':1024,'dinov2_vitg14':1536}
        if cfg.SEMISUPNET.DINO_HEAD == 'integrated':
            dino_size = dino_out_dim_dict[cfg.SEMISUPNET.DINO_MODEL]
            feat_key = list(backbone.output_shape().keys())[-1]
            dino_spec = {feat_key: ShapeSpec(channels = dino_size , stride=16)}
            proposal_generator = build_proposal_generator(cfg, dino_spec)
            roi_heads = build_roi_heads(cfg, dino_spec)
        else:
            proposal_generator = build_proposal_generator(cfg, backbone.output_shape())
            roi_heads = build_roi_heads(cfg, backbone.output_shape())
        return {
            "backbone": backbone,
            "proposal_generator": proposal_generator,
            "roi_heads": roi_heads,
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
            "align_proposals": cfg.SEMISUPNET.ALIGN_PROPOSALS,
            "proj_type": cfg.SEMISUPNET.DINO_HEAD,
            "dino_out_dim": dino_out_dim_dict[cfg.SEMISUPNET.DINO_MODEL]
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, backbone_only=False,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if backbone_only:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            return features
        elif (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            if 0:
                pass
            else:
                # self.D_img.train()
                # source_label = 0
                # target_label = 1
                # images = self.preprocess_image(batched_inputs)
                images_s, images_t = self.preprocess_image_train(batched_inputs)

                features = self.backbone(images_s.tensor)
                if self.integrated_proj:
                    new_feat = self.proj_layer(features[self.dis_type])
                    features_s = grad_reverse(new_feat)
                else:
                    features_s = grad_reverse(features[self.dis_type])

                # import pdb
                # pdb.set_trace()
            
                D_img_out_s = self.D_img(features_s)
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

                features_t = self.backbone(images_t.tensor)
                if self.integrated_proj:
                    new_feat_t = self.proj_layer(features_t[self.dis_type])
                    features_t = grad_reverse(new_feat_t)
                else:
                    features_t = grad_reverse(features_t[self.dis_type])
                # features_t = grad_reverse(features_t['p2'])
                D_img_out_t = self.D_img(features_t)
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

                # import pdb
                # pdb.set_trace()

                losses = {}
                losses["loss_D_img_s"] = loss_D_img_s
                losses["loss_D_img_t"] = loss_D_img_t
                return losses, [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features_ = self.backbone(images.tensor)
        features = {}
        if self.integrated_proj:
            features[self.dis_type] = self.proj_layer(features_[self.dis_type])
        else:
            features[self.dis_type] = features_[self.dis_type]
        # features = self.backbone(images.tensor)
        # if self.integrated_proj:
        #     features[self.dis_type] = self.proj_layer(features[self.dis_type])

        # if self.dis_type == 'res4':
        #     features = {key: features[key]/1000 for key in features.keys()}

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            #print([x.item() for x in proposal_losses.values()])
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator( 
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features_ = self.backbone(images.tensor)
        features = {}
        if self.integrated_proj:
            features[self.dis_type] = self.proj_layer(features_[self.dis_type])
        else:
            features[self.dis_type] = features_[self.dis_type]

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]

            results, _ = self.roi_heads(images, features, proposals, None)
        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results


@META_ARCH_REGISTRY.register()
class TwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):
    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False
    ):
        if (not self.training) and (not val_mode):
            return self.inference(batched_inputs)

        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)

        if branch == "supervised":
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images, features, proposals_rpn, gt_instances, branch=branch
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator(
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)  # notice that we do not use any target in ROI head to do inference !
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            return {}, proposals_rpn, proposals_roih, ROI_predictions

        elif branch == "val_loss":

            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances, compute_val_loss=True
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                gt_instances,
                branch=branch,
                compute_val_loss=True,
            )

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            return losses, [], [], None


@META_ARCH_REGISTRY.register()
class ProbDATwoStagePseudoLabGeneralizedRCNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        bbox_cov_loss: str,
        bbox_cov_num_samples: int = 4,
        bbox_cov_type: str,
        cfg,
        prob_iou,
        select_iou,
        use_gt_proposals,
        use_gt_proposals_only,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.iter = 0

        # self.bbox_cov_loss = bbox_cov_loss
        # self.compute_bbox_cov = self.bbox_cov_loss != 'none'
        # self.bbox_cov_num_samples = bbox_cov_num_samples
        # self.bbox_cov_type = bbox_cov_type
        # if bbox_cov_type == 'diagonal':
        #     self.bbox_cov_dims = 4
        # else:
        #     self.bbox_cov_dims = 10

        # # Define custom probabilistic head
        # self.roi_heads.box_predictor = ProbabilisticFastRCNNOutputLayers(
        #     cfg,
        #     self.roi_heads.box_head.output_shape,
        #     False,
        #     None,
        #     None,
        #     self.compute_bbox_cov,
        #     self.bbox_cov_loss,
        #     self.bbox_cov_type,
        #     self.bbox_cov_dims,
        #     self.bbox_cov_num_samples)


        # Send to device
        # self.to(self.device)

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()

        self.prob_iou = prob_iou
        self.select_iou = select_iou
        self.test_with_gt_prop = False

        self.roi_heads.use_gt_proposals = use_gt_proposals
        self.roi_heads.use_gt_proposals_only = use_gt_proposals_only

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            "bbox_cov_loss": cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NAME,
            "bbox_cov_num_samples": cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.NUM_SAMPLES,
            "bbox_cov_type": cfg.MODEL.PROBABILISTIC_MODELING.BBOX_COV_LOSS.COVARIANCE_TYPE,
            "cfg": cfg,
            "prob_iou": cfg.MODEL.PROBABILISTIC_MODELING.PROB_IOU,
            "select_iou": cfg.MODEL.PROBABILISTIC_MODELING.SELECT_IOU2,
            "use_gt_proposals": cfg.SEMISUPNET.USE_GT_PROPOSALS,
            "use_gt_proposals_only": cfg.SEMISUPNET.USE_GT_PROPOSALS_ONLY,
            # "dis_loss_ratio": cfg.xxx,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, use_gt_only=False
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs, gt_proposals=self.test_with_gt_prop)

        source_label = 0
        target_label = 1

        if branch == "domain":
            # self.D_img.train()
            # source_label = 0
            # target_label = 1
            # images = self.preprocess_image(batched_inputs)
            images_s, images_t = self.preprocess_image_train(batched_inputs)

            features = self.backbone(images_s.tensor)

            # import pdb
            # pdb.set_trace()
           
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            features_t = self.backbone(images_t.tensor)
            
            features_t = grad_reverse(features_t[self.dis_type])
            # features_t = grad_reverse(features_t['p2'])
            D_img_out_t = self.D_img(features_t)
            loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            # import pdb
            # pdb.set_trace()

            losses = {}
            losses["loss_D_img_s"] = loss_D_img_s
            losses["loss_D_img_t"] = loss_D_img_t
            return losses, [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        if "instances_gt" in batched_inputs[0]:
            gt_instances_fixed = [x["instances_gt"].to(self.device) for x in batched_inputs]
        else:
            gt_instances_fixed = None

        features = self.backbone(images.tensor)

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            self.iter += 1
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )
            # roi_head lower branch
            proposals_roih, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
                unsup=False,
                current_step=self.iter,
                targets_gt=gt_instances_fixed,
            )
            self.proposals_rpn_temp = proposals_rpn
            self.proposals_roih_temp = proposals_roih

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
                unsup=True,
                targets_gt=gt_instances_fixed,
                use_gt_only=use_gt_only,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            if 1:
                proposals_rpn, _ = self.proposal_generator( 
                    images, features, None, compute_loss=False
                )
            else:
                proposals_rpn, _, self.scores, self.keeps = self.proposal_generator( 
                    images, features, None, compute_loss=False
                )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
                unsup=True,
                # prob_iou=self.prob_iou,
                # select_iou=self.select_iou,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch


    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        gt_proposals: bool = False,
    ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        # gt_proposals = True

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if gt_proposals:
            proposals = [x['instances'] for x in batched_inputs]
            if type(features) == dict:
                features = [features]
            for img, feat in zip(proposals, features):
                img.proposal_boxes = img.gt_boxes.to(device=feat['vgg0'].device)
                img.objectness_logits = torch.ones_like(img.gt_classes).to(device=feat['vgg0'].device)
            results, _ = self.roi_heads(images, features[0], proposals, None)

        elif detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)

        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results
    
from adapteacher.engine.dino_extractor import DinoV2VitFeatureExtractor
from detectron2.layers import ShapeSpec
@META_ARCH_REGISTRY.register()
class DINOgenRCNNN(GeneralizedRCNN):

    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        align_proposals: bool,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()


        if align_proposals:
            self.roi_heads.align_proposals = True
        else:
            self.roi_heads.align_proposals = False

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = DinoV2VitFeatureExtractor_wrapper(cfg)
        return {
            "backbone": backbone,
            "proposal_generator": build_proposal_generator(cfg, backbone.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
            "align_proposals": cfg.SEMISUPNET.ALIGN_PROPOSALS,
        }

    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, backbone_only=False,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if backbone_only:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            return features
        elif (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            if 0:
                pass
            else:
                # self.D_img.train()
                # source_label = 0
                # target_label = 1
                # images = self.preprocess_image(batched_inputs)
                images_s, images_t = self.preprocess_image_train(batched_inputs)

                features = self.backbone(images_s.tensor)

                # import pdb
                # pdb.set_trace()
            
                features_s = grad_reverse(features[self.dis_type])
                D_img_out_s = self.D_img(features_s)
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

                features_t = self.backbone(images_t.tensor)
                
                features_t = grad_reverse(features_t[self.dis_type])
                # features_t = grad_reverse(features_t['p2'])
                D_img_out_t = self.D_img(features_t)
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

                # import pdb
                # pdb.set_trace()

                losses = {}
                losses["loss_D_img_s"] = loss_D_img_s
                losses["loss_D_img_t"] = loss_D_img_t
                return losses, [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        # if self.dis_type == 'res4':
        #     features = {key: features[key]/1000 for key in features.keys()}

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator( 
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()

    def visualize_training(self, batched_inputs, proposals, branch=""):
        """
        This function different from the original one:
        - it adds "branch" to the `vis_name`.

        A function used to visualize images and proposals. It shows ground truth
        bounding boxes on the original image and up to 20 predicted object
        proposals on the original image. Users can implement different
        visualization functions for different models.

        Args:
            batched_inputs (list): a list that contains input to the model.
            proposals (list): a list that contains predicted proposals. Both
                batched_inputs and proposals should have the same length.
        """
        from detectron2.utils.visualizer import Visualizer

        storage = get_event_storage()
        max_vis_prop = 20

        for input, prop in zip(batched_inputs, proposals):
            img = input["image"]
            img = convert_image_to_rgb(img.permute(1, 2, 0), self.input_format)
            v_gt = Visualizer(img, None)
            v_gt = v_gt.overlay_instances(boxes=input["instances"].gt_boxes)
            anno_img = v_gt.get_image()
            box_size = min(len(prop.proposal_boxes), max_vis_prop)
            v_pred = Visualizer(img, None)
            v_pred = v_pred.overlay_instances(
                boxes=prop.proposal_boxes[0:box_size].tensor.cpu().numpy()
            )
            prop_img = v_pred.get_image()
            vis_img = np.concatenate((anno_img, prop_img), axis=1)
            vis_img = vis_img.transpose(2, 0, 1)
            vis_name = (
                "Left: GT bounding boxes "
                + branch
                + ";  Right: Predicted proposals "
                + branch
            )
            storage.put_image(vis_name, vis_img)
            break  # only visualize one image in a batch

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        gt_proposals: bool = False,
        ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        # gt_proposals = True

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)

        if gt_proposals:
            proposals = [x['instances'] for x in batched_inputs]
            if type(features) == dict:
                features = [features]
            for img, feat in zip(proposals, features):
                img.proposal_boxes = img.gt_boxes.to(device=feat['feat_dino'].device)
                img.objectness_logits = torch.ones_like(img.gt_classes).to(device=feat['feat_dino'].device)
            results, _ = self.roi_heads(images, features[0], proposals, None)

        elif detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)

        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results
    
class DinoV2VitFeatureExtractor_wrapper(DinoV2VitFeatureExtractor):
    def __init__(self, cfg, output_layer='dino_out'):
        super(DinoV2VitFeatureExtractor_wrapper, self).__init__(cfg, model_name=cfg.SEMISUPNET.DINO_MODEL, normalize_feature=False, freeze=True)
        self.output_layer = output_layer
        self._out_feature_channels = {self.output_layer:self.encoder.blocks[-1].norm2.bias.shape[0]}
        self._out_feature_strides = {self.output_layer:self.patch_size}
        self.size_divisibility = 0
        self.padding_constraints = {}
    
    def output_shape(self):
        output = ShapeSpec(channels = self._out_feature_channels[self.output_layer], stride=self._out_feature_strides[self.output_layer])
        return {self.output_layer: output}

    def forward(self, x):
        x = x[:,[2,1,0],:,:]
        batch_size, _, height, width = x.size()
        # check image dims divisible by patch_size
        assert (height % self.patch_size) == 0
        assert (width % self.patch_size) == 0
        f_height = height // self.patch_size
        f_width = width // self.patch_size

        x = self.encoder.get_intermediate_layers(x)[0] # batch_size, num_patches, self.embed_dim
        if "v2" not in self.model_name:
            x = x[:,1:,:] # remove class token

        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=2)

        x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        return {self.output_layer: x_grid_features}

@META_ARCH_REGISTRY.register()
class TwoTrunksRCNNN(GeneralizedRCNN):
    @configurable
    def __init__(
        self,
        *,
        backbone: Backbone,
        backbone_dino: Backbone,
        backbone_proj_head: nn.Module,
        proposal_generator: nn.Module,
        roi_heads: nn.Module,
        pixel_mean: Tuple[float],
        pixel_std: Tuple[float],
        input_format: Optional[str] = None,
        vis_period: int = 0,
        dis_type: str,
        align_proposals: bool,
        # dis_loss_weight: float = 0,
    ):
        """
        Args:
            backbone: a backbone module, must follow detectron2's backbone interface
            proposal_generator: a module that generates proposals using backbone features
            roi_heads: a ROI head that performs per-region computation
            pixel_mean, pixel_std: list or tuple with #channels element, representing
                the per-channel mean and std to be used to normalize the input image
            input_format: describe the meaning of channels of input. Needed by visualization
            vis_period: the period to run visualization. Set to 0 to disable.
        """
        super(GeneralizedRCNN, self).__init__()
        self.backbone = backbone
        self.backbone_proj_head = backbone_proj_head
        self.backbone_dino = backbone_dino.eval()
        self.proposal_generator = proposal_generator
        self.roi_heads = roi_heads

        self.input_format = input_format
        self.vis_period = vis_period
        if vis_period > 0:
            assert input_format is not None, "input_format is required for visualization!"

        self.register_buffer("pixel_mean", torch.tensor(pixel_mean).view(-1, 1, 1), False)
        self.register_buffer("pixel_std", torch.tensor(pixel_std).view(-1, 1, 1), False)
        assert (
            self.pixel_mean.shape == self.pixel_std.shape
        ), f"{self.pixel_mean} and {self.pixel_std} have different shapes!"
        # @yujheli: you may need to build your discriminator here

        self.dis_type = dis_type
        self.D_img = None
        # self.D_img = FCDiscriminator_img(self.backbone._out_feature_channels['res4']) # Need to know the channel
        
        # self.D_img = None
        self.D_img = FCDiscriminator_img(self.backbone_dino._out_feature_channels[self.dis_type]) # Need to know the channel
        # self.bceLoss_func = nn.BCEWithLogitsLoss()


        if align_proposals:
            self.roi_heads.align_proposals = True
        else:
            self.roi_heads.align_proposals = False

    def build_discriminator(self):
        self.D_img = FCDiscriminator_img(self.backbone_dino._out_feature_channels[self.dis_type]).to(self.device) # Need to know the channel

    @classmethod
    def from_config(cls, cfg):
        backbone = build_backbone(cfg)
        backbone_dino = DinoV2VitFeatureExtractor_wrapper(cfg, output_layer=cfg.MODEL.RPN.IN_FEATURES[-1])
        backone_proj_head = ProjBackbone(cfg,backbone,dino_dim=768)
        return {
            "backbone": backbone,
            "backbone_dino": backbone_dino,
            "backbone_proj_head": backone_proj_head,
            "proposal_generator": build_proposal_generator(cfg, backbone_dino.output_shape()),
            "roi_heads": build_roi_heads(cfg, backbone_dino.output_shape()),
            "input_format": cfg.INPUT.FORMAT,
            "vis_period": cfg.VIS_PERIOD,
            "pixel_mean": cfg.MODEL.PIXEL_MEAN,
            "pixel_std": cfg.MODEL.PIXEL_STD,
            "dis_type": cfg.SEMISUPNET.DIS_TYPE,
            # "dis_loss_ratio": cfg.xxx,
            "align_proposals": cfg.SEMISUPNET.ALIGN_PROPOSALS,
        }
    
    def preprocess_image_train(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        """
        Normalize, pad and batch the input images.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        images_t = [x["image_unlabeled"].to(self.device) for x in batched_inputs]
        images_t = [(x - self.pixel_mean) / self.pixel_std for x in images_t]
        images_t = ImageList.from_tensors(images_t, self.backbone.size_divisibility)

        return images, images_t

    def forward(
        self, batched_inputs, branch="supervised", given_proposals=None, val_mode=False, backbone_only=False,
    ):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.
                For now, each item in the list is a dict that contains:

                * image: Tensor, image in (C, H, W) format.
                * instances (optional): groundtruth :class:`Instances`
                * proposals (optional): :class:`Instances`, precomputed proposals.

                Other information that's included in the original dicts, such as:

                * "height", "width" (int): the output resolution of the model, used in inference.
                  See :meth:`postprocess` for details.

        Returns:
            list[dict]:
                Each dict is the output for one input image.
                The dict contains one key "instances" whose value is a :class:`Instances`.
                The :class:`Instances` object has the following keys:
                "pred_boxes", "pred_classes", "scores", "pred_masks", "pred_keypoints"
        """
        if self.D_img == None:
            self.build_discriminator()
        if backbone_only:
            images = self.preprocess_image(batched_inputs)
            features = self.backbone(images.tensor)
            return features
        elif (not self.training) and (not val_mode):  # only conduct when testing mode
            return self.inference(batched_inputs)

        source_label = 0
        target_label = 1

        if branch == "domain":
            if 0:
                pass
            else:
                # self.D_img.train()
                # source_label = 0
                # target_label = 1
                # images = self.preprocess_image(batched_inputs)
                images_s, images_t = self.preprocess_image_train(batched_inputs)

                features = self.backbone(images_s.tensor)
                features = {self.dis_type:self.backbone_proj_head(features[self.dis_type])}

                # import pdb
                # pdb.set_trace()
            
                features_s = grad_reverse(features[self.dis_type])
                D_img_out_s = self.D_img(features_s)
                loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

                features_t = self.backbone(images_t.tensor)
                
                features_t = grad_reverse(features_t[self.dis_type])
                # features_t = grad_reverse(features_t['p2'])
                D_img_out_t = self.D_img(features_t)
                loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

                # import pdb
                # pdb.set_trace()

                losses = {}
                losses["loss_D_img_s"] = loss_D_img_s
                losses["loss_D_img_t"] = loss_D_img_t
                return losses, [], [], None

        # self.D_img.eval()
        images = self.preprocess_image(batched_inputs)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        else:
            gt_instances = None

        features = self.backbone(images.tensor)
        features = {self.dis_type:self.backbone_proj_head(features[self.dis_type])}
        # if self.dis_type == 'res4':
        #     features = {key: features[key]/1000 for key in features.keys()}

        # TODO: remove the usage of if else here. This needs to be re-organized
        if branch == "supervised":
            features_s = grad_reverse(features[self.dis_type])
            D_img_out_s = self.D_img(features_s)
            loss_D_img_s = F.binary_cross_entropy_with_logits(D_img_out_s, torch.FloatTensor(D_img_out_s.data.size()).fill_(source_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            features_dino = self.backbone_dino(images.tensor)

            proposals_rpn_d, proposal_losses_d = self.proposal_generator(
                images, features_dino, gt_instances
            )
            proposal_losses_d = {k+'_trunk':v for k,v in proposal_losses_d.items()}


            # roi_head lower branch
            _, detector_losses_d = self.roi_heads(
                images,
                features,
                proposals_rpn_d,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )
            detector_losses_d = {k+'_trunk':v for k,v in detector_losses_d.items()}

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            losses.update(detector_losses_d)
            losses.update(proposal_losses_d)
            losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "supervised_target":

            # features_t = grad_reverse(features_t[self.dis_type])
            # D_img_out_t = self.D_img(features_t)
            # loss_D_img_t = F.binary_cross_entropy_with_logits(D_img_out_t, torch.FloatTensor(D_img_out_t.data.size()).fill_(target_label).to(self.device))

            
            # Region proposal network
            proposals_rpn, proposal_losses = self.proposal_generator(
                images, features, gt_instances
            )

            # roi_head lower branch
            _, detector_losses = self.roi_heads(
                images,
                features,
                proposals_rpn,
                compute_loss=True,
                targets=gt_instances,
                branch=branch,
            )

            # visualization
            if self.vis_period > 0:
                storage = get_event_storage()
                if storage.iter % self.vis_period == 0:
                    self.visualize_training(batched_inputs, proposals_rpn, branch)

            losses = {}
            losses.update(detector_losses)
            losses.update(proposal_losses)
            # losses["loss_D_img_t"] = loss_D_img_t*0.001
            # losses["loss_D_img_s"] = loss_D_img_s*0.001
            return losses, [], [], None

        elif branch == "unsup_data_weak":
            """
            unsupervised weak branch: input image without any ground-truth label; output proposals of rpn and roi-head
            """
            # Region proposal network
            proposals_rpn, _ = self.proposal_generator( 
                images, features, None, compute_loss=False
            )

            # roi_head lower branch (keep this for further production)
            # notice that we do not use any target in ROI head to do inference!
            proposals_roih, ROI_predictions = self.roi_heads(
                images,
                features,
                proposals_rpn,
                targets=None,
                compute_loss=False,
                branch=branch,
            )

            # if self.vis_period > 0:
            #     storage = get_event_storage()
            #     if storage.iter % self.vis_period == 0:
            #         self.visualize_training(batched_inputs, proposals_rpn, branch)

            return {}, proposals_rpn, proposals_roih, ROI_predictions
        elif branch == "unsup_data_strong":
            raise NotImplementedError()
        elif branch == "val_loss":
            raise NotImplementedError()
    
    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
        gt_proposals: bool = False,
        ):
        """
        Run inference on the given inputs.

        Args:
            batched_inputs (list[dict]): same as in :meth:`forward`
            detected_instances (None or list[Instances]): if not None, it
                contains an `Instances` object per image. The `Instances`
                object contains "pred_boxes" and "pred_classes" which are
                known boxes in the image.
                The inference will then skip the detection of bounding boxes,
                and only predict other per-ROI outputs.
            do_postprocess (bool): whether to apply post-processing on the outputs.

        Returns:
            When do_postprocess=True, same as in :meth:`forward`.
            Otherwise, a list[Instances] containing raw network outputs.
        """
        # gt_proposals = True

        assert not self.training

        images = self.preprocess_image(batched_inputs)
        features = self.backbone(images.tensor)
        features = {self.dis_type:self.backbone_proj_head(features[self.dis_type])}

        if detected_instances is None:
            if self.proposal_generator is not None:
                proposals, _ = self.proposal_generator(images, features, None)
            else:
                assert "proposals" in batched_inputs[0]
                proposals = [x["proposals"].to(self.device) for x in batched_inputs]
            results, _ = self.roi_heads(images, features, proposals, None)

        else:
            detected_instances = [x.to(self.device) for x in detected_instances]
            results = self.roi_heads.forward_with_given_boxes(features, detected_instances)

        if do_postprocess:
            assert not torch.jit.is_scripting(), "Scripting is not supported for postprocess."
            return GeneralizedRCNN._postprocess(results, batched_inputs, images.image_sizes)
        return results

def ProjBackbone(cfg,backbone,dino_dim=768):
    cnn_dim = backbone.output_shape()[list(backbone.output_shape().keys())[0]].channels
    if cfg.SEMISUPNET.TRUNK_PROJ == 'MLP':
        proj_dim = cfg.SEMISUPNET.DINO_PROJ_DIM   
        projection_layer = nn.Sequential(nn.Conv2d(cnn_dim, proj_dim, 1, 1),
                                                   nn.ReLU(),
                                                   nn.Conv2d(proj_dim, dino_dim, 1, 1),
                                                   nn.ReLU())
    else:
        projection_layer = nn.Sequential(nn.Conv2d(cnn_dim, dino_dim, 1, 1))
    return projection_layer