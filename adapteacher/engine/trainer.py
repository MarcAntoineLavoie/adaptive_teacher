# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import time
import logging
import torch
from torch.nn.parallel import DistributedDataParallel
from fvcore.nn.precise_bn import get_bn_modules
import numpy as np
from collections import OrderedDict

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.engine import DefaultTrainer, SimpleTrainer, TrainerBase
from detectron2.engine.train_loop import AMPTrainer
from detectron2.utils.events import EventStorage
from detectron2.evaluation import verify_results, DatasetEvaluators
# from detectron2.evaluation import COCOEvaluator, verify_results, DatasetEvaluators

from detectron2.data.dataset_mapper import DatasetMapper
from detectron2.engine import hooks
from detectron2.structures.boxes import Boxes
from detectron2.structures.instances import Instances
from detectron2.utils.env import TORCH_VERSION
from detectron2.data import MetadataCatalog

from adapteacher.data.build import (
    build_detection_semisup_train_loader,
    build_detection_test_loader,
    build_detection_semisup_train_loader_two_crops,
    build_detection_unlabel_train_loader,
)
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate, DatasetMapperWithWeakAugs, DatasetMapperWithStrongAugs, DatasetMapperTwoCropSeparate_detect, DatasetMapper_test
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator

from .probe import OpenMatchTrainerProbe
import copy


from detectron2.evaluation import (
    DatasetEvaluator,
    #inference_on_dataset,
    print_csv_format,
    verify_results,
)



from detectron2.utils.comm import get_world_size
from collections import abc
from contextlib import ExitStack, contextmanager
from torch import nn
import datetime
from detectron2.utils.logger import log_every_n_seconds
import json
from detectron2.data import DatasetCatalog
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.structures.boxes import BoxMode
from detectron2.structures.boxes import pairwise_iou, pairwise_intersection
from math import comb
from adapteacher.modeling.prob_rcnn import ProbROIHeadsPseudoLab

import geomloss
import numpy
import os
from itertools import chain
import csv
# from adapteacher.evaluation.grad_cam import GradCAM
# import cv2

from adapteacher.engine.dino_extractor import DinoV2VitFeatureExtractor, DinoAlignHead
from detectron2.structures.masks import polygons_to_bitmask, BitMasks
from PIL import Image
from fvcore.transforms.transform import PadTransform, HFlipTransform
import wandb
from detectron2.data.dataset_mapper import DatasetMapper
from math import ceil
import pickle

# pedestrian		0,93,192+x
# rider			0,97,170+x
# car			0,101,144+x
# truck			0,105,120+x
# bus			0,109,96+x
# train			0,121,24+x
# motorcycle		0,125,0+x
# bicycle			0,128,238+x

ACDC_SEG_INSTANCE_LABELS = {'person': [0, [0, 93, 192]], 'rider': [1, [0, 97, 170]], 'car':[2, [0, 101, 144]], 'truck': [3, [0, 105, 102]],
                            'bus': [4, [0, 109, 96]], 'train': [5, [0, 121, 24]], 'motorcycle': [6, [0, 125, 0]], 'bicycle': [7, [0, 128, 238]]}
ACDC_PIX2CLASS = {93:0,97:1,101:2,105:3,109:4,121:5,125:6,128:7}

# Supervised-only Trainer
class BaselineTrainer(DefaultTrainer):
    def __init__(self, cfg):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.checkpointer = DetectionCheckpointer(
            model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.register_hooks(self.build_hooks())

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            # if TORCH_VERSION >= (1, 7):
            #     self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    def train_loop(self, start_iter: int, max_iter: int):
        """
        Args:
            start_iter, max_iter (int): See docs above
        """
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()
                for self.iter in range(start_iter, max_iter):
                    self.before_step()
                    self.run_step()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    def run_step(self):
        self._trainer.iter = self.iter

        assert self.model.training, "[SimpleTrainer] model was changed to eval mode!"
        start = time.perf_counter()

        data = next(self._trainer._data_loader_iter)
        data_time = time.perf_counter() - start

        record_dict, _, _, _ = self.model(data, branch="supervised")

        num_gt_bbox = 0.0
        for element in data:
            num_gt_bbox += len(element["instances"])
        num_gt_bbox = num_gt_bbox / len(data)
        record_dict["bbox_num/gt_bboxes"] = num_gt_bbox

        loss_dict = {}
        for key in record_dict.keys():
            if key[:4] == "loss" and key[-3:] != "val":
                loss_dict[key] = record_dict[key]

        losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        return build_detection_semisup_train_loader(cfg, mapper=None)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """
        Returns:
            iterable
        """
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        """
        Build a list of default hooks, including timing, evaluation,
        checkpointing, lr scheduling, precise BN, writing events.

        Returns:
            list[HookBase]:
        """
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                cfg.TEST.EVAL_PERIOD,
                self.model,
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results():
            self._last_eval_results = self.test(self.cfg, self.model)
            return self._last_eval_results

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD, test_and_save_results))

        if comm.is_main_process():
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def _write_metrics(self, metrics_dict: dict):
        """
        Args:
            metrics_dict (dict): dict of scalar metrics
        """
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }
        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)


# Adaptive Teacher Trainer
class ATeacherTrainer(DefaultTrainer):
    def __init__(self, cfg, wandb_run=None):
        """
        Args:
            cfg (CfgNode):
        Use the custom checkpointer, which loads other backbone models
        with matching heuristics.
        """
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())
        data_loader = self.build_train_loader(cfg)

        # create an student model
        model = self.build_model(cfg)

        if cfg.SEMISUPNET.USE_DINO and not cfg.SEMISUPNET.DINO_BASE:
            self.dino_layer = cfg.SEMISUPNET.DIS_TYPE
            self.branch = "supervised"
            self.use_dino = True
            self.cnn_feat = {}
            if "vgg" in cfg.MODEL.BACKBONE.NAME:
                cnn_dim = [*model.backbone.modules()][-3].num_features
            elif 'dino' in cfg.MODEL.BACKBONE.NAME or 'BiCephal' in cfg.MODEL.META_ARCHITECTURE:
                cnn_dim = 768
            else:
                cnn_dim = [*model.backbone.modules()][-1].num_features
            model.dino_head = DinoV2VitFeatureExtractor(cfg, model_name=cfg.SEMISUPNET.DINO_MODEL, normalize_feature=cfg.SEMISUPNET.DINO_LOSS_NORM).eval()
            # model.dino_head = DinoV2VitFeatureExtractor(cfg, cnn_dim, model_name='dinov2_vitb14', normalize_feature=cfg.SEMISUPNET.DINO_LOSS_NORM).eval()
            # model.dino_head = DinoV2VitFeatureExtractor(cfg, cnn_dim, model_name='dino_vitb16', normalize_feature=cfg.SEMISUPNET.DINO_LOSS_NORM).eval()
            # model.dino_head = DinoV2VitFeatureExtractor(cfg, cnn_dim, model_name='dino_vitb8', normalize_feature=cfg.SEMISUPNET.DINO_LOSS_NORM).eval()
            dino_dim = [*model.dino_head.modules()][-2].normalized_shape[0]
            model.dino_align = DinoAlignHead(cfg, cnn_dim, dino_dim, normalize_feature=model.dino_head.normalize_feature)
            self._register_input_hook(model, 'proposal_generator')
            self.dino_loss_weight = cfg.SEMISUPNET.DINO_LOSS_WEIGHT
            self.dino_loss_weight_target = cfg.SEMISUPNET.DINO_LOSS_WEIGHT_TARGET
            model.dino_head = model.dino_head.to((torch.device(cfg.MODEL.DEVICE)))
            model.dino_align = model.dino_align.to((torch.device(cfg.MODEL.DEVICE)))
            self.dino_sam_masks = cfg.SEMISUPNET.DINO_SAM_MASK
            if cfg.SEMISUPNET.DINO_SAM_MASK:
                label_data_mask_file = './datasets/cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train/sam_masks.pkl'
                with open(label_data_mask_file, 'rb') as f_in:
                    self.label_data_masks = pickle.loads(f_in)
                if 'city' in cfg.DATASETS.TRAIN_UNLABEL[0]:
                    split = '_RAW_CITYSCAPES_SPLITS'
                elif 'ACDC' in cfg.DATASETS.TRAIN_UNLABEL[0]:
                    split = '_RAW_ACDC_SPLITS'
                elif 'BDD' in cfg.DATASETS.TRAIN_UNLABEL[0]:
                    split = '_RAW_BDD_SPLITS'
                unlabel_data_mask_file = '/'.join(DatasetCatalog['ACDC_val_rain'].__globals__[split][cfg.DATASETS.TRAIN_UNLABEL[0]], 'sam_masks.pkl')
                with open(unlabel_data_mask_file, 'rb') as f_in:
                    self.unlabel_data_masks = pickle.loads(f_in)                
                
        else:
            self.use_dino = False
        
        if type(cfg.SEMISUPNET.DINO_TARGET_PSEUDOGT) == str:
            self.use_dino_PL = True
            file_in = cfg.SEMISUPNET.DINO_TARGET_PSEUDOGT
            with open(file_in, 'rb') as f_in:
                temp_dict = pickle.load(f_in)
            self.dino_pseudogt = {}
            for img in temp_dict:
                self.dino_pseudogt[img['image_id']] = img
        else:
            self.use_dino_PL = False
        
        self.align_only_iter = cfg.SEMISUPNET.DINO_ALIGN_ONLY_UNTIL

        if type(cfg.SEMISUPNET.DINO_PSEUDOGT_SWAP) == str:
            assert type(cfg.SEMISUPNET.DINO_PSEUDOGT_SWAP_ITER) == int
            self.PL_swap = cfg.SEMISUPNET.DINO_PSEUDOGT_SWAP
            self.PL_swap_iter = cfg.SEMISUPNET.DINO_PSEUDOGT_SWAP_ITER
        else:
            self.PL_swap = None

        self.easy_dino_only = cfg.SEMISUPNET.DINO_ALIGN_EASY_ONLY       
            
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )
            if self.use_dino:
                model.dino_head = model.module.dino_head
                model.dino_align = model.module.dino_align

        TrainerBase.__init__(self)
        self._trainer = (AMPTrainer if cfg.SOLVER.AMP.ENABLED else SimpleTrainer)(
            model, data_loader, optimizer
        )
        self.scheduler = self.build_lr_scheduler(cfg, optimizer)

        # Ensemble teacher and student model is for model saving and loading
        ensem_ts_model = EnsembleTSModel(model_teacher, model)

        self.checkpointer = DetectionTSCheckpointer(
            ensem_ts_model,
            cfg.OUTPUT_DIR,
            optimizer=optimizer,
            scheduler=self.scheduler,
        )
            # self.checkpointer_best = hooks.BestCheckpointer(
            #     ensem_ts_model,   
            #     cfg.OUTPUT_DIR,
            #     optimizer=optimizer,
            #     scheduler=self.scheduler,
            # )
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg

        self.loss_dict = {}

        self.probe = OpenMatchTrainerProbe(cfg) 
        
        # self.register_hooks(self.build_hooks_final())
        self.register_hooks(self.build_hooks())
        # self.register_hooks(self.build_hooks_final())

        self.prob_iou = cfg.MODEL.PROBABILISTIC_MODELING.PROB_IOU
        self.select_iou = cfg.MODEL.PROBABILISTIC_MODELING.SELECT_IOU
        self.align_proposals = cfg.SEMISUPNET.ALIGN_PROPOSALS
        if self.align_proposals:
            if 'module' in self.model.__dict__['_modules']:
                model_student = self.model.module
            else:
                model_student = self.model

            if cfg.SEMISUPNET.ALIGN_USE_BG:
                n_labels = 9
            else:
                n_labels = 8
            if 'vgg4' in cfg.MODEL.RPN.IN_FEATURES:
                feat_dim = 512
            else:
                feat_dim = 512
            model_student.roi_heads.build_queues(n_classes=n_labels, n_samples=200, feat_dim=feat_dim, 
                        base_count=cfg.SEMISUPNET.ALIGN_BASE_COUNT,
                        # use_prototypes=cfg.SEMISUPNET.ALIGN_PROTOTYPES, only_prototypes=cfg.SEMISUPNET.ALIGN_PROTOTYPES_ONLY,
                        )
            model_student.roi_heads.align_proposals = self.align_proposals
            model_student.roi_heads.current_proposals = {}
            model_student.roi_heads.use_bg = cfg.SEMISUPNET.ALIGN_USE_BG
            model_student.roi_heads.sampling = cfg.SEMISUPNET.ALIGN_SUBSAMPLING
            model_student.roi_heads.points_per_proposals = cfg.SEMISUPNET.ALIGN_POINTS_PER_PROPOSALS
            model_student.roi_heads.align_gt_proposals = cfg.SEMISUPNET.ALIGN_GT_PROPOSALS

            self.proj_head = model_student.roi_heads.box_predictor.proj_head

            self.model_teacher.roi_heads.align_proposals = self.align_proposals
            self.model_teacher.roi_heads.current_proposals = {}

            temperature  = cfg.SEMISUPNET.ALIGN_PARAM
            n_labels = 9
            select = 'all'
            if cfg.SEMISUPNET.ALIGN_LOSS == 'MMD':
                loss_func = geomloss.SamplesLoss(loss='energy')
                self.align_loss = SinkLoss(self.proj_head, loss_func=loss_func, scale=cfg.SEMISUPNET.ALIGN_WEIGHT,
                                            intra_align=cfg.SEMISUPNET.ALIGN_INTRA, use_negatives=cfg.SEMISUPNET.ALIGN_USE_NEGATIVES)
            elif cfg.SEMISUPNET.ALIGN_LOSS == 'contrast':
                self.align_loss = ContrastLoss(self.proj_head, n_labels, select, temperature=temperature, scale=cfg.SEMISUPNET.ALIGN_WEIGHT,
                                                base_temp=cfg.SEMISUPNET.ALIGN_PARAM_BASE, intra_align=cfg.SEMISUPNET.ALIGN_INTRA,
                                                scale_count=cfg.SEMISUPNET.ALIGN_SCALE_COUNT)
        self.use_gt_proposals = cfg.SEMISUPNET.USE_GT_PROPOSALS
        self.use_gt_proposals_only = cfg.SEMISUPNET.USE_GT_PROPOSALS_ONLY
        self.align_gt_proposals = cfg.SEMISUPNET.ALIGN_GT_PROPOSALS
        self.thresh_both = cfg.SEMISUPNET.PSEUDO_THRESH_BOTH
        self.thresh_object = cfg.SEMISUPNET.PSEUDO_OBJ_THRESH


        self.eval_pseudo_labels = cfg.SEMISUPNET.EVAL_PSEUDO_LABELS
        if self.eval_pseudo_labels:
            self.pseudo_subsampling = 1
            classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcylce', 'bicycle', 'all']
            stats = ['gt', 'raw50', 'raw75', 'rawn25', 'sel50', 'sel75', 'seln25']
            self.pseudo_counts = torch.zeros(len(classes)*len(stats))
            self.pseudo_stats_file = ('/').join((cfg.OUTPUT_DIR,'pseudo_stats.csv'))
            if not os.path.isfile(self.pseudo_stats_file):
                columns = ['iter'] + list(chain.from_iterable([['_'.join((x,y)) for y in stats] for x in classes]))
                # header = (', ').join(columns)
                with open(self.pseudo_stats_file, 'w') as f_out:
                    csv_writer = csv.writer(f_out, delimiter=' ')
                    csv_writer.writerow(columns)

        if wandb_run is not None:
            self.log_wandb = True
            self.wandb_run = wandb_run
        else:
            self.log_wandb = False

        # self.target_layer_name = ['backbone.vgg2.8','backbone.vgg3.8','backbone.vgg4.8','proposal_generator.rpn_head.conv']
        # self.activations_grads = []
        # self._register_grad_hook()
        # self.activations = []
        # self.gradient = []

    def _get_activations_hook(self, module, input, output):
        # self.activations = output
        self.activations.append(output)

    def _get_grads_hook(self, module, input_grad, output_grad):
        # self.gradient = output_grad[0]
        self.gradient.append(output_grad[0])

    def _register_grad_hook(self):
        for (name, module) in self.model_teacher.named_modules():
        # for (name, module) in self.model.named_modules():
            # if name == self.target_layer_name:
            if name in self.target_layer_name:
                self.activations_grads.append(module.register_forward_hook(self._get_activations_hook))
                self.activations_grads.append(module.register_backward_hook(self._get_grads_hook))
        return True
        print(f"Layer {self.target_layer_name} not found in Model!")
    
    def _get_rcnn_input_hook(self, module, input, output):
        self.cnn_feat[self.branch] = input[1][self.cfg.MODEL.RPN.IN_FEATURES[0]]

    def _register_input_hook(self, model, target_layer):
        for (name, module) in model.named_modules():
            if name == target_layer:
                module.register_forward_hook(self._get_rcnn_input_hook)
        return True

    def _postprocess_cam(self, raw_cam, img_width, img_height):
        cam_orig = np.sum(raw_cam, axis=0)  # [H,W]
        cam_orig = np.maximum(cam_orig, 0)  # ReLU
        cam_orig -= np.min(cam_orig)
        cam_orig /= np.max(cam_orig)
        cam = cv2.resize(cam_orig, (img_width, img_height))
        return cam, cam_orig

    def resume_or_load(self, resume=True):
        """
        If `resume==True` and `cfg.OUTPUT_DIR` contains the last checkpoint (defined by
        a `last_checkpoint` file), resume from the file. Resuming means loading all
        available states (eg. optimizer and scheduler) and update iteration counter
        from the checkpoint. ``cfg.MODEL.WEIGHTS`` will not be used.
        Otherwise, this is considered as an independent training. The method will load model
        weights from the file `cfg.MODEL.WEIGHTS` (but will not load other states) and start
        from iteration 0.
        Args:
            resume (bool): whether to do resume or not
        """
        checkpoint = self.checkpointer.resume_or_load(
            self.cfg.MODEL.WEIGHTS, resume=resume
        )
        # if self.model.dis_type == 'res4':
        #     self.model.backbone.stem.weight /= 1000
        if resume and self.checkpointer.has_checkpoint():
            self.start_iter = checkpoint.get("iteration", -1) + 1
            # The checkpoint stores the training iteration that just finished, thus we start
            # at the next iteration (or iter zero if there's no checkpoint).
        if isinstance(self.model, DistributedDataParallel):
            # broadcast loaded data/model from the first rank, because other
            # machines may not have access to the checkpoint file
            # if TORCH_VERSION >= (1, 7):
            #     self.model._sync_params_and_buffers()
            self.start_iter = comm.all_gather(self.start_iter)[0]

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None, allow_cached=True):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            # use_prob = True if cfg.MODEL.META_ARCHITECTURE == 'ProbDATwoStagePseudoLabGeneralizedRCNN' else False
            use_prob = False
            # allow_cached = False
            evaluator_list.append(COCOEvaluator(
                dataset_name, output_dir=output_folder, allow_cached=allow_cached, use_prob=use_prob))
        elif evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)
        elif evaluator_type == "pascal_voc_water":
            return PascalVOCDetectionEvaluator(dataset_name, target_classnames=["bicycle", "bird", "car", "cat", "dog", "person"])
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]

        return DatasetEvaluators(evaluator_list)

    @classmethod
    def build_train_loader(cls, cfg):
        # mapper = DatasetMapperTwoCropSeparate(cfg, True)
        mapper = DatasetMapperTwoCropSeparate_detect(cfg, True, keep_tf_data=True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
        # self.test_DINO()
        self.train_loop(self.start_iter, self.max_iter)
        if hasattr(self, "_last_eval_results") and comm.is_main_process():
            verify_results(self.cfg, self._last_eval_results)
            return self._last_eval_results

    def train_loop(self, start_iter: int, max_iter: int):
        logger = logging.getLogger(__name__)
        logger.info("Starting training from iteration {}".format(start_iter))

        self.iter = self.start_iter = start_iter
        self.max_iter = max_iter

        with EventStorage(start_iter) as self.storage:
            try:
                self.before_train()

                for self.iter in range(start_iter, max_iter):
                    # self.model.iter = self.iter
                    self.before_step()
                    self.run_step_full_semisup()
                    self.after_step()
            except Exception:
                logger.exception("Exception during training:")
                raise
            finally:
                self.after_train()

    # =====================================================
    # ================== Pseduo-labeling ==================
    # =====================================================
    def threshold_bbox(self, proposal_bbox_inst, thres=0.7, proposal_type="roih"):
        if proposal_type == "rpn":
            valid_map = proposal_bbox_inst.objectness_logits > thres

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.proposal_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.objectness_logits = proposal_bbox_inst.objectness_logits[
                valid_map
            ]
        elif proposal_type == "roih":
            if self.thresh_both:
                valid_map = torch.logical_and(proposal_bbox_inst.scores > thres, proposal_bbox_inst.rpn_score > self.thresh_object)
            else:
                valid_map = proposal_bbox_inst.scores > thres
            # valid_map2 = (proposal_bbox_inst.iou > thres) * (proposal_bbox_inst.rpn_score > 0.3)
            # overlap = sum(valid_map==valid_map2)**2/(sum(valid_map)*sum(valid_map))

            # create instances containing boxes and gt_classes
            image_shape = proposal_bbox_inst.image_size
            new_proposal_inst = Instances(image_shape)

            # create box
            new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
            new_boxes = Boxes(new_bbox_loc)

            # add boxes to instances
            new_proposal_inst.gt_boxes = new_boxes
            new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
            new_proposal_inst.gt_scores = proposal_bbox_inst.scores[valid_map]
            if 'iou' in proposal_bbox_inst._fields.keys():
                new_proposal_inst.gt_iou = proposal_bbox_inst.iou[valid_map]

        elif proposal_type == "dino":
            valid_map = proposal_bbox_inst.gt_scores > thres
            new_proposal_inst = proposal_bbox_inst[valid_map]
        return new_proposal_inst, 0
    
    def threshold_self(self, proposal_bbox_inst, thres=0.7):
        valid_map = (proposal_bbox_inst.iou > thres) * (proposal_bbox_inst.rpn_score > 0.3)
        valid_map2 = proposal_bbox_inst.scores > 0.8
        overlap = sum(torch.logical_and(valid_map, valid_map2))/(len(proposal_bbox_inst)+1e-12)

        # create instances containing boxes and gt_classes
        image_shape = proposal_bbox_inst.image_size
        new_proposal_inst = Instances(image_shape)

        # create box
        new_bbox_loc = proposal_bbox_inst.pred_boxes.tensor[valid_map, :]
        new_boxes = Boxes(new_bbox_loc)

        # add boxes to instances
        new_proposal_inst.gt_boxes = new_boxes
        new_proposal_inst.gt_classes = proposal_bbox_inst.pred_classes[valid_map]
        new_proposal_inst.gt_scores = proposal_bbox_inst.scores[valid_map]
        if 'iou' in proposal_bbox_inst._fields.keys():
            new_proposal_inst.gt_iou = proposal_bbox_inst.iou[valid_map]

        return new_proposal_inst, overlap

    def process_pseudo_label(
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method="", gt_labels=None,
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst_raw in proposals_rpn_unsup_k:
            # thresholding
            if self.select_iou:
                if proposal_type == 'roih':
                    proposal_bbox_inst, overlap = self.threshold_self(proposal_bbox_inst_raw, thres=cur_threshold,)
                else:
                    proposal_bbox_inst, overlap = self.threshold_bbox(
                    proposal_bbox_inst_raw, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == "thresholding":
                proposal_bbox_inst, overlap = self.threshold_bbox(
                    proposal_bbox_inst_raw, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        if gt_labels is not None and self.eval_pseudo_labels:
            self.check_pseudo_labels(proposals_rpn_unsup_k, list_instances, gt_labels)
        return list_instances, num_proposal_output, overlap

    def hold_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_datum['instances_gt'] = label_datum['instances']
        return label_data

    def remove_label(self, label_data):
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                del label_datum["instances"]
        return label_data

    def add_label(self, unlabled_data, label):
        for unlabel_datum, lab_inst in zip(unlabled_data, label):
            unlabel_datum["instances"] = lab_inst
        return unlabled_data
    
    def get_label(self, label_data):
        label_list = []
        for label_datum in label_data:
            if "instances" in label_datum.keys():
                label_list.append(copy.deepcopy(label_datum["instances"]))
        
        return label_list
    
    def test_with_gen(self, cfg, model, evaluators=None, gen_labels=False, gen_dir=''):
        """
        Evaluate the given model. The given model is expected to already contain
        weights to evaluate.

        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                ``cfg.DATASETS.TEST``.
            gen_labels: if True, saves generated proposals to file
            gen_dir: path to save generated proposals

        Returns:
            dict: a dict of result metrics
        """
        logger = logging.getLogger(__name__)
        if isinstance(evaluators, DatasetEvaluator):
            evaluators = [evaluators]
        if evaluators is not None:
            assert len(cfg.DATASETS.TEST) == len(evaluators), "{} != {}".format(
                len(cfg.DATASETS.TEST), len(evaluators)
            )

        results = OrderedDict()
        for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
            data_loader = self.build_test_loader(cfg, dataset_name)
            print(idx)
            if idx > 10:
                print('done')
                break
            # When evaluators are passed in as arguments,
            # implicitly assume that evaluators can be created before data_loader.
            if evaluators is not None:
                evaluator = evaluators[idx]
            else:
                try:
                    evaluator = self.build_evaluator(cfg, dataset_name)
                except NotImplementedError:
                    logger.warn(
                        "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                        "or implement its `build_evaluator` method."
                    )
                    results[dataset_name] = {}
                    continue
            name_split = dataset_name.split('_')
            if name_split[0] == 'ACDC' and name_split[1] == 'train' and gen_labels:
                output_file = gen_dir + 'dino_anno_{}.pkl'.format(dataset_name)
                results_i, outputs = inference_on_dataset(model, data_loader, evaluator, get_outs=True)
                with open(output_file, 'wb') as f_out:
                    pickle.dump(outputs, f_out)
            if name_split[0] == 'cityscapes' and name_split[-1] == 'train' and gen_labels:
                output_file = gen_dir + 'dino_anno_{}.pkl'.format(dataset_name)
                results_i, outputs = inference_on_dataset(model, data_loader, evaluator, get_outs=True)
                with open(output_file, 'wb') as f_out:
                    pickle.dump(outputs, f_out)
            if name_split[0] == 'BDD' and name_split[-1] == 'train' and gen_labels:
                output_file = gen_dir + 'dino_anno_{}.pkl'.format(dataset_name)
                results_i, outputs = inference_on_dataset(model, data_loader, evaluator, get_outs=True)
                with open(output_file, 'wb') as f_out:
                    pickle.dump(outputs, f_out)
            else:
                results_i = inference_on_dataset(model, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]
        return results
    
    # def get_label_test(self, label_data):
    #     label_list = []
    #     for label_datum in label_data:
    #         if "instances" in label_datum.keys():
    #             label_list.append(label_datum["instances"])

    # =====================================================
    # =================== Training Flow ===================
    # =====================================================

    def run_step_full_semisup(self):
        self._trainer.iter = self.iter
        assert self.model.training, "[UBTeacherTrainer] model was changed to eval mode!"
        start = time.perf_counter()
        data = next(self._trainer._data_loader_iter)
        # data_q and data_k from different augmentations (q:strong, k:weak)
        # label_strong, label_weak, unlabed_strong, unlabled_weak
        if self.cfg.INPUT.CLEAN_DETECTIONS:
            label_data_q, label_data_k, label_regions, unlabel_data_q, unlabel_data_k, unlabel_regions = data
            label_data_q, _ = self.clean_detections(label_data_q, label_regions)
            unlabel_data_q, old_boxes = self.clean_detections(unlabel_data_q, unlabel_regions, output_old=True)
        elif self.cfg.SEMISUPNET.USE_DINO:
            label_data_q, label_data_k, label_regions, unlabel_data_q, unlabel_data_k, unlabel_regions = data
        else:
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data
            label_regions = None
            unlabel_regions = None

        data_time = time.perf_counter() - start

        # print(self.iter, self.model.iter)

        if 'module' in self.model.__dict__['_modules']:
            self.model.module.roi_heads.keep_proposals = {}
            self.model.module.roi_heads.keep_stats = False
            self.model_teacher.roi_heads.keep_stats = False
        else:
            self.model.roi_heads.keep_proposals = {}
            self.model.roi_heads.keep_stats = False
            self.model_teacher.roi_heads.keep_stats = False
            
        # burn-in stage (supervised training with labeled data)

        if self.cfg.SEMISUPNET.FREEZE_POSTPRETRAIN and self.iter == (self.cfg.SEMISUPNET.PRETRAIN_STEPS):
            if 'module' in self.model.__dict__['_modules']:
                model = self.model.module
                for param in model.backbone.parameters():
                    param.requires_grad = False
                self.model = DistributedDataParallel(
                    model, device_ids=[comm.get_local_rank()], broadcast_buffers=False, find_unused_parameters=False)
                if self.use_dino:
                    self.model.dino_head = self.model.module.dino_head
                    self.model.dino_align = self.model.module.dino_align
            else:
                for param in self.model.backbone.parameters():
                    param.requires_grad = False

        if self.iter % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            self._update_teacher_model(
                keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:
            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            if self.align_gt_proposals:
                label_data_q = self.hold_label(label_data_q)

            self.branch = "supervised"
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            if self.align_proposals:
                # proposals_t = self.model.roi_heads.keep_proposals["supervised_target"]
                loss_align = self.align_proposals_loss()
                if not self.cfg.SEMISUPNET.ALIGN_INTRA:
                    loss_align['loss_align'] *= 1e-12
                record_dict.update(loss_align)

            if self.use_dino:
                if self.easy_dino_only:
                    easy_feat = self.model.dino_head(label_data_k)
                    dino_feat = torch.cat([easy_feat,easy_feat])
                else:
                    dino_feat = self.model.dino_head(label_data_q)
                cnn_feat = self.model.dino_align(self.cnn_feat[self.branch], dino_feat)
                if 0:#self.cfg.SEMISUPNET.DINO_SOURCE_BG_WEIGHT != 1.0 or self.cfg.INPUT.USE_RANDOM_NOISE:
                    mask = self.get_fg_mask_torch(label_data_q, noise_regions=label_regions, thresh=self.cfg.SEMISUPNET.DINO_SOURCE_FG_THRESH, bg_weight=self.cfg.SEMISUPNET.DINO_SOURCE_BG_WEIGHT)
                    dino_loss = self.model.dino_align.dino_loss(cnn_feat, dino_feat, fg_mask=mask, gt_data=label_data_q) * self.dino_loss_weight
                elif self.dino_sam_masks:
                    file_names = [x['file_name'] for x in label_data_q]
                    rle_masks = [x['masks'] for x in self.label_data_masks if x['file_name'] in file_names]
                    dino_loss = self.model.dino_align.dino_loss(cnn_feat, dino_feat, gt_data=rle_masks) * self.dino_loss_weight    
                else:
                    dino_loss = self.model.dino_align.dino_loss(cnn_feat, dino_feat, gt_data=label_data_q) * self.dino_loss_weight
                record_dict['loss_dino'] = dino_loss

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
                    if self.iter < self.cfg.SEMISUPNET.PRETRAIN_STEPS and "dino" not in key:
                        loss_dict[key] *= 0
                    if 'trunk' in key:
                        loss_dict[key] *= self.cfg.SEMISUPNET.TRUNK_SCALE
                                    
                    # if loss_dict[key] > 10:
                    #     loss_dict[key] = loss_dict[key] / loss_dict[key].item() * 10

            losses = sum(loss_dict.values())

        else:
            # if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
            #     # update copy the the whole model
            #     self._update_teacher_model(keep_rate=0.00)
            #     # self.model.build_discriminator()

            # elif (
            #     self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            # ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
            #     self._update_teacher_model(
            #         keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)
            

            #  0. remove unlabeled data labels
            if self.align_gt_proposals or self.eval_pseudo_labels or (self.use_dino and self.cfg.SEMISUPNET.DINO_TARGET_MASK):
                label_data_q = self.hold_label(label_data_q)
                label_data_k = self.hold_label(label_data_k)
                unlabel_data_q = self.hold_label(unlabel_data_q)
                unlabel_data_k = self.hold_label(unlabel_data_k)
                hold_labels = unlabel_data_k
            else:
                hold_labels = None
            if not self.use_gt_proposals:
                unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)


                # single_box = proposals_rpn_unsup_k[0].objectness_logits[0]
                # single_box.backward()
                # gradient = self.gradient[0].cpu().data.numpy()  # [C,H,W]
                # activations = self.activations[0].cpu().data.numpy()  # [C,H,W]
                # weight = np.mean(gradient, axis=(1, 2))  # [C]

                # cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
                # cam, cam_orig = self._postprocess_cam(cam, unlabel_data_k[0]["width"], unlabel_data_k[0]["height"])

                # from adapteacher.evaluation.grad_cam import *
                # with GradCAM(model=self.model_teacher,
                #             target_layers=target_layers,
                #             use_cuda=torch.cuda.is_available()) as cam:
                #     grayscale_cam = cam(input_tensor=input_tensor,
                #                         targets=targets)[0, :]
                #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']  
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

            swap_PL = False
            if self.use_dino_PL:
                if self.PL_swap == 'full':
                    if self.iter > self.PL_swap_iter:
                        swap_PL = True
                elif self.PL_swap == 'half':
                    if self.iter > self.PL_swap_iter and self.iter % 2:
                        swap_PL = True

            if not self.use_dino_PL or swap_PL:
                #  1. generate the pseudo-label using teacher model
                with torch.no_grad():
                    (
                        _,
                        proposals_rpn_unsup_k,
                        proposals_roih_unsup_k,
                        _,
                    ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

                #  2. Pseudo-labeling
                joint_proposal_dict = {}
                joint_proposal_dict["proposals_rpn"] = proposals_rpn_unsup_k
                #Process pseudo labels and thresholding
                (
                    pesudo_proposals_rpn_unsup_k,
                    nun_pseudo_bbox_rpn,
                    _
                ) = self.process_pseudo_label(
                    proposals_rpn_unsup_k, cur_threshold, "rpn", "thresholding"
                )
                # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,pesudo_proposals_rpn_unsup_k,'pred',True)
                # record_dict.update(analysis_pred)

                joint_proposal_dict["proposals_pseudo_rpn"] = pesudo_proposals_rpn_unsup_k
                # Pseudo_labeling for ROI head (bbox location/objectness)
                pesudo_proposals_roih_unsup_k, _, overlap = self.process_pseudo_label(
                    proposals_roih_unsup_k, cur_threshold, "roih", "thresholding", gt_labels=hold_labels
                )
                joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
                record_dict.update({'iou_overlap':overlap})

            else:
                # boxes = self.dino_pseudogt[x['image_id']]['instances_dino'].pred_boxes
                instances = [self.dino_pseudogt[x['image_id']]['instances_dino'] for x in unlabel_data_q]
                boxes = [(x['tf_data'].apply_box(y.pred_boxes),y.scores,y.pred_classes) for x,y in zip(unlabel_data_q,instances)]
                dino_pseudo_labels = []
                for i in range(len(instances)):
                    new_instances = Instances(gt_unlabel_k[i].image_size)
                    new_instances.gt_boxes = Boxes(torch.tensor(boxes[i][0]))
                    new_instances.gt_scores = boxes[i][1]
                    new_instances.gt_classes = boxes[i][2]
                    dino_pseudo_labels.append(new_instances)

                joint_proposal_dict = {}
                pseudo_proposals_dino, _, overlap = self.process_pseudo_label(dino_pseudo_labels, cur_threshold, "dino", "thresholding")
                joint_proposal_dict["proposals_pseudo_roih"] = pseudo_proposals_dino
                record_dict.update({'iou_overlap':overlap})

            # 3. add pseudo-label to unlabeled data

            if not self.use_gt_proposals:
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            if self.cfg.INPUT.CLEAN_DETECTIONS:
                unlabel_data_q, old_pseudo_boxes = self.clean_detections(unlabel_data_q, unlabel_regions, output_old=True)

            # if any(any(i in x['instances_gt'].gt_classes.tolist() for i in [1,7]) for x in unlabel_data_q):
            # # if any(any(i in x['instances_gt'].gt_classes.tolist() for i in [3,4,5]) for x in unlabel_data_q):
            #     a = 1

            all_label_data = label_data_q + label_data_k
            if not self.use_dino_PL or swap_PL:
                all_unlabel_data = unlabel_data_q
            else:
                all_unlabel_data = unlabel_data_q + unlabel_data_k

            # 4. input both strongly and weakly augmented labeled data into student model
            self.branch = "supervised"
            if 1:
                record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised", #use_gt_only=False
                )
                record_dict.update(record_all_label_data)
            else:
                with torch.no_grad():
                    record_all_label_data, _, _, _ = self.model(
                    all_label_data, branch="supervised"
                    )
                    record_dict.update(record_all_label_data)

            if self.use_dino:
                if self.easy_dino_only:
                    easy_feat = self.model.dino_head(label_data_k)
                    dino_feat = torch.cat([easy_feat,easy_feat])
                else:
                    dino_feat = self.model.dino_head(all_label_data)
                cnn_feat = self.model.dino_align(self.cnn_feat[self.branch], dino_feat)
                if 0:#self.cfg.SEMISUPNET.DINO_SOURCE_BG_WEIGHT != 1.0 or self.cfg.INPUT.USE_RANDOM_NOISE:
                    mask = self.get_fg_mask_torch(all_label_data, noise_regions=label_regions, thresh=self.cfg.SEMISUPNET.DINO_SOURCE_FG_THRESH, bg_weight=self.cfg.SEMISUPNET.DINO_SOURCE_BG_WEIGHT)
                    dino_loss = self.model.dino_align.dino_loss(cnn_feat, dino_feat, fg_mask=mask, gt_data=all_label_data) * self.dino_loss_weight
                else:
                    dino_loss = self.model.dino_align.dino_loss(cnn_feat, dino_feat, gt_data=all_label_data) * self.dino_loss_weight
                record_dict['loss_dino'] = dino_loss

            # 5. input strongly augmented unlabeled data into model
            self.branch = "supervised_target"
            record_all_unlabel_data, _, _, _ = self.model(
                # all_unlabel_data, branch="supervised_target", use_gt_only=self.use_gt_proposals_only
                all_unlabel_data, branch="supervised_target", #use_gt_only=self.use_gt_proposals_only
            )
            if self.use_dino:
                if self.easy_dino_only and len(all_unlabel_data) > len(unlabel_data_k):
                    easy_feat = self.model.dino_head(unlabel_data_k)
                    dino_feat = torch.cat([easy_feat,easy_feat])
                else:
                    dino_feat = self.model.dino_head(all_unlabel_data)
                dino_feat = self.model.dino_head(all_unlabel_data)
                cnn_feat = self.model.dino_align(self.cnn_feat[self.branch], dino_feat)
                if 0:#self.cfg.INPUT.USE_RANDOM_NOISE:
                    mask = self.get_fg_mask_torch(all_unlabel_data, noise_regions=unlabel_regions, thresh=self.cfg.SEMISUPNET.DINO_SOURCE_FG_THRESH, bg_weight=self.cfg.SEMISUPNET.DINO_SOURCE_BG_WEIGHT, has_segm=False)
                    dino_loss_pseudo = self.model.dino_align.dino_loss(cnn_feat, dino_feat, fg_mask=mask, gt_data=all_unlabel_data) * self.dino_loss_weight_target
                else:
                    dino_loss_pseudo = self.model.dino_align.dino_loss(cnn_feat, dino_feat, gt_data=all_unlabel_data) * self.dino_loss_weight_target
                record_dict['loss_dino_pseud'] = dino_loss_pseudo

            new_record_all_unlabel_data = {}
            for key in record_all_unlabel_data.keys():
                new_record_all_unlabel_data[key + "_pseudo"] = record_all_unlabel_data[
                    key
                ]
            record_dict.update(new_record_all_unlabel_data)

            if self.align_proposals:
                # proposals_t = self.model.roi_heads.keep_proposals["supervised_target"]
                loss_align = self.align_proposals_loss()
                record_dict.update(loss_align)


            # 6. input weakly labeled data (source) and weakly unlabeled data (target) to student model
            # give sign to the target data

            for i_index in range(len(unlabel_data_k)):
                # unlabel_data_item = {}
                for k, v in unlabel_data_k[i_index].items():
                    # label_data_k[i_index][k + "_unlabeled"] = v
                    label_data_k[i_index][k + "_unlabeled"] = v
                # unlabel_data_k[i_index] = unlabel_data_item

            all_domain_data = label_data_k
            # all_domain_data = label_data_k + unlabel_data_k
            record_all_domain_data, _, _, _ = self.model(all_domain_data, branch="domain")
            record_dict.update(record_all_domain_data)


            # weight losses
            loss_dict = {}
            if self.iter < self.align_only_iter:
                unsup_loss_weight = 0
            else:
                unsup_loss_weight = self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if torch.isnan(record_dict[key]):
                        record_dict[key] = 0
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            unsup_loss_weight
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                        if loss_dict[key] > 0.1:
                            print(key)
                            loss_dict[key] = loss_dict[key] / loss_dict[key].item() * 0.1
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1
                    
                    if 'trunk' in key:
                        loss_dict[key] *= self.cfg.SEMISUPNET.TRUNK_SCALE

                    if self.iter < self.cfg.SEMISUPNET.PRETRAIN_STEPS and "dino" not in key:
                        loss_dict[key] *= 0
                    
                    # if loss_dict[key] > 10:
                    #     loss_dict[key] = loss_dict[key] / loss_dict[key].item() * 10

            losses = sum(loss_dict.values())

        metrics_dict = record_dict
        metrics_dict["data_time"] = data_time
        self._write_metrics(metrics_dict)

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

    def _write_metrics(self, metrics_dict: dict):
        metrics_dict = {
            k: v.detach().cpu().item() if isinstance(v, torch.Tensor) else float(v)
            for k, v in metrics_dict.items()
        }

        # gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)
        # all_hg_dict = comm.gather(hg_dict)

        if comm.is_main_process():
            if "data_time" in all_metrics_dict[0]:
                # data_time among workers can have high variance. The actual latency
                # caused by data_time is the maximum among workers.
                data_time = np.max([x.pop("data_time")
                                   for x in all_metrics_dict])
                self.storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }

            # append the list
            loss_dict = {}
            for key in metrics_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = metrics_dict[key]
                    if key in self.loss_dict:
                        self.loss_dict[key].append(metrics_dict[key])
                    else:
                        self.loss_dict[key] = [metrics_dict[key]]

            total_losses_reduced = sum(loss for loss in loss_dict.values())

            self.storage.put_scalar("total_loss", total_losses_reduced)
            if self.log_wandb:
                self.wandb_run.log(metrics_dict)
            if len(metrics_dict) > 1:
                self.storage.put_scalars(**metrics_dict)

    @torch.no_grad()
    def _update_teacher_model(self, keep_rate=0.9996):
        if comm.get_world_size() > 1:
            student_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
        else:
            student_model_dict = self.model.state_dict()

        new_teacher_dict = OrderedDict()
        for key, value in self.model_teacher.state_dict().items():
            if key in student_model_dict.keys():
                new_teacher_dict[key] = (
                    student_model_dict[key] *
                    (1 - keep_rate) + value * keep_rate
                )
            else:
                raise Exception("{} is not found in student model".format(key))

        self.model_teacher.load_state_dict(new_teacher_dict)

    @torch.no_grad()
    def _copy_main_model(self):
        # initialize all parameters
        if comm.get_world_size() > 1:
            rename_model_dict = {
                key[7:]: value for key, value in self.model.state_dict().items()
            }
            self.model_teacher.load_state_dict(rename_model_dict)
        else:
            self.model_teacher.load_state_dict(self.model.state_dict())

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name)

    def build_hooks(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        # Do PreciseBN before checkpointer, because it updates the model and need to
        # be saved by checkpointer.
        # This is not always the best: if checkpointing has a different frequency,
        # some checkpoints may have more precise statistics than others.
        if comm.is_main_process():
            ret.append(
                hooks.PeriodicCheckpointer(
                    self.checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD
                )
            )

        def test_and_save_results_student():
            self._last_eval_results_student = self.test(self.cfg, self.model)
            _last_eval_results_student = {
                k + "_student": self._last_eval_results_student[k]
                for k in self._last_eval_results_student.keys()
            }
            if self.log_wandb:
                self.wandb_run.log(_last_eval_results_student)
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
            if self.log_wandb:
                self.wandb_run.log(self._last_eval_results_teacher)
            return self._last_eval_results_teacher

        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_student))
        ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
                   test_and_save_results_teacher))
        # ret.append(hooks.EvalHook(cfg.TEST.EVAL_PERIOD,
        #            test_and_save_results_student))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret

    def clean_detections(self, data_strong, regions, output_old=False):
        if output_old:
            old_boxes = [x['instances'].gt_boxes.clone() for x in data_strong]
        else:
            old_boxes = None
        for idx in range(len(data_strong)):
            bbox_regions = torch.tensor([x for x in regions[idx] if x])
            if not bbox_regions.shape[0]:
                continue

            bbox_gt = data_strong[idx]['instances'].gt_boxes.tensor
            bbox_regions[:,2:] = bbox_regions[:,2:] + bbox_regions[:,:2]
            bbox_regions = bbox_regions.to(device=bbox_gt.device)

            m = bbox_gt.shape[0]
            n = bbox_regions.shape[0]
            if n > 0:
                cs1 = torch.zeros((m,n,4)).to(device=bbox_gt.device)
                cs1[:,:,2:] = torch.min(bbox_gt[:, None, 2:], bbox_regions[:, 2:])
                cs1[:,:,:2] = torch.max(bbox_gt[:, None, :2], bbox_regions[:, :2])
                dcs1 = cs1[:,:,2:] - cs1[:,:,:2]
                check1 = (dcs1 > 0).all(dim=2, keepdim=True)
                cs1 = cs1*check1
                intersection = (dcs1*check1).prod(dim=2).sum(dim=1)

                if n > 1:
                    n2 = comb(n,2)
                    cs2 = torch.zeros((m,n2,4)).to(device=bbox_gt.device)
                    lv1 = 0
                    for lv2 in range(n-1):
                        for lv3 in range(lv2+1,n):    
                            cs2[:,lv1,2:] = torch.min(cs1[:, lv2, 2:], cs1[:, lv3, 2:])
                            cs2[:,lv1,:2] = torch.max(cs1[:, lv2, :2], cs1[:, lv3, :2])
                            lv1 += 1
                    dcs2 = cs2[:,:,2:] - cs2[:,:,:2]
                    check2 = (dcs2 > 0).all(dim=2, keepdim=True)
                    cs2 = cs2*check2
                    intersection = intersection - (dcs2*check2).prod(dim=2).sum(dim=1)

                    if n > 2:
                        n3 = comb(n,3)
                        cs3 = torch.zeros((m,1,4)).to(device=bbox_gt.device)
                        # lv1 = 0
                        # for lv2 in range(n2-2):
                        #     for lv2 in range(lv2,n3-1):
                        #         for lv3 in range(lv3,n3):    
                                # cs2[:,k,2:] = torch.min(cs1[:, i, 2:], cs1[:, j, 2:])
                                # cs2[:,k,:2] = torch.max(cs1[:, i, :2], cs1[:, j, :2])
                                # k += 1
                        cs3[:,0,2:] = torch.min(cs2[:, 0, 2:], cs1[:, 2, 2:]) 
                        cs3[:,0,:2] = torch.min(cs2[:, 0, :2], cs1[:, 2, :2])
                        dcs3 = cs3[:,:,2:] - cs3[:,:,:2]
                        check3 = (dcs3 > 0).all(dim=2, keepdim=True)
                        cs3 = cs3*check3
                        intersection = intersection + (dcs3*check3).prod(dim=2).sum(dim=1)

                areas = (bbox_gt[:,2:] - bbox_gt[:,:2]).prod(dim=1)
                valid_boxes = (intersection / areas) < self.cfg.INPUT.MAX_OCCLUSION
                # if not all(valid_boxes) and output_old:
                #     a=1

                deltas = dcs1*check1
                new_bboxes = bbox_gt.clone()
                for lv1 in range(m):
                    for lv2 in range(n):
                        if deltas[lv1,lv2,:].any():
                            if (bbox_regions[lv2,1] <= new_bboxes[lv1,1] and bbox_regions[lv2,3] >= new_bboxes[lv1,3]):
                                if (bbox_regions[lv2,0] <= new_bboxes[lv1,0]) and (bbox_regions[lv2,2] <= new_bboxes[lv1,2]):
                                    new_bboxes[lv1,0] = bbox_regions[lv2,2]
                                elif (bbox_regions[lv2,0] >= new_bboxes[lv1,0]) and (bbox_regions[lv2,2] >= new_bboxes[lv1,2]):
                                    new_bboxes[lv1,2] = bbox_regions[lv2,0]
                            
                            if (bbox_regions[lv2,0] <= new_bboxes[lv1,0] and bbox_regions[lv2,2] >= new_bboxes[lv1,2]):
                                if (bbox_regions[lv2,1] <= new_bboxes[lv1,1]) and (bbox_regions[lv2,3] <= new_bboxes[lv1,3]):
                                    new_bboxes[lv1,1] = bbox_regions[lv2,3]
                                elif (bbox_regions[lv2,1] >= new_bboxes[lv1,1]) and (bbox_regions[lv2,3] >= new_bboxes[lv1,3]):
                                    new_bboxes[lv1,3] = bbox_regions[lv2,1]
                
                data_strong[idx]['instances'].gt_boxes = Boxes(new_bboxes)
                # if sum(valid_boxes) < len(valid_boxes):
                #     a = 1
                data_strong[idx]['instances'] = data_strong[idx]['instances'][valid_boxes]
            
        return data_strong, old_boxes

    def build_hooks_final(self):
        cfg = self.cfg.clone()
        cfg.defrost()
        cfg.DATALOADER.NUM_WORKERS = 0  # save some memory and time for PreciseBN

        ret = [
            hooks.IterationTimer(),
            hooks.LRScheduler(self.optimizer, self.scheduler),
            hooks.PreciseBN(
                # Run at the same freq as (but before) evaluation.
                cfg.TEST.EVAL_PERIOD,
                self.model,
                # Build a new data loader to not affect training
                self.build_train_loader(cfg),
                cfg.TEST.PRECISE_BN.NUM_ITER,
            )
            if cfg.TEST.PRECISE_BN.ENABLED and get_bn_modules(self.model)
            else None,
        ]

        def eval_relative():
            # return self.test_relative()
            return self.test_DINO()

        ret.append(hooks.EvalHook(0,
                   eval_relative))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
    
    def test_DINO(self):
        self.model = self.model.eval()
        self.branch = "supervised"
        n_imgs = 400

        test_loaders = []
        for dataset_name in self.cfg.DATASETS.TEST:
        # for dataset_name in self.cfg.DATASETS.TEST[2:]:
            test_loaders.append(iter(build_detection_test_loader(self.cfg, dataset_name, mapper=DatasetMapper_test(self.cfg, True))))
        n_datasets = len(test_loaders)

        mask_first = False
        use_cnn = True
        use_bbox = False
        with torch.no_grad():
            instance_feats = []
            instance_cnn_feats = []
            instance_cnn_project_feats = []
            instance_class = []
            instance_area = []
            instance_dataset = []
            instance_city = []
            instance_file = []
            instances_per_dataset = []
            # for lv1 in range(n_imgs):
            #     if not lv1 % 10:
            #         print(lv1)
            n_instances = 0
            for lv2 in range(6):
                n_img = 0
                print(self.cfg.DATASETS.TEST[lv2])
                for idx, data in enumerate(test_loaders[lv2]):
                    if n_img >= n_imgs:
                        break
                    if not len(data[0]['instances']):
                        continue
                    n_img += 1
                    if mask_first:
                        cnn_dims = np.floor(np.array(data[0]['image'].shape[1:])/32).astype(int)
                        curr_masks = self.get_fg_mask(data, patch_size=self.model.dino_head.patch_size,cnn_dims=cnn_dims, box_mask=use_bbox)[0]
                        new_data = [{'image':(data[0]['image']*x[1]).long()} for x in curr_masks]
                        n_masks = len(new_data)
                        n_max = 2
                        if n_masks > n_max:
                            curr_feats = []
                            curr_feats_cnn = []
                            curr_feats_cnn_project = []
                            n_batch = ceil(n_masks/n_max)
                            for i in range(n_batch):
                                id1 = i*n_max
                                id2 = id1 + n_max
                                dino_feat = self.model.dino_head(new_data[id1:id2])
                                cnn_feat = self.model.backbone(self.model.preprocess_image(new_data[id1:id2]).tensor)[self.dino_layer]
                                cnn_rescale_feat = self.model.dino_align(cnn_feat, dino_feat)
                                masks_temp = curr_masks[id1:id2]
                                curr_feats += [(dino_feat[idx,:,:,:].detach().cpu().numpy()*x[2]).sum(axis=(1,2)) for idx, x in enumerate(masks_temp)]
                                curr_feats_cnn += [(cnn_feat[idx,:,:,:].detach().cpu().numpy()*x[-1]).sum(axis=(1,2)) for idx, x in enumerate(masks_temp)]
                                curr_feats_cnn_project += [(cnn_rescale_feat[idx,:,:,:].detach().cpu().numpy()*x[2]).sum(axis=(1,2)) for idx, x in enumerate(masks_temp)]
                        else:
                            dino_feat = self.model.dino_head(new_data)
                            cnn_feat = self.model.backbone(self.model.preprocess_image(new_data).tensor)[self.dino_layer]
                            cnn_rescale_feat = self.model.dino_align(cnn_feat, dino_feat)
                            curr_feats = [(dino_feat[idx,:,:,:].detach().cpu().numpy()*x[2]).sum(axis=(1,2)) for idx, x in enumerate(curr_masks)]
                            curr_feats_cnn = [(cnn_feat[idx,:,:,:].detach().cpu().numpy()*x[-1]).sum(axis=(1,2)) for idx, x in enumerate(curr_masks)]
                            curr_feats_cnn_project = [(cnn_rescale_feat[idx,:,:,:].detach().cpu().numpy()*x[2]).sum(axis=(1,2)) for idx, x in enumerate(curr_masks)]
                    else:
                        dino_feat = self.model.dino_head(data)
                        cnn_feat = self.model.backbone(self.model.preprocess_image(data).tensor)[self.dino_layer]
                        cnn_rescale_feat = self.model.dino_align(cnn_feat, dino_feat)
                        curr_masks = self.get_fg_mask(data, patch_size=self.model.dino_head.patch_size,cnn_dims=cnn_feat.shape[2:], box_mask=use_bbox)[0]
                        curr_feats = [(dino_feat.detach().cpu().numpy()*x[2]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]
                        curr_feats_cnn = [(cnn_feat.detach().cpu().numpy()*x[-1]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]
                        curr_feats_cnn_project = [(cnn_rescale_feat.detach().cpu().numpy()*x[2]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]
                    # feats = self.model([data], branch="supervised", backbone_only=True)
                    # cnn_feat = self.model.dino_align(feats['vgg4'], dino_feat)
                    # loss_, sim_ = self.model.dino_align.dino_loss(cnn_feat, dino_feat, return_sim=True)
                    # source_sims.append(sim_s.squeeze().detach().cpu().numpy())
                    # if any([np.linalg.norm(x) < 0.000001 for x in curr_feats]):
                    #     a=1
                    curr_feats = [x/np.linalg.norm(x) for x in curr_feats]
                    instance_feats += curr_feats
                    if use_cnn:
                        norms = np.array([np.linalg.norm(x) for x in curr_feats_cnn])
                        if any(norms < 0.00000001):
                            a=1
                        curr_feats_cnn = [x/np.linalg.norm(x) for x in curr_feats_cnn]
                        instance_cnn_feats += curr_feats_cnn
                        curr_feats_cnn_project = [x/np.linalg.norm(x) for x in curr_feats_cnn_project]
                        instance_cnn_project_feats += curr_feats_cnn_project
                    instance_class += [x[0] for x in curr_masks]
                    instance_area += [x[2].sum() for x in curr_masks]
                    instance_dataset += [self.cfg.DATASETS.TEST[lv2] for x in curr_masks]
                    instance_file += [data[0]['file_name'] for x in curr_masks]
                    if 'cityscapes' in data[0]['file_name']:
                        city = data[0]['file_name'].split('/')[-2]
                        instance_city += [city for x in curr_masks]
                    else:
                        instance_city += ['' for x in curr_masks]
                instances_per_dataset.append(len(instance_area)-n_instances)
                n_instances = len(instance_area)
        
            instance_feats = np.array(instance_feats)
            instance_dataset = np.array(instance_dataset)
            instance_class = np.array(instance_class)
            data_dict = {'instance_feats':instance_feats, 'instance_class':instance_class, 'instance_area':instance_area, 'instance_dataset':instance_dataset,
                         'instance_city':instance_city, 'instance_file':instance_file, 'instances_per_dataset':instances_per_dataset,
                         'instance_cnn_feats':instance_cnn_feats, 'instance_cnn_project_feats':instance_cnn_project_feats}
            file_out = 'dino_feats_resnet50c4_mlp_20k.pkl'
            with open(file_out, 'wb') as f_out:
                pickle.dump(data_dict, f_out)

            permutations = [(x,y) for x in self.cfg.DATASETS.TEST for y in range(8)]
            ids = [np.where((instance_dataset==x[0]) * (instance_class==x[1]))[0] for x in permutations]
            datasets = self.cfg.DATASETS.TEST
            classes = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
            colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray']
            shapes = ['.','x','+','^','v']
            legend =  [x + ' ' + y for x in datasets for y in classes]


            offset = instances_per_dataset[0]
            sim_mat = np.matmul(instance_feats,instance_feats.T)
            k = 5
            k_NN = np.argpartition(sim_mat[:,:offset],-k)[:,-k:]
            k_NN_ = np.take_along_axis(k_NN, np.argsort(np.take_along_axis(sim_mat[:,:offset], k_NN, axis=1), axis=1), axis=1)
            n = len(instance_feats)
            confusions_acdc2city = []
            acdc_ids = np.core.defchararray.find(instance_dataset,'ACDC')!=-1
            k_closest_class = instance_class[k_NN_]
            for i in range(8):
                ids_acdc_class = (instance_class==i)*acdc_ids
                k_closest_single = k_closest_class[ids_acdc_class,:].flatten()
                confusions_acdc2city.append(np.array([(k_closest_single==x).sum() for x in range(8)])/len(k_closest_single))


            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            offset = 0
            model_ = TSNE(n_components=2)
            tsne_data = model_.fit_transform(instance_feats[offset:,:])
            plt.figure()
            lv3 = -1
            n0 = 8*lv3+8
            for i in range(n0,len(legend)):
                perm = permutations[i]
                if perm[1] == 0:
                    lv3 += 1
                ids_curr = ids[i]-offset
                plt.plot(tsne_data[ids_curr,0],tsne_data[ids_curr,1],linestyle='none',marker=shapes[lv3],color=colors[perm[1]])
            
            plt.legend(legend)
            plt.show()
            a=1
                # source_strong, source_weak, rs1, target_strong, target_weak, sr2 = data

                # _, _ = self.model(source_weak, branch="supervised")
                # dino_feat = self.model.dino_head(source_weak)
                # cnn_feat = self.model.dino_align(self.cnn_feat["supervised"], dino_feat)
                # loss_s, sim_s = self.model.dino_align.dino_loss(cnn_feat, dino_feat, return_sim=True)
                # source_sims.append(sim_s.squeeze().detach().cpu().numpy())
                # curr_masks = self.get_fg_mask(source_weak)
                # source_names = source_names + [x['file_name'] for x in source_weak]
                # # source_masks.append(curr_masks)
                # for i in range(len(curr_masks)):
                #     for j in range(len(curr_masks[i])):
                #         mask_feat = torch.nn.functional.normalize((dino_feat[i,:,:,:].detach().cpu() * torch.tensor(curr_masks[i][j][2])).sum(1).sum(1), dim=0)
                #         source_instance_feats.append((curr_masks[i][j][0], mask_feat, curr_masks[i][j][2].sum()))

                # _, _ = self.model(target_weak, branch="supervised")
                # dino_feat = self.model.dino_head(target_weak)
                # cnn_feat = self.model.dino_align(self.cnn_feat["supervised"], dino_feat)
                # loss_t, sim_t = self.model.dino_align.dino_loss(cnn_feat, dino_feat, return_sim=True)
                # target_sims.append(sim_t.squeeze().detach().cpu().numpy())
                # curr_masks = self.get_fg_mask(target_weak)
                # target_names = target_names + [x['file_name'] for x in target_weak]
                # # target_masks.append(curr_masks)
                # for i in range(len(curr_masks)):
                #     for j in range(len(curr_masks[i])):
                #         mask_feat = torch.nn.functional.normalize((dino_feat[i,:,:,:].detach().cpu() * torch.tensor(curr_masks[i][j][2])).sum(1).sum(1), dim=0)
                #         target_instance_feats.append((curr_masks[i][j][0], mask_feat, curr_masks[i][j][2].sum()))

            feat_s = [[] for x in range(8)]
            size_s = [[] for x in range(8)]
            for i in range(len(source_instance_feats)):
                id = source_instance_feats[i][0]
                feat_s[id].append(source_instance_feats[i][1])
                size_s[id].append(source_instance_feats[i][2])
            feat_s = [torch.vstack(x) for x in feat_s]

            feat_t = [[] for x in range(8)]
            size_t = [[] for x in range(8)]
            for i in range(len(target_instance_feats)):
                id = target_instance_feats[i][0]
                feat_t[id].append(target_instance_feats[i][1])
                size_t[id].append(target_instance_feats[i][2])
            feat_t = [torch.vstack(x) for x in feat_t]
            a = 1
            
            legend = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']
            colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink','tab:gray']
            import matplotlib.pyplot as plt
            from sklearn.manifold import TSNE
            feat_all = feat_s + feat_t
            feat_ = np.concatenate([x.numpy() for x in feat_all])
            model_ = TSNE(n_components=2,perplexity=5)
            tsne_data = model_.fit_transform(feat_)
            plt.figure()
            x1 = 0
            for i in range(len(feat_s)):
                x2 = x1 + len(feat_s[i])
                data_ = tsne_data[x1:x2,:]
                x1 = x2
                plt.plot(data_[:,0],data_[:,1],'.',color=colors[i])
                plt.legend(legend)
            
            for i in range(len(feat_t)):
                x2 = x1 + len(feat_t[i])
                data_ = tsne_data[x1:x2,:]
                x1 = x2
                plt.plot(data_[:,0],data_[:,1],'x',color=colors[i])
            plt.title('T-SNE Cityscapes to ACDC Rain')
            plt.tight_layout()
            plt.show()

        # a = 1
        
        import matplotlib.pyplot as plt
        sm_pos = np.concatenate(source_masks, axis=0).reshape(-1) > thresh
        tm_pos = np.concatenate(target_masks, axis=0).reshape(-1) > thresh
        source_angs = np.arccos(np.concatenate(source_sims,axis=0).reshape(-1))*180/np.pi
        target_angs = np.arccos(np.concatenate(target_sims,axis=0).reshape(-1))*180/np.pi
        
        sfg_mean = source_angs[sm_pos].mean()
        sbg_mean = source_angs[~sm_pos].mean()
        tfg_mean = target_angs[tm_pos].mean()
        tbg_mean = target_angs[~tm_pos].mean()

        fig1, ax1 = plt.subplots(1,2,sharey=True,figsize=[8,4])
        ax1[0].hist(source_angs[~sm_pos], bins=np.arange(0,101,1), alpha=0.6, density=True, label="BG Source, avg={0:.0f}".format(sbg_mean), cumulative=0)
        ax1[0].hist(source_angs[sm_pos], bins=np.arange(0,101,1), alpha=0.6, density=True, label="FG Source, avg={0:.0f}".format(sfg_mean), cumulative=0)
        ax1[0].set_xlabel('Angle')
        ax1[0].set_ylabel('Density')
        ax1[0].legend()
        ax1[0].grid(True)
        ax1[1].hist(target_angs[~tm_pos], bins=np.arange(0,101,1), alpha=0.6, density=True, label="BG Target, avg={0:.0f}".format(tbg_mean), cumulative=0)
        ax1[1].hist(target_angs[tm_pos], bins=np.arange(0,101,1), alpha=0.6, density=True, label="FG Target, avg={0:.0f}".format(tfg_mean), cumulative=0)
        ax1[1].set_xlabel('Angle')
        # ax1[1].set_ylabel('Density')
        ax1[1].legend()
        ax1[1].grid(True)

        plt.title('Angle Distribution, weight=0.1, 40k iter.')
        plt.show()

    def get_fg_mask(self, data, patch_size=14, cnn_dims=[17,34], box_mask=True):
        out = []
        for img in data:
            if box_mask:
                masks = box2mask(data=data,dino_patch=patch_size,cnn_dims=cnn_dims)
                out.append(masks)
            # elif 'segm_file' in img.keys():
            #     file_split = img['file_name'].split('/')[:5]
            #     file_split[3:] = ['gt_panoptic_trainval','gt_panoptic']
            #     seg_file = '/'.join(file_split) + '/' + img['segm_file'].rsplit('_',1)[0] + '_panoptic.png'
            #     tfs = img['tf_data']
            #     pixels = np.array(Image.open(seg_file))
            #     h, w, c = pixels.shape
            #     pixels_flat = np.unique(pixels.reshape(h*w,c), axis=0)
            #     pixels_flat = pixels_flat[pixels_flat[:,1]>0,:]
            #     mask_class = [ACDC_PIX2CLASS[x[1]] for x in pixels_flat]
            #     masks_big = [tfs.apply_segmentation(np.all(pixels==x,axis=2).astype(float)) for x in pixels_flat]
            #     masks_small = [torch.nn.functional.avg_pool2d(torch.tensor(np.copy(x)).unsqueeze(0),patch_size).squeeze().numpy() for x in masks_big]
            #     masks_cnn = [torch.nn.functional.interpolate(torch.tensor(np.copy(x)).unsqueeze(0).unsqueeze(0),size=cnn_dims,mode='bicubic',antialias=True).clamp(min=0, max=1).squeeze().numpy() for x in masks_big]
            #     masks = [(x,y,z,img['file_name'],u) for x,y,z,u in zip(mask_class, masks_big, masks_small,masks_cnn)]
            #     masks = [x for x in masks if x[1].sum()>25]
            #     out.append(masks)
            elif 'gt_masks' in img['instances']._fields.keys():
                if type(img['instances'].gt_masks) == BitMasks:
                    cnn_interp = [torch.nn.functional.interpolate(x.float().unsqueeze(0).unsqueeze(0),size=cnn_dims,mode='bicubic',antialias=True) for x in img['instances'].gt_masks]
                    masks = [(y.item(),x.float(),torch.nn.functional.avg_pool2d(x.float().unsqueeze(0),patch_size).squeeze().numpy(),img['file_name'],z.clamp(min=0, max=1).squeeze().numpy()) for x,y,z in zip(img['instances'].gt_masks,img['instances'].gt_classes,cnn_interp)] 
                else:
                    c, h, w = img['image'].shape
                    polygons = [polygons_to_bitmask(x, h, w).astype(float) for x in img['instances'].gt_masks.polygons]
                    cnn_interp = [torch.nn.functional.interpolate(torch.tensor(x).unsqueeze(0).unsqueeze(0),size=cnn_dims,mode='bicubic',antialias=True) for x in img['instances'].gt_masks]
                    masks = [(y.item(),x,torch.nn.functional.avg_pool2d(torch.tensor(x).unsqueeze(0),patch_size).squeeze().numpy(),img['file_name']) for x,y,z in zip(polygons,img['instances'].gt_classes,cnn_interp)] 
                # mask_fg = sum([polygons_to_bitmask(x, h, w) for x in img['instances'].gt_masks.polygons]).astype(bool).astype(float)
                # mask_small = torch.nn.functional.avg_pool2d(torch.tensor(mask_fg).unsqueeze(0),patch_size).squeeze().numpy()
                out.append(masks)
            # elif 'segmentation' in img['instnaces']._fields.keys():
        return out

    def get_fg_mask_torch(self, data, noise_regions=None, thresh=0.2, bg_weight=1.0, has_segm=True):
        """
        Generates foreground mask from segmentation GT and removes random noise and padding regions
        Args:
            data (list): list of input augmented images with gt instances and augmentation data
            noise_region (list): list of XYWH coords of the random noise regions. Unscaled by the DINO scaling.
            thresh (float): for each patch, assign to background or regions not evaluated if <20% of component pixels are FG
            bg_weight (float): DINO align loss downscaled by weight if a background patch.

        Returns:
            list: list of per image downsampled mask to scale DINO align loss per patch.
        """
        
        patch_size = self.cfg.INPUT.DINO_PATCH_SIZE
        out = []
        for i,img in enumerate(data):
            c, h, w = img['image'].shape
            if len(img['instances']) and has_segm:
                mask = sum([polygons_to_bitmask(x, h, w) for x in img['instances'].gt_masks.polygons]).astype(bool).astype(float)
                mask_small = torch.nn.functional.avg_pool2d(torch.tensor(mask).unsqueeze(0),patch_size).squeeze()
                mask_vals = torch.where(mask_small >= thresh, 1, bg_weight)
            else:
                mask_vals = torch.ones(int(h/patch_size),int(w/patch_size))

            mask_valid = torch.ones(h,w)
            if noise_regions is not None and i < len(noise_regions):
                polygons_noise = [np.array([x[0],x[1],x[0]+x[2],x[1],x[0]+x[2],x[1]+x[3],x[0],x[1]+x[3],x[0],x[1]]) for x in noise_regions[i] if x]
                mask_valid = mask_valid * ~polygons_to_bitmask(polygons_noise, h, w)
            
            tf_pad = [id for id,x in enumerate(img['tf_data']) if isinstance(x,PadTransform)]
            is_flip = any([isinstance(x,HFlipTransform) for x in img['tf_data']])
            if tf_pad:
                pad_id = tf_pad[0]
                scale = h / img['height']
                pad = (np.array([img['tf_data'][pad_id].x1, img['tf_data'][pad_id].y1])*scale)#.floor()
                if is_flip:
                    polygons_pad = [[0,0, pad[0],0, pad[0],h-pad[1], w,h-pad[1], w,h, 0,h, 0,0]]
                else:
                    polygons_pad = [[w-pad[0],0, w,0, w,h, 0,h, 0,h-pad[1], w-pad[0],h-pad[1], w-pad[0],0]]
                mask_valid = mask_valid * ~polygons_to_bitmask(polygons_pad, h, w)

            mask_valid_small = torch.nn.functional.avg_pool2d(mask_valid.unsqueeze(0),patch_size).squeeze()
            mask_eval = torch.where(mask_valid_small >= thresh, 1, 0)

            out.append(mask_vals*mask_eval)
        return torch.stack(out)

    def test_relative(self):

        # """
        # Evaluate the given model. The given model is expected to already contain
        # weights to evaluate.

        # Args:
        #     cfg (CfgNode):
        #     model (nn.Module):
        #     evaluators (list[DatasetEvaluator] or None): if None, will call
        #         :meth:`build_evaluator`. Otherwise, must have the same length as
        #         ``cfg.DATASETS.TEST``.

        # Returns:
        #     dict: a dict of result metrics
        # """
        logger = logging.getLogger(__name__)

        results = OrderedDict()
        datasets_val = ['cityscapes_val', 'cityscapes_foggy_val']

        cfg = self.cfg.clone()
        cfg.defrost()
        # cfg.INPUT.MIN_SIZE_TRAIN = (1024,)
        # cfg.INPUT.MIN_SIZE_TEST = (1024,)
        # cfg.INPUT.MAX_SIZE_TRAIN = (2048,)
        # cfg.INPUT.MAX_SIZE_TEST = (2048,)
        # cfg.INPUT.RANDOM_FLIP = "none"
        cfg.INPUT.CROP.ENABLED = False


        # Test weak aug and generate pseudo labels
        mapper = DatasetMapperWithWeakAugs(cfg, True)
        data_loader = build_detection_unlabel_train_loader(cfg, mapper=mapper)
        evaluator = self.build_evaluator(cfg, 'cityscapes_foggy_train')
        print('Running on cityscapes_foggy_train weak')
        results_i = self.inference_and_pseudo_label(self.model_teacher, data_loader, evaluator)
        results['cityscapes_foggy_train_weak'] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format('cityscapes_foggy_train'))
            print_csv_format(results_i)


        # Test strong augss on GT and pseudo labels
        mapper = DatasetMapperWithStrongAugs(cfg, True)
        data_loader = build_detection_unlabel_train_loader(cfg, mapper=mapper)
        evaluator = self.build_evaluator(self.cfg, 'cityscapes_foggy_train')
        evaluator_pseudo = self.build_evaluator(cfg, 'cityscapes_foggy_pseudo_labels', allow_cached=False)

        print('Running on cityscapes_foggy_train strong')
        results_i, results_pseudo = self.inference_on_dataset_pseudo(self.model, data_loader, evaluator, evaluator_pseudo=evaluator_pseudo)
        results['cityscapes_foggy_train_strong'] = results_i
        if comm.is_main_process():
            assert isinstance(
                results_i, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_i
            )
            logger.info("Evaluation results for {} in csv format:".format('cityscapes_foggy_train'))
            print_csv_format(results_i)

        results['cityscapes_foggy_pseudo_strong'] = results_pseudo
        if comm.is_main_process():
            assert isinstance(
                results_pseudo, dict
            ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                results_pseudo
            )
            logger.info("Evaluation results for {} in csv format:".format('cityscapes_foggy_train'))
            print_csv_format(results_pseudo)



        for idx, dataset_name in enumerate(datasets_val):
            data_loader = self.build_test_loader(cfg, dataset_name)

            try:
                evaluator = self.build_evaluator(cfg, dataset_name)
            except NotImplementedError:
                logger.warn(
                    "No evaluator found. Use `DefaultTrainer.test(evaluators=)`, "
                    "or implement its `build_evaluator` method."
                )
                results[dataset_name] = {}
                continue

            print('Running on {}'.format(dataset_name))
            results_i = inference_on_dataset(self.model_teacher, data_loader, evaluator)
            results[dataset_name] = results_i
            if comm.is_main_process():
                assert isinstance(
                    results_i, dict
                ), "Evaluator must return a dict on the main process. Got {} instead.".format(
                    results_i
                )
                logger.info("Evaluation results for {} in csv format:".format(dataset_name))
                print_csv_format(results_i)

        if len(results) == 1:
            results = list(results.values())[0]

        outfile = 'results.json'
        import os
        file_out = '/'.join([os.getcwd(),self.cfg.OUTPUT_DIR,outfile])
        with open(file_out, 'w') as f_out:
            json.dump(results, f_out)

        return results


    def inference_and_pseudo_label(self,
        model, data_loader, evaluator
    ):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        file_out = self.cfg.OUTPUT_DIR + '/inference/pseudo_labels.json'
        if os.path.isfile(file_out)*0:
            with  open(file_out, 'r') as f_in:
                pseudo_dicts = json.load(f_in)
            results = None
        else:
            num_devices = get_world_size()
            logger = logging.getLogger(__name__)
            logger.info("Start inference on {} batches".format(len(data_loader)))

            total = len(data_loader)  # inference data loader must have a fixed length
            if evaluator is None:
                # create a no-op evaluator
                evaluator = DatasetEvaluators([])
            if isinstance(evaluator, abc.MutableSequence):
                evaluator = DatasetEvaluators(evaluator)
            evaluator.reset()

            num_warmup = min(5, total - 1)
            start_time = time.perf_counter()
            total_data_time = 0
            total_compute_time = 0
            total_eval_time = 0
            pseudo_dicts = []
            with ExitStack() as stack:
                if isinstance(model, nn.Module):
                    stack.enter_context(inference_context(model))
                stack.enter_context(torch.no_grad())

                start_data_time = time.perf_counter()
                for idx, inputs in enumerate(data_loader):
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0

                    start_compute_time = time.perf_counter()
                    outputs = model(inputs)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()
                    total_compute_time += time.perf_counter() - start_compute_time

                    curr_thresh = self.cfg.SEMISUPNET.BBOX_THRESHOLD
                    pred_insts = outputs[0]['instances'][outputs[0]['instances'].scores > curr_thresh]
                    annotations = []
                    for idx2 in range(pred_insts.pred_boxes.tensor.shape[0]):
                        annotation = {'iscrowd': False,
                                    'category_id': pred_insts.pred_classes[idx2].item(),
                                    'bbox': tuple(pred_insts.pred_boxes.tensor[idx2,:].tolist()),
                                    'bbox_mode': BoxMode.XYXY_ABS,}
                        annotations.append(annotation)

                    pred_dict = {'annotations': annotations}
                    for key in inputs[0].keys():
                        # if key != 'image':
                        if key not in ['instances', 'image']:
                            pred_dict[key] = inputs[0][key]

                    pseudo_dicts.append(pred_dict)

                    start_eval_time = time.perf_counter()
                    evaluator.process(inputs, outputs)
                    total_eval_time += time.perf_counter() - start_eval_time

                    iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    data_seconds_per_iter = total_data_time / iters_after_start
                    compute_seconds_per_iter = total_compute_time / iters_after_start
                    eval_seconds_per_iter = total_eval_time / iters_after_start
                    total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                    # if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    if not idx % 100:
                        print(idx)
                        eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                        log_every_n_seconds(
                            logging.INFO,
                            (
                                f"Inference done {idx + 1}/{total}. "
                                f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                                f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                                f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                                f"Total: {total_seconds_per_iter:.4f} s/iter. "
                                f"ETA={eta}"
                            ),
                            n=5,
                        )
                    start_data_time = time.perf_counter()

            file_out = self.cfg.OUTPUT_DIR + '/inference/pseudo_labels.json'
            with open(file_out, 'w') as f_out:
                json.dump(pseudo_dicts, f_out)

            # Measure the time only for this worker (before the synchronization barrier)
            total_time = time.perf_counter() - start_time
            total_time_str = str(datetime.timedelta(seconds=total_time))
            # NOTE this format is parsed by grep
            logger.info(
                "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                    total_time_str, total_time / (total - num_warmup), num_devices
                )
            )
            total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
            logger.info(
                "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                    total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
                )
            )

            results = evaluator.evaluate()
            # An evaluator may return None when not in main process.
            # Replace it by an empty dict instead to make it easier for downstream code to handle
            if results is None:
                results = {}

            # cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD
            # self.gen_eval_pseudo_label(outputs, cur_threshold)

        dataset_name = 'cityscapes_foggy_pseudo_labels'
        if 'cityscapes_foggy_pseudo_labels' in DatasetCatalog:
            DatasetCatalog.pop('cityscapes_foggy_pseudo_labels')
        DatasetCatalog.register(dataset_name, lambda: load_pseudo_dicts(file_out))
        meta = MetadataCatalog.get('cityscapes_foggy_train').as_dict()
        meta['name'] = dataset_name
        meta['gt_dir'] = file_out.rsplit('/',1)[0] + '/'
        if 'json_file' in meta.keys():
            meta.pop('json_file')
        MetadataCatalog.get(dataset_name).set(**meta)


        return results

    def inference_on_dataset_pseudo(self, model, data_loader, evaluator, evaluator_pseudo=None):
        """
        Run model on the data_loader and evaluate the metrics with evaluator.
        Also benchmark the inference speed of `model.__call__` accurately.
        The model will be used in eval mode.

        Args:
            model (callable): a callable which takes an object from
                `data_loader` and returns some outputs.

                If it's an nn.Module, it will be temporarily set to `eval` mode.
                If you wish to evaluate a model in `training` mode instead, you can
                wrap the given model and override its behavior of `.eval()` and `.train()`.
            data_loader: an iterable object with a length.
                The elements it generates will be the inputs to the model.
            evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
                but don't want to do any evaluation.

        Returns:
            The return value of `evaluator.evaluate()`
        """
        if evaluator_pseudo is None:
            results = inference_on_dataset(model, data_loader, evaluator)
            return results, None

        num_devices = get_world_size()
        logger = logging.getLogger(__name__)
        logger.info("Start inference on {} batches".format(len(data_loader)))

        total = len(data_loader)  # inference data loader must have a fixed length
        if evaluator is None:
            # create a no-op evaluator
            evaluator = DatasetEvaluators([])
        if isinstance(evaluator, abc.MutableSequence):
            evaluator = DatasetEvaluators(evaluator)
        evaluator.reset()
        evaluator_pseudo.reset()

        num_warmup = min(5, total - 1)
        start_time = time.perf_counter()
        total_data_time = 0
        total_compute_time = 0
        total_eval_time = 0
        with ExitStack() as stack:
            if isinstance(model, nn.Module):
                stack.enter_context(inference_context(model))
            stack.enter_context(torch.no_grad())

            start_data_time = time.perf_counter()
            for idx, inputs in enumerate(data_loader):
                total_data_time += time.perf_counter() - start_data_time
                if idx == num_warmup:
                    start_time = time.perf_counter()
                    total_data_time = 0
                    total_compute_time = 0
                    total_eval_time = 0

                start_compute_time = time.perf_counter()
                outputs = model(inputs)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                total_compute_time += time.perf_counter() - start_compute_time

                start_eval_time = time.perf_counter()
                evaluator.process(inputs, outputs)
                evaluator_pseudo.process(inputs, outputs)
                total_eval_time += time.perf_counter() - start_eval_time

                iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                data_seconds_per_iter = total_data_time / iters_after_start
                compute_seconds_per_iter = total_compute_time / iters_after_start
                eval_seconds_per_iter = total_eval_time / iters_after_start
                total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
                if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                    eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    log_every_n_seconds(
                        logging.INFO,
                        (
                            f"Inference done {idx + 1}/{total}. "
                            f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                            f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                            f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                            f"Total: {total_seconds_per_iter:.4f} s/iter. "
                            f"ETA={eta}"
                        ),
                        n=5,
                    )
                start_data_time = time.perf_counter()

        # Measure the time only for this worker (before the synchronization barrier)
        total_time = time.perf_counter() - start_time
        total_time_str = str(datetime.timedelta(seconds=total_time))
        # NOTE this format is parsed by grep
        logger.info(
            "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_time_str, total_time / (total - num_warmup), num_devices
            )
        )
        total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
        logger.info(
            "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
                total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
            )
        )

        results = evaluator.evaluate()
        results_pseudo = evaluator_pseudo.evaluate()
        # An evaluator may return None when not in main process.
        # Replace it by an empty dict instead to make it easier for downstream code to handle
        if results is None:
            results = {}
        return results, results_pseudo
    
    def align_proposals_loss(self):
        if 'module' in self.model.__dict__['_modules']:
            logits = self.model.module.roi_heads.current_proposals
        else:
            logits = self.model.roi_heads.current_proposals
        
        loss_align = self.align_loss(logits)

        # losses = []

        # for label in range(n):
        #     idx_s = sample_s[1] == label
        #     sample_label_s = sample_s[0][idx_s,:]
        #     idx_t = sample_t[1] == label
        #     sample_label_t = sample_t[0][idx_t,:]
        #     losses.append(self.align_loss(sample_label_s, sample_label_t))
        # self.loss_align = {'loss_align': sum(losses) / n}

        return {'loss_align': loss_align}

    def check_pseudo_labels(self, proposals_raw, proposals_select, proposals_gt, n_classes=8):
        labels = ['gt', 'raw50', 'raw75', 'rawn25', 'sel50', 'sel75', 'seln25']
        counts = torch.zeros((len(labels), n_classes+1)).to(device=proposals_raw[0].pred_boxes.device)

        for img in range(len(proposals_raw)):
            raw_boxes = proposals_raw[img].pred_boxes
            select_boxes = proposals_select[img].gt_boxes
            gt_boxes = proposals_gt[img]['instances_gt'].gt_boxes.to(device=select_boxes.device)
            raw_labels = proposals_raw[img].pred_classes
            select_labels = proposals_select[img].gt_classes
            gt_labels = proposals_gt[img]['instances_gt'].gt_classes.to(device=select_boxes.device)

            raw_ids = [torch.where(raw_labels == x)[0].to(device=select_boxes.device) for x in range(n_classes)]
            select_ids = [torch.where(select_labels == x)[0].to(device=select_boxes.device) for x in range(n_classes)]
            gt_ids = [torch.where(gt_labels == x)[0].to(device=select_boxes.device) for x in range(n_classes)]

            pos_raw = torch.eq(gt_labels, raw_labels.unsqueeze(1))
            match_raw = pairwise_iou(raw_boxes, gt_boxes)
            if all([x > 0 for x in match_raw.shape]):
                max_raw, ids_raw = torch.max(match_raw, 1, keepdim=True)
                max_only_raw = torch.zeros_like(match_raw)
                max_only_raw.scatter_(1, ids_raw, max_raw)
                max_only_raw_pos = max_only_raw * pos_raw
                morp_050 = max_only_raw_pos >= 0.5
                morp_075 = max_only_raw_pos >= 0.75
                max_only_raw_neg = max_only_raw * ~pos_raw
                morn_025 = max_only_raw_neg >= 0.25

                for id in range(len(gt_ids)):
                    counts[1,id] += morp_050[:,gt_ids[id]].sum()
                    counts[2,id] += morp_075[:,gt_ids[id]].sum()
                    counts[3,id] += morn_025[raw_ids[id],:].sum()
                counts[1,-1] += morp_050.sum()
                counts[2,-1] += morp_075.sum()
                counts[3,-1] += morn_025.sum()
            
            pos_select = torch.eq(gt_labels, select_labels.unsqueeze(1))
            match_select = pairwise_iou(select_boxes, gt_boxes)
            if all([x > 0 for x in match_select.shape]):
                max_select, ids_select = torch.max(match_select, 1, keepdim=True)
                max_only_select = torch.zeros_like(match_select)
                max_only_select.scatter_(1, ids_select, max_select)
                max_only_select_pos = max_only_select * pos_select
                mosp_050 = max_only_select_pos >= 0.5
                mosp_075 = max_only_select_pos >= 0.75
                max_only_select_neg = max_only_select * ~pos_select
                mosn_025 = max_only_select_neg >= 0.25

                for id in range(len(gt_ids)):
                    counts[4,id] += mosp_050[:,gt_ids[id]].sum()
                    counts[5,id] += mosp_075[:,gt_ids[id]].sum()
                    counts[6,id] += mosn_025[select_ids[id],:].sum()

                counts[4,-1] += mosp_050.sum()
                counts[5,-1] += mosp_075.sum()
                counts[6,-1] += mosn_025.sum()

            for id in range(len(gt_ids)):
                counts[0,id] += gt_ids[id].shape[0]

            counts[0,-1] += gt_labels.shape[0]


        self.pseudo_counts += counts.transpose(0,1).flatten().cpu()
        if (self.iter % self.pseudo_subsampling) == (self.pseudo_subsampling - 1):
            row = [self.iter] + self.pseudo_counts.tolist() 
            with open(self.pseudo_stats_file, 'a', newline='') as f_out:
                csv_writer = csv.writer(f_out, delimiter=' ')
                csv_writer.writerow(row)
            self.pseudo_counts *= 0

    
def load_pseudo_dicts(filename):
    with open(filename, 'r') as f_in:
        dicts = json.load(f_in)

    return dicts

@contextmanager
def inference_context(model):
    """
    A context where the model is temporarily changed to eval mode,
    and restored to previous mode afterwards.

    Args:
        model: a torch Module
    """
    training_mode = model.training
    model.eval()
    yield
    model.train(training_mode)


# class NormedLinear(nn.Module):
#     def __init__(self, feat_dim, temp):
#         super(NormedLinear, self).__init__()
#         self.weight = torch.nn.Parameter(torch.Tensor(feat_dim, feat_dim))
#         self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
#         self.temp = temp

#     def forward(self, x):
#         out = nn.functional.normalize(x, dim=1).mm(nn.functional.normalize(self.weight, dim=0)) / self.temp
#         return out


class ContrastLoss(nn.Module):
    def __init__(self, proj_head, n_labels, select, temperature=0.07, intra_align=False, scale=1.0, base_temp=None, scale_count=False):
        super(ContrastLoss, self).__init__()
        self.proj_head = proj_head
        self.n_labels = n_labels
        self.select = select
        self.temp = temperature
        self.intra_align = intra_align
        self.criterion = nn.CrossEntropyLoss()
        self.scale = scale
        if base_temp is None:
            self.base_temp = self.temp
        else:
            self.base_temp = base_temp
        self.scale_count = scale_count

    def forward(self, logits):
        labels_s, feat_s = logits['supervised']
        if 'supervised_target' in logits.keys():
            labels_t, feat_t = logits['supervised_target']
        else:
            labels_t, feat_t = logits['supervised']

        if self.select == 'all':
            labels_s = torch.cat(labels_s)
            feat_s = torch.cat(feat_s)
            nfeat_s = self.proj_head(feat_s)

            labels_t = torch.cat(labels_t)
            feat_t = torch.cat(feat_t)
            nfeat_t = self.proj_head(feat_t)

        # elif self.select == 'background':
        #     labels_s = labels_s
        #     feat_s = torch.cat(feat_s)
        #     labels_t = torch.cat(labels_t)
        #     feat_t = torch.cat(feat_t)

        if self.intra_align:
            feat_1 = torch.cat((nfeat_s, nfeat_t))
            labels_1 = torch.cat((labels_s, labels_t))
            feat_2 = torch.cat((nfeat_s, nfeat_t))
            labels_2 = torch.cat((labels_s, labels_t))
            # if self.scale_count:

        else:
            feat_1 = nfeat_s
            labels_1 = labels_s
            feat_2 = nfeat_t
            labels_2 = labels_t
        
        logits = torch.matmul(feat_2, feat_1.T)/self.temp
        targets = torch.eq(labels_2, labels_1.unsqueeze(1)).to(device=logits.device)
        exp_logits = torch.exp(logits)
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        mean_log_prob_pos = (targets * log_prob).sum(1) / (targets.sum(1) + 1e-12)
        loss = -mean_log_prob_pos.mean() * self.scale * (self.temp / self.base_temp)
       
        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # test = self.criterion()

        return loss

class SinkLoss(nn.Module):
    def __init__(self, proj_head, intra_align=False, scale=1.0, loss_func=None, use_negatives=False):
        super(SinkLoss, self).__init__()
        self.proj_head = proj_head
        self.intra_align = intra_align
        self.scale = scale
        self.loss_func = loss_func
        self.select = "all"
        self.use_negatives = use_negatives

    def forward(self, logits):
        labels_s, feat_s = logits['supervised']
        if 'supervised_target' in logits.keys():
            labels_t, feat_t = logits['supervised_target']
        else:
            return 0
            # labels_t, feat_t = logits['supervised']

        if self.select == 'all':
            labels_s = torch.cat(labels_s)
            feat_s = torch.cat(feat_s)
            nfeat_s = self.proj_head(feat_s)

            labels_t = torch.cat(labels_t)
            feat_t = torch.cat(feat_t)
            nfeat_t = self.proj_head(feat_t)

        n_classes = len(logits['supervised'][0])
        losses = []
        for i in range(n_classes):
            ids_t = torch.where(labels_t == i)[0]
            ids_sp = torch.where(labels_s == i)[0]
            ids_sn = torch.where(labels_s != i)[0]
            loss_i = self.loss_func(nfeat_t[ids_t,:],nfeat_s[ids_sp, :]) #- self.loss_func(nfeat_t[ids_t,:],nfeat_s[ids_sn, :])*self.use_negatives
            # losses.append(torch.where(loss_i>0, loss_i, 0))
            losses.append(loss_i)

        loss = sum(losses)/len(losses)*self.scale
       
        return loss
    
def temp_plots():
    import matplotlib.pyplot as plt
    import matplotlib
    from detectron2.utils.visualizer import Visualizer
    names = ['person','rider','car', 'truck', 'bus', 'train', 'mcycle','bcycle']
    curr_id = 0
    curr_data = unlabel_data_k
    # curr_data = all_label_data
    img_ = curr_data[curr_id]['image'].transpose(0,1).transpose(1,2)

    test_v = Visualizer(img_[:, :, [2,1,0]])
    temp = curr_data[curr_id]['instances'].gt_boxes
    labels = [names[x] for x in curr_data[curr_id]['instances'].gt_classes.tolist()]
    # temp = proposals_roih_unsup_k[curr_id].pred_boxes
    temp.tensor = temp.tensor.cpu()
    test_v.overlay_instances(boxes=temp, labels=labels)
    img = test_v.get_output().get_image()
    temp2 = [x.tolist() for x in temp]
    patches = [matplotlib.patches.Rectangle((x[0],x[1]),x[2]-x[0],x[3]-x[1],color='red',alpha=0.6) for x  in temp2]

    curr_data = label_data_k
    # curr_data = all_label_data
    img_ = curr_data[curr_id]['image'].transpose(0,1).transpose(1,2)
    test_v = Visualizer(img_[:, :, [2,1,0]])
    temp = curr_data[curr_id]['instances'].gt_boxes
    labels = [names[x] for x in curr_data[curr_id]['instances'].gt_classes.tolist()]
    # temp = proposals_roih_unsup_k[curr_id].pred_boxes
    temp.tensor = temp.tensor.cpu()
    test_v.overlay_instances(boxes=temp, labels=labels)
    img2 = test_v.get_output().get_image()
    temp2 = [x.tolist() for x in temp]
    patches2 = [matplotlib.patches.Rectangle((x[0],x[1]),x[2]-x[0],x[3]-x[1],color='red',alpha=0.6) for x  in temp2]

    test_v2 = Visualizer(img_[:, :, [2,1,0]])
    temp2 = curr_data[curr_id]['instances_gt'].gt_boxes
    labels2 = [names[x] for x in curr_data[curr_id]['instances_gt'].gt_classes.tolist()]
    # test_v2 = Visualizer(label_data_k[curr_id]['image'].transpose(0,1).transpose(1,2)[:, :, [2,1,0]])
    # temp2 = label_data_k[curr_id]['instances'].gt_boxes
    temp2.tensor = temp2.tensor.cpu()
    test_v2.overlay_instances(boxes=temp2, labels=labels2)
    img2 = test_v2.get_output().get_image()

    # test_v3= Visualizer(img_[:, :, [2,1,0]])
    # temp3 = old_pseudo_boxes[curr_id]
    # temp3.tensor = temp3.tensor.cpu()
    # test_v3.overlay_instances(boxes=temp3)
    # img3 = test_v3.get_output().get_image()

    test_v3 = Visualizer(img_[:, :, [2,1,0]])
    ids3 = torch.where(torch.logical_and(proposals_roih_unsup_k[curr_id].rpn_score > -5.0, proposals_roih_unsup_k[curr_id].scores > 0.5))[0]
    temp3 = proposals_roih_unsup_k[curr_id][ids3].pred_boxes
    labels3 = [names[x] for x in proposals_roih_unsup_k[curr_id][ids3].pred_classes.tolist()]
    temp3.tensor = temp3.tensor.cpu()
    # ids = torch.where(self.model.proposals_roih_temp[curr_id].gt_classes<8)[0]
    # temp3 = self.model.proposals_roih_temp[curr_id].proposal_boxes[ids]
    # logits_temp = self.model.proposals_roih_temp[curr_id].class_logits[ids]
    # logits3 = torch.gather(logits_temp, 1, self.model.proposals_roih_temp[curr_id].gt_classes[ids].unsqueeze(1))
    # new_ids = torch.where(logits3>0.8)[0]
    # temp3.tensor = temp3.tensor.cpu()[new_ids,:]
    # labels3 = self.model.proposals_roih_temp[curr_id].gt_classes[ids][new_ids].tolist()
    test_v3.overlay_instances(boxes=temp3, labels=labels3)
    # boxes = Boxes(torch.cat((temp3[sel_id].tensor, self.model.proposals_roih_temp[curr_id].gt_boxes[sel_id].tensor.detach().cpu())))
    # test_v3.overlay_instances(boxes=boxes, labels=[labels3[sel_id], self.model.proposals_roih_temp[curr_id].gt_classes[sel_id].detach().cpu().item()])
    img3 = test_v3.get_output().get_image()

    n = 100
    labels4 = list(range(n))
    # n = 100
    # labels4 = [n]
    test_v4 = Visualizer(img_[:, :, [2,1,0]])
    # temp4 = pesudo_proposals_rpn_unsup_k[curr_id].gt_boxes
    temp4 = proposals_rpn_unsup_k[curr_id].proposal_boxes
    # temp4 = self.model.proposals_rpn_temp[curr_id].proposal_boxes
    temp4.tensor = temp4.tensor.cpu()
    test_v4.overlay_instances(boxes=temp4[:n], labels=labels4)
    # test_v4.overlay_instances(boxes=temp4[n], labels=labels4)
    img4 = test_v4.get_output().get_image()

    fig, ax =  plt.subplots(1,2,figsize=[10,3])
    ax[0].imshow(img2)
    for patch in patches2: ax[0].add_patch(patch)
    ax[1].imshow(img)
    for patch in patches: ax[1].add_patch(patch)
    plt.tight_layout()
    plt.show()

    plt.figure();plt.imshow(img)
    plt.figure();plt.imshow(img2)
    plt.figure();plt.imshow(img3)
    plt.figure();plt.imshow(img4)
    plt.show()

def temp_pseudo_labels():
    import matplotlib.pyplot as plt
    from copy import copy

    file_in = '/home/marc/.adapt/output/test_v2_nom080_checks/pseudo_stats.csv'
    # data_, names = np.genfromtxt(file_in, names=True)
    data_ = np.genfromtxt(file_in, skip_header=1)
    data_ = data_[:40000,:]
    data_[1::2,0] = data_[1::2,0] + 0.5
    data2 = copy(data_)
    m = 0.01
    for i in range(1,data2.shape[0]):
        data2[i,1:] = data2[i-1,1:]*(1-m) + m*data_[i,1:]
    # ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcylce', 'bicycle', 'all']
    # iter person_gt person_raw50 person_raw75 person_rawn25 person_sel50 person_sel75 person_seln25
    # plt.figure();plt.plot(data2[:,0],data2[:,1]);plt.plot(data2[:,0],data2[:,2]);plt.plot(data2[:,0],data2[:,5]);plt.plot(data2[:,0],data2[:,7])
    plt.figure();plt.plot(data2[:,0],data2[:,15]);plt.plot(data2[:,0],data2[:,16]);plt.plot(data2[:,0],data2[:,19]);plt.plot(data2[:,0],data2[:,21])
    # plt.figure();plt.plot(data2[:,0],data2[:,22]);plt.plot(data2[:,0],data2[:,23]);plt.plot(data2[:,0],data2[:,26]);plt.plot(data2[:,0],data2[:,28])
    plt.figure();plt.plot(data2[:,0],data2[:,29]);plt.plot(data2[:,0],data2[:,30]);plt.plot(data2[:,0],data2[:,33]);plt.plot(data2[:,0],data2[:,35])


    plt.rcParams.update({'font.size': 10})
    fig, ax = plt.subplots(1,2,figsize=[8,4])
    ax[0].plot(data2[:,0],data2[:,15],label='GT');ax[0].plot(data2[:,0],data2[:,16],label='Raw PL @50');ax[0].plot(data2[:,0],data2[:,19],label='Filtered PL @50');ax[0].plot(data2[:,0],data2[:,21],label='False Pos. @25')
    ax[0].set_title('Car Pseudo-Labels, Foggy Cityscapes');ax[0].set_xlabel('Iterations');ax[0].set_ylabel('Label Count');ax[0].grid(True);ax[0].legend()
    ax[1].plot(data2[:,0],data2[:,29],label='GT');ax[1].plot(data2[:,0],data2[:,30],label='Raw PL @50');ax[1].plot(data2[:,0],data2[:,33],label='Filtered PL @50');ax[1].plot(data2[:,0],data2[:,35],label='False Pos. @25')
    ax[1].set_title('Bus Pseudo-Labels, Foggy Cityscapes');ax[1].set_xlabel('Iterations');ax[1].set_ylabel('Label Count');ax[1].grid(True);ax[1].legend()
    plt.tight_layout()
    
    plt.rcParams.update({'font.size': 12})
    labels = ['person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcylce', 'bicycle']
    counts_cs = [17919, 1781, 26963, 484, 379, 168, 737, 3675]
    counts_acdc_rain = [237, 21, 1433, 31, 37, 37, 77, 93]

    from matplotlib import gridspec
    fig = plt.figure(figsize=[8,4])
    gs = gridspec.GridSpec(1, 3, width_ratios=[1,1,0.4]) 
    ax0 = plt.subplot(gs[0]);ax1 = plt.subplot(gs[1]);ax2=plt.subplot(gs[2])
    ax0.pie(counts_cs,radius=1.3);ax0.set_title('Cityscapes labels, \n total=52106')
    pie = ax1.pie(counts_acdc_rain,radius=1.3);ax1.set_title('ACDC Rain labels, \n total=1966')
    ax2.axis("off")
    ax2.legend(pie[0],labels,loc='right')
    plt.tight_layout()
    plt.show()


    # plt.figure();plt.pie(counts_acdc_rain, labels=labels)
    # plt.show() 

# def temp_cam():
#     self.gradient = []
#     import cv2
#     # sel_id = 4
#     # sel_id2 = self.model.proposals_roih_temp[curr_id].orig_box[new_ids][sel_id]
#     grad_layer = 2
#     # c,h,w = unlabel_data_k[curr_id]['image'].shape
#     c,h,w = all_label_data[curr_id]['image'].shape
#     # single_box = proposals_rpn_unsup_k[curr_id].objectness_logits[sel_id]
#     # single_box = self.model.proposals_rpn_temp[curr_id][12].objectness_logits
#     maxes = self.model.proposal_generator.pred_objectness_logits[0][curr_id,:,:,:].max(dim=0)[0]
#     single_box = maxes[2,5]
#     # single_box = self.model.proposal_generator.pred_objectness_logits[0][curr_id,:,:,:][5,7,26]
#     # single_box = logits3[new_ids][sel_id]
#     single_box.backward(retain_graph=True)
#     self.gradient.reverse()
#     gradient = self.gradient[grad_layer][curr_id,:,:,:].squeeze().cpu().data.numpy()  # [C,H,W]
#     activations = self.activations[grad_layer][curr_id,:,:,:].squeeze().cpu().data.numpy()  # [C,H,W]
#     weight = np.mean(gradient, axis=(1, 2))  # [C]
#     cam = activations * weight[:, np.newaxis, np.newaxis]  # [C,H,W]
#     cam, cam_orig = self._postprocess_cam(cam, w, h)

#     test_cam = Visualizer(cam*255)
#     test_cam.overlay_instances(boxes=temp4[:n], labels=labels4)

#     test_both = Visualizer(img_[:, :, [2,1,0]]*cam[:,:,np.newaxis])
#     test_both.overlay_instances(boxes=temp4[:n], labels=labels4)
#     img_both = test_both.get_output().get_image()

#     img_cam = test_cam.get_output().get_image()
#     plt.figure();plt.imshow(img_cam)
#     # plt.figure();plt.imshow(img_both)
#     plt.figure();plt.imshow(img4)
#     plt.show()
    
#     curr_id = 2
#     test = self.activations[0][curr_id,:,:,:].sum(dim=0).cpu().data.numpy()
#     plt.figure();plt.imshow(test)
#     test = self.activations[1][curr_id,:,:,:].sum(dim=0).cpu().data.numpy()
#     plt.figure();plt.imshow(test)
#     test = self.activations[2][curr_id,:,:,:].sum(dim=0).cpu().data.numpy()
#     plt.figure();plt.imshow(test)
#     test = self.activations[3][curr_id,:,:,:].sum(dim=0).cpu().data.numpy()
#     plt.figure();plt.imshow(test)
#     plt.show()



        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # test = self.criterion()

        # vals = []
        # n = labels_s.shape[0]
        # intra_source = torch.matmul(feat_1, feat_1.T).detach().cpu() - torch.eye(n)
        # intra_target = torch.matmul(feat_2, feat_2.T).detach().cpu() - torch.eye(n)
        # inter = torch.matmul(feat_1, feat_2.T).detach().cpu()
        # labels_source = [labels_1[intra_source.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]
        # labels_target = [labels_2[intra_target.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]
        # labels_inter = [labels_1[inter.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]

        # order = [7,4,2,6,0,1,5,3]
        # import numpy as np

        # tols_source = np.array([intra_source[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]
        # tols_target = np.array([intra_target[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]
        # tols_inter = np.array([inter[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]

        # unif1 = -torch.log(torch.exp(-2*(feat_1.detach().cpu().unsqueeze(1) - feat_1.detach().cpu().unsqueeze(0)).norm(dim = 2)**2).mean())
        # unif2 = -torch.log(torch.exp(-2*(feat_2.detach().cpu().unsqueeze(1) - feat_2.detach().cpu().unsqueeze(0)).norm(dim = 2)**2).mean())
        # unif3 = -torch.log(torch.exp(-2*(feat_1.detach().cpu().unsqueeze(1) - feat_2.detach().cpu().unsqueeze(0)).norm(dim = 2)**2).mean())

        # vals = torch.zeros((3,8,8))
        # for i in range(8):
        #     vals[0,i,:] = torch.tensor([(labels_source[i] == x).sum() for x in range(8)])
        #     vals[1,i,:] = torch.tensor([(labels_target[i] == x).sum() for x in range(8)])
        #     vals[2,i,:] = torch.tensor([(labels_inter[i] == x).sum() for x in range(8)])

        # import matplotlib.pyplot as plt
        # vals_new = vals[:,order,:]
        # vals_new = vals_new[:,:,order]
        # cumsum = torch.cumsum(vals_new[0,:,:], axis = 1).numpy()
        # names = ['bicycle','bus','car', 'mcycle', 'person', 'rider', 'train','truck']
        # colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink', 'tab:gray']
        # fig, ax = plt.subplots()
        # for i in range(8):
        #     ax.bar(names, cumsum[:,7-i], color=colors[7-i], label=names[7-i])

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels))
        # plt.xlabel('Class of source queue feature')
        # plt.ylabel('Class of closest neighbor in source domain')

        # # plt.legend()
        # plt.tight_layout()

        # cumsum = torch.cumsum(vals_new[1,:,:], axis = 1).numpy()
        # names = ['bicycle','bus','car', 'mcycle', 'person', 'rider', 'train','truck']
        # colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink', 'tab:gray']
        # fig, ax = plt.subplots()
        # for i in range(8):
        #     ax.bar(names, cumsum[:,7-i], color=colors[7-i], label=names[7-i])

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels))
        # plt.xlabel('Class of target queue feature')
        # plt.ylabel('Class of closest neighbor in target domain')

        # # plt.legend()
        # plt.tight_layout()

        # cumsum = torch.cumsum(vals_new[2,:,:], axis = 1).numpy()
        # names = ['bicycle','bus','car', 'mcycle', 'person', 'rider', 'train','truck']
        # colors = ['tab:blue','tab:orange','tab:green','tab:red','tab:purple','tab:brown','tab:pink', 'tab:gray']
        # fig, ax = plt.subplots()
        # for i in range(8):
        #     ax.bar(names, cumsum[:,7-i], color=colors[7-i], label=names[7-i])

        # handles, labels = ax.get_legend_handles_labels()
        # ax.legend(reversed(handles), reversed(labels))
        # plt.xlabel('Class of target queue feature')
        # plt.ylabel('Class of closest neighbor in source domain')

        # # plt.legend()
        # plt.tight_layout()

        # plt.show()

# def temp():
#     import matplotlib.pyplot as plt
#     from detectron2.utils import visualizer

#     i = 1
#     fig_q = visualizer.VisImage(unlabel_data_q[i]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1])
#     fig_k = visualizer.VisImage(unlabel_data_k[i]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1])

#     unlabel_data_k[i]['instances'].pred_boxes = unlabel_data_k[i]['instances'].gt_boxes
#     unlabel_data_k[i]['instances'].pred_classes = unlabel_data_k[i]['instances'].gt_classes
#     unlabel_data_q[i]['instances'].pred_boxes = unlabel_data_q[i]['instances'].gt_boxes
#     unlabel_data_q[i]['instances'].pred_classes = unlabel_data_q[i]['instances'].gt_classes
#     fig_q_full = visualizer.Visualizer(unlabel_data_q[i]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1]).draw_instance_predictions(unlabel_data_q[i]['instances'])
#     fig_k_full = visualizer.Visualizer(unlabel_data_k[i]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1]).draw_instance_predictions(unlabel_data_k[i]['instances'])

#     plt.figure()
#     plt.imshow(fig_q_full.get_image())
#     plt.figure()
#     plt.imshow(fig_q_full.get_image())
#     plt.show()


#     import matplotlib.pyplot as plt
#     from detectron2.utils import visualizer

#     output_instances = outputs[0]['instances'][outputs[0]['instances'].scores > 0.5]
#     for key in output_instances.get_fields().keys():
#         if key == 'pred_boxes':
#             output_instances.get_fields()[key].tensor = output_instances.get_fields()[key].tensor.detach().cpu()
#         else:
#             output_instances.get_fields()[key] = output_instances.get_fields()[key].detach().cpu()
#     inputs[0]['instances'].pred_boxes = inputs[0]['instances'].gt_boxes
#     inputs[0]['instances'].pred_classes = inputs[0]['instances'].gt_classes
#     data = json.load(open('output/at_scaled/inference/pseudo_labels.json', 'r'))

#     fig = visualizer.VisImage(inputs[0]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1])
#     fig_full = visualizer.Visualizer(inputs[0]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1]).draw_instance_predictions(inputs[0]['instances'])
#     fig_pseud = visualizer.Visualizer(inputs[0]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1]).draw_dataset_dict(data[idx])
#     fig_pred = visualizer.Visualizer(inputs[0]['image'].cpu().transpose(0,2).transpose(0,1).numpy()[:,:,::-1]).draw_instance_predictions(output_instances)
#     plt.figure()
#     plt.imshow(fig_pred.get_image())
#     plt.tight_layout()
#     plt.figure()
#     plt.imshow(fig_pseud.get_image())
#     plt.tight_layout()
#     plt.figure()
#     plt.imshow(fig_full.get_image())
#     plt.tight_layout()
#     plt.show()



# import matplotlib.pyplot as plt

# dict_class = dict([(x.split('-')[1], []) for x in results['cityscapes_val']['bbox'].keys() if '-' in x])
# run_order = []
# # labels = list(dict_class.keys())
# for run in results.keys():
#     run_order.append(run)
#     for label in results[run]['bbox'].keys():
#         if '-' in label:
#             curr_class = label.split('-')[1]
#             dict_class[curr_class].append(results[run]['bbox'][label])

# vals = np.array(list(dict_class.values()))
# order = [3,4,0,1,2]
# order2 = np.argsort(list(dict_class.keys()))
# vals = vals[:,order]
# vals = vals[order2,:]
# class_labels = [list(dict_class.keys())[x] for x in [7, 4, 2, 6, 0, 1, 5, 3]]
# run_labels = [run_order[x] for x in order]

# N = len(dict_class)
# ind = np.arange(N) 
# width = 1/(len(order)+2)
# plt.figure()
# for i in range(len(order)):
#     plt.bar(ind+width*i, vals[:,i], width)
  
# plt.xlabel("Class")
# plt.ylabel('Score')
# plt.xticks(ind+2*width, class_labels)
# plt.legend(run_labels)
# plt.tight_layout()

# file_in = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/output/at_scaled/metrics.json'
# with open(file_in, 'r') as f_in:
#     temp = json.load(f_in)

# old_vals = np.zeros((8,5))
# order_old = [4,3,2,1,0]
# i = 0
# j = -1
# old_key = None
# for key in temp.keys():
#     if key[0] == 'c' and '-' in key:
#         curr_key = key.split('/')[0]
#         if old_key != curr_key:
#             old_key = curr_key
#             j += 1
#             i = 0
#         old_vals[i,j] = temp[key]
#         i += 1

# old_vals = old_vals[:,order_old]

# N = len(dict_class)
# ind = np.arange(N) 
# width = 1/(len(order)+2)
# plt.figure()
# for i in range(len(order)):
#     plt.bar(ind+width*i, old_vals[:,i], width)
  
# plt.xlabel("Class")
# plt.ylabel('Score')
# plt.xticks(ind+2*width, class_labels)
# plt.legend(run_labels)
# plt.tight_layout()

# N = len(dict_class)
# ind = np.arange(N) 
# width = 1/(len(order)+2)
# plt.figure()
# for i in range(len(order)):
#     plt.bar(ind+width*i, vals[:,i] - old_vals[:,i], width)
  
# plt.xlabel("Class")
# plt.ylabel('Score')
# plt.xticks(ind+2*width, class_labels)
# plt.legend(run_labels)
# plt.tight_layout()


# vals_t = nfeat_t[ids_t,:]
# vals_s = nfeat_s[ids_sp, :]

# dist1 = torch.cdist(vals_t,vals_s).mean()
# dist2 = torch.cdist(vals_t,vals_t).mean()
# dist3 = torch.cdist(vals_s,vals_s).mean()
# final = dist1*2-dist2-dist3


def temps():
    import matplotlib.pyplot as plt
    import matplotlib
    from detectron2.utils.visualizer import Visualizer
    names = ['person','rider','car', 'truck', 'bus', 'train', 'mcycle','bcycle']
    curr_id = 0
    curr_data = label_data_k
    # curr_data = all_label_data
    img_ = curr_data[curr_id]['image'].transpose(0,1).transpose(1,2)

    test_v2 = Visualizer(img_[:, :, [2,1,0]])
    temp2 = curr_data[curr_id]['instances_gt'].gt_boxes
    labels2 = [names[x] for x in curr_data[curr_id]['instances_gt'].gt_classes.tolist()]
    # test_v2 = Visualizer(label_data_k[curr_id]['image'].transpose(0,1).transpose(1,2)[:, :, [2,1,0]])
    # temp2 = label_data_k[curr_id]['instances'].gt_boxes
    temp2.tensor = temp2.tensor.cpu()
    test_v2.overlay_instances(boxes=temp2, labels=labels2)
    img2 = test_v2.get_output().get_image()

    target_point = [265,442]
    cnn_point = [8,14]
    dino_point = [20,29]
    cnn_feat1 = torch.nn.functional.normalize(cnn_feat1,dim=1)
    cnn_feat2 = torch.nn.functional.normalize(cnn_feat2,dim=1)
    sim2 = (cnn_feat2 * cnn_feat1[0,:,8,14].view(1,-1,1,1)).sum(dim=1)

    sim_d = (dino_feat1 * dino_feat1[0,:,20,29].view(1,-1,1,1)).sum(dim=1)

    (dino_features * selected_feature.view(1, -1, 1, 1)).sum(dim=1)

    import matplotlib.pyplot as plt
    import numpy as np
    file_paths = ['/'.join(('./plots/runs',x)) for x in os.listdir('./plots/runs')]
    runs = [np.genfromtxt(x,skip_header=1,delimiter=',') for x in file_paths]
    runs_2 = [runs[x] for x in [1,2,0,5,4,3]]
    legend = ['Pure S','DI','DI + PL','DINO 0.5 & 0.1', 'DINO 0.5 & 0.5', 'DINO 0.5 & 0.1 + PL']

    plt.rcParams.update({'font.size': 12})
    plt.figure()
    for idx,run in enumerate(runs_2): plt.plot(run[:,1],run[:,2],label=legend[idx])
    plt.xlabel('Iterations');plt.ylabel('Mean Accuracy @50');plt.legend();plt.tight_layout();plt.grid(True)
    plt.show()

def temp_run():
    import detectron2
    for name, module in self.model.named_modules():
        module_type = type(module)
        if module_type == torch.nn.modules.conv.Conv2d or module_type == detectron2.layers.wrappers.Conv2d:
            print(name, module.weight.sum().item())
        elif module_type == torch.nn.modules.batchnorm.BatchNorm2d:
            print(name, module.running_mean.sum().item())

def test_object_coloring():
    import matplotlib.pyplot as plt
    import pickle
    from segment_anything import sam_model_registry, SamPredictor
    

    self.model = self.model.eval()
    self.branch = "supervised"
    n_imgs = 100

    test_loaders = []
    for dataset_name in self.cfg.DATASETS.TEST:
        print(dataset_name)
        test_loaders.append(iter(build_detection_test_loader(self.cfg, dataset_name, mapper=DatasetMapper_test(self.cfg, True))))
    


    prototypes = np.zeros((5,8,data['instance_feats'].shape[1]))
    for lv1 in range(len(test_loaders)):
        for lv2 in range(8):
            ids = np.where((data['instance_dataset'] == self.cfg.DATASETS.TEST[lv1]) * (data['instance_class'] == lv2))[0]
            prototype = np.sum(data['instance_feats'][ids,:],axis=0)
            prototypes[lv1,lv2,:] = prototype / np.linalg.norm(prototype)

    proto_flat = np.swapaxes(prototypes,0,1).reshape(40,-1)
    sims_proto = np.matmul(proto_flat,proto_flat.T)
     
    img_data = next(test_loaders[1])
    img_ = img_data[0]['image'].detach().cpu().transpose(0,1).transpose(1,2).numpy()[:,:,[2,1,0]]
    img_dino = self.model.dino_head(img_data).squeeze(0).detach().cpu().numpy()
    c,h,w = img_dino.shape
    # sims = img_dino * prototypes[1,2,:][:,np.newaxis,np.newaxis]
    sims_new = np.matmul(prototypes[1,6,:][np.newaxis,:],img_dino.reshape(768,h*w)).reshape(1,h,w).squeeze()
    sims_old = np.matmul(prototypes[0,6,:][np.newaxis,:],img_dino.reshape(768,h*w)).reshape(1,h,w).squeeze()

    data_id = 0
    dataset_names = self.cfg.DATASETS.TEST[data_id]
    class_num = 2
    ids = np.where((np.isin(data['instance_dataset'], dataset_names)) * (data['instance_class'] == class_num))[0]
    test = data['instance_feats'][ids,:]
    sim_mat = np.matmul(test,test.T) - np.eye(test.shape[0])

    # plt.figure();plt.hist(sim_mat.flatten(),bins=np.arange(-0.1,1.1,0.1))
    sim_mat2 = np.matmul(test,prototypes[data_id,class_num,:][:,np.newaxis])
    sim_mat3 = np.matmul(test,prototypes[0,class_num,:][:,np.newaxis])




    from detectron2.utils.visualizer import Visualizer
    img_data = next(test_loaders[1])

    img_temp = img_data[0]['image'].detach().cpu().transpose(0,1).transpose(1,2)[:,:,[2,1,0]]
    images = self.model.preprocess_image(img_data)
    features = self.model.backbone(images.tensor)
    features_ = [features['vgg4']]
    pred_objectness_logits, pred_anchor_deltas = self.model.proposal_generator.rpn_head(features_)
    anchors = self.model.proposal_generator.anchor_generator(features_)

    pred_objectness_logits = [
        # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
        score.permute(0, 2, 3, 1).flatten(1)
        for score in pred_objectness_logits
    ]
    pred_anchor_deltas = [
        # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
        x.view(
            x.shape[0], -1, self.model.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1]
        )
        .permute(0, 3, 4, 1, 2)
        .flatten(1, -2)
        for x in pred_anchor_deltas
    ]
    proposals, keeps, proposal_idx, proposals_old = self.model.proposal_generator.predict_proposals(
        anchors, pred_objectness_logits, pred_anchor_deltas, images.image_sizes
    )
    deltas = (proposals_old[0].squeeze() - anchors[0].tensor).detach().cpu()
    sorted_ids = deltas[final_ids[:20],:].sum(dim=1).abs().argsort()
    final_ids = proposal_idx[0][0,keeps[0]].detach().cpu()

    test = proposals_old[0].squeeze().reshape(17,30,15,4)
    temp = test[8,3,:,:].detach().cpu()


    test_v = Visualizer(img_temp)
    temp = proposals[0][sorted_ids[:12]].proposal_boxes.tensor.detach().cpu()
    test_v.overlay_instances(boxes=temp, labels=range(temp.shape[0]))
    img2 = test_v.get_output().get_image()

    proposals_roih, ROI_predictions = self.model.roi_heads(
        images,
        features,
        proposals,
        targets=None,
        compute_loss=False,
        branch="unsup_data_weak",
    )

    temp2 = proposals_roih[0].pred_boxes.tensor.detach().cpu()
    test_v = Visualizer(img_temp)
    test_v.overlay_instances(boxes=temp2, labels=range(temp2.shape[0]))
    img3 = test_v.get_output().get_image()



    # # roi_head lower branch
    # _, detector_losses = self.roi_heads(
    #     images,
    #     features,
    #     proposals_rpn,
    #     compute_loss=True,
    #     targets=gt_instances,
    #     branch=branch,
    # ))

    

# pedestrian		0,93,192+x
# rider			0,97,170+x
# car			0,101,144+x
# truck			0,105,120+x
# bus			0,109,96+x
# train			0,121,24+x
# motorcycle		0,125,0+x
# bicycle			0,128,238+x

def box2mask(data,dino_patch,cnn_dims):
    c,h,w = data[0]['image'].shape
    masks = []
    scale_cnn = np.floor(h/cnn_dims[0]).astype(int)
    scales = [1,dino_patch,scale_cnn]
    for id, box in enumerate(data[0]['instances'].gt_boxes):
        box_class = data[0]['instances'].gt_classes[id].item()
        # box = box
        masks_box = []
        for scale in scales:
            xs = torch.arange(np.floor(h/scale))
            ys = torch.arange(np.floor(w/scale))
            corners = box/scale
            
            x_min = xs+1-corners[1]
            x_max = corners[3]-xs
            y_min = ys+1-corners[0]
            y_max = corners[2]-ys

            x_scale_min = torch.where(x_min < 1, x_min, 1).clamp(0,1)
            x_scale_max = torch.where(x_max < 1, x_max, 1).clamp(0,1)
            y_scale_min = torch.where(y_min < 1, y_min, 1).clamp(0,1)
            y_scale_max = torch.where(y_max < 1, y_max, 1).clamp(0,1)
            masks_box.append(torch.matmul((x_scale_min*x_scale_max).unsqueeze(1),(y_scale_min*y_scale_max).unsqueeze(0)))
        
        masks.append((box_class, masks_box[0].float(), masks_box[1].float().numpy(), data[0]['file_name'], masks_box[2].float().numpy()))

    return masks

def ttemp_():
    import cv2
    image = cv2.imread('/home/marc/.adapt/datasets/acdc/rgb_anon_trainvaltest/rgb_anon/rain/train/GP010400/GP010400_frame_000543_rgb_anon.png')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(10,10))
    plt.imshow(image)
    plt.axis('on')
    plt.show()
    ## Selecting objects with SAM

    # import sys
    # sys.path.append("..")
    from segment_anything import sam_model_registry, SamPredictor, SamAutomaticMaskGenerator

    sam_checkpoint = "../../segment-anything/segment_anything/weights/sam_vit_h_4b8939.pth"
    model_type = "vit_h"

    device = "cuda"

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    mask_generator = SamAutomaticMaskGenerator(sam, points_per_side=20)
    masks = mask_generator.generate(image)


def temp123():
    import matplotlib.pyplot as plt
    import matplotlib
    from detectron2.utils.visualizer import Visualizer

    # scale = (1330+672*2)/4096
    scale = (1330+742*2)/(1080*2+1920)

    names = ['person','rider','car', 'truck', 'bus', 'train', 'mcycle','bcycle']
    img_ = tens2img(unlabel_data_q[0]['image'])
    boxes1 = unlabel_data_q[0]['instances_gt'].gt_boxes.tensor.numpy()
    labels1 = unlabel_data_q[0]['instances_gt'].gt_classes.tolist()
    tfs = unlabel_data_q[0]['tf_data']
    ids = torch.where(self.dino_pseudogt[unlabel_data_q[0]['image_id']]['instances_dino'].scores > 0.75)[0]
    boxes2_ = self.dino_pseudogt[unlabel_data_q[0]['image_id']]['instances_dino'][ids].pred_boxes.numpy()
    boxes2 = tfs.apply_box(boxes2_)
    labels2 = self.dino_pseudogt[unlabel_data_q[0]['image_id']]['instances_dino'][ids].pred_classes.tolist()

    # img_ = inputs[0]['image'].transpose(0,1).transpose(1,2)
    # temp1 = inputs[0]['instances'].gt_boxes
    # labels1 = [names[x] for x in inputs[0]['instances'].gt_classes.tolist()]
    # temp1.tensor = temp1.tensor.cpu()

    test_v = Visualizer(img_)
    test_v.overlay_instances(boxes=boxes1, labels=labels1)
    img1 = test_v.get_output().get_image()

    test_v = Visualizer(img_)
    test_v.overlay_instances(boxes=boxes2, labels=labels2)
    img2 = test_v.get_output().get_image()

    plt.figure();plt.imshow(img1)
    plt.figure();plt.imshow(img2)
    plt.show()

def tens2img(x):
    return x.transpose(0,1).transpose(1,2).cpu().numpy()[:,:,[2,1,0]]


    # import matplotlib.pyplot as plt
    # import matplotlib
    # from detectron2.utils.visualizer import Visualizer
    # names = ['person','rider','car', 'truck', 'bus', 'train', 'mcycle','bcycle']
    # curr_id = 0
    # curr_data = unlabel_data_k
    # # curr_data = all_label_data
    # img_ = curr_data[curr_id]['image'].transpose(0,1).transpose(1,2)

    # test_v = Visualizer(img_[:, :, [2,1,0]])
    # temp = curr_data[curr_id]['instances'].gt_boxes
    # labels = [names[x] for x in curr_data[curr_id]['instances'].gt_classes.tolist()]
    # temp2 = gt_unlabel_k[curr_id].gt_boxes
    # labels2 = [names[x] for x in gt_unlabel_k[curr_id].gt_classes.tolist()]
    # # temp = proposals_roih_unsup_k[curr_id].pred_boxes
    # temp.tensor = temp.tensor.cpu()
    # test_v.overlay_instances(boxes=temp, labels=labels)
    # img = test_v.get_output().get_image()
    # test_v2 = Visualizer(img_[:, :, [2,1,0]])
    # temp2.tensor = temp2.tensor.cpu()
    # test_v2.overlay_instances(boxes=temp2, labels=labels2)
    # img2 = test_v2.get_output().get_image()

# def test1245(a):
#     self.model = self.model.eval()
#     from PIL import Image 
#     with torch.no_grad():
        
#         image_path = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes/leftImg8bit/train/aachen/aachen_000010_000019_leftImg8bit.png'
#         img1_ = np.array(Image.open(image_path).resize((1596,798)))
#         img1_[392:406,490:504,:] = [255,100,200]
#         img1 = torch.tensor(img1_)[:,:798,[2,1,0]].transpose(1,2).transpose(0,1)
#         # image_path = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/cityscapes_foggy/leftImg8bit/train/aachen/aachen_000012_000019_leftImg8bit_foggy_beta_0.02.png'
#         image_path = '/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/datasets/bdd/images/train/0a3e70d1-a515ffaf.jpg'
#         img2_ = np.array(Image.open(image_path).resize((1596,798)))
#         img2 = torch.tensor(img2_)[:,-798:,[2,1,0]].transpose(1,2).transpose(0,1)
#         img_dict = [{'image':img1},{'image':img2}]
#         dino_feat = self.model.dino_head(img_dict).detach().cpu()
#         cnn_feat = self.model.backbone(self.model.preprocess_image(img_dict).tensor)['res5'].detach().cpu()
#         ndino = torch.nn.functional.normalize(dino_feat,dim=1).transpose(1,3).transpose(1,2)
#         ncnn = torch.nn.functional.normalize(dino_feat,dim=1).transpose(1,3).transpose(1,2)
#         point = ndino[0,28,35,:]
#         dino2 = (ndino*point).sum(dim=-1).numpy()
#         pointc = ncnn[0,25,32,:]
#         cnn2 = (ncnn*pointc).sum(dim=-1).numpy()


#         plt.figure()
#         plt.imshow(dino2[0,:,:])
#         plt.figure()
#         plt.imshow(img1_[:,:798,:])
#         plt.figure()
#         plt.imshow(dino2[1,:,:])
#         plt.figure()
#         plt.imshow(img1_[:,-798:,:])

#         plt.rcParams["font.family"] = "serif"
#         plt.rcParams.update({'font.size': 14})
#         fig, axs = plt.subplots(2,2)
#         axs[0,0].imshow(img1_[:,:798,:])
#         axs[0,1].imshow(img2_[:,-798:,:])
#         axs[1,0].imshow(dino2[0,:,:])
#         axs[1,1].imshow(dino2[1,:,:])
#         axs[0,0].set_axis_off()
#         axs[0,1].set_axis_off()
#         axs[1,0].set_axis_off()
#         axs[1,1].set_axis_off()
#         axs[0,0].set_title('Cityscapes')
#         axs[0,1].set_title('BDD100k')
#         plt.show()

#         plt.figure()
#         plt.imshow(dino2[1,:,:],vmax=1)
#         plt.figure()
#         plt.imshow(cnn2[0,:,:])
#         plt.figure()
#         plt.imshow(cnn2[1,:,:],vmax=1)
#         plt.show()
    

#     cnn_rescale_feat = self.model.dino_align(cnn_feat, dino_feat)
#     curr_masks = self.get_fg_mask(data, patch_size=self.model.dino_head.patch_size,cnn_dims=cnn_feat.shape[2:], box_mask=use_bbox)[0]
#     curr_feats = [(dino_feat.detach().cpu().numpy()*x[2]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]
#     curr_feats_cnn = [(cnn_feat.detach().cpu().numpy()*x[-1]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]
#     curr_feats_cnn_project = [(cnn_rescale_feat.detach().cpu().numpy()*x[2]).squeeze(0).sum(axis=(1,2)) for x in curr_masks]


from typing import List, Union
def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None], get_outs=False
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        if isinstance(model, nn.Module):
            stack.enter_context(inference_context(model))
        stack.enter_context(torch.no_grad())

        start_data_time = time.perf_counter()
        if get_outs:
            output_list = []
        for idx, inputs in enumerate(data_loader):
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if get_outs:
                instances = Instances(outputs[0]['instances'].image_size)
                instances.pred_boxes = outputs[0]['instances'].pred_boxes.tensor.cpu()
                instances.scores = outputs[0]['instances'].scores.cpu()
                instances.pred_classes = outputs[0]['instances'].pred_classes.cpu()
                out_dict = {'file_name':inputs[0]['file_name'], 'image_id':inputs[0]['image_id'], 'height':inputs[0]['height'], 'width':inputs[0]['width'], 'instances_dino':instances}
                output_list.append(out_dict)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    if get_outs:
        return results, output_list
    else:
        return results