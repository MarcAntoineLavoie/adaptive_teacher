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
from adapteacher.data.dataset_mapper import DatasetMapperTwoCropSeparate, DatasetMapperWithWeakAugs, DatasetMapperWithStrongAugs, DatasetMapperTwoCropSeparate_detect
from adapteacher.engine.hooks import LossEvalHook
from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel
from adapteacher.checkpoint.detection_checkpoint import DetectionTSCheckpointer
from adapteacher.solver.build import build_lr_scheduler
from adapteacher.evaluation import PascalVOCDetectionEvaluator, COCOEvaluator

from .probe import OpenMatchTrainerProbe
import copy


from detectron2.evaluation import (
    DatasetEvaluator,
    inference_on_dataset,
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
    def __init__(self, cfg):
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
        optimizer = self.build_optimizer(cfg, model)

        # create an teacher model
        model_teacher = self.build_model(cfg)
        self.model_teacher = model_teacher

        # For training, wrap with DDP. But don't need this for inference.
        if comm.get_world_size() > 1:
            model = DistributedDataParallel(
                model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
            )

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
        self.register_hooks(self.build_hooks_final())

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
            model_student.roi_heads.build_queues(n_classes=n_labels, n_samples=200, feat_dim=512, base_count=cfg.SEMISUPNET.ALIGN_BASE_COUNT)
            model_student.roi_heads.align_proposals = self.align_proposals
            model_student.roi_heads.current_proposals = {}
            model_student.roi_heads.use_bg = cfg.SEMISUPNET.ALIGN_USE_BG
            model_student.roi_heads.sampling = cfg.SEMISUPNET.ALIGN_SUBSAMPLING
            model_student.roi_heads.points_per_proposals = cfg.SEMISUPNET.ALIGN_POINTS_PER_PROPOSALS

            self.proj_head = model_student.roi_heads.box_predictor.proj_head

            self.model_teacher.roi_heads.align_proposals = self.align_proposals
            self.model_teacher.roi_heads.current_proposals = {}

            temperature  = cfg.SEMISUPNET.ALIGN_PARAM
            n_labels = 9
            select = 'all'
            if cfg.SEMISUPNET.ALIGN_LOSS == 'MMD':
                loss_func = geomloss.SamplesLoss(loss='energy')
                self.align_loss = SinkLoss(self.proj_head, loss_func=loss_func, scale=cfg.SEMISUPNET.ALIGN_WEIGHT, intra_align=cfg.SEMISUPNET.ALIGN_INTRA)
            elif cfg.SEMISUPNET.ALIGN_LOSS == 'contrast':
                self.align_loss = ContrastLoss(self.proj_head, n_labels, select, temperature=temperature, scale=cfg.SEMISUPNET.ALIGN_WEIGHT,
                                                base_temp=cfg.SEMISUPNET.ALIGN_PARAM_BASE, intra_align=cfg.SEMISUPNET.ALIGN_INTRA,
                                                scale_count=cfg.SEMISUPNET.ALIGN_SCALE_COUNT)
            self.use_gt_proposals = cfg.SEMISUPNET.USE_GT_PROPOSALS

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
        mapper = DatasetMapperTwoCropSeparate_detect(cfg, True)
        return build_detection_semisup_train_loader_two_crops(cfg, mapper)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        return build_lr_scheduler(cfg, optimizer)

    def train(self):
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
        self, proposals_rpn_unsup_k, cur_threshold, proposal_type, psedo_label_method=""
    ):
        list_instances = []
        num_proposal_output = 0.0
        for proposal_bbox_inst in proposals_rpn_unsup_k:
            # thresholding
            if self.select_iou:
                if proposal_type == 'roih':
                    proposal_bbox_inst, overlap = self.threshold_self(proposal_bbox_inst, thres=cur_threshold,)
                else:
                    proposal_bbox_inst, overlap = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            elif psedo_label_method == "thresholding":
                proposal_bbox_inst, overlap = self.threshold_bbox(
                    proposal_bbox_inst, thres=cur_threshold, proposal_type=proposal_type
                )
            else:
                raise ValueError("Unkown pseudo label boxes methods")
            num_proposal_output += len(proposal_bbox_inst)
            list_instances.append(proposal_bbox_inst)
        num_proposal_output = num_proposal_output / len(proposals_rpn_unsup_k)
        return list_instances, num_proposal_output, overlap

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
            label_data_q = self.clean_detections(label_data_q, label_regions)
            unlabel_data_q = self.clean_detections(unlabel_data_q, unlabel_regions)
        else:
            label_data_q, label_data_k, unlabel_data_q, unlabel_data_k = data

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
        if self.iter < self.cfg.SEMISUPNET.BURN_UP_STEP:

            # input both strong and weak supervised data into model
            label_data_q.extend(label_data_k)
            record_dict, _, _, _ = self.model(
                label_data_q, branch="supervised")

            if self.align_proposals:
                # proposals_t = self.model.roi_heads.keep_proposals["supervised_target"]
                loss_align = self.align_proposals_loss()
                if not self.cfg.SEMISUPNET.ALIGN_INTRA:
                    loss_align['loss_align'] *= 1e-12
                record_dict.update(loss_align)

            # weight losses
            loss_dict = {}
            for key in record_dict.keys():
                if key[:4] == "loss":
                    loss_dict[key] = record_dict[key] * 1
            losses = sum(loss_dict.values())

        else:
            if self.iter == self.cfg.SEMISUPNET.BURN_UP_STEP:
                # update copy the the whole model
                self._update_teacher_model(keep_rate=0.00)
                # self.model.build_discriminator()

            elif (
                self.iter - self.cfg.SEMISUPNET.BURN_UP_STEP
            ) % self.cfg.SEMISUPNET.TEACHER_UPDATE_ITER == 0:
                self._update_teacher_model(
                    keep_rate=self.cfg.SEMISUPNET.EMA_KEEP_RATE)

            record_dict = {}

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 
            gt_unlabel_k = self.get_label(unlabel_data_k)
            # gt_unlabel_q = self.get_label_test(unlabel_data_q)
            

            #  0. remove unlabeled data labels
            if not self.use_gt_proposals:
                unlabel_data_q = self.remove_label(unlabel_data_q)
            unlabel_data_k = self.remove_label(unlabel_data_k)

            #  1. generate the pseudo-label using teacher model
            with torch.no_grad():
                (
                    _,
                    proposals_rpn_unsup_k,
                    proposals_roih_unsup_k,
                    _,
                ) = self.model_teacher(unlabel_data_k, branch="unsup_data_weak")

            ######################## For probe #################################
            # import pdb; pdb. set_trace() 

            # probe_metrics = ['compute_fp_gtoutlier', 'compute_num_box']
            # probe_metrics = ['compute_num_box']  
            # analysis_pred, _ = self.probe.compute_num_box(gt_unlabel_k,proposals_roih_unsup_k,'pred')
            # record_dict.update(analysis_pred)
            ######################## For probe END #################################

            #  2. Pseudo-labeling
            cur_threshold = self.cfg.SEMISUPNET.BBOX_THRESHOLD

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
                proposals_roih_unsup_k, cur_threshold, "roih", "thresholding"
            )
            joint_proposal_dict["proposals_pseudo_roih"] = pesudo_proposals_roih_unsup_k
            record_dict.update({'iou_overlap':overlap})

            # 3. add pseudo-label to unlabeled data

            if not self.use_gt_proposals:
                unlabel_data_q = self.add_label(
                    unlabel_data_q, joint_proposal_dict["proposals_pseudo_roih"]
                )
            unlabel_data_k = self.add_label(
                unlabel_data_k, joint_proposal_dict["proposals_pseudo_roih"]
            )

            all_label_data = label_data_q + label_data_k
            all_unlabel_data = unlabel_data_q

            # 4. input both strongly and weakly augmented labeled data into student model
            record_all_label_data, _, _, _ = self.model(
                all_label_data, branch="supervised"
            )
            record_dict.update(record_all_label_data)

            # 5. input strongly augmented unlabeled data into model
            record_all_unlabel_data, _, _, _ = self.model(
                all_unlabel_data, branch="supervised_target"
            )
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
            for key in record_dict.keys():
                if key.startswith("loss"):
                    if key == "loss_rpn_loc_pseudo" or key == "loss_box_reg_pseudo":
                        # pseudo bbox regression <- 0
                        loss_dict[key] = record_dict[key] * 0
                    elif key[-6:] == "pseudo":  # unsupervised loss
                        loss_dict[key] = (
                            record_dict[key] *
                            self.cfg.SEMISUPNET.UNSUP_LOSS_WEIGHT
                        )
                    elif (
                        key == "loss_D_img_s" or key == "loss_D_img_t"
                    ):  # set weight for discriminator
                        # import pdb
                        # pdb.set_trace()
                        loss_dict[key] = record_dict[key] * self.cfg.SEMISUPNET.DIS_LOSS_WEIGHT #Need to modify defaults and yaml
                    else:  # supervised loss
                        loss_dict[key] = record_dict[key] * 1

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
            return _last_eval_results_student

        def test_and_save_results_teacher():
            self._last_eval_results_teacher = self.test(
                self.cfg, self.model_teacher)
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

    def clean_detections(self, data_strong, regions):
        for idx in range(len(data_strong)):
            bbox_regions = torch.tensor([x for x in regions[idx] if x])
            if not bbox_regions.shape[0]:
                continue
            bbox_regions[:,2:] = bbox_regions[:,2:] + bbox_regions[:,:2]
            bbox_gt = data_strong[idx]['instances'].gt_boxes.tensor

            m = bbox_gt.shape[0]
            n = bbox_regions.shape[0]
            if n > 0:
                cs1 = torch.zeros((m,n,4))
                cs1[:,:,2:] = torch.min(bbox_gt[:, None, 2:], bbox_regions[:, 2:])
                cs1[:,:,:2] = torch.max(bbox_gt[:, None, :2], bbox_regions[:, :2])
                dcs1 = cs1[:,:,2:] - cs1[:,:,:2]
                check1 = (dcs1 > 0).all(dim=2, keepdim=True)
                cs1 = cs1*check1
                intersection = (dcs1*check1).prod(dim=2).sum(dim=1)

                if n > 1:
                    n2 = comb(n,2)
                    cs2 = torch.zeros((m,n2,4))
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
                        cs3 = torch.zeros((m,1,4))
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
            
        return data_strong    

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
            return self.test_relative()

        ret.append(hooks.EvalHook(0,
                   eval_relative))

        if comm.is_main_process():
            # run writers in the end, so that evaluation metrics are written
            ret.append(hooks.PeriodicWriter(self.build_writers(), period=20))
        return ret
    
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
    def __init__(self, proj_head, intra_align=False, scale=1.0, loss_func=None):
        super(SinkLoss, self).__init__()
        self.proj_head = proj_head
        self.intra_align = intra_align
        self.scale = scale
        self.loss_func = loss_func
        self.select = "all"

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

        n_classes = len(logits['supervised'][0])
        losses = []
        for i in range(n_classes):
            ids_t = torch.where(labels_t == i)[0]
            ids_sp = torch.where(labels_s == i)[0]
            ids_sn = torch.where(labels_s != i)[0]
            loss_i = self.loss_func(nfeat_t[ids_t,:],nfeat_s[ids_sp, :]) - self.loss_func(nfeat_t[ids_t,:],nfeat_s[ids_sn, :])
            # losses.append(torch.where(loss_i>0, loss_i, 0))
            losses.append(loss_i)

        loss = sum(losses)/len(losses)*self.scale
       
        return loss

        # # compute mean of log-likelihood over positive
        # mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        # log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # test = self.criterion()

        # vals = []
        # intra_source = torch.matmul(feat_1, feat_1.T).detach().cpu() - torch.eye(1600)
        # intra_target = torch.matmul(feat_2, feat_2.T).detach().cpu() - torch.eye(1600)
        # inter = torch.matmul(feat_1, feat_2.T).detach().cpu()
        # labels_source = [labels_1[intra_source.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]
        # labels_target = [labels_2[intra_target.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]
        # labels_inter = [labels_1[inter.max(dim=0)[1][200*i:200+200*i]] for i in range(8)]

        # order = [7,4,2,6,0,1,5,3]
        # import numpy as np

        # tols_source = np.array([intra_source[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]
        # tols_target = np.array([intra_target[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]
        # tols_inter = np.array([inter[200*i:200+200*i, 200*i:200+200*i].sum()/199/200 for i in range(8)])[order]

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