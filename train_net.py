#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch

from adapteacher import add_ateacher_config
from adapteacher.engine.trainer import ATeacherTrainer, BaselineTrainer

# hacky way to register
from adapteacher.modeling.meta_arch.rcnn import TwoStagePseudoLabGeneralizedRCNN, DAobjTwoStagePseudoLabGeneralizedRCNN
from adapteacher.modeling.meta_arch.vgg import build_vgg_backbone  # noqa
from adapteacher.modeling.proposal_generator.rpn import PseudoLabRPN
from adapteacher.modeling.roi_heads.roi_heads import StandardROIHeadsPseudoLab
import adapteacher.data.datasets.builtin

from adapteacher.modeling.meta_arch.ts_ensemble import EnsembleTSModel

from math import floor 
import os

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_ateacher_config(cfg)
    if args.use_old_cfg and os.path.isfile('/'.join([os.getcwd(),args.output_dir,'config.yaml'])):
        cfg.merge_from_file('/'.join([args.output_dir,'config.yaml']))
        cfg.OUTPUT_DIR = args.output_dir
        cfg.SOLVER.IMG_PER_BATCH_LABEL = 2
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 2
        # cfg.TEST.EVAL_PERIOD = 20
        # cfg.DATASETS.TEST = ("cityscapes_val",)
    else:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg = scale_configs(cfg)
    cfg['DATALOADER']['NUM_WORKERS'] = 8
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def scale_configs(cfg):
    base_batch = 16
    curr_batch = cfg['SOLVER']['IMG_PER_BATCH_LABEL']
    curr_batch = 6
    ratio = curr_batch / base_batch
    base_iteration = cfg['SOLVER']['MAX_ITER']
    curr_iter = int(round(base_iteration / ratio, -4))
    base_lr = cfg['SOLVER']['BASE_LR']
    curr_lr = base_lr * ratio
    base_steps = cfg['SOLVER']['STEPS']
    curr_steps = tuple([int(round(x/ratio, -4)) for x in base_steps])
    base_burn = cfg['SEMISUPNET']['BURN_UP_STEP']
    curr_burn = int(round(base_burn / ratio,-4))
    base_eval = cfg['TEST']['EVAL_PERIOD']
    curr_eval = base_eval * round(1/ratio)
    base_ckpt = cfg['SOLVER']['CHECKPOINT_PERIOD']
    curr_ckpt = base_ckpt * floor(1/ratio)

    cfg['SOLVER']['MAX_ITER'] = curr_iter
    cfg['SOLVER']['BASE_LR'] = curr_lr
    cfg['SOLVER']['STEPS'] = curr_steps
    cfg['SEMISUPNET']['BURN_UP_STEP'] = curr_burn
    cfg['TEST']['EVAL_PERIOD'] = curr_eval
    cfg['SOLVER']['CHECKPOINT_PERIOD'] = curr_ckpt

    return cfg


def main(args):
    cfg = setup(args)
    if cfg.SEMISUPNET.Trainer == "ateacher":
        Trainer = ATeacherTrainer
    elif cfg.SEMISUPNET.Trainer == "baseline":
        Trainer = BaselineTrainer
    else:
        raise ValueError("Trainer Name is not found.")

    if args.eval_only:
        if cfg.SEMISUPNET.Trainer == "ateacher":
            model = Trainer.build_model(cfg)
            model_teacher = Trainer.build_model(cfg)
            ensem_ts_model = EnsembleTSModel(model_teacher, model)

            DetectionCheckpointer(
                ensem_ts_model, save_dir=cfg.OUTPUT_DIR
            ).resume_or_load(cfg.MODEL.WEIGHTS, resume=args.resume)
            # res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            # DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(   
	     cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == "__main__":
    args = default_argument_parser().parse_args()

    #   --num-gpus 8\
    #   --config configs/faster_rcnn_VGG_cross_city.yaml\
    #   OUTPUT_DIR output/exp_city

    # args.num_gpus = 1
    args.config_file = './configs/faster_rcnn_VGG_cross_city_prob.yaml'
    # args.config_file = './configs/faster_rcnn_VGG_cross_city.yaml'
    # args.config_file = './configs/faster_rcnn_VGG_cross_city_test.yaml'
    # args.config_file = './configs/faster_rcnn_R101_cross_clipart_v2.yaml'
    args.resume = False
    # args.resume = True

    # args.OUTPUT_DIR = './output/temp1'

    # args.eval_only = True

    # args.output_dir = 'output/test_v2_align_contrast100'
    # args.output_dir = 'output/test_v2_iou70_min30_temp'
    # args.output_dir = 'output/test_v2_iou70_min30'
    # args.output_dir = 'output/test_v2_7030_L1'
    # args.output_dir = 'output/test_v2_align_contrast010_noDA'
    # args.output_dir = 'output/test_v2_align_contrast010/'
    # args.output_dir = 'output/test_v2_align_contrast100/'
    # args.output_dir = 'output/test_v2_align_contrast010_temp100/'
    # args.output_dir = 'output/test_v2_align_contrast010_gtprops/'
    # args.output_dir = 'output/test_v2_short_align010_centre_gtprops_mmd/'
    # args.output_dir = 'output/test_dino000_cityonly/'

    # args.use_old_cfg = True
    args.use_old_cfg = False

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
