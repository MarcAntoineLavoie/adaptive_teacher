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
from random import randint
import sys
import wandb
import yaml

from collections.abc import MutableMapping
def flatten_dict(dictionary, parent_key='', separator='.'):
    items = []
    for key, value in dictionary.items():
        new_key = parent_key + separator + key if parent_key else key
        if isinstance(value, MutableMapping):
            items.extend(flatten_dict(value, new_key, separator=separator).items())
        else:
            items.append((new_key, value))
    return dict(items)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.set_new_allowed(True)
    add_ateacher_config(cfg)
    cfg.merge_from_file('/'.join([args.output_dir,'config.yaml']))
    cfg.merge_from_list(args.opts)
    cfg.INPUT.MIN_SIZE_TRAIN = (800,)
    cfg.DATASETS.TEST = ("{}".format(args.gen_dataset),)
    cfg.OUTPUT_DIR = args.output_dir
    if cfg.SEMISUPNET.DINO_BASE:
        scale = cfg.INPUT.DINO_PATCH_SIZE
        if cfg.INPUT.MAX_SIZE_TEST % scale:
            cfg.INPUT.MAX_SIZE_TEST =  floor(cfg.INPUT.MAX_SIZE_TEST / scale)*scale
    cfg.freeze()
    # default_setup(cfg, args)
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
            res = Trainer.test_with_gen(Trainer, cfg, ensem_ts_model.modelTeacher, gen_labels=args.gen_labels, gen_dir=args.gen_dir)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            # DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(   
	     cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--gen-labels", default=False, help="for labeller, set to True to generate labels in eval mode")
    parser.add_argument("--gen-dir", default='', help="for labeller, dir for generated labels")
    parser.add_argument("--gen-dataset", default='cityscapes_foggy_train', help="dataset name for which labels are to be generated")
    parser.add_argument("--output-dir", default=None, help="path to saved checkpoint dir")
    args = parser.parse_args()
    url_parts = args.dist_url.rsplit(':',1)
    url_parts[1] = str(randint(0,1000) + int(url_parts[1]))
    args.dist_url = (':').join(url_parts)

    args.eval_only = True
    args.gen_labels = True
    args.gen_dir = ''
    args.output_dir = '/media/marc/data_checks1/results_dino/dino_head/dino_twin_dinov2_nom_vitl_v1/'


    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
