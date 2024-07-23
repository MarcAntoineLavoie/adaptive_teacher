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
    if args.use_old_cfg: #os.path.isfile('/'.join([os.getcwd(),args.output_dir,'config.yaml'])):
        cfg.merge_from_file('/'.join([args.output_dir,'config.yaml']))
        cfg.OUTPUT_DIR = args.output_dir
        cfg.SOLVER.IMG_PER_BATCH_LABEL = 2
        cfg.SOLVER.IMG_PER_BATCH_UNLABEL = 2
        # cfg.TEST.EVAL_PERIOD = 20
        # cfg.DATASETS.TEST = ("cityscapes_val","cityscapes_foggy_val","ACDC_val_rain","ACDC_val_fog")
        # cfg.DATASETS.TEST = ("cityscapes_val",)
        cfg.DATASETS.TEST = ("cityscapes_val","ACDC_val_rain")
    else:
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # cfg = scale_configs(cfg)
    cfg['DATALOADER']['NUM_WORKERS'] = 8
    # cfg.SOLVER.CHECKPOINT_PERIOD = 50
    # cfg['MODEL']['PIXEL_STD'] = [1.0, 1.0, 1.0]
    if args.acdc_type is not None:
        cfg.DATASETS.TEST = ("cityscapes_val","ACDC_val_{}".format(args.acdc_type))
        cfg.DATASETS.TRAIN_UNLABEL = ("ACDC_train_{}".format(args.acdc_type),)
    # cfg.DATASETS.TRAIN_UNLABEL = ("cityscapes_foggy_train")
    cfg.DATASETS.TEST = ("cityscapes_val","ACDC_val_fog","ACDC_val_night","ACDC_val_rain","ACDC_val_snow")
    # cfg.INPUT.MIN_SIZE_TRAIN = (800,)
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
            res = Trainer.test(cfg, ensem_ts_model.modelTeacher)
            # res = Trainer.test(cfg, ensem_ts_model.modelStudent)

        else:
            model = Trainer.build_model(cfg)
            DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            # DetectionCheckpointer(model, save_dir=args.output_dir).resume_or_load(   
	     cfg.MODEL.WEIGHTS, resume=args.resume
            )
            res = Trainer.test(cfg, model)
        return res

    split_name = cfg.OUTPUT_DIR.rsplit('/',1)
    name = split_name[-1]
    run_dir = split_name[0]
    # name = "config_test"
    conf_file='/'.join((cfg.OUTPUT_DIR,'config.yaml'))
    with open(conf_file) as yaml_in:
        config_dict = yaml.safe_load(yaml_in)
        if conf_file is None:
            flat_dict = {}
        else:
            flat_dict = flatten_dict(config_dict)
    run_id_file = '/'.join((cfg.OUTPUT_DIR,'run_id.txt'))
    if os.path.isfile(run_id_file):
        with open(run_id_file, "r") as text_file:
            run_id = text_file.read().rstrip()
    else:
        run_id = os.urandom(4).hex()
        with open(run_id_file, "w") as text_file:
            print(run_id, file=text_file)
    if args.use_wandb:
        run = wandb.init(
            # set the wandb project where this run will be logged
            project="test_dino",
            name=name,
            dir=run_dir,
            # id=run_id,
            # resume="allow",
            # track hyperparameters and run metadata
            config=flat_dict
        )
    else:
        run = None

    trainer = Trainer(cfg, wandb_run=run)
    trainer.resume_or_load(resume=args.resume)
    out = trainer.train()

    if args.use_wandb:
        run.finish()

    return out

if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--acdc-type", default=None, help="acdc run type")
    parser.add_argument("--use-wandb", default=False, help="use wandb to log run")
    args = parser.parse_args()
    url_parts = args.dist_url.rsplit(':',1)
    url_parts[1] = str(randint(0,1000) + int(url_parts[1]))
    args.dist_url = (':').join(url_parts)
    args.use_wandb=True

    #   --num-gpus 8
    #   --config configs/faster_rcnn_VGG_cross_city.yaml\
    #   OUTPUT_DIR output/exp_city

    # args.num_gpus = 1
    # args.config_file = './configs/faster_rcnn_VGG_cross_city_prob.yaml'
    # args.config_file = './configs/faster_rcnn_VGG_cross_city_tiny.yaml'
    # args.config_file = './configs/faster_rcnn_VGG_cross_city.yaml'
    # args.config_file = './configs/faster_rcnn_VGG_cross_city_test.yaml'
    # args.config_file = './configs/faster_rcnn_R101_cross_clipart_v2.yaml'
    # args.config_file = './configs/faster_rcnn_RES_cross_city_tiny.yaml'
    # args.resume = False
    args.resume = True

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
    # args.output_dir = 'output/dino/test_dino_nom_dino050_smask020_w010/'
    # args.output_dir = 'output/dino/test_dino_rain_dino050_010/'
    # args.output_dir = 'output/dino/test_dino_rain_dino050_050/'
    # args.output_dir = 'output/dino/test_dino_nom_dino010/'
    # args.output_dir = 'output/dino/test_dino_nom_dino050/'
    # args.output_dir = 'output/dino/test_dino_nom_dino100/'
    # args.output_dir = 'output/dino/test_nom_noDAINST/'
    # args.output_dir = 'output/test_dinonom_DA/'
    # args.output_dir = 'output/dino/test_dino_rain_dino050_010_PL/'
    # args.output_dir = 'output/test_v2_nom080_checks/'
    # args.output_dir = 'output/dino/test_dino_rain_DA/'
    # args.output_dir = 'output/dino/test_dino_rain_DA_PL/'
    # args.output_dir = 'output/dino/test_nom_noDAINST/'
    # args.output_dir = 'output/dino/dgx_dino/test_dino_rain_dino050_000_augs_dgx/'
    # args.output_dir = 'output/dino/dgx_dino/test_dino_nom_augs_dgx/'
    # args.output_dir = 'output/dino/dgx_dino/test_dino_TR_rain_DA_PL_augs_TU/'
    # args.output_dir = 'output/dino/dgx_dino/test_dino_TR_rain_DA_PL_augs_TU/'
    # args.output_dir = 'output/dino/dino_tests/test_no_pseudo/'
    # args.output_dir = 'output/dino/dino_tests/test_all_pseudo/'
    # args.output_dir = 'output/dino/dino_tests/test_print6000/'
    # args.output_dir = 'output/dino/dino_teacher_rate/dino_nom050_EMA0999_v1/'

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
