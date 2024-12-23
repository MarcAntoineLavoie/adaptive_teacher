from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyConfig, instantiate
from detectron2.engine import (
    default_argument_parser,
    default_setup,
)
from detectron2.engine.defaults import create_ddp_model

def build_eva_model(config_file='/home/marc/Documents/trailab_work/uda_detect/adaptive_teacher/dino_eva/configs/dino-eva-02/new_dino_eva_02_vitdet_b_4attn_1024_lrd0p7_4scale_12ep.py'):
    args = default_argument_parser().parse_args()
    cfg = LazyConfig.load(config_file)
    default_setup(cfg, args)
    model = instantiate(cfg.model)
    model.to(cfg.train.device)
    model = create_ddp_model(model)
    DetectionCheckpointer(model).load(cfg.train.init_checkpoint)

    return model

