# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from detectron2.checkpoint.c2_model_loading import align_and_update_state_dicts
from detectron2.checkpoint import DetectionCheckpointer

# for load_student_model
from typing import Any
from fvcore.common.checkpoint import _strip_prefix_if_present, _IncompatibleKeys

import logging
from collections import OrderedDict
from torch.nn.parallel import DistributedDataParallel
import os
from detectron2.utils import comm
from urllib.parse import urlparse

class DetectionTSCheckpointer(DetectionCheckpointer):
    def _load_model(self, checkpoint):
        if checkpoint.get("__author__", None) == "Caffe2":
            # pretrained model weight: only update student model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                model_state_dict = self.model.modelStudent.state_dict()
                renamed_ckpt = align_and_update_state_dicts(
                    model_state_dict,
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
                # checkpoint["model"] = model_state_dict
                checkpoint["model"] = renamed_ckpt

            # for non-caffe2 models, use standard ways to load it
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif all("vgg" in x for x in checkpoint["model"].keys()):
            # pretrained vgg weights, update student model
            model_state_dict = self.model.modelStudent.state_dict()
            renamed_ckpt = align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            # checkpoint["model"] = model_state_dict
            checkpoint["model"] = renamed_ckpt

            # for non-caffe2 models, use standard ways to load it
            incompatible = self._load_student_model(checkpoint)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif 'modelStudent.model.transformer.encoder.layers.0.norms.0.weight' in checkpoint['model'].keys(): # old form in checkpoint
            if 'modelStudent.transformer.encoder.layers.0.norms.0.weight' in self.model.state_dict().keys(): # new form in model
                new_key = []
                new_vals = []
                for key, value in checkpoint['model'].items():
                    new_key.append(key.replace('.model.','.'))
                    new_vals.append(value)
                checkpoint['model'] = OrderedDict(zip(new_key,new_vals))
            incompatible = super()._load_model(checkpoint)
            return incompatible

        elif 'backbone.net.rope_win.freqs_cos' in checkpoint['model'].keys(): # eva02 full
            incompatible = self._load_student_model(checkpoint, wrapped_model=True)

            model_buffers = dict(self.model.modelStudent.model.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif 'lm_head.bias' in checkpoint['model'].keys(): # transformer
            new_key = []
            new_vals = []
            for key, value in checkpoint['module'].items():
                new_key.append('backbone.net.' + key)
                new_vals.append(value)
            checkpoint['model'] = OrderedDict(zip(new_key,new_vals))
            incompatible = self._load_student_model(checkpoint, wrapped_model=True)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        elif "cls_token" in checkpoint["model"].keys():
            # pretrained vgg weights, update student model
            model_state_dict = self.model.modelStudent.backbone.state_dict()
            renamed_ckpt = align_and_update_state_dicts(
                model_state_dict,
                checkpoint["model"],
                c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
            )
            # checkpoint["model"] = model_state_dict
            checkpoint["model"] = renamed_ckpt

            # for non-caffe2 models, use standard ways to load it
            incompatible = self._load_student_model(checkpoint, backbone_only=True)

            model_buffers = dict(self.model.modelStudent.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

        else:  # whole model
            if checkpoint.get("matching_heuristics", False):
                self._convert_ndarray_to_tensor(checkpoint["model"])
                # convert weights by name-matching heuristics
                model_state_dict = self.model.state_dict()
                align_and_update_state_dicts(
                    model_state_dict,
                    checkpoint["model"],
                    c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
                )
                checkpoint["model"] = model_state_dict 
            # for non-caffe2 models, use standard ways to load it
            incompatible = super()._load_model(checkpoint)

            model_buffers = dict(self.model.named_buffers(recurse=False))
            for k in ["pixel_mean", "pixel_std"]:
                # Ignore missing key message about pixel_mean/std.
                # Though they may be missing in old checkpoints, they will be correctly
                # initialized from config anyway.
                if k in model_buffers:
                    try:
                        incompatible.missing_keys.remove(k)
                    except ValueError:
                        pass
            return incompatible

    def _load_student_model(self, checkpoint: Any, backbone_only=False, wrapped_model=False) -> _IncompatibleKeys:  # pyre-ignore
        checkpoint_state_dict = checkpoint.pop("model")
        self._convert_ndarray_to_tensor(checkpoint_state_dict)

        # if the state_dict comes from a model that was wrapped in a
        # DataParallel or DistributedDataParallel during serialization,
        # remove the "module" prefix before performing the matching.
        _strip_prefix_if_present(checkpoint_state_dict, "module.")

        # work around https://github.com/pytorch/pytorch/issues/24139
        if backbone_only:
            model_state_dict = self.model.modelStudent.backbone.state_dict()        
        if wrapped_model:
            model_state_dict = self.model.modelStudent.model.state_dict()
        else:
            model_state_dict = self.model.modelStudent.state_dict()
        incorrect_shapes = []
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    incorrect_shapes.append((k, shape_checkpoint, shape_model))
                    checkpoint_state_dict.pop(k)
        # pyre-ignore        
        if backbone_only:
            incompatible = self.model.modelStudent.backbone.load_state_dict(checkpoint_state_dict, strict=False)
        if wrapped_model:
            incompatible = self.model.modelStudent.model.load_state_dict(checkpoint_state_dict, strict=False)
        else:
            incompatible = self.model.modelStudent.load_state_dict(checkpoint_state_dict, strict=False)
        # incompatible = self.model.modelStudent.load_state_dict(
        #     checkpoint_state_dict, strict=False
        # )
        return _IncompatibleKeys(
            missing_keys=incompatible.missing_keys,
            unexpected_keys=incompatible.unexpected_keys,
            incorrect_shapes=incorrect_shapes,
        )
    
    def load1(self, path, *args, **kwargs):
        assert self._parsed_url_during_load is None
        need_sync = False
        logger = logging.getLogger(__name__)
        logger.info("[DetectionCheckpointer] Loading from {} ...".format(path))

        if path and isinstance(self.model, DistributedDataParallel):
            path = self.path_manager.get_local_path(path)
            has_file = os.path.isfile(path)
            all_has_file = comm.all_gather(has_file)
            if not all_has_file[0]:
                raise OSError(f"File {path} not found on main worker.")
            if not all(all_has_file):
                logger.warning(
                    f"Not all workers can read checkpoint {path}. "
                    "Training may fail to fully resume."
                )
                # TODO: broadcast the checkpoint file contents from main
                # worker, and load from it instead.
                need_sync = True
            if not has_file:
                path = None  # don't load if not readable

        if path:
            parsed_url = urlparse(path)
            self._parsed_url_during_load = parsed_url
            path = parsed_url._replace(query="").geturl()  # remove query from filename
            path = self.path_manager.get_local_path(path)
        ret = self.load2(path, *args, **kwargs)

        if need_sync:
            logger.info("Broadcasting model states from main worker ...")
            self.model._sync_params_and_buffers()
        self._parsed_url_during_load = None  # reset to None
        return ret

    def load2(self, path, checkpointables=None):
        """
        Load from the given checkpoint.

        Args:
            path (str): path or url to the checkpoint. If empty, will not load
                anything.
            checkpointables (list): List of checkpointable names to load. If not
                specified (None), will load all the possible checkpointables.
        Returns:
            dict:
                extra data loaded from the checkpoint that has not been
                processed. For example, those saved with
                :meth:`.save(**extra_data)`.
        """
        if not path:
            # no checkpoint provided
            self.logger.info("No checkpoint found. Initializing model from scratch")
            return {}
        self.logger.info("[Checkpointer] Loading from {} ...".format(path))
        if not os.path.isfile(path):
            path = self.path_manager.get_local_path(path)
            assert os.path.isfile(path), "Checkpoint {} not found!".format(path)

        checkpoint = self._load_file(path)
        if 'eva' in path.rsplit('/',1)[1]:
            new_key = []
            new_vals = []
            for key, value in checkpoint['module'].items():
                new_key.append('backbone.net.' + key)
                new_vals.append(value)
            checkpoint['model'] = OrderedDict(zip(new_key,new_vals))
        incompatible = self._load_model(checkpoint)
        if (
            incompatible is not None
        ):  # handle some existing subclasses that returns None
            self._log_incompatible_keys(incompatible)

        for key in self.checkpointables if checkpointables is None else checkpointables:
            if key in checkpoint:
                self.logger.info("Loading {} from {} ...".format(key, path))
                obj = self.checkpointables[key]
                obj.load_state_dict(checkpoint.pop(key))

        # return any further checkpoint data
        return checkpoint


# class DetectionCheckpointer(Checkpointer):
#     """
#     Same as :class:`Checkpointer`, but is able to handle models in detectron & detectron2
#     model zoo, and apply conversions for legacy models.
#     """

#     def __init__(self, model, save_dir="", *, save_to_disk=None, **checkpointables):
#         is_main_process = comm.is_main_process()
#         super().__init__(
#             model,
#             save_dir,
#             save_to_disk=is_main_process if save_to_disk is None else save_to_disk,
#             **checkpointables,
#         )

#     def _load_file(self, filename):
#         if filename.endswith(".pkl"):
#             with PathManager.open(filename, "rb") as f:
#                 data = pickle.load(f, encoding="latin1")
#             if "model" in data and "__author__" in data:
#                 # file is in Detectron2 model zoo format
#                 self.logger.info("Reading a file from '{}'".format(data["__author__"]))
#                 return data
#             else:
#                 # assume file is from Caffe2 / Detectron1 model zoo
#                 if "blobs" in data:
#                     # Detection models have "blobs", but ImageNet models don't
#                     data = data["blobs"]
#                 data = {k: v for k, v in data.items() if not k.endswith("_momentum")}
#                 return {"model": data, "__author__": "Caffe2", "matching_heuristics": True}

#         loaded = super()._load_file(filename)  # load native pth checkpoint
#         if "model" not in loaded:
#             loaded = {"model": loaded}
#         return loaded

#     def _load_model(self, checkpoint):
#         if checkpoint.get("matching_heuristics", False):
#             self._convert_ndarray_to_tensor(checkpoint["model"])
#             # convert weights by name-matching heuristics
#             model_state_dict = self.model.state_dict()
#             align_and_update_state_dicts(
#                 model_state_dict,
#                 checkpoint["model"],
#                 c2_conversion=checkpoint.get("__author__", None) == "Caffe2",
#             )
#             checkpoint["model"] = model_state_dict
#         # for non-caffe2 models, use standard ways to load it
#         super()._load_model(checkpoint)