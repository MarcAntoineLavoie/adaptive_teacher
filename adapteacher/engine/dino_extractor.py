import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
from PIL import Image
from detectron2.structures.masks import polygons_to_bitmask, BitMasks, PolygonMasks

class dino_preprocessing():
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self, pixel_mean, pixel_std, is_RGB=True):
        if is_RGB:
            pixel_mean = torch.tensor(pixel_mean)[[2,1,0]]
            pixel_std = torch.tensor(pixel_std)[[2,1,0]]

        normalize = T.Normalize(mean=pixel_mean, std=pixel_std)        
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)

class DinoV2VitFeatureExtractor(nn.Module):
    """
    DINO V2 Vision Transformer Feature Extractor.
    """
    def __init__(self, cfg, cnn_dim, model_name='dinov2_vitb14', normalize_feature=True):
        super(DinoV2VitFeatureExtractor, self).__init__()
        if 'vgg' in cfg.MODEL.BACKBONE:
            self.preprocessing = dino_preprocessing(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD, is_RGB=True)
        else:
            pixel_std = [57.375, 57.120, 58.395]
            self.preprocessing = dino_preprocessing(cfg.MODEL.PIXEL_MEAN, pixel_std, is_RGB=True)
        self.is_RGB = True
        self.normalize_feature = normalize_feature
        if "v2" not in model_name:
            # 'dino_vitb16'
            self.model_name = model_name
            # self.encoder = torch.hub.load('facebookresearch/dino:main', model_name)
            # local_dir = '/home/marc/.cache/torch/hub/facebookresearch_dino_main'
            local_dir = "adapteacher/engine/dinov1/hub/facebookresearch_dino_main"
            self.encoder = torch.hub.load(local_dir, source='local', model=model_name, path=model_name)
            for param in self.encoder.parameters():
                param.requires_grad = False
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = int(model_name.rsplit('vit',1)[-1][1:])
            assert (cfg.INPUT.DINO_PATCH_SIZE == self.patch_size), f'Config patch size is {cfg.INPUT.DINO_PATCH_SIZE} while loaded model has a patch size of {self.patch_size}'
        else:
            dino_v2_models = {
                "dinov2_vits14": (14, 384, dinov2_vits14), # patch_size, output dims, function name to create model
                "dinov2_vitb14": (14, 768, dinov2_vitb14),
                "dinov2_vitl14": (14, 1024, dinov2_vitl14),
                "dinov2_vitg14": (14, 1536, dinov2_vitg14),
            }
            # model name to model weights
            name_to_weights = {"dinov2_vits14": "dinov2_vits14_pretrain.pth",
                            "dinov2_vitb14": "dinov2_vitb14_pretrain.pth",
                            "dinov2_vitl14": "dinov2_vitl14_pretrain.pth",
                            "dinov2_vitg14": "dinov2_vitg14_pretrain.pth"
            }
            # load model on cpu
            self.model_name = model_name
            assert (
                self.model_name in dino_v2_models.keys()
            ), f"class DinoV2VitFeatureExtractor(nn.Module): is only available for {dino_v2_models.keys()}"
            path_to_pretrained_weights = "adapteacher/engine/dino_weights/" + model_name + "_pretrain.pth"
            assert (
                os.path.exists(path_to_pretrained_weights)
            ), f"DINO v2 pretrained model path {path_to_pretrained_weights} does not exist!"
            print(f"Model Path: {path_to_pretrained_weights}")
            
            patch_size, embed_dim, model_func_name = dino_v2_models[self.model_name]
            # load model
            self.encoder = model_func_name(pretrained=False)
            self.encoder.load_state_dict(torch.load(path_to_pretrained_weights))
            for param in self.encoder.parameters():
                param.requires_grad = False
            # ensure 
            assert self.encoder.embed_dim == embed_dim
            self.embed_dim = self.encoder.embed_dim
            self.patch_size = patch_size


    def forward(self, x):
        x = torch.stack([img['image'] for img in x], dim=0)[:,[2,1,0],:,:].float()
        x = self.preprocessing(x).to(device=next(self.encoder.parameters()).device)
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

        return x_grid_features

class DinoAlignHead(nn.Module):
    def __init__(self, cnn_dim, dino_dim, normalize_feature=True, head_type="Linear", instance_masks=False):
        super(DinoAlignHead, self).__init__()
        self.normalize_feature = normalize_feature
        self.instance_masks = instance_masks
        if head_type=='attention':
            self.projection_layer = MHALayer(cnn_dim, dino_dim)
        elif head_type=='MLP':
            self.projection_layer = nn.Sequential(nn.Conv2d(cnn_dim, 512, 1, 1),
                                                   nn.ReLU(),
                                                   nn.Conv2d(512, dino_dim, 1, 1))
        else:
            self.projection_layer = nn.Conv2d(cnn_dim, dino_dim, 1, 1)

    def forward(self, feat_cnn, feat_dino):
        return self.project_RCNN_feat(feat_cnn, feat_dino)
    
    def project_RCNN_feat(self, feat_cnn, feat_dino):
        h, w = feat_dino.shape[2:]
        feat_cnn = self.projection_layer(feat_cnn)
        feat_cnn = F.interpolate(feat_cnn, (h,w), mode='bilinear')
        if self.normalize_feature:
            feat_cnn = F.normalize(feat_cnn, p=2, dim=1)
        return feat_cnn
    
    def dino_loss(self, feat_cnn, feat_dino, return_sim=False, fg_mask=None, gt_data=None):
        if self.instance_masks and gt_data is not None:
            device = feat_cnn.device
            dino_instances = []
            cnn_instances = []
            for idx, img in enumerate(gt_data):
                h,w = img['image'].shape[1:]
                if type(img['instances'].gt_masks) is PolygonMasks:
                    gt_masks = torch.tensor(np.concatenate([np.expand_dims(polygons_to_bitmask(x,h,w).astype(float),0) for x in img['instances'].gt_masks.polygons])).to(device=device)
                else:
                    gt_masks = img['instances'].gt_masks.tensor
                scaled_masks = torch.nn.functional.interpolate(gt_masks.unsqueeze(1),size=feat_dino.shape[2:],mode='bicubic',antialias=True)
                dino_instances.append((scaled_masks * feat_dino[idx,:,:,:]).sum(dim=2).sum(dim=2))
                cnn_instances.append((scaled_masks * feat_cnn[idx,:,:,:]).sum(dim=2).sum(dim=2))
            dino_instances = torch.nn.functional.normalize(torch.cat(dino_instances), dim=1)
            cnn_instances = torch.nn.functional.normalize(torch.cat(cnn_instances), dim=1)
            sim = torch.matmul(cnn_instances.unsqueeze(-2), dino_instances.unsqueeze(-1))
            loss = (1-sim).mean()
                # scaled_masks = [torch.nn.functional.interpolate(x.float().unsqueeze(0).unsqueeze(0),size=feat_dino.shape,mode='bicubic',antialias=True) for x in img['instances'].gt_masks]
                # dino_feats = [feat_dino[id,:,:,:]*mask for mask in ]
        else:
            if self.normalize_feature:
                feat_cnn = feat_cnn.permute((0,2,3,1)).unsqueeze(-2)
                feat_dino = feat_dino.permute((0,2,3,1)).unsqueeze(-1)
                sim = torch.matmul(feat_cnn, feat_dino)
                if fg_mask is not None:
                    loss = ((1-sim.squeeze())*fg_mask.to(device=sim.device)).mean()
                else:
                    loss = (1-sim).mean()
            else:
                sim = torch.norm(feat_cnn-feat_dino, dim=1)
                if fg_mask is not None:
                    loss = (sim*fg_mask.to(device=sim.device)).mean() / 50
                else:
                    loss = sim.mean() / 50

        if return_sim:
            return loss, sim
        else:
            return loss
        
class MHALayer(nn.Module):
    def __init__(self, cnn_dim, dino_dim):
        super(MHALayer, self).__init__()

        self.attn_layer = nn.MultiheadAttention(cnn_dim, num_heads=4, batch_first=True)
        self.projection = nn.Conv2d(cnn_dim, dino_dim, 1, 1)

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.reshape(b,c,h*w).transpose(1,2)
        x, _ = self.attn_layer(x, x, x, need_weights=False)
        x = self.projection(x.transpose(1,2).reshape(b,c,h,w))
        return x 
        
