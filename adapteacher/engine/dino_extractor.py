import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import numpy as np
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
from PIL import Image
from detectron2.structures.masks import polygons_to_bitmask, BitMasks, PolygonMasks
import detectron2.utils.comm as comm

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
    def __init__(self, cfg, model_name='dinov2_vitb14', normalize_feature=True, freeze=True):
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
            if freeze:
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
    def __init__(self, cfg, cnn_dim, dino_dim, normalize_feature=True):
        super(DinoAlignHead, self).__init__()
        head_type = cfg.SEMISUPNET.DINO_HEAD
        self.instance_masks = cfg.SEMISUPNET.DINO_INSTANCE_MASK
        self.loss_type = cfg.SEMISUPNET.DINO_ALIGN_LOSS
        self.normalize_feature = normalize_feature
        self.proj_dim = cfg.SEMISUPNET.DINO_PROJ_DIM
        if cfg.SEMISUPNET.DINO_PROJ_GELU:
            nl_layer = nn.GELU()
        else:
            nl_layer = nn.ReLU()
        if head_type=='attention':
            self.projection_layer = MHALayer(cnn_dim, dino_dim)
        elif head_type=='MLP':
            self.projection_layer = nn.Sequential(nn.Conv2d(cnn_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, dino_dim, 1, 1))
        elif head_type=='MLP3':
            self.projection_layer = nn.Sequential(nn.Conv2d(cnn_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, self.proj_dim, 1, 1),
                                                   nl_layer,
                                                   nn.Conv2d(self.proj_dim, dino_dim, 1, 1))
        else:
            self.projection_layer = nn.Conv2d(cnn_dim, dino_dim, 1, 1)
        
        if self.loss_type == 'contrast' :
            self.scale_loss = False
            self.queue_length = cfg.SEMISUPNET.DINO_CONT_QUEUE_LENGTH
            self.contrast_temp = cfg.SEMISUPNET.DINO_CONT_TEMP
            self.default_temp = 0.1
            self.curr_id = 0
            self.register_buffer("queue_dino", torch.randn(self.queue_length,dino_dim))
            self.register_buffer("queue_cnn", torch.randn(self.queue_length,dino_dim))
            self.ignore_closest = cfg.SEMISUPNET.DINO_CONT_IGNORE_CLOSEST

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
                    if not len(img['instances'].gt_masks):
                        continue
                    gt_masks = torch.tensor(np.concatenate([np.expand_dims(polygons_to_bitmask(x,h,w).astype(float),0) for x in img['instances'].gt_masks.polygons])).to(device=device)
                else:
                    gt_masks = img['instances'].gt_masks.tensor
                scaled_masks = torch.nn.functional.interpolate(gt_masks.unsqueeze(1),size=feat_dino.shape[2:],mode='bicubic',antialias=True)
                ids = torch.where(scaled_masks.squeeze(1).sum(1).sum(1))[0]
                scaled_masks = scaled_masks[ids,:,:,:]
                dino_instances.append((scaled_masks * feat_dino[idx,:,:,:]).sum(dim=2).sum(dim=2))
                cnn_instances.append((scaled_masks * feat_cnn[idx,:,:,:]).sum(dim=2).sum(dim=2))
            if self.normalize_feature:
                dino_instances = torch.nn.functional.normalize(torch.cat(dino_instances), dim=1)
                cnn_instances = torch.nn.functional.normalize(torch.cat(cnn_instances), dim=1)
                if self.loss_type == 'similarity':
                    sim = torch.matmul(cnn_instances.unsqueeze(-2), dino_instances.unsqueeze(-1))
                    loss = (1-sim).mean()
                elif self.loss_type == "contrast":
                    loss, sim = self.contrast_loss(dino_instances, cnn_instances)
                    # scaled_masks = [torch.nn.functional.interpolate(x.float().unsqueeze(0).unsqueeze(0),size=feat_dino.shape,mode='bicubic',antialias=True) for x in img['instances'].gt_masks]
                    # dino_feats = [feat_dino[id,:,:,:]*mask for mask in ]
            else:
                dino_instances = torch.cat(dino_instances,dim=0)
                cnn_instances = torch.cat(cnn_instances,dim=0)
                sim = torch.linalg.norm(dino_instances-cnn_instances, dim=1, ord=2)
                loss = sim.mean() / 100
        else:
            if self.normalize_feature:
                feat_cnn = feat_cnn.permute((0,2,3,1)).unsqueeze(-2)
                feat_dino = feat_dino.permute((0,2,3,1)).unsqueeze(-1)
                if self.loss_type == 'similarity':
                    sim = torch.matmul(feat_cnn, feat_dino)
                    if fg_mask is not None:
                        loss = ((1-sim.squeeze())*fg_mask.to(device=sim.device)).mean()
                    else:
                        loss = (1-sim).mean()
                elif self.loss_type == "contrast":
                    loss, sim = self.contrast_loss(feat_dino, feat_cnn)
            else:
                sim = torch.linalg.norm(feat_cnn-feat_dino, dim=1, ord=2)
                if fg_mask is not None:
                    loss = (sim*fg_mask.to(device=sim.device)).mean() / 100
                else:
                    loss = sim.mean() / 100

        if return_sim:
            return loss, sim
        else:
            return loss
    
    def contrast_loss(self, dino_instances, cnn_instances):
        with torch.no_grad():
            dino_queue = torch.clone(self.queue_dino).to(device=dino_instances.device)
            dino_queue /= dino_queue.norm(dim=1).unsqueeze(1)
            self.update_queue(dino_instances)
        dino_full = torch.cat((dino_instances, dino_queue))

        sim = torch.matmul(cnn_instances, dino_full.T)
        sims_scaled = sim / self.contrast_temp
        if self.scale_loss:
            sims_scaled -= torch.max(sims_scaled, dim=1, keepdim=True)
        exp_sim = torch.exp(sims_scaled)
        if self.ignore_closest:
            with torch.no_grad():
                k = int(self.ignore_closest/100 * sim.shape[1])
                n = sim.shape[0]
                neg_sim = torch.clone(exp_sim)
                neg_sim[range(n),range(n)] = -10
                vals, ids = torch.topk(neg_sim,k,sorted=False)
                mask = torch.ones_like(exp_sim)
                mask.scatter_(1,ids,0)
            exp_sim=mask*exp_sim
        log_prob = sims_scaled - torch.log(exp_sim.sum(1, keepdim=True))
        loss = -torch.diagonal(log_prob).mean() * (self.contrast_temp/self.default_temp)
        return loss, sim[:,:sim.shape[0]]
    
    @torch.no_grad()
    def update_queue(self, dino_instances):
        if comm.get_world_size() > 1:
            all_instances = gather_across_devices(dino_instances)
        else:
            all_instances = dino_instances
        n = all_instances.shape[0]
        if (self.curr_id+n) >= self.queue_length:
            n1 = self.queue_length - self.curr_id
            self.queue_dino[self.curr_id:,:] = all_instances[:n1,:]
            n2 = n-n1
            self.queue_dino[:n2,:] = all_instances[n1:,:]
            self.curr_id = n1
        else:
            self.queue_dino[self.curr_id:self.curr_id+n,:] = all_instances
            self.curr_id += n

@torch.no_grad()
def gather_across_devices(tensor1, pad=1024):
    """
    From https://github.com/zhangyifei01/MoCo-v2-SupContrast/blob/main/moco/builder_in.py#L185
    """
    if pad:
        n, c =  tensor1.shape
        n_tensor = torch.tensor(n,device=tensor1.device)
        pad_tensor = torch.zeros((pad, c), device=tensor1.device, dtype=tensor1.dtype)
        pad_tensor[:n,:] = tensor1
        tensor1 = pad_tensor
    tensors_gather = [torch.ones_like(tensor1) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor1, async_op=False)
    n_gather = [torch.ones_like(n_tensor) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(n_gather, n_tensor, async_op=False)
    unpad_tensor = [x[:y,:] for x,y in zip(tensors_gather, n_gather)]

    output = torch.cat(unpad_tensor, dim=0)
    return output


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
        
