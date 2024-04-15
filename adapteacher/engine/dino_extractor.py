import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from dinov2.hub.backbones import dinov2_vits14, dinov2_vitb14, dinov2_vitl14, dinov2_vitg14
from PIL import Image

class dino_preprocessing():
    """
    Use the ImageNet preprocessing.
    """

    def __init__(self, pixel_mean, pixel_std, is_RGB=False):
        if is_RGB:
            pixel_mean = pixel_mean.reverse()
            pixel_std = pixel_std.reverse()

        normalize = T.Normalize(mean=pixel_mean, std=pixel_std)        
        self.preprocessing_img = normalize

    def __call__(self, image):
        return self.preprocessing_img(image)

class DinoV2VitFeatureExtractor(nn.Module):
    """
    DINO V2 Vision Transformer Feature Extractor.
    """
    def __init__(self, cfg, cnn_dim, model_name='dinov2_vitb14'):
        super(DinoV2VitFeatureExtractor, self).__init__()
        self.preprocessing = dino_preprocessing(cfg.MODEL.PIXEL_MEAN, cfg.MODEL.PIXEL_STD, is_RGB=False)
        self.is_RGB = False
        self.normalize_feature = True
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
        path_to_pretrained_weights = os.path.join("adapteacher/engine/dino_weights/dinov2_vitb14_pretrain.pth")
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

        if self.normalize_feature:
            x = F.normalize(x, p=2, dim=2)

        x_grid_features = x.contiguous().transpose(1, 2).contiguous().view(batch_size, self.embed_dim, f_height, f_width)

        return x_grid_features

class DinoAlignHead(nn.Module):
    def __init__(self, cnn_dim, dino_dim, normalize_feature=True):
        super(DinoAlignHead, self).__init__()
        self.normalize_feature = normalize_feature
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
    
    def dino_loss(self, feat_cnn, feat_dino, return_sim=False):
        feat_cnn = feat_cnn.permute((0,2,3,1)).unsqueeze(-2)
        feat_dino = feat_dino.permute((0,2,3,1)).unsqueeze(-1)
        sim = torch.matmul(feat_cnn, feat_dino)
        loss = (1-sim).mean()
        if return_sim:
            return loss, sim
        else:
            return loss
        
