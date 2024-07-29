'''
Code for loading models trained with CosPlace as a global features extractor
for geolocalization through image retrieval.
Multiple models are available with different backbones. Below is a summary of
models available (backbone : list of available output descriptors
dimensionality). For example you can use a model based on a ResNet50 with
descriptors dimensionality 1024.
    ResNet18:  [32, 64, 128, 256, 512]
    ResNet50:  [32, 64, 128, 256, 512, 1024, 2048]
    ResNet101: [32, 64, 128, 256, 512, 1024, 2048]
    ResNet152: [32, 64, 128, 256, 512, 1024, 2048]
    VGG16:     [    64, 128, 256, 512]

CosPlace paper: https://arxiv.org/abs/2204.02287
'''

import torch
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


class SALAD(BaseModel):
    default_conf = {
        'backbone': 'ResNet50',
        'fc_output_dim' : 2048
    }
    required_inputs = ['image']
    def _init(self, conf, resize_type="dino_v2_resize"):
        self.net = torch.hub.load(
            "serizba/salad", "dinov2_salad",
            # backbone=conf['backbone'],
            # fc_output_dim=conf['fc_output_dim']
        ).eval()

        self.resize_type = resize_type
        
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        self.norm_rgb = tvf.Normalize(mean=mean, std=std)

    def _forward(self, data):
        image = self.norm_rgb(data['image'])
        if self.resize_type == "dino_v2_resize":
            b, c, h, w = image.shape
            # DINO wants height and width as multiple of 14, therefore resize them
            # to the nearest multiple of 14
            h = round(h / 14) * 14
            w = round(w / 14) * 14
            image = tvf.functional.resize(image, [h, w], antialias=True)
        if isinstance(self.resize_type, int):
            img_size = self.resize_type
            image = tvf.functional.resize(image, [img_size, img_size], antialias=True)
        desc = self.net(image)
        return {
            'global_descriptor': desc,
        }
