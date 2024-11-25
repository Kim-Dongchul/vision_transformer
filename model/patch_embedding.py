import math

import torch
from torch.nn import Module, Flatten, Parameter, Linear
from torchvision.transforms.functional import crop


class PatchEmbedding(Module):
    def __init__(self, img_shape, patch_size, device):
        super(PatchEmbedding, self).__init__()
        self.batch_size = img_shape[0]
        self.channel_size = img_shape[1]
        self.height_size = img_shape[2]
        self.width_size = img_shape[3]
        self.patch_size = patch_size
        self.n = math.ceil(self.height_size/self.patch_size) * math.ceil(self.width_size/self.patch_size)
        self.dim = self.channel_size*self.patch_size**2
        self.flat = Flatten()
        self.cls_token = Parameter(torch.zeros(
            (self.batch_size, self.dim),
            requires_grad=True,
            device=device
        ), requires_grad=True)
        self.position_embedding = Parameter(torch.zeros(
            (self.n + 1, self.channel_size * self.patch_size**2),
            requires_grad=True,
            device=device
        ), requires_grad=True)

    def forward(self, img):
        crop_images = [self.cls_token, ]
        cur_width = 0
        cur_height = 0

        while cur_height < self.height_size:
            while cur_width < self.width_size:
                crop_img = crop(img, cur_height, cur_width, self.patch_size, self.patch_size)
                flat = self.flat(crop_img)
                crop_images.append(flat)
                cur_width += self.patch_size
            cur_height += self.patch_size
            cur_width = 0

        patch_embedding = torch.stack(crop_images, dim=1)
        return patch_embedding + self.position_embedding
