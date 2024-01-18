import os
import math
import torch
import numpy as np
from .utils import *
import torch.nn as nn
from .VQ import VectorQuantize
import torch.nn.functional as F
from collections import namedtuple
from transformers import CLIPProcessor, CLIPModel

class GroupNorm(nn.Module):
    def __init__(self, channels):
        super(GroupNorm, self).__init__()
        self.gn = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6, affine=True)

    def forward(self, x):
        return self.gn(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.block = nn.Sequential(
            GroupNorm(in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            GroupNorm(out_channels),
            Swish(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        )

        if in_channels != out_channels:
            self.channel_up = nn.Conv2d(in_channels, out_channels, 1, 1, 0)

    def forward(self, x):
        if self.in_channels != self.out_channels:
            return self.channel_up(x) + self.block(x)
        else:
            return x + self.block(x)

class CLIPUpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(CLIPUpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=7.0)
        return self.conv(x)

class UpSampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 1, 1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2.0)
        return self.conv(x)


class DownSampleBlock(nn.Module):
    def __init__(self, channels):
        super(DownSampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels, 3, 2, 0)

    def forward(self, x):
        pad = (0, 1, 0, 1)
        x = F.pad(x, pad, mode="constant", value=0)
        return self.conv(x)


class NonLocalBlock(nn.Module):
    def __init__(self, channels):
        super(NonLocalBlock, self).__init__()
        self.in_channels = channels

        self.gn = GroupNorm(channels)
        self.q = nn.Conv2d(channels, channels, 1, 1, 0)
        self.k = nn.Conv2d(channels, channels, 1, 1, 0)
        self.v = nn.Conv2d(channels, channels, 1, 1, 0)
        self.proj_out = nn.Conv2d(channels, channels, 1, 1, 0)

    def forward(self, x):
        h_ = self.gn(x)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        b, c, h, w = q.shape

        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)
        k = k.reshape(b, c, h*w)
        v = v.reshape(b, c, h*w)

        attn = torch.bmm(q, k)
        attn = attn * (int(c)**(-0.5))
        attn = F.softmax(attn, dim=2)
        attn = attn.permute(0, 2, 1)

        A = torch.bmm(v, attn)
        A = A.reshape(b, c, h, w)

        return x + A

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        channels = [512, 256, 256, 128]
        attn_resolutions = [16]
        num_res_blocks = 3
        resolution = 16

        in_channels = channels[0]
        layers = [nn.Conv2d(1024, in_channels, 3, 1, 1),
                  ResidualBlock(in_channels, in_channels),
                  NonLocalBlock(in_channels),
                  ResidualBlock(in_channels, in_channels)]

        for i in range(len(channels)):
            out_channels = channels[i]
            for j in range(num_res_blocks):
                layers.append(ResidualBlock(in_channels, out_channels))
                in_channels = out_channels
                if resolution in attn_resolutions:
                    layers.append(NonLocalBlock(in_channels))
            if i != 0:
                layers.append(UpSampleBlock(in_channels))
                resolution *= 2
            else:
                layers.append(CLIPUpSampleBlock(in_channels))
                resolution *= 7

        layers.append(GroupNorm(in_channels))
        layers.append(Swish())
        layers.append(nn.Conv2d(in_channels, 3, 3, 1, 1))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)



class VQCLIP(nn.Module):
    def __init__(self):
        super(VQCLIP, self).__init__()

        # ENCODER
        self.clip_encoder = CLIPModel.from_pretrained(CLIPLARGE_LOCAL_PATH)
        for param in self.clip_encoder.parameters(): param.requires_grad_(False);
        self.clip_encoder = self.clip_encoder.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIPLARGE_LOCAL_PATH)
        self.image_mean = torch.tensor(self.clip_processor.image_processor.image_mean).view(1,-1,1,1)
        self.image_std = torch.tensor(self.clip_processor.image_processor.image_std).view(1,-1,1,1)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=2, stride=2, padding=0),
                                        nn.GELU(),
                                        nn.Conv2d(1024, 1024, kernel_size=2, stride=2, padding=0),
                                        nn.GELU())
                                        

        # pre_quant/post_quant
        self.pre_quant_conv = torch.nn.Conv2d(1024, 1024, 1)
        self.post_quant_conv = torch.nn.Conv2d(1024, 1024, 1)

        # [VQ] Method
                
        # Normal VQ
        self.VQ = VectorQuantize(
                    dim = 1024,
                    codebook_size = 1024,
                    decay = 0.8,             
                    commitment_weight = 1)

        # DECODER
        self.clip_decoder = Decoder()

    @staticmethod
    def to_image(x):
        return x.view(x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1])), x.shape[2]).permute(0, 3, 1, 2)
    
    @staticmethod
    def to_feat(x):
        return x.view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)

    def forward(self, batched_inputs, accel):

        max_num_instances = 20

        masks_list = []
        masked_image_list = []
        for batch in batched_inputs:
            masks = batch['instances'].gt_masks
            masked_images = masks.unsqueeze(1).repeat(1, 3, 1, 1) * batch['image']
            masked_images = masks.unsqueeze(1).repeat(1, 3, 1, 1).bool().float() * 255
            masked_image_list.append(masked_images)
            masks_list.append(masks.bool().float())
        masked_image_tensor = torch.cat(masked_image_list, dim=0)
        mask_tensor = torch.cat(masks_list, dim=0)

        id = torch.randperm(len(masked_image_tensor))[:max_num_instances]
        shuffled_masked_image_tensor = masked_image_tensor[id]
        shuffled_mask_tensor = mask_tensor[id]

        clip_inputs = self.clip_processor(images=shuffled_masked_image_tensor, return_tensors='pt')
        clip_embeds = self.clip_encoder.vision_model(pixel_values=clip_inputs.pixel_values.to(accel.device), output_hidden_states=True).hidden_states[-2][:,1:]
        bottleneck_embeds = self.bottleneck(self.pre_quant_conv(self.to_image(clip_embeds)))
        quantized, indices, commit_loss = self.VQ(bottleneck_embeds)
        recov_x = self.clip_decoder(self.post_quant_conv(quantized)).squeeze(1)
        denorm_recov_x = (self.image_std.to(accel.device) * recov_x + self.image_mean.to(accel.device)).mean(dim=1)

        l1_rec_loss = torch.abs(denorm_recov_x - shuffled_mask_tensor).mean()
        l2_rec_loss = F.mse_loss(denorm_recov_x, shuffled_mask_tensor)
        bce_loss = F.binary_cross_entropy_with_logits(denorm_recov_x, shuffled_mask_tensor)


        # Visualization
        # i=0
        # a = shuffled_mask_tensor[i].cpu().numpy()
        # b = denorm_recov_x[i]>0

        return recov_x, indices, commit_loss + l1_rec_loss + l2_rec_loss + bce_loss

def prepare_clip():
    return VQCLIP()