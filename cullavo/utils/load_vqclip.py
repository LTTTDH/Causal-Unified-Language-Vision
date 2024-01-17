import os
import math
import torch
import numpy as np
from .LFQ import LFQ
from .utils import *
import torch.nn as nn
from .VQ import VectorQuantize
import torch.nn.functional as F
from collections import namedtuple
from transformers import CLIPProcessor, CLIPModel

def nonlinearity(x):
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv = torch.nn.Conv2d(in_channels,
                                    in_channels,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        x = self.conv(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if temb_channels > 0:
            self.temb_proj = torch.nn.Linear(temb_channels,
                                             out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        self.conv2 = torch.nn.Conv2d(out_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        if temb is not None:
            h = h + self.temb_proj(nonlinearity(temb))[:,:,None,None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)


    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = q.reshape(b,c,h*w)
        q = q.permute(0,2,1)   # b,hw,c
        k = k.reshape(b,c,h*w) # b,c,hw
        w_ = torch.bmm(q,k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b,c,h*w)
        w_ = w_.permute(0,2,1)   # b,hw,hw (first hw of k, second of q)
        h_ = torch.bmm(v,w_)     # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = h_.reshape(b,c,h,w)

        h_ = self.proj_out(h_)

        return x+h_


class Decoder(nn.Module):
    def __init__(self, ch=128, out_ch=1, ch_mult=(1,2,4,8,16), num_res_blocks=2,
                 attn_resolutions=[16], dropout=0.0,
                 resolution=336, z_channels=1024, give_pre_end=False):
        super().__init__()
        self.ch = ch
        self.temb_ch = 0
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.give_pre_end = give_pre_end

        # compute in_ch_mult, block_in and curr_res at lowest res
        block_in = ch*ch_mult[self.num_resolutions-1]
        curr_res = resolution // 2**(self.num_resolutions-1)
        self.z_shape = (1,z_channels,curr_res,curr_res)

        # z to block_in
        self.conv_in = torch.nn.Conv2d(z_channels,
                                       block_in,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            for i_block in range(self.num_res_blocks+1):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in)
                curr_res = curr_res * 2
            self.up.insert(0, up) # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, z):
        #assert z.shape[1:] == self.z_shape[1:]
        self.last_z_shape = z.shape

        # timestep embedding
        temb = None

        # z to block_in
        h = self.conv_in(z)

        # middle
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](h, temb)
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        if self.give_pre_end:
            return h

        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class VQCLIP(nn.Module):
    def __init__(self):
        super(VQCLIP, self).__init__()

        # ENCODER
        self.clip_encoder = CLIPModel.from_pretrained(CLIPLARGE_LOCAL_PATH)
        for param in self.clip_encoder.parameters(): param.requires_grad_(False)
        self.clip_encoder = self.clip_encoder.eval()
        self.clip_processor = CLIPProcessor.from_pretrained(CLIPLARGE_LOCAL_PATH)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1),
                                        nn.GELU(),
                                        nn.Conv2d(1024, 1024, kernel_size=4, stride=2, padding=1),
                                        nn.GELU())
                                        

        # pre_quant/post_quant
        self.pre_quant_conv = torch.nn.Conv2d(1024, 1024, 1)
        self.post_quant_conv = torch.nn.Conv2d(1024, 1024, 1)

        # [VQ] Method

        # LFQ
        # self.LFQ = LFQ(
        #             codebook_size = 65536,      
        #             dim = 1024,
        #             entropy_loss_weight = 1,
        #             diversity_gamma = 1.)    
        
        # Normal VQ
        self.LFQ = VectorQuantize(
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

    def forward(self, x, accel):
        x_label = F.interpolate(x[:,0].clone().unsqueeze(1).float(), size=(96,96), mode='bicubic').squeeze(1).bool().float()
        x_to_image = x.bool().float() * 255
        clip_inputs = self.clip_processor(images=x_to_image, return_tensors='pt')
        x_norm_label = F.interpolate(clip_inputs.pixel_values.to(accel.device)[:,0].unsqueeze(1), size=(96,96), mode='bicubic').squeeze(1)
        clip_embeds = self.clip_encoder.vision_model(pixel_values=clip_inputs.pixel_values.to(accel.device), output_hidden_states=True).hidden_states[-2][:,1:]
        bottleneck_embeds = self.bottleneck(self.to_image(clip_embeds))
        quantized, indices, commit_loss = self.LFQ(self.to_feat(bottleneck_embeds))
        recov_x = self.clip_decoder(self.to_image(quantized)).squeeze(1)

        rec_loss = F.mse_loss(recov_x, x_norm_label)
        bce_loss = F.binary_cross_entropy_with_logits(recov_x, x_label)

        return recov_x, indices, commit_loss + rec_loss + bce_loss

def prepare_clip():
    return VQCLIP()