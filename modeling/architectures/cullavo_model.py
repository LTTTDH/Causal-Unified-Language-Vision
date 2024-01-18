# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Xueyan Zou (xueyan@cs.wisc.edu), Ziyi Dou, Jianwei Yang
# --------------------------------------------------------

from typing import Tuple
import random

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from timm.models.layers import trunc_normal_
from detectron2.structures import Boxes, ImageList, Instances, BitMasks, BoxMode
from detectron2.utils.memory import retry_if_cuda_oom
from detectron2.data import MetadataCatalog

from .build import register_model
from ..utils import configurable, get_class_names
from ..modules import sem_seg_postprocess, bbox_postprocess
from ..language.loss import vl_similarity

# CuLLaVO
from cullavo.load_cullavo import prepare_cullavo
from cullavo.utils.load_vqclip import prepare_clip

class CuLLaVO(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        cullavo_model, # CuLLaVO
        cullavo_processor, # CuLLaVO
        vq_clip,
    ):
        super().__init__()
        self.cullavo_model = cullavo_model # CuLLaVO
        self.cullavo_processor = cullavo_processor # CuLLaVO
        self.vq_clip = vq_clip

    @classmethod
    def from_config(cls, cfg):

        # VQ-CLIP
        if cfg['VQCLIP']['LOAD_VQCLIP']:
            vq_clip = prepare_clip()
        else:
            vq_clip = None

        # CuLLaVO
        if cfg['LLM']['LOAD_LLM']:
            cullavo_model, cullavo_processor = prepare_cullavo(bits=cfg['LLM']['BITS'],
                                                               grad_ckpt=cfg['LLM']['GRAD_CKPT'],
                                                               lora=cfg['LLM']['LORA'])
        else:
            cullavo_model, cullavo_processor = None, None


        return {
            "cullavo_model": cullavo_model, # CuLLaVO
            "cullavo_processor": cullavo_processor, # CuLLaVO
            "vq_clip": vq_clip,
            }

    @property
    def device(self):
        return self.pixel_mean.device

    def forward(self, batched_inputs, accel, mode=None):

        # a = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # b = batched_inputs[0]['instances'].gt_masks[0].unsqueeze(2).cpu().numpy()
        # c = batched_inputs[0]['groundings']['masks'][0].unsqueeze(2).cpu().numpy()
        # for i in range(batched_inputs[0]['instances'].gt_masks.shape[0]):
        #     a[torch.where(batched_inputs[0]['instances'].gt_masks[i].float() == 1)] = 128
        
        if self.training:
            if not self.cullavo_model:
                return self.forward_vqclip(batched_inputs, accel)

            return self.forward_seg_with_cullavo(batched_inputs, accel)
        else:
            if mode == 'retrieval':
                return self.evaluate_retrieval(batched_inputs)
            elif mode == 'captioning':
                return self.evaluate_captioning(batched_inputs)
            elif mode == 'classification':
                return self.evaluate_classification(batched_inputs)
            elif mode == 'grounding_refcoco':
                # return self.evaluate_grounding(batched_inputs)
                return self.evaluate_grounding_with_llm(batched_inputs)
            else:
                # return self.evaluate(batched_inputs)
                return self.evaluate_with_llm(batched_inputs, accel)

    # CuLLaVO
    def forward_vqclip(self, batched_inputs, accel):
        max_num_instances = 10

        masks_list = []
        masked_image_list = []
        for batch in batched_inputs:
            masks = batch['instances'].gt_masks
            masked_images = masks.unsqueeze(1).repeat(1, 3, 1, 1) * batch['image']
            masked_image_list.append(masked_images)
            masks_list.append(masks)
        masked_image_tensor = torch.cat(masked_image_list, dim=0)
        mask_tensor = torch.cat(masks_list, dim=0)

        id = torch.randperm(len(masked_image_tensor))[:max_num_instances]
        shuffled_masked_image_tensor = masked_image_tensor[id]
        shuffled_mask_tensor = mask_tensor[id]

        # Visualization
        # a = masked_image[0].permute(1,2,0).cpu().numpy()
        # b = batch['image'].permute(1,2,0).cpu().numpy()
        # c = masks[6].cpu().numpy()
        # d = images[1].permute(1,2,0).cpu().numpy()
        # e = self.vq_clip(images, accel)[0][0].permute(1,2,0).detach().float().cpu().numpy()
        # n = lambda x: (x-x.min()) / (x.max()-x.min())
        # f = n(d)
        return {'loss_clip': self.vq_clip(shuffled_masked_image_tensor, shuffled_mask_tensor, accel)[2]}

    # CuLLaVO
    def forward_seg_with_cullavo(self, batched_inputs, accel):
        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.train_process(batched_inputs, self.cullavo_processor, accel.device)
        cullavo_outputs = self.cullavo_model(**cullavo_inputs)
        return {'loss_llm': cullavo_outputs.loss}

    def evaluate_with_llm(self, batched_inputs, accel):
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [(x - self.pixel_mean) / self.pixel_std for x in images]
        
        images = ImageList.from_tensors(images, self.size_divisibility)


        a = F.interpolate(torch.stack([x['image'] for x in batched_inputs]), size=(336,336))
        new_batched_inputs = [{'image': aa} for aa in a]
        b = a[0].permute(1,2,0).cpu().numpy()

        # CuLLaVO: llm preparation
        # cullavo_inputs = self.cullavo_model.eval_process(batched_inputs, processor=self.cullavo_processor, device=accel.device)
        cullavo_inputs = self.cullavo_model.custom_process(new_batched_inputs, prompt="An image is evenly divided into the 576 number of patches. The patches is labeled from 1 to 576 in a sequential manner. "
                                                           "Starting from the top-left corner, we assign the patches to labels row-wise, moving from left to right, and then proceeding to the next row until the entire image is covered. "
                                                           "USER: what is the object name for label sets (5,6,7,8,9,10).\nAnswer the object name only ASSISTANT:", processor=self.cullavo_processor, device=accel.device)
        with torch.inference_mode():
            generate_ids = self.cullavo_model.generate(**{k:v.to(accel.device) for k,v in cullavo_inputs.items()}, do_sample=False, temperature=0, max_new_tokens=200, use_cache=True)
        decoded_text = self.cullavo_processor.batch_decode(generate_ids)[0]

        # BOX visualizer
        from detectron2.utils.visualizer import Visualizer
        img = batched_inputs[0]['image'].permute(1,2,0).cpu().numpy()
        vis = Visualizer(new_batched_inputs[0]['image'].permute(1,2,0).cpu().numpy())
        out = vis.draw_box(torch.tensor([0.75, 0.55, 0.9, 0.72])*336).get_image()
        

        #  CuLLaVO 
        # mask_cls_results = res_outputs["pred_logits"]
        # mask_pred_results = res_outputs["pred_masks"]
        # box_pred_results = [None for _ in range(len(mask_pred_results))]
        # caption_pred_results = [None for _ in range(len(mask_pred_results))]

        # # upsample masks
        # mask_pred_results = F.interpolate(
        #     mask_pred_results,
        #     size=(images.tensor.shape[-2], images.tensor.shape[-1]),
        #     mode="bicubic",
        #     align_corners=False,
        #     antialias=True
        # )

        # input_size = mask_pred_results.shape[-2:]
        # keep_sem_bgd = self.metadata.keep_sem_bgd if hasattr(self.metadata, 'keep_sem_bgd') else False
        # del outputs

        # processed_results = []
        # for mask_cls_result, mask_pred_result, box_pred_result, caption_pred_result, input_per_image, image_size in zip(
        #     mask_cls_results, mask_pred_results, box_pred_results, caption_pred_results, batched_inputs, images.image_sizes
        # ):
        #     height = input_per_image.get("height", image_size[0])
        #     width = input_per_image.get("width", image_size[1])
        #     processed_results.append({})

        #     if self.sem_seg_postprocess_before_inference:
        #         mask_pred_result = retry_if_cuda_oom(sem_seg_postprocess)(
        #             mask_pred_result, image_size, height, width
        #         )
        #         mask_cls_result = mask_cls_result.to(mask_pred_result)

        #     # semantic segmentation inference
        #     if self.semantic_on:
        #         r = retry_if_cuda_oom(self.semantic_inference)(mask_cls_result, mask_pred_result, keep_sem_bgd)
        #         if not self.sem_seg_postprocess_before_inference:
        #             r = retry_if_cuda_oom(sem_seg_postprocess)(r, image_size, height, width)
        #         processed_results[-1]["sem_seg"] = r

        #     # panoptic segmentation inference
        #     if self.panoptic_on:
        #         panoptic_r = retry_if_cuda_oom(self.panoptic_inference)(mask_cls_result, mask_pred_result)
        #         processed_results[-1]["panoptic_seg"] = panoptic_r
            
        #     # LBK Visualization
        #     # cmap = self.create_pascal_label_colormap()
        #     # c = cmap[panoptic_r[0].cpu()]
            
        #     # instance segmentation inference
        #     if self.instance_on:
        #         if self.task_switch['bbox']:
        #             box_pred_result = bbox_postprocess(box_pred_result, input_size, image_size, height, width)
        #         instance_r = retry_if_cuda_oom(self.instance_inference)(mask_cls_result, mask_pred_result, box_pred_result)
        #         processed_results[-1]["instances"] = instance_r
        #     if self.task_switch['caption']:
        #         processed_results[-1]["captions"] = caption_pred_result
        #         processed_results[-1]["masks"] = mask_pred_result

        # return processed_results

@register_model
def get_cullavo_model(cfg, **kwargs):
    return CuLLaVO(cfg)