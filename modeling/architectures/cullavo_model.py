import torch
from torch import nn
from torch.nn import functional as F
from detectron2.structures import ImageList

from .build import register_model
from ..utils import configurable

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
            if not self.cullavo_model: return self.forward_vqclip(batched_inputs, accel)
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
        return {'loss_clip': self.vq_clip(batched_inputs, accel)[2]}

    # CuLLaVO
    def forward_seg_with_cullavo(self, batched_inputs, accel):
        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.train_process(batched_inputs, self.cullavo_processor, self.vq_clip, accel.device)
        cullavo_outputs = self.cullavo_model(**cullavo_inputs)
        return {'loss_llm': cullavo_outputs.loss}

    def evaluate_with_llm(self, batched_inputs, accel):

        interpolated_images = F.interpolate(torch.stack([x['image'] for x in batched_inputs]), size=(336,336))

        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[0], 
                                                         prompt="can you show segmentation index of person?\nAnswer segmentation index", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        with torch.inference_mode():
            generate_ids = self.cullavo_model.generate(**cullavo_inputs, do_sample=False, temperature=0, max_new_tokens=100, num_beams=5, no_repeat_ngram_size=2, use_cache=True)
        decoded_text = self.cullavo_processor.batch_decode(generate_ids)[0]

        # BOX visualizer
        from detectron2.utils.visualizer import Visualizer
        img = interpolated_images[0].permute(1,2,0).cpu().numpy()
        vis = Visualizer(img)
        out = vis.draw_box(torch.tensor([0.6, 0.41, 0.7, 0.62])*336).get_image()
        

        # CuLLaVO 
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