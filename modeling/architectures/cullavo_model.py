import torch
from torch import nn
from torch.nn import functional as F

from .build import register_model
from ..utils import configurable

# CuLLaVO
from cullavo.load_cullavo import prepare_cullavo, add_adapter_for_step2

class CuLLaVO(nn.Module):

    @configurable
    def __init__(
        self,
        *,
        cfg,
        cullavo_model, # CuLLaVO
        cullavo_processor, # CuLLaVO
    ):
        super().__init__()
        self.cfg = cfg
        self.cullavo_model = cullavo_model # CuLLaVO
        self.cullavo_processor = cullavo_processor # CuLLaVO

    @classmethod
    def from_config(cls, cfg):

        # CuLLaVO
        if cfg['LLM']['LOAD_LLM']:
            cullavo_model, cullavo_processor = prepare_cullavo(bits=cfg['LLM']['BITS'],
                                                               grad_ckpt=cfg['LLM']['GRAD_CKPT'],
                                                               lora=cfg['LLM']['LORA'])
            # CuLLaVO
            if cfg['NAME'] == 'cullavo_step2.yaml': add_adapter_for_step2(cullavo_model)
        else:
            cullavo_model, cullavo_processor = None, None

        return {
            "cfg": cfg,
            "cullavo_model": cullavo_model, # CuLLaVO
            "cullavo_processor": cullavo_processor, # CuLLaVO
            }

    def forward(self, batched_inputs, accel, mode=None):
        if self.training:
            if self.cfg['NAME']=='cullavo_step0s.yaml':
                return self.forward_step0(batched_inputs, accel)
            elif self.cfg['NAME']=='cullavo_step1.yaml':
                return self.forward_step1(batched_inputs, accel)
            elif self.cfg['NAME']=='cullavo_step2.yaml':
                return self.forward_step2(batched_inputs, accel)
        else:
            if self.cfg['NAME']=='cullavo_step2_pre.yaml':
                return self.forward_step2_pre(batched_inputs, accel)
            
            return self.evaluate_with_llm(batched_inputs, accel)

    # CuLLaVO - STEP1 - Finetuning for Object Understanding
    def forward_step1(self, batched_inputs, accel):
        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.step1_process(batched_inputs, self.cullavo_processor, accel.device)

        # TRY-EXCEPT HANDLING
        if cullavo_inputs['input_ids']!=None:
            cullavo_outputs = self.cullavo_model(**cullavo_inputs)
            return {'loss_llm': cullavo_outputs.loss}
        else:
            return {'loss_llm': torch.tensor([0]).to(accel.device)}

    
    # CuLLaVO - STEP2 - Finetuning for Instruction tuning based on Object Understanding
    def forward_step2_pre(self, batched_inputs, accel):
        # CuLLaVO: llm preparation
        new_json_list = self.cullavo_model.step2_preprocess(batched_inputs, self.cullavo_processor, accel.device)
        return new_json_list

    # CuLLaVO - STEP2 - Finetuning for Instruction tuning based on Object Understanding
    def forward_step2(self, batched_inputs, accel):
        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.step2_process(batched_inputs, self.cullavo_processor, accel.device)
        cullavo_outputs = self.cullavo_model(**cullavo_inputs)
        return {'loss_llm': cullavo_outputs.loss}

    def evaluate_with_llm(self, batched_inputs, accel):

        interp_image_list = []
        for x in batched_inputs:
            x_interp = F.interpolate(x['image'].unsqueeze(0), size=(336,336))
            interp_image_list.append(x_interp)
        interpolated_images = torch.cat(interp_image_list, dim=0)

        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[5], 
                                                         prompt=f"provide multiple object names with their numbering index and the objects' bounding box coordinates in this image.", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[0], 
                                                         prompt="provide multiple coordinates of multiple bounding boxes corresponding multiple objects in this image.", 
                                                         aux_prompt="USER: provide multiple object names and their numbering index in this image. ASSISTANT: Sure, it is person (#1), tv (#1), microwave (#1), chair (#1), chair (#2), chair (#3), chair (#4), chair (#5), potted plant (#1), potted plant (#2), potted plant (#3). There are 10 objects in this image.",
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[2], 
                                                         prompt=f"provide multiple bounding box coordinates corresponding window in this image.", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[2], 
                                                         prompt=f"provide multiple coordinates of multiple bounding boxes corresponding bed in this image", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[2], 
                                                         prompt=f"Is there any red bounding box in this image?", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[2], 
                                                         prompt=f"How many books are in this image?", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[2], 
                                                         prompt=f"provide object name for bounding box coordinate [0.000, 0.580, 0.607, 1.000]", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        cullavo_inputs = self.cullavo_model.eval_process(images=torch.from_numpy(out).permute(2,0,1), 
                                                         prompt=f"provide multiple bounding box colors in this image",
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)
        
        filt = "Sure, it is person (#1), refrigerator (#1), couch (#1), tv (#1), potted plant (#1), potted plant (#2), vase (#1). There are 7 objects in this image."


        # CuLLaVO: llm preparation
        cullavo_inputs = self.cullavo_model.eval_process(images=interpolated_images[0], 
                                                         prompt=f"provide multiple object names with their numbering index and the objects' bounding box coordinates in this image.", 
                                                         processor=self.cullavo_processor, 
                                                         device=accel.device)


        with torch.inference_mode():
            generate_ids = self.cullavo_model.generate(**cullavo_inputs, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, max_new_tokens=1000, use_cache=True)
        decoded_text = self.cullavo_processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
        
        # BOX visualizer
        from detectron2.structures import Boxes
        from detectron2.utils.visualizer import Visualizer
        img = interpolated_images[5].permute(1,2,0).cpu().numpy()
        vis = Visualizer(img)
        vis._default_font_size=16
        out = vis.draw_box(torch.tensor([0.423, 0.851, 0.991, 0.988])*336, alpha=1, edge_color='red').get_image()


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