import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.utils.generic import ModelOutput
from transformers import LlavaForConditionalGeneration
from utils.constants import COCO_PANOPTIC_CLASSES

import cv2
import numpy as np
from skimage import measure
import pycocotools.mask as mask_util
import torch.nn.functional as F


@dataclass
class CullavoCausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    image_hidden_states: Optional[Tuple[torch.FloatTensor]] = None


class CuLLaVOModel(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, position_ids
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(input_ids[:, -1] == torch.tensor(self.pad_token_id))
        # 1. Create a mask to know where special image tokens are
        special_image_token_mask = input_ids == self.config.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        # Compute the maximum embed dimension
        max_embed_dim = (num_special_image_tokens.max() * (num_image_patches - 1)) + sequence_length
        batch_indices, non_image_indices = torch.where(input_ids != self.config.image_token_index)

        # 2. Compute the positions where text should be written
        # Calculate new positions for text tokens in merged image-text sequence.
        # `special_image_token_mask` identifies image tokens. Each image token will be replaced by `nb_text_tokens_per_images - 1` text tokens.
        # `torch.cumsum` computes how each image token shifts subsequent text token positions.
        # - 1 to adjust for zero-based indexing, as `cumsum` inherently increases indices by one.
        new_token_positions = torch.cumsum((special_image_token_mask * (num_image_patches - 1) + 1), -1) - 1
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]  # offset for left padding
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        # 3. Create the full embedding, already padded to the maximum position
        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim, dtype=inputs_embeds.dtype, device=inputs_embeds.device
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim, dtype=attention_mask.dtype, device=inputs_embeds.device
        )

        # 4. Fill the embeddings based on the mask. If we have ["hey" "<image>", "how", "are"]
        # we need to index copy on [0, 577, 578, 579] for the text and [1:576] for the image features
        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[batch_indices, non_image_indices]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[batch_indices, non_image_indices]

        # 5. Fill the embeddings corresponding to the images. Anything that is still zeros needs filling
        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= image_to_overwrite.cumsum(-1) - 1 >= nb_image_pad[:, None]

        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"The input provided to the model are wrong. The number of image tokens is {torch.sum(special_image_token_mask)} while"
                f" the number of image given to the model is {num_images}. This prevents correct indexing and breaks batch generation."
            )

        final_embedding[image_to_overwrite] = image_features.contiguous().reshape(-1, embed_dim)
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_((final_attention_mask == 0), 1)
        return final_embedding, final_attention_mask, position_ids        


    @staticmethod
    def seq2string(seq):
        out='['
        for i, x in enumerate(seq):
            out+=f"{x}"
            if i!=len(seq)-1: out+=','
        out+=']'
        return out

    # @staticmethod
    # def seq2string(seq):
    #     out='['
    #     for i, (x, y) in enumerate(seq):
    #         if x=='<background>':
    #             out+=f"(<background>,{y})"
    #         else:
    #             out+=f"(<object>,{y})"
    #         if i!=len(seq)-1: out+=','
    #     out+=']'
    #     return out

    @staticmethod
    def seq2mask(seq, height=64, width=64):
        out = ''
        for i, x in enumerate(seq):
            if i==0:
                if x=='<background>':
                    j=0
                elif x=='<object>':
                    j=1
                continue
                    
            if j%2==0:
                out += '0'*x
            else:
                out += '1'*x
            j+=1
        return torch.tensor(list(map(int, [*out]))).reshape(height, width)
    
    # @staticmethod
    # def seq2mask(seq, height=64, width=64):
    #     out = ''
    #     for x, y in seq:
    #         if x=='<background>':
    #             out += '0'*y
    #         else:
    #             out += '1'*y
    #     return torch.tensor(list(map(int, [*out]))).reshape(height, width)
    
    @staticmethod
    def mask2seq(small_mask):
        
        def bfs_search_and_make_seq(flat_mask):
            len_flat_mask = len(flat_mask)
            out = []
            visit = []
            for i in range(len_flat_mask):
                
                # efficient visit
                if i in visit: continue
                
                # Background
                if flat_mask[i] == False:
                    # visit register
                    visit.append(i)
                    
                    # number
                    num=1
                    if num+i<len_flat_mask:
                        while not flat_mask[i+num]:
                            visit.append(i+num)
                            num+=1
                            if num+i>=len_flat_mask:
                                break
                    # out.append(('<background>', num))
                    if i==0:
                        out.append('<background>')
                        out.append(num)
                    else:
                        out.append(num)
                # Object
                else:
                    # visit register
                    visit.append(i)
                    
                    # number
                    num=1
                    if num+i<len_flat_mask:
                        while flat_mask[i+num]:
                            visit.append(i+num)
                            num+=1
                            if num+i>=len_flat_mask:
                                break
                    # out.append(('<object>',num))
                    if i==0:
                        out.append('<object>')
                        out.append(num)
                    else:
                        out.append(num)
            return out
            
        return bfs_search_and_make_seq(small_mask.flatten())
    
    @staticmethod
    def poly2mask(poly, height=1024, width=1024):
        # LBK + Detectron2
        def scale(x):
            x[:,0] = x[:,0] * width
            x[:,1] = x[:,1] * height
            return x
        poly_numpy = [x.reshape(-1) for x in list(map(scale, [np.stack(x) for x in poly]))]
        rle = mask_util.frPyObjects(poly_numpy, height, width)
        rle = mask_util.merge(rle)
        return torch.from_numpy(mask_util.decode(rle)[:, :]).bool()
    
    @staticmethod
    def mask2poly(mask, height=1024, width=1024):
        # LBK + Detectron2
        # cv2.RETR_CCOMP flag retrieves all the contours and arranges them to a 2-level
        # hierarchy. External contours (boundary) of the object are placed in hierarchy-1.
        # Internal contours (holes) are placed in hierarchy-2.
        # cv2.CHAIN_APPROX_NONE flag gets vertices of polygons from contours.
        mask = np.ascontiguousarray(mask)  # some versions of cv2 does not support incontiguous arr
        res = cv2.findContours(mask.astype("uint8"), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
        hierarchy = res[-1]
        if hierarchy is None:  # empty mask
            return [], False
        has_holes = (hierarchy.reshape(-1, 4)[:, 3] >= 0).sum() > 0
        res = res[-2]
        res = [x.flatten() for x in res]
        # These coordinates from OpenCV are integers in range [0, W-1 or H-1].
        # We add 0.5 to turn them into real-value coordinate space. A better solution
        # would be to first +0.5 and then dilate the returned polygon by 0.5.
        res = [x + 0.5 for x in res if len(x) >= 6]
        
        # normalizing and reshaping
        def resize_and_scale(x):
            x = x.reshape(-1, 2)
            x[:,0] = x[:,0] / width
            x[:,1] = x[:,1] / height
            return x
        res = [np.round(resize_and_scale(x), 3) for x in res]        
        return res, has_holes


    def label_generator(self, input_ids, device):
        add_point = 24*24-1
        labels = torch.ones_like(input_ids) * self.config.ignore_index
        image_labels = torch.ones([input_ids.shape[0], add_point]).to(device) * self.config.ignore_index
        labels = torch.cat([labels, image_labels], dim=1).to(torch.int64)
        x_gens, y_gens = torch.where(input_ids == 32002) # <GEN>
        x_gen_ends, y_gen_ends = torch.where(input_ids == 32003) # </GEN>
        x_grds, y_grds = torch.where(input_ids == 32004) # <GRD>
        x_grd_ends, y_grd_ends = torch.where(input_ids == 32005) # </GRD>
        
        for x_gen, y_gen, x_gen_end, y_gen_end in zip(x_gens, y_gens, x_gen_ends, y_gen_ends):
            assert x_gen == x_gen_end
            labels[x_gen, y_gen+1+add_point: y_gen_end+1+add_point] = input_ids[x_gen, y_gen+1: y_gen_end+1]
            
        for x_grd, y_grd, x_grd_end, y_grd_end in zip(x_grds, y_grds, x_grd_ends, y_grd_ends):
            assert x_grd == x_grd_end
            labels[x_grd, y_grd+1+add_point: y_grd_end+1+add_point] = input_ids[x_grd, y_grd+1: y_grd_end+1]
            
        return labels

    def preprocess(
        self,
        inputs,
        mode=None,
        *,
        processor,
        device
    ):
        if mode=='eval':
            batched_cullavo_prompt=[]
            for input in inputs:

                # cullavo prompt prefix
                cullavo_prompt = "If an image is given, you will predict classes, boxes, and binary masks in order on the image. "
            
                # cullavo prompt init
                cullavo_prompt += "USER: <image>\n<GEN>"
                
                # making batched cullavo prompt
                batched_cullavo_prompt.append(cullavo_prompt)
            
            '''For Final Outputs'''
            cullavo_inputs = \
            processor(text=batched_cullavo_prompt, images=torch.stack([input['image'] for input in inputs]), padding=True, return_tensors="pt")
            
            # [1] input_ids
            input_ids = cullavo_inputs.input_ids.to(device)
            
            # [2] pixel values
            pixel_values = cullavo_inputs.pixel_values.to(device)
            
            # [3] attention_mask
            attention_mask = cullavo_inputs.attention_mask.to(device)

            return {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask,}
                    
        # fix num
        fix_num = 20
         
        # initialization
        input_ids = None
        pixel_values  = None
        attention_mask = None
        position_ids = None
        past_key_values = None
        inputs_embeds = None
        vision_feature_layer = None
        vision_feature_select_strategy = None
        labels = None
        use_cache = None
        output_attentions = None
        output_hidden_states = None
        return_dict = None
        
        # Instance example
        # inputs[0]['instances'].gt_classes
        # inputs[0]['instances'].is_things
        # m = inputs[0]['instances'].gt_masks[1].cpu().numpy()
        # inputs[0]['instances'].gt_boxes.tensor
        
        # Grounding exmaple
        # m = inputs[0]['groundings']['masks'][0].float().unsqueeze(2).cpu().numpy()
        # t = inputs[0]['groundings']['texts']
        # img = inputs[0]['image'].permute(1,2,0).cpu().numpy()
        
        # Drawing BBox & Seg
        # import numpy as np
        # from utils.constants import COCO_PANOPTIC_CLASSES
        # from detectron2.utils.visualizer import Visualizer
        # vis = Visualizer(inputs[0]['image'].permute(1,2,0).cpu().numpy())
        # out = vis.draw_box(inputs[0]['instances'].gt_boxes.tensor[2]).get_image()
        # out = vis.overlay_instances(boxes=inputs[0]['instances'].gt_boxes.tensor * 1024,
        #                             masks=inputs[0]['instances'].gt_masks,
        #                             labels=[COCO_PANOPTIC_CLASSES[c] for c in inputs[0]['instances'].gt_classes]).get_image()
        
        batched_cullavo_prompt=[]
        for input in inputs:
            # Shape
            _,H,W=input['image'].shape
             
            if input['groundings']['mode']=='no text':
             
                # cullavo prompt prefix
                cullavo_prompt = "If an image is given, you will predict classes, boxes, and binary masks in order on the image. "
                # cullavo_prompt += "The output format of classes is string and it is located between <CLASS> and </CLASS>. "
                # cullavo_prompt += "The output format of boxes is a list and it is located between <BOX> and </BOX>. "
                # cullavo_prompt += "The output format of instances is a list with tuple and it is located between <INST> and </INST>.\n"
             
                # cullavo prompt init
                cullavo_prompt += "USER:<image>, ASSISTANT:<GEN>"
                
                # CLASSES
                classes = [COCO_PANOPTIC_CLASSES[c].replace('-merged','').replace('-other','').replace('-stuff','') for c in input['instances'].gt_classes]
                
                # BOXES
                input['instances'].gt_boxes.scale(1/W,1/H)
                boxes = input['instances'].gt_boxes.tensor
                
                # MASKS
                masks = input['instances'].gt_masks
                small_masks = F.interpolate(masks.float().unsqueeze(1), size=(64, 64), mode='bicubic').squeeze(1).bool()
                
                # LBK Visualization
                # m = masks[0]
                # sm = small_masks[0]
            
                # Making cullavo prompt
                for cls, box, small_mask, mask in zip(classes[:fix_num], boxes[:fix_num], small_masks[:fix_num], masks[:fix_num]):
                    if small_masks.sum()==0: continue
                    _box = list(map(lambda x: round(x, 3), box.tolist()))
                    _seq = self.mask2seq(small_mask)
                    # _mask = self.seq2mask(_seq) # Recover Mask
                    cullavo_prompt+=f"<CLASS>{cls}</CLASS><BOX>{_box}</BOX><INST>{self.seq2string(_seq)}</INST>"
                    
                # cullavo prompt ending
                cullavo_prompt+="</GEN></s>\n"

            # Vision Grounding 
            elif input['groundings']['mode']=='text':
                
                # cullavo prompt prefix
                cullavo_prompt = "If an image and texts are given, "                
                cullavo_prompt += "you will predict classes, boxes, and binary instance mask corresponding the given text in order on image. "
                
                # cullavo prompt init
                cullavo_prompt += "USER: <image>, "
                
                # TEXTS
                texts = [str(x) for x in input['groundings']['texts']]
                
                # CLASSES    
                classes = input['groundings']['classes']
                
                # BOXES
                from detectron2.structures import Boxes
                boxes = Boxes(input['groundings']['boxes'])
                boxes.scale(1/W, 1/H)
                
                # MASKS
                masks = input['groundings']['masks'].bool()
                small_masks = F.interpolate(masks.float().unsqueeze(1), size=(64, 64), mode='bicubic').squeeze(1).bool()
                
                # LBK visualization
                # img = input['image'].permute(1,2,0).cpu().numpy()
                # m = masks[2]
                # s = small_masks[0]
                
                # Making cullavo prompt for vision grounding
                for cls, box, small_mask, text in zip(classes[:fix_num], boxes[:fix_num], small_masks[:fix_num], texts[:fix_num]):
                    if small_masks.sum()==0: continue
                    _box = list(map(lambda x: round(x, 3), box.tolist()))
                    _seq = self.mask2seq(small_mask)
                    # _mask = self.seq2mask(_seq) # Recover Mask
                    cullavo_prompt+=f"<SET>{text}</SET>, ASSISTANT:<GRD><CLASS>{cls}</CLASS><BOX>{_box}</BOX><INST>{self.seq2string(_seq)}</INST></GRD></s>"
            
            # making batched cullavo prompt
            batched_cullavo_prompt.append(cullavo_prompt)
            
        '''For Final Outputs'''
        cullavo_inputs = \
        processor(text=batched_cullavo_prompt, images=torch.stack([input['image'] for input in inputs]), padding=True, return_tensors="pt")
        
        # [1] input_ids
        input_ids = cullavo_inputs.input_ids.to(device)
        
        # [2] pixel values
        pixel_values = cullavo_inputs.pixel_values.to(device)
        
        # [3] attention_mask
        attention_mask = cullavo_inputs.attention_mask.to(device)
                
        # [4] labels
        labels = self.label_generator(input_ids, device)

        return {"input_ids": input_ids,\
            "pixel_values": pixel_values,\
            "attention_mask": attention_mask,\
            "position_ids": position_ids,\
            "past_key_values": past_key_values,\
            "inputs_embeds": inputs_embeds,\
            "vision_feature_layer": vision_feature_layer,\
            "vision_feature_select_strategy": vision_feature_select_strategy,\
            "labels": labels,\
            "use_cache": use_cache,\
            "output_attentions": output_attentions,\
            "output_hidden_states": output_hidden_states,\
            "return_dict": return_dict}
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        pixel_values: torch.FloatTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        vision_feature_layer: Optional[int] = None,
        vision_feature_select_strategy: Optional[str] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CullavoCausalLMOutputWithPast]:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        vision_feature_layer = (
            vision_feature_layer if vision_feature_layer is not None else self.config.vision_feature_layer
        )
        vision_feature_select_strategy = (
            vision_feature_select_strategy
            if vision_feature_select_strategy is not None
            else self.config.vision_feature_select_strategy
        )

        if inputs_embeds is None:
            # 1. Extra the input embeddings
            inputs_embeds = self.get_input_embeddings()(input_ids)

            # 2. Merge text and images
            if pixel_values is not None and input_ids.shape[1] != 1:
                image_outputs = self.vision_tower(pixel_values, output_hidden_states=True)
                # this is not memory efficient at all (output_hidden_states=True) will save all the hidden stated.
                selected_image_feature = image_outputs.hidden_states[vision_feature_layer]

                if vision_feature_select_strategy == "default":
                    selected_image_feature = selected_image_feature[:, 1:]
                elif vision_feature_select_strategy == "full":
                    selected_image_feature = selected_image_feature
                else:
                    raise ValueError(
                        f"Unexpected select feature strategy: {self.config.vision_feature_select_strategy}"
                    )

                image_features = self.multi_modal_projector(selected_image_feature)
                inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
                    image_features.to(torch.bfloat16), inputs_embeds, input_ids, attention_mask, position_ids
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, 0, :, 0]
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value == 0)
                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # TODO and FIXME The part of LBK Overload!
                    # Zero-out the places where we don't need to attend
                    for a, b in zip(batch_index, non_attended_tokens):
                        if 0<=a<extended_attention_mask.shape[0] and 0<=b<extended_attention_mask.shape[1]:
                            extended_attention_mask[a, b] = 0

                    attention_mask = torch.cat((attention_mask, extended_attention_mask), dim=1)
                    position_ids = torch.sum(attention_mask, dim=1).unsqueeze(-1) - 1

        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        logits = outputs[0]

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            if attention_mask is not None:
                shift_attention_mask = attention_mask[..., 1:]
                shift_logits = logits[..., :-1, :][shift_attention_mask.to(logits.device) != 0].contiguous()
                shift_labels = labels[..., 1:][shift_attention_mask.to(labels.device) != 0].contiguous()
            else:
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(shift_logits.device)
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CullavoCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )