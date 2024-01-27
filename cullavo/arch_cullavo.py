import torch
import random
import torch.nn as nn
from copy import deepcopy
from .utils.utils import *
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.utils.generic import ModelOutput
from transformers import LlavaForConditionalGeneration
from utils.constants import COCO_PANOPTIC_CLASSES
from detectron2.utils.visualizer import Visualizer

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
    
    @staticmethod
    def make_system_prompt(processor, device, ignore_index):
        # system prompt
        system_prompt = "A chat between a curious human and an artificial intelligence assistant. "+\
                        "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        
        # concat system prompt and image prompt
        cullavo_prompt = system_prompt + "<image>"
        length = processor.tokenizer(cullavo_prompt, return_tensors='pt').input_ids[0].shape[0]
        cullavo_label = torch.tensor([ignore_index]*(length+575)).to(device)

        return cullavo_prompt, cullavo_label

    @staticmethod
    def make_and_add_prompt_and_label(cullavo_prompt, cullavo_label, prompt, answer, processor, device, ignore_index):
        
        # indent
        prompt = " USER: " + prompt + " ASSISTANT:"

        # Only Prompt Length
        length = processor.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0].shape[0]

        # Concat Prompt + Answer Length + stop token
        prompt = prompt + " " + str(answer) + "</s>"

        # input_ids and 
        label_ids = processor.tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids[0]
        label_ids[:length]=ignore_index
        
        # Concat previous prompt + current prompt
        cullavo_prompt += prompt
        cullavo_label = torch.tensor(cullavo_label.tolist() + label_ids.tolist()).to(device)
        
        return cullavo_prompt, cullavo_label

    def eval_process(
        self,
        images,
        aux_prompt=None,
        prompt=None,
        processor=None,
        device=None):
        batched_cullavo_prompt=[]

        # cullavo prompt prefix
        cullavo_prompt, _ = self.make_system_prompt(processor, device, self.config.ignore_index)

        # CLASS
        if aux_prompt:
            cullavo_prompt += f" {aux_prompt} USER: {prompt} ASSISTANT:"
        else:
            cullavo_prompt += f" USER: {prompt} ASSISTANT:"
        
        '''For Final Outputs'''
        cullavo_inputs = processor(text=cullavo_prompt, images=images, padding=True, return_tensors="pt")
        
        # [1] input_ids
        input_ids = cullavo_inputs.input_ids.to(device)
        
        # [2] pixel values
        pixel_values = cullavo_inputs.pixel_values.to(device)
        
        # [3] attention_mask
        attention_mask = cullavo_inputs.attention_mask.to(device)

        return {"input_ids": input_ids, "pixel_values": pixel_values, "attention_mask": attention_mask,}


    def step1_process(
        self,
        inputs,
        processor,
        device,
        fix_num=10
    ):
    
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
        
        # deepcopy
        _color_list= deepcopy(color_list)

        # shuffle color_list
        random.shuffle(_color_list)

        # Drawing BBox & Seg
        # import numpy as np
        # from utils.constants import COCO_PANOPTIC_CLASSES
        # from detectron2.utils.visualizer import Visualizer
        # img = inputs[0]['image'].permute(1,2,0).cpu().numpy()
        # vis = Visualizer(inputs[0]['image'].permute(1,2,0).cpu().numpy())
        # out = vis.overlay_instances(boxes=inputs[0]['instances'].gt_boxes.tensor.cpu()).get_image()
        #                             masks=inputs[0]['instances'].gt_masks,
        #                             labels=[COCO_PANOPTIC_CLASSES[c] for c in inputs[0]['instances'].gt_classes]).get_image()
        
        batched_boxed_image=[]
        batched_cullavo_prompt=[]
        batched_cullavo_label=[]
        for input in inputs:

            # only consider thing object
            thing_index_list = []
            for index, is_things in enumerate(input['instances'].is_things):
                if is_things: thing_index_list.append(index)
            thing_index_tensor = torch.tensor(thing_index_list)[:len(_color_list)]
            if len(thing_index_tensor)==0: continue # Exception Handling

            # CLASSES - thing
            thing_class_ids = input['instances'].gt_classes[thing_index_tensor]
            thing_classes = [COCO_PANOPTIC_CLASSES[c.item()].replace('-merged','').replace('-other','').replace('-stuff','') for c in thing_class_ids]
            unique_thing_class_ids = thing_class_ids.unique()
            unique_thing_classes = [COCO_PANOPTIC_CLASSES[c.item()].replace('-merged','').replace('-other','').replace('-stuff','') for c in unique_thing_class_ids]

            # BOXES - thing
            _,H,W = input['image'].shape
            input['instances'].gt_boxes.scale(1/W,1/H)
            thing_boxes = input['instances'].gt_boxes.tensor[thing_index_tensor]

            # BOX Image
            vis = Visualizer(input['image'].cpu().permute(1,2,0))
            vis._default_font_size = 4 # box edge font
            boxed_image = vis.overlay_instances(boxes=(thing_boxes*H).cpu().numpy(),
                                                assigned_colors=_color_list[:len(thing_index_tensor)],
                                                alpha=1).get_image()

            # cullavo prompt prefix
            cullavo_prompt, cullavo_label = self.make_system_prompt(processor, device, self.config.ignore_index)
            
            # IMAGE -> CLASS, BOX
            prompt = f"provide multiple object names with their numbering index and the objects' bounding box coordinates in this image."
            if len(thing_classes)==1:
                answer = f"Sure, it is {classesboxes2string(thing_classes, thing_boxes)}. There is an object in this image."
            else:
                answer = f"Sure, it is {classesboxes2string(thing_classes, thing_boxes)}. There are {len(thing_classes)} objects in this image."
            cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt,
                                                                            cullavo_label=cullavo_label, 
                                                                            prompt=prompt,
                                                                            answer=answer,
                                                                            processor=processor,
                                                                            device=device,
                                                                            ignore_index=self.config.ignore_index)
            # Rolling Dice
            rolling_dice = torch.randint(high=2, low=0, size=(1,)).item()
            if rolling_dice==0:
                # IMAGE -> COLOR
                prompt = f"provide multiple bounding box colors in this image"
                if len(_color_list[:len(thing_index_tensor)])==1:
                    answer = f"Sure, it is {list2string(_color_list[:len(thing_index_tensor)])} color. There is a bounding box in this image."
                else:
                    answer = f"Sure, it is {list2string(_color_list[:len(thing_index_tensor)])} color. There are {len(_color_list[:len(thing_index_tensor)])} bounding boxes in this image."

                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                cullavo_label=cullavo_label, 
                                                                                prompt=prompt,
                                                                                answer=answer,
                                                                                processor=processor,
                                                                                device=device,
                                                                                ignore_index=self.config.ignore_index)
            elif rolling_dice==1:
                # CLASS -> COLOR
                # Class selection Rolling dice 
                rolling_dice = torch.randint(high=len(unique_thing_class_ids), low=0, size=(1,)).item()
                selected_class_id = unique_thing_class_ids[rolling_dice]
                selected_class = unique_thing_classes[rolling_dice]
                selected_index = torch.where(thing_class_ids==selected_class_id)
                selected_classes = [thing_classes[i.item()] for i in selected_index[0]]
                selected_boxes = thing_boxes[selected_index]
                selected_colors = [_color_list[i.item()] for i in selected_index[0]]

                prompt = f"provide multiple colors for multiple bounding boxes corresponding {selected_class} in this image"
                if len(selected_classes)==1:
                    answer = f"Sure, it is {classescolors2string(selected_classes, selected_colors)} color. There is a bounding box in this image."
                else:
                    answer = f"Sure, it is {classescolors2string(selected_classes, selected_colors)} color. There are {len(selected_classes)} bounding boxes in this image."

                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                cullavo_label=cullavo_label, 
                                                                                prompt=prompt,
                                                                                answer=answer,
                                                                                processor=processor,
                                                                                device=device,
                                                                                ignore_index=self.config.ignore_index)
            else:
                raise Exception("This is unexpected error")


            # Random shuffling
            rand_int = torch.randperm(len(thing_boxes))[:fix_num]

            # Making cullavo prompt
            for r_int in rand_int:
                cls = thing_classes[r_int]
                box = thing_boxes[r_int]
                color = _color_list[r_int]

                # Rolling dice 
                rolling_dice = torch.randint(high=2, low=0, size=(1,)).item()
                if rolling_dice == 0:
                    # Color -> BOX
                    prompt = f"provide a bounding box coordinate of {color} color bounding box."
                    answer = f"Sure, it is {box2string(box)}. There is a {color} color bounding box"
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt,
                                                                                    answer=answer,
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                elif rolling_dice == 1:
                    # BOX -> Color
                    prompt = f"provide a bouding box color of bounding box coordinate {box2string(box)}."
                    answer = f"Sure, it is {color} color."
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt,
                                                                                    answer=answer,
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                else:
                    raise Exception("This is unexpected error")
            
                # Rolling dice 
                rolling_dice = torch.randint(high=2, low=0, size=(1,)).item()
                if rolling_dice == 0:
                    # BOX -> CLASS
                    prompt = f"provide object name for bounding box coordinate {box2string(box)}."
                    answer = f"Sure, it is {cls}."
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt,
                                                                                    answer=answer,
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                elif rolling_dice == 1:
                    # COLOR -> CLASS
                    prompt = f"provide object name for {color} bounding box."
                    answer = f"Sure, it is {cls}."
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt,
                                                                                    answer=answer,
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                else:
                    raise Exception("This is unexpected error")
            
            # making batched cullavo prompt
            batched_boxed_image.append(torch.from_numpy(boxed_image).permute(2, 0, 1).to(device))
            batched_cullavo_prompt.append(cullavo_prompt)
            batched_cullavo_label.append(cullavo_label)
        
        # TRY-EXCEPT Handling
        if len(batched_cullavo_prompt) == 0: return {"input_ids": None}

        '''For Final Outputs'''
        cullavo_inputs = \
        processor(text=batched_cullavo_prompt, images=torch.stack(batched_boxed_image), padding=True, return_tensors="pt")

        # [1] input_ids
        input_ids = cullavo_inputs.input_ids.to(device)

        # [2] pixel values
        pixel_values = cullavo_inputs.pixel_values.to(device)

        # [3] attention_mask
        attention_mask = cullavo_inputs.attention_mask.to(device)

        # [4] labels
        labels = torch.nn.utils.rnn.pad_sequence(batched_cullavo_label, batch_first=True, padding_value=self.config.ignore_index)

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

    def step2_preprocess(self, 
                        batched_inputs, 
                        processor, 
                        device):

        # new json list
        new_json_list = []
        for batch in batched_inputs:

            if not 'image' in batch.keys():
                new_json_list.append({'id': batch['question_id'], 'conversations': batch['question']})


            # interpolation
            interp_image = F.interpolate(batch['image'].unsqueeze(0), size=(336,336)).squeeze(0)
                    
            # CuLLaVO: llm preparation
            cullavo_inputs = self.eval_process(images=interp_image, 
                                            prompt=f"provide multiple object names with their numbering index and the objects' bounding box coordinates in this image.", 
                                            processor=processor, 
                                            device=device)
            # Generation
            with torch.inference_mode():
                generate_ids = self.generate(**cullavo_inputs, do_sample=True, temperature=0.9, top_k=50, top_p=0.95, max_new_tokens=1000, use_cache=True)
            decoded_text = processor.batch_decode(generate_ids, skip_special_tokens=True)[0]
            
            # Box parsing
            box_tensor, box_flag = box_parser(decoded_text)

            # Class parsing
            class_list, class_flag = class_parser(decoded_text)

            # TRY-EXECPTION HANDLING
            if box_flag + class_flag: continue

            # Visualize
            # img = interp_image.permute(1,2,0).cpu().numpy()
            # vis = Visualizer(img)
            # vis._default_font_size = 4
            # out = vis.overlay_instances(boxes=box_tensor*336,
            #                             labels=class_list, 
            #                             assigned_colors=color_list[:box_tensor.shape[0]],
            #                             alpha=1).get_image()

            # making new json for CuLLaVO Dataset
            new_json_list.append({'id': batch['question_id'], 'image': batch['image_id'], 'conversations': batch['question'], 'boxes': box_tensor.cpu().tolist(), 'classes': class_list})

        return new_json_list

    def step2_process(self, 
                      batched_inputs, 
                      processor, 
                      device):

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

        # IMAGE/QUESTION/ANSWER PREPROCESS
        images_list = []
        batched_cullavo_prompt = []
        batched_cullavo_label = []
        for batch in batched_inputs:
            # image append
            images_list.append(batch['image'])
            
            # cullavo prompt prefix
            cullavo_prompt, cullavo_label = self.make_system_prompt(processor, device, self.config.ignore_index)
            
            # Drawing BBox & Seg
            # from detectron2.utils.visualizer import Visualizer
            # img = batch['image'].permute(1,2,0).cpu().numpy()
            # vis = Visualizer(batch['image']['image'].permute(1,2,0).cpu().numpy())
            # out = vis.draw_box(torch.tensor([0.391, 0.478, 0.885, 0.987])*1024).get_image()

            # CuLLaVO: llm preparation for Generation
            # cullavo_inputs = self.eval_process(images=batch['image'], 
            #                                     prompt="provide the name of objects in this image", 
            #                                     processor=processor, 
            #                                     device=device)
            
            # # self.language_model.set_adapter(["step1","step2"])
            # self.language_model.disable_adapters()
            # with torch.inference_mode():
            #     generate_ids = self.generate(**cullavo_inputs, do_sample=False, temperature=0, max_new_tokens=30, use_cache=True)
            # decoded_text = processor.batch_decode(generate_ids)[0]


            # make prompt and answer
            for k in range(len(batch['question'])//2):
                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                cullavo_label=cullavo_label, 
                                                                prompt=batch['question'][2*k]['value'] if k!=0 else batch['question'][2*k]['value'].replace('<image>').strip(),
                                                                answer=batch['question'][2*k+1]['value'],
                                                                processor=processor,
                                                                device=device,
                                                                ignore_index=self.config.ignore_index)
                
            # making batched cullavo prompt
            batched_cullavo_prompt.append(cullavo_prompt)
            batched_cullavo_label.append(cullavo_label)


        '''For Final Outputs'''
        cullavo_inputs = \
        processor(text=batched_cullavo_prompt, images=torch.stack(images_list), padding=True, return_tensors="pt")
        
        # [1] input_ids
        input_ids = cullavo_inputs.input_ids.to(device)
        
        # [2] pixel values
        pixel_values = cullavo_inputs.pixel_values.to(device)
        
        # [3] attention_mask
        attention_mask = cullavo_inputs.attention_mask.to(device)
                
        # [4] labels
        labels = torch.nn.utils.rnn.pad_sequence(batched_cullavo_label, batch_first=True, padding_value=self.config.ignore_index)


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

        # CuLLaVO
        self.vision_tower.eval()

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
                inputs_embeds, attention_mask, _, position_ids = self._merge_input_ids_with_image_features(
                    image_features, inputs_embeds, input_ids, attention_mask, labels
                )
                if labels is None:
                    labels = torch.full_like(attention_mask, self.config.ignore_index).to(torch.long)
            else:
                # In case input_ids.shape[1] == 1 & pixel_values==None & past_key_values != None, we are in the case of
                # generation with cache
                if past_key_values is not None and pixel_values is not None and input_ids.shape[1] == 1:
                    # Retrieve the first layer to inspect the logits and mask out the hidden states
                    # that are set to 0
                    first_layer_past_key_value = past_key_values[0][0][:, :, :, 0]

                    # Sum all dimensions of head_dim (-2) to avoid random errors such as: https://github.com/huggingface/transformers/pull/28032#issuecomment-1863691941
                    batch_index, non_attended_tokens = torch.where(first_layer_past_key_value.float().sum(-2) == 0)

                    # Get the target length
                    target_seqlen = first_layer_past_key_value.shape[-1] + 1

                    extended_attention_mask = torch.ones(
                        (attention_mask.shape[0], target_seqlen - attention_mask.shape[1]),
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )

                    # Filter out only the tokens that can be un-attended, this can happen
                    # if one uses Llava + Fused modules where the cache on the
                    # first iteration is already big enough, or if one passes custom cache
                    valid_indices = non_attended_tokens < extended_attention_mask.size(-1)
                    new_batch_index = batch_index[valid_indices]
                    new_non_attended_tokens = non_attended_tokens[valid_indices]

                    # Zero-out the places where we don't need to attend
                    extended_attention_mask[new_batch_index, new_non_attended_tokens] = 0

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