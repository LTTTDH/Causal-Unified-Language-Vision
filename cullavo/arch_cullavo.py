import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
from transformers.utils.generic import ModelOutput
from transformers import LlavaForConditionalGeneration
from utils.constants import COCO_PANOPTIC_CLASSES

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
    def classes2string(classes):
        out = ''
        for i, x in enumerate(classes):
            out+=f"{x}"
            if i!=len(classes)-1: out+=', '
        return out

    @staticmethod
    def boxes2string(boxes):
        out = ''
        for i, x in enumerate(boxes):
            out+=CuLLaVOModel.box2string(x)
            if i!=len(boxes)-1: out+=', '
        return out

    @staticmethod
    def box2string(box):
        out = '['
        for i, x in enumerate(box):
            out+=f"{round(x.item(), 3):.3f}"
            if i!=len(box)-1: out+=', '
        out += ']'
        return out

    @staticmethod
    def segs2string(segs):
        out = ''
        for i, x in enumerate(segs):
            out+=CuLLaVOModel.seg2string(x)
            if i!=len(segs)-1: out+=', '
        return out

    @staticmethod
    def seg2string(seg):
        out = '['
        for i, x in enumerate(seg):
            out+=f"{x}"
            if i!=len(seg)-1: out+=', '
        out+=']'
        return out

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
        prompt,
        processor,
        device):
        batched_cullavo_prompt=[]

        # cullavo prompt prefix
        cullavo_prompt, _ = self.make_system_prompt(processor, device, self.config.ignore_index)

        # CLASS
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


    def train_process(
        self,
        inputs,
        processor,
        vq_clip,
        device
    ):                    
        # fix num
        fix_num = 5
         
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
        # out = vis.draw_box(torch.tensor([0.391, 0.478, 0.885, 0.987])*1024).get_image()
        # out = vis.overlay_instances(boxes=inputs[0]['instances'].gt_boxes.tensor * 1024,
        #                             masks=inputs[0]['instances'].gt_masks,
        #                             labels=[COCO_PANOPTIC_CLASSES[c] for c in inputs[0]['instances'].gt_classes]).get_image()
        
        batched_cullavo_prompt=[]
        batched_cullavo_label=[]
        removed_index_list=[]
        for input_index, input in enumerate(inputs):
            # Shape
            _,H,W=input['image'].shape
             
            # cullavo prompt prefix
            cullavo_prompt, cullavo_label = self.make_system_prompt(processor, device, self.config.ignore_index)
            
            # CLASSES
            class_ids = input['instances'].gt_classes
            classes = [COCO_PANOPTIC_CLASSES[c].replace('-merged','').replace('-other','').replace('-stuff','') for c in input['instances'].gt_classes]
            unique_class_ids = class_ids.unique()
            unique_classes = [COCO_PANOPTIC_CLASSES[c].replace('-merged','').replace('-other','').replace('-stuff','') for c in unique_class_ids]
            
            # BOXES
            input['instances'].gt_boxes.scale(1/W,1/H)
            boxes = input['instances'].gt_boxes.tensor
            
            # MASKS
            masks = input['instances'].gt_masks.bool().float()

            # MASKED IMAGES
            masked_image = masks.unsqueeze(1).repeat(1, 3, 1, 1) * input['image']

            # TRY-EXCEPT
            if masked_image.shape[0]==0:
                removed_index_list.append(input_index)
                continue
            
            # MASK INDICES by VQ-CLIP
            mask_indices = vq_clip.quantize(masked_image, device)
            
            # LBK Visualization
            # m = masks[3]
            # sm = small_masks[0]
            # seq = self.mask2seq(small_masks[0])
            # a = self.seq2mask(seq)
        
            # Rolling dice 
            rolling_dice = torch.randint(high=3, low=0, size=(1,)).item()
            if rolling_dice==0:
                # [OPT1] IMG -> CLS
                prompt = f"provide the class name of object in this image"
                if len(unique_classes)==1:
                    answer = f"Sure, there is one object in this image. Its class name is {self.classes2string(unique_classes)}."
                else:
                    answer = f"Sure, there are {len(unique_classes)} objects in this image. Their class names of multiple objects are {self.classes2string(unique_classes)}."
                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                cullavo_label=cullavo_label, 
                                                                                prompt=prompt,
                                                                                answer=answer,
                                                                                processor=processor,
                                                                                device=device,
                                                                                ignore_index=self.config.ignore_index)
            elif rolling_dice==1:
                # Rolling dice 
                rolling_dice = torch.randint(high=len(unique_class_ids), low=0, size=(1,)).item()
                selected_class_id = unique_class_ids[rolling_dice]
                selected_class = unique_classes[rolling_dice]
                selected_index = torch.where(class_ids==selected_class_id)
                selected_boxes = boxes[selected_index]

                prompt = f"provide bounding box coordinate of {selected_class} in this image"
                if len(selected_boxes)==1:
                    answer = f"Sure, there is one bounding box coordinate [x_min, y_min, x_max, y_max] for {selected_class} in this image. This bounding box coordinate of {selected_class} is {self.boxes2string(selected_boxes)}."
                else:
                    answer = f"Sure, there are {len(selected_boxes)} bounding box coordinates [x_min, y_min, x_max, y_max] for {selected_class} in this image. Multiple bounding box coordiantes of {selected_class} is {self.boxes2string(selected_boxes)}."
                # [OPT2] CLS -> BOX
                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                cullavo_label=cullavo_label, 
                                                                                prompt=prompt,
                                                                                answer=answer,
                                                                                processor=processor,
                                                                                device=device,
                                                                                ignore_index=self.config.ignore_index)
            elif rolling_dice==2:
                # Rolling dice 
                rolling_dice = torch.randint(high=len(unique_class_ids), low=0, size=(1,)).item()
                selected_class_id = unique_class_ids[rolling_dice]
                selected_class = unique_classes[rolling_dice]
                selected_index = torch.where(class_ids==selected_class_id)
                selected_segs = mask_indices[selected_index]

                prompt = f"provide segmentation token sequence of {selected_class} in this image."
                answer = f"{self.segs2string(selected_segs)}"

                if len(selected_segs)==1:
                    answer = f"Sure, there is one segmentation token sequence having 49 number of integer numbers from 0 to 63 for {selected_class} in this image. This segmentation token index is {self.segs2string(selected_segs)}."
                else:
                    answer = f"Sure, there are {len(selected_segs)} segmentation token sequences having 49 number of integer numbers from 0 to 63 for {selected_class} in this image. Multiple segmentation token indexes are {self.segs2string(selected_segs)}."

                # [OPT3] CLS -> SEG
                cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                cullavo_label=cullavo_label, 
                                                                                prompt=prompt,
                                                                                answer=answer,
                                                                                processor=processor,
                                                                                device=device,
                                                                                ignore_index=self.config.ignore_index)



            # Random shuffling
            rand_int = torch.randperm(masked_image.shape[0])[:fix_num]

            # Making cullavo prompt
            for r_int in rand_int:
                if masks[r_int].sum()==0: continue
                cls = classes[r_int]
                box = boxes[r_int]
                seg = mask_indices[r_int]
                
                # Rolling dice 
                rolling_dice = torch.randint(high=3, low=0, size=(1,)).item()
                if rolling_dice==0:
                    prompt = f"provide bounding box coordinate of object corresponding to segmentation token sequence {self.seg2string(seg)}."
                    answer = f"Sure, segmentation token sequence having 49 number of integer numbers from 0 to 63 {self.seg2string(seg)} corresponds to bounding box coordinate [x_min, y_min, x_max, y_max] of object {self.box2string(box)}."
                    # [OPT1] SEG -> BOX
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt,
                                                                                    answer=answer,
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)

                elif rolling_dice==1:
                    prompt = f"provide the class name of the object that segmentation token sequence {self.seg2string(seg)} represents."
                    answer = f"Sure, segmentation token sequence having 49 number of integer numbers from 0 to 63 {self.seg2string(seg)} represents {cls}."
                    # [OPT2] SEG -> CLS
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt, 
                                                                                    answer=answer, 
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                elif rolling_dice==2:
                    prompt = f"provide segmentation token sequence of object corresponding to bounding box coordinate {self.box2string(box)} and the name of object {cls}."
                    answer = f"Sure, [x_min, y_min, x_max, y_max] bounding box coordinate {self.box2string(box)} and the name of object {cls} corresponds to segmentation token sequence having 49 number of integer numbers from 0 to 63 {self.seg2string(seg)}."
                    # [OPT3] BOX, CLASS -> SEG
                    cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                    cullavo_label=cullavo_label, 
                                                                                    prompt=prompt, 
                                                                                    answer=answer, 
                                                                                    processor=processor,
                                                                                    device=device,
                                                                                    ignore_index=self.config.ignore_index)
                else:
                    raise Exception("This is unexpected error")

            # Vision Grounding 
            if input['groundings']['mode']=='text':
                
                # TEXTS
                texts = [str(x) for x in input['groundings']['texts']]
                
                # CLASSES    
                classes = input['groundings']['classes']
                
                # BOXES
                from detectron2.structures import Boxes
                boxes = Boxes(input['groundings']['boxes'])
                boxes.scale(1/W, 1/H)                

                # MASKS
                masks = input['groundings']['masks'].bool().float()

                # MASKED IMAGES
                masked_image = masks.unsqueeze(1).repeat(1, 3, 1, 1) * input['image']
                
                # TRY-EXCEPT
                if masked_image.shape[0]==0:
                    removed_index_list.append(input_index)
                    continue

                # MASK INDICES by VQ-CLIP
                mask_indices = vq_clip.quantize(masked_image, device)

                # LBK visualization
                # img = input['image'].permute(1,2,0).cpu().numpy()
                # m = masks[2]
                # s = small_masks[0]
                
                # Random shuffling
                rand_int = torch.randperm(masked_image.shape[0])[:fix_num]

                # Making cullavo prompt
                for r_int in rand_int:
                    if masks[r_int].sum()==0: continue
                    cls = classes[r_int]
                    box = boxes.tensor[r_int]
                    seg = mask_indices[r_int]
                    txt = texts[r_int]
                    

                    # Rolling dice 
                    rolling_dice = torch.randint(high=2, low=0, size=(1,)).item()
                    if rolling_dice==0:
                        prompt = f"provide segmentation token sequence of object that the referring text represents: '{txt}'"
                        answer = f"Sure, the referring text '{txt}' represents segmentation token sequence having 49 number of integer numbers from 0 to 63 {self.seg2string(seg)}."
                        # [OPT1] TXT -> SEG
                        cullavo_prompt, cullavo_label = self.make_and_add_prompt_and_label(cullavo_prompt=cullavo_prompt, 
                                                                                        cullavo_label=cullavo_label, 
                                                                                        prompt=prompt, 
                                                                                        answer=answer, 
                                                                                        processor=processor,
                                                                                        device=device, 
                                                                                        ignore_index=self.config.ignore_index)
                    elif rolling_dice==1:
                        prompt = f"provide bounding box coordinate describing the referring text: '{txt}'"
                        answer = f"Sure, the referring text '{txt}' represents [x_min, y_min, x_max, y_max] bounding box coordinate {self.box2string(box)}"
                        # [OPT2] TXT -> BOX
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
            batched_cullavo_prompt.append(cullavo_prompt)
            batched_cullavo_label.append(cullavo_label)
            
        '''For Final Outputs'''
        cullavo_inputs = \
        processor(text=batched_cullavo_prompt, images=torch.stack([input['image'] for input_index, input in enumerate(inputs) if input_index not in removed_index_list]), padding=True, return_tensors="pt")
        
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