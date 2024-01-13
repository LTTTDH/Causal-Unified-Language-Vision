# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import logging
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators

from trainer.default_trainer import DefaultTrainer

from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from trainer.utils.misc import move_batch_to_device

from .utils.misc import hook_opt
import torch.distributed as dist
from cullavo.utils.utils import *

logger = logging.getLogger(__name__)


from transformers import AutoProcessor, LlavaForConditionalGeneration

class LLaVAVQATESTPipeline:
    def __init__(self, opt):
        self._opt = opt

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ) -> Union[DataLoader, iterators.CheckpointableIterator]:
        distributed = self._opt['world_size'] > 1
        if is_evaluation:
            if not hasattr(self, 'valid_loader'):
                dataloaders = build_eval_dataloader(self._opt)
                self.valid_loader = dataloaders
            else:
                dataloaders = self.valid_loader
            idx = 0 if dataset_label=='dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
            dataloader = dataloaders[idx]
            self.evaluator_total = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
        else:
            if not hasattr(self, 'train_loader'):
                dataloader = build_train_dataloader(self._opt)
                self.train_loader = dataloader
                # logger.info(f'num of train samples: {len(dataloader)}')
            else:
                dataloader = self.train_loader
                
            # temp solution for lr scheduler
            steps_total = len(self.train_loader)
            steps_acc = self._opt['LLM']['GRAD_CUM']
            steps_update = steps_total // steps_acc
            self._opt["LR_SCHEDULER_PARAMS"]["steps_update_per_epoch"] = steps_update
        return dataloader

    @staticmethod
    def all_gather(data, world_size):
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, data, group=None)
        return output

    @staticmethod
    def eval_freeze(model):
        model.eval()
        for param in model.parameters(): param.requires_grad = False

    @staticmethod
    def print_dtype(model):
        for param in model.parameters(): print(param.dtype)

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        self._opt = hook_opt(self._opt)
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}

        # LLaVA 8Bit compression
        llava_model = LBK.from_pretrained(LLAVA_LOCAL_PATH, load_in_8bit=True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        llava_processor = AutoProcessor.from_pretrained(LLAVA_LOCAL_PATH)
        self.eval_freeze(llava_model)
        
        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            self.evaluator_total.reset()
            
            # accelerate wrapping
            llava_model, eval_batch_gen = trainer.accel.prepare(llava_model, eval_batch_gen)
            
            with torch.no_grad():
                prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True, disable=not trainer.accel.is_local_main_process)
                for idx, batch in prog_bar:

                    batch = move_batch_to_device(batch, trainer.accel.device)

                    # Visualization
                    # a = batch[0]['image'].permute(1,2,0).cpu().numpy()
                    
                    # LLaVA Process
                    prompt = ["A chat between a curious human and an artificial intelligence assistant. "
                              "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                              f"USER: <image>\n{batch[0]['question']}\nAnswer the question using a single word or phrase. ASSISTANT:"]
                    llava_inputs = llava_processor(text=prompt, images=batch[0]['image'], return_tensors="pt")
                    
                    # Generate
                    with torch.inference_mode():
                        generate_ids = llava_model.generate(**{k:v.to(trainer.accel.device) for k,v in llava_inputs.items()}, do_sample=False, temperature=0, max_new_tokens=128, use_cache=True)
                    decoded_text = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split('ASSISTANT:')[-1].strip()

                    # Bounding Box Prompt & Visualization
                    # prompt = ["A chat between a curious human and an artificial intelligence assistant. "
                    #           "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                    #           "<image> USER: what is the coordinate of bounding box for the pitcher in this image ASSISTANT:"]
                    # llava_inputs = llava_processor(text=prompt, images=batch[0]['image'], return_tensors="pt")
                    # with torch.inference_mode():
                    #     generate_ids = llava_model.generate(**{k:v.to(trainer.accel.device) for k,v in llava_inputs.items()}, do_sample=False, temperature=0, max_new_tokens=128, use_cache=True)
                    # decoded_text = llava_processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].split('ASSISTANT:')[-1].strip()
                    # from detectron2.utils.visualizer import Visualizer
                    # import torch.nn.functional as F
                    # vis = Visualizer(255* ((llava_inputs.pixel_values-llava_inputs.pixel_values.min())/(llava_inputs.pixel_values.max()-llava_inputs.pixel_values.min())).squeeze(0).permute(1,2,0).cpu().numpy())
                    # out = vis.draw_box(torch.tensor(eval(decoded_text))*336).get_image()

                    # VQA evaluate process
                    self.evaluator_total.process(batch, {'question_id': batch[0]["question_id"], 'text': decoded_text})
                    
        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        return scores


from typing import List, Optional, Tuple, Union
from transformers.models.llava.modeling_llava import LlavaCausalLMOutputWithPast

class LBK(LlavaForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        
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
    ) -> Union[Tuple, LlavaCausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
                config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
                (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:

        Example:

        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import AutoProcessor, LlavaForConditionalGeneration

        >>> model = LlavaForConditionalGeneration.from_pretrained("llava-hf/llava-1.5-7b-hf")
        >>> processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")

        >>> prompt = "<image>\nUSER: What's the content of the image?\nASSISTANT:"
        >>> url = "https://www.ilankelman.org/stopsigns/australia.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)

        >>> inputs = processor(text=text, images=image, return_tensors="pt")

        >>> # Generate
        >>> generate_ids = model.generate(**inputs, max_length=30)
        >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        "There seems to be a stop sign"
        ```"""

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
                    image_features, inputs_embeds, input_ids, attention_mask, position_ids
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

                    # TODO and FIXME The part of LBK Overload
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

        return LlavaCausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )