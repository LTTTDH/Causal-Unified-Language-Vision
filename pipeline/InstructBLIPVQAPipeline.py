# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import logging
import time
import datetime

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import DataLoader
from typing import Tuple, Dict, List, Union
from infinibatch import iterators

from trainer.default_trainer import DefaultTrainer

from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from utils.constants import COCO_SEMANTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_opt
import torch.distributed as dist
from cullavo.utils.utils import *

logger = logging.getLogger(__name__)


from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

class InstructBLIPVQAPipeline:
    def __init__(self, opt):
        self._opt = opt
        self.data_classes = COCO_SEMANTIC_CLASSES

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
            self.evaluator = [build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR']) for _ in range(len(self.data_classes))]
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
            steps_acc = self._opt['GRADIENT_ACCUMULATE_STEP']
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

        # LLAMA2 8Bit compression
        from transformers import AutoTokenizer, LlamaForCausalLM
        llama2_model = LlamaForCausalLM.from_pretrained(LLAMA2_LOCAL_PATH, load_in_8bit=True, device_map=self._opt['rank'], torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        llama2_tokenizer = AutoTokenizer.from_pretrained(LLAMA2_LOCAL_PATH)
        self.eval_freeze(llama2_model)
        
        # BLIP2 8Bit compression
        instructblip_model = InstructBlipForConditionalGeneration.from_pretrained(BLIP2_LOCAL_PATH, load_in_8bit=True, device_map=self._opt['rank'], torch_dtype=torch.bfloat16)
        instructblip_processor = InstructBlipProcessor.from_pretrained(BLIP2_LOCAL_PATH)
        self.eval_freeze(instructblip_model)
        
        # CLIP
        import torch.nn.functional as F
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained(CLIPLARGE_LOCAL_PATH)
        clip_processor = CLIPProcessor.from_pretrained(CLIPLARGE_LOCAL_PATH)
        clip_model = clip_model.cuda()
        self.eval_freeze(clip_model)

        # CLIP Text
        text_inputs = clip_processor(text=self.data_classes, return_tensors="pt", padding=True)
        text = clip_model.text_model(**{k:v.cuda() for k, v in text_inputs.items()})[1]
        text = clip_model.text_projection(text)
        norm_text = F.normalize(text, dim=1)
        
        # n_image_list = []
        n_image_list = [0 for _ in range(len(self.data_classes))]
        
        # CSV
        if self._opt['rank'] == 0:
            import csv
            with open("problem_experiment/instructblip_vqa.csv", "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['CLASS'] + ['Accuracy'] + ['n_image'])
        
        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            for x in self.evaluator: x.reset()
            self.evaluator_total.reset()
            with torch.no_grad():
                prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True)
                for idx, batch in prog_bar:

                    batch = move_batch_to_device(batch, self._opt['device'])
                    if self._opt['FP16']:
                        # in FP16 mode, DeepSpeed casts the model to FP16, so the input needs to be manually casted to FP16
                        batch = cast_batch_to_half(batch)

                    # Visualization
                    # a = batch[0]['image'].flip(0).permute(1,2,0).cpu().numpy()
                    
                    # LLAMA2
                    llama2_prompt = f"Choose object the question asks" +\
                                    "ex) what color is the man's shirt? shirt. " +\
                                    "ex) how many bikes have helmets? helmets. " +\
                                    f"ex) where are the dogs looking at? dogs. ex) {batch[0]['captions'][0]}"
                    llama2_inputs = llama2_tokenizer(llama2_prompt, return_tensors="pt")

                    # LLAMA2 In-Context Generation
                    with torch.inference_mode():
                        llama2_generate_ids = llama2_model.generate(llama2_inputs.input_ids.cuda(), max_new_tokens=10, do_sample=True, top_p=0.9, temperature=0.9, pad_token_id=llama2_tokenizer.eos_token_id)
                    llama2_text = llama2_tokenizer.batch_decode(llama2_generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0][len(llama2_prompt):].strip()
                    
                    # CLIP Text
                    text_inputs = clip_processor(text=llama2_text.split('.')[0], return_tensors="pt", padding=True)
                    text_embed = clip_model.text_model(**{k:v.cuda()for k, v in text_inputs.items()})[1]
                    text_embed = clip_model.text_projection(text_embed)
                    norm_text_embed = F.normalize(text_embed, dim=1)
                    
                    # CLIP SCORE
                    score = norm_text_embed @ norm_text.T
                    clip_value, clip_index = score.topk(k=1, dim=1)
                    clip_index = clip_index[clip_value.argmax()]

                    # InstructBLIP Process
                    prompt = [f"Question: {b['captions'][0]} Answer:" for b in batch]
                    instructblip_inputs = instructblip_processor(text=prompt, images=torch.stack([b['image'] for b in batch]), return_tensors="pt")
                    
                    # Generate
                    with torch.inference_mode():
                        generate_ids = instructblip_model.generate(**{k:v.cuda() for k,v in instructblip_inputs.items()}, max_new_tokens=10, min_length=1, num_beams=5, length_penalty=-1)
                    decoded_text = instructblip_processor.batch_decode(generate_ids, skip_special_tokens=True)[0].strip()
                    
                    # VQA evaluate process
                    self.evaluator[clip_index[0]].process(batch, {'question_id': batch[0]["question_ids"][0], 'text': decoded_text})
                    self.evaluator_total.process(batch, {'question_id': batch[0]["question_ids"][0], 'text': decoded_text})
                    n_image_list[clip_index[0]] += 1
                    
                    # Fast Computation
                    if idx > len(eval_batch_gen) * 0.1: break
            
        # DDP communication
        if self._opt['world_size'] > 1:
            dist.barrier()
            new_n_image_list = []
            for x in n_image_list:
                new_n_image_list.append(sum(self.all_gather(x, self._opt['world_size'])))
            n_image_list = new_n_image_list

        # Class-wise Result Write on CSV
        if self._opt['world_size'] > 1: dist.barrier()
        for i, x in enumerate(self.evaluator):
            if n_image_list[i]==0:
                if self._opt['rank'] == 0:
                    with open("problem_experiment/instructblip_vqa.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + ['NaN'] + [n_image_list[i]])
            else:
                results = x.evaluate()
                if self._opt['rank'] == 0:
                    with open("problem_experiment/instructblip_vqa.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + [results['accuracy']] + [n_image_list[i]])
            if self._opt['world_size'] >1: dist.barrier()
        
        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        if self._opt['rank'] == 0:
            with open("problem_experiment/instructblip_vqa.csv", "a+", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['ALL'] + [results['accuracy']] + [sum(n_image_list)])
        return scores