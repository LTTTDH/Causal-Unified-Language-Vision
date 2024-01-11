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
from utils.constants import IMAGENET_CLASSES
from trainer.utils.misc import move_batch_to_device

from .utils.misc import hook_opt
import torch.distributed as dist
from cullavo.utils.utils import *

logger = logging.getLogger(__name__)


from transformers import Blip2Processor, Blip2ForConditionalGeneration

class BLIP2ImageNetPipeline:
    def __init__(self, opt):
        self._opt = opt
        self.data_classes = IMAGENET_CLASSES

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ) -> Union[DataLoader, iterators.CheckpointableIterator]:
        # distributed = self._opt['world_size'] > 1
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
        if hasattr(model, "eval"):
            model.eval()
            for param in model.parameters(): param.requires_grad = False
        else:
            model.requires_grad = False

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
        
        # BLIP2 8Bit compression
        blip2_model = Blip2ForConditionalGeneration.from_pretrained(BLIP2_LOCAL_PATH, load_in_8bit=True, torch_dtype=torch.bfloat16)
        blip2_processor = Blip2Processor.from_pretrained(BLIP2_LOCAL_PATH)
        self.eval_freeze(blip2_model)

        # n_image_list = []
        n_image_list = [0 for _ in range(len(self.data_classes))]
        
        # CSV
        if trainer.accel.is_main_process:
            import csv
            with open("problem_experiment/blip2_imagenet.csv", "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['CLASS'] + ['Accuracy'] + ['n_image'])
        
        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            for x in self.evaluator: x.reset()
            self.evaluator_total.reset()

            # accelerate wrapping
            blip2_model, eval_batch_gen = trainer.accel.prepare(blip2_model, eval_batch_gen)

            with torch.no_grad():
                prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True)
                for idx, batch in prog_bar:
                    batch = move_batch_to_device(batch, trainer.accel.device)

                    # Only First process to get norm_text for all classes in ImageNet!
                    if idx == 0:
                        # BLIP2 Text
                        blip2_text_inputs = blip2_processor(text=[f'This is {cl}' for cl in self.data_classes], images=None, padding=True, return_tensors="pt")
                        with torch.inference_mode():
                            query_outputs = blip2_model.qformer(
                                input_ids=blip2_text_inputs.qformer_input_ids.to(trainer.accel.device),                         # [num classes, padded len]/ [1000, 14]
                                attention_mask=blip2_text_inputs.qformer_attention_mask.to(trainer.accel.device),      
                                return_dict=blip2_model.config.use_return_dict,
                            )
                        text_embed = query_outputs[1]                  # [num classes, dim_emb]/ [1000, 768]
                        import torch.nn.functional as F
                        norm_text = F.normalize(text_embed, dim=1)

                    # Visualization
                    # a = batch[1]['image'].permute(1,2,0).cpu().numpy()
                    
                    # blip2 inputs
                    blip2_img_inputs = blip2_processor(text=None, images=[x["image"] for x in batch], return_tensors="pt") # pixel_values: [batch size, C, H ,W]

                    # BLIP2 Process
                    with torch.inference_mode():
                        vision_outputs = blip2_model.vision_model(
                            pixel_values=blip2_img_inputs["pixel_values"].to(trainer.accel.device),
                            return_dict=blip2_model.config.use_return_dict,
                        )                                                                                                                # last_hidden_state, pooler_output
                        query_outputs = blip2_model.qformer(
                            input_ids=None,
                            attention_mask=None,
                            query_embeds=blip2_model.query_tokens.expand(vision_outputs[0].shape[0], -1, -1),                     # [batch size, 32, 768]
                            encoder_hidden_states=vision_outputs[0],
                            encoder_attention_mask=torch.ones(vision_outputs[0].size()[:-1], dtype=torch.long, device=vision_outputs[0].device),
                            return_dict=blip2_model.config.use_return_dict,
                        )
                    img_embed = query_outputs[1]
                    norm_img_embed = F.normalize(img_embed, dim=1)                                      # [batch size , dim_emb]

                    # SCORE MATRIX
                    score = norm_img_embed @ norm_text.T                                                # [batch size, num classes]

                    img_gt = torch.tensor([x["class_id"] for x in batch], device=trainer.accel.device)  # [batch size]
                    # Classification evaluate process
                    for idx, gt in enumerate(img_gt):
                        self.evaluator[gt].process([{'class_id': img_gt[idx]}], [{'pred_class': score[idx]}])
                        self.evaluator_total.process([{'class_id': img_gt[idx]}], [{'pred_class': score[idx]}])
                        n_image_list[gt] += 1
                    
                    # # Fast Computation
                    # if idx > len(eval_batch_gen) * 0.1: break
            
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
                if trainer.accel.is_main_process:
                    with open("problem_experiment/blip2_imagenet.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + ['NaN'] + [n_image_list[i]])
            else:
                results = x.evaluate()
                if trainer.accel.is_main_process:
                    with open("problem_experiment/blip2_imagenet.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + [results['class']['top1']] + [n_image_list[i]])
            if self._opt['world_size'] >1: dist.barrier()
        
        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        if trainer.accel.is_main_process:
            with open("problem_experiment/blip2_imagenet.csv", "a+", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['ALL'] + [results['class']['top1']] + [sum(n_image_list)])
        return scores
    