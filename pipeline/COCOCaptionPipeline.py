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

from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog

from modeling import build_model
from modeling.utils import get_class_names
from modeling.BaseModel import BaseModel
from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from utils.distributed import is_main_process
from utils.constants import COCO_SEMANTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_metadata, hook_switcher, hook_opt
import torch.distributed as dist
from cullavo.utils.utils import *

logger = logging.getLogger(__name__)


class COCOCaptionPipeline:
    def __init__(self, opt):
        self._opt = opt
        self.data_classes = COCO_SEMANTIC_CLASSES

    def initialize_model(self):
        model_name = "default"
        model = build_model(self._opt)
        model.train()

        # if is_main_process():
        #     logger.info(model)

        raw_models = {model_name: BaseModel(self._opt, model)}
        return raw_models

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
            from copy import copy
            memory_evaluator = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
            self.evaluator = [copy(memory_evaluator) for _ in range(len(self.data_classes))]
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
    def forward_func(trainer, batch):
        loss = trainer.models['default'](batch)
        return loss

    def forward_step(
        self,
        trainer: DefaultTrainer,
        batch,
        grad_acc_batches: List,
        grad_acc_index: int,
        is_distributed: bool,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        loss_info, sample_size_info, extra_info = {}, {}, {}
        batch = move_batch_to_device(batch, self._opt['device'])
        if self._opt['FP16']:
            # in FP16 mode, DeepSpeed casts the model to FP16, so the input needs to be manually casted to FP16
            batch = cast_batch_to_half(batch)
        loss = trainer.compute_loss(self.forward_func, batch)
        loss_info = {k: v.detach().item() for k,v in loss.items()}
        sample_size_info = {'num_samples': len(batch)}
        loss = sum(loss for loss in loss.values())
        trainer.backward_loss(loss, model_names=['default'])
        trainer.update_model(model_name='default')
        return loss_info, sample_size_info, extra_info

    @staticmethod
    def all_gather(data, world_size):
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, data, group=None)
        return output

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        model = trainer.raw_models['default'].eval()
        self._opt = hook_opt(self._opt)
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}
        summary = {}

        # CLIP
        import torch.nn.functional as F
        from transformers import CLIPProcessor, CLIPModel
        clip_model = CLIPModel.from_pretrained(CLIPLARGE_LOCAL_PATH)
        clip_processor = CLIPProcessor.from_pretrained(CLIPLARGE_LOCAL_PATH)
        clip_model = clip_model.cuda()
        
        # CLIP Text
        text_inputs = clip_processor(text=[f"a photo of {cl}" for cl in self.data_classes], return_tensors="pt", padding=True)
        text = clip_model.text_model(**{k:v.cuda()for k, v in text_inputs.items()})[1]
        text = clip_model.text_projection(text)
        norm_text = F.normalize(text, dim=1)
        
        # n_image_list = []
        n_image_list = [0 for _ in range(len(self.data_classes))]
        
        # CSV
        if self._opt['rank'] == 0:
            import csv
            with open("problem_experiment/coco_caption.csv", "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['CLASS', 'Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4', 'METEOR', 'ROUGE_L', 'CIDEr', 'n_image'])
        
        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            for x in self.evaluator: x.reset()
            self.evaluator_total.reset()
            with torch.no_grad():
                names = get_class_names(dataset_label)
                model.model.metadata = MetadataCatalog.get(dataset_label)
                model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                eval_type = model.model.metadata.evaluator_type
                if 'background' in names:
                    model.model.sem_seg_head.num_classes = len(names) - 1
                model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(names, is_eval=True)
                hook_switcher(model, dataset_label)
                prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True)
                for idx, batch in prog_bar:
                    batch = move_batch_to_device(batch, self._opt['device'])
                    if self._opt['FP16']:
                        # in FP16 mode, DeepSpeed casts the model to FP16, so the input needs to be manually casted to FP16
                        batch = cast_batch_to_half(batch)
                                                
                    # Visualization
                    # a = batch[7]['image'].flip(0).permute(1,2,0).cpu().numpy()
                    
                    # CLIP Vision
                    vision_inputs = clip_processor(images=torch.stack([b['image'].flip(0) for b in batch]), return_tensors="pt", padding=True)
                    vision_embed = clip_model.vision_model(**{k:v.cuda()for k, v in vision_inputs.items()})[1]
                    vision_embed = clip_model.visual_projection(vision_embed)
                    norm_vision_embed = F.normalize(vision_embed, dim=1)
                    
                    # CLIP SCORE
                    score = norm_vision_embed @ norm_text.T
                    clip_index = score.topk(k=1, dim=1)[1]
                    
                    # Model Prediction
                    outputs = model(batch, mode=eval_type)
                    
                    # Captioning evaluate process
                    for i in range(clip_index.shape[0]):
                        for j in range(clip_index.shape[1]):
                            self.evaluator[clip_index[i][j]].process([batch[i]], [outputs[i]])
                            n_image_list[clip_index[i][j]] += 1
                    self.evaluator_total.process(batch, outputs)
                            
                model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()

        # DDP communication
        if self._opt['world_size'] >1:
            dist.barrier()
            new_n_image_list = []
            for x in n_image_list:
                new_n_image_list.append(sum(self.all_gather(x, self._opt['world_size'])))
            n_image_list = new_n_image_list        

        # Class-wise Result Write on CSV
        if self._opt['world_size'] >1: dist.barrier()
        for i, x in enumerate(self.evaluator):
            if n_image_list[i]==0:
                if self._opt['rank'] == 0:
                    with open("problem_experiment/coco_caption.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + ['NaN']*7 + [n_image_list[i]])
            else:
                results = x.evaluate()
                if self._opt['rank'] == 0:
                    with open("problem_experiment/coco_caption.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{self.data_classes[i]}'] + [v for v in results.values()] + [n_image_list[i]])
            if self._opt['world_size'] >1: dist.barrier()
            
        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        if self._opt['rank'] == 0:
            with open("problem_experiment/coco_caption.csv", "a+", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['ALL'] + [v for v in results.values()] + [sum(n_image_list)])


        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        return scores