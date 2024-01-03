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

logger = logging.getLogger(__name__)


class RefCOCOPipeline:
    def __init__(self, opt):
        self._opt = opt

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
            self.evaluator = [build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR']) for _ in range(len(COCO_SEMANTIC_CLASSES))]
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
        
        # n_image_list = []
        n_image_list = [0 for _ in range(len(COCO_SEMANTIC_CLASSES))]
        
        # CSV
        if self._opt['rank'] == 0:
            import csv
            with open("problem_experiment/ref_coco.csv", "w") as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['CLASS', 'precision@0.5', 'precision@0.6', 'precision@0.7', 'precision@0.8', 'precision@0.9', 'cIoU', 'mIoU', 'n_image'])
        
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

                    # Model Prediction
                    outputs = model(batch, mode=eval_type)
                    
                    # Referring evaluate process
                    for i in range(len(batch[0]['grounding_info'])):
                        category_id = batch[0]['grounding_info'][i]['category_id']
                        self.evaluator[category_id-1].process(
                            [{'groundings': {'masks': batch[0]['groundings']['masks'][i].unsqueeze(0)}}],
                            [{'grounding_mask': outputs[0]['grounding_mask'][i].unsqueeze(0)}])
                        n_image_list[category_id-1] += 1
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
                    with open("problem_experiment/ref_coco.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{COCO_SEMANTIC_CLASSES[i]}'] + ['NaN']*7 + [n_image_list[i]])
            else:
                results = x.evaluate()
                if self._opt['rank'] == 0:
                    with open("problem_experiment/ref_coco.csv", "a+", newline='') as f:
                        csv_writer = csv.writer(f)
                        csv_writer.writerow([f'{COCO_SEMANTIC_CLASSES[i]}'] + [v for v in results['grounding'].values()] + [n_image_list[i]])
            if self._opt['world_size'] >1: dist.barrier()
        
        # Total Result Write on CSV
        results = self.evaluator_total.evaluate()
        if self._opt['rank'] == 0:
            with open("problem_experiment/ref_coco.csv", "a+", newline='') as f:
                csv_writer = csv.writer(f)
                csv_writer.writerow(['ALL'] + [v for v in results['grounding'].values()] + [sum(n_image_list)])

        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        return scores