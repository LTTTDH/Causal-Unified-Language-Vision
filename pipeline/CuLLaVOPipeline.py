# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import json
import logging

import torch
from tqdm import tqdm
from typing import Tuple, Dict

from trainer.default_trainer import DefaultTrainer

from modeling import build_model
from modeling.BaseModel import BaseModel
from datasets import build_eval_dataloader, build_train_dataloader
from trainer.utils.misc import move_batch_to_device
import torch.distributed as dist

logger = logging.getLogger(__name__)


class CuLLaVOPipeline:
    def __init__(self, opt):
        self._opt = opt

    def initialize_model(self):
        model = build_model(self._opt)
        model.train()
        model = BaseModel(self._opt, model)
        return model

    def get_dataloaders(
        self, trainer: DefaultTrainer,
        dataset_label: str,
        is_evaluation: bool
    ):
        distributed = self._opt['world_size'] > 1
        if is_evaluation:
            if not hasattr(self, 'valid_loader'):
                dataloaders = build_eval_dataloader(self._opt)
                self.valid_loader = dataloaders
            else:
                dataloaders = self.valid_loader
            idx = 0 if dataset_label=='dev' else self._opt['DATASETS']['TEST'].index(dataset_label)
            dataloader = dataloaders[idx]
        else:
            if not hasattr(self, 'train_loader'):
                dataloader = build_train_dataloader(self._opt)
                self.train_loader = dataloader
                # logger.info(f'num of train samples: {len(dataloader)}')
            else:
                dataloader = self.train_loader
                
            # temp solution for lr scheduler
            steps_total = len(self.train_loader)
            steps_acc = self._opt['OPTIMIZER']['GRAD_CUM']
            steps_update = steps_total // steps_acc
            self._opt["LR_SCHEDULER_PARAMS"]["steps_update_per_epoch"] = steps_update
        return dataloader

    @staticmethod
    def all_gather(data, world_size):
        output = [None for _ in range(world_size)]
        dist.all_gather_object(output, data, group=None)
        return output

    @staticmethod
    def forward_func(trainer, batch):
        loss = trainer.model(batch, trainer.accel)
        return loss

    def forward_step(
        self,
        trainer: DefaultTrainer,
        batch,
    ) -> Tuple[Dict[str, float], Dict[str, int], Dict]:
        loss_info, sample_size_info, extra_info = {}, {}, {}
        batch = move_batch_to_device(batch, trainer.accel.device)
        loss = trainer.compute_loss(self.forward_func, batch)
        loss_info = {k: v.detach().item() for k,v in loss.items()}
        sample_size_info = {'num_samples': len(batch)}
        loss = sum(loss for loss in loss.values())
        trainer.accel.wait_for_everyone() # Wait for Sync
        if loss.requires_grad: # TRY-EXCEPT HANDLING
            trainer.backward_loss(loss)
            if trainer.accel.sync_gradients:
                trainer.accel.clip_grad_norm_(trainer.model.parameters(), self._opt['OPTIMIZER']['GRAD_MAX'])
            trainer.update_model()
        return loss_info, sample_size_info, extra_info

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
    ) -> Tuple[Dict, Dict[str, float], bool]:
        
        model = trainer.model.eval()
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}

        new_json_dict_list_extend = []
        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            
            # accelerate wrapping
            model, eval_batch_gen = trainer.accel.prepare(model, eval_batch_gen)
            try:
                model = model.module # DDP
            except:
                pass # Deepspeed or Not DDP
            
            with torch.no_grad():
                prog_bar = tqdm(enumerate(eval_batch_gen), total=len(eval_batch_gen), leave=True, disable=not trainer.accel.is_local_main_process)
                for idx, batch in prog_bar:
                    batch = move_batch_to_device(batch, trainer.accel.device)
                    new_json_list = model(batch, accel=trainer.accel)
                    new_json_dict_list_extend.extend(new_json_list)
        
        trainer.accel.wait_for_everyone()
        if self._opt['world_size'] > 1:
            temp = self.all_gather(new_json_dict_list_extend, self._opt['world_size'])
            new_json_dict_list_extend = [] 
            for t in temp: new_json_dict_list_extend += t

        # New Dataset
        with open(f"/mnt/hard/lbk-cvpr/dataset/ShareGPT4V/data/sharegpt4v/lbk.json", "w") as f:
            json.dump(new_json_dict_list_extend, f)
        return scores