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

from trainer.default_trainer import DefaultTrainer

from detectron2.evaluation import inference_on_dataset
from detectron2.utils.logger import log_every_n_seconds
from detectron2.data import MetadataCatalog

from modeling import build_model
from modeling.utils import get_class_names
from modeling.BaseModel import BaseModel
from datasets import build_evaluator, build_eval_dataloader, build_train_dataloader
from utils.distributed import is_main_process
from utils.constants import COCO_PANOPTIC_CLASSES
from trainer.utils.misc import move_batch_to_device, cast_batch_to_half

from .utils.misc import hook_metadata, hook_switcher, hook_opt

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
            self.evaluator = build_evaluator(self._opt, self._opt['DATASETS']['TEST'][idx], self._opt['SAVE_DIR'])
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
        trainer.backward_loss(loss)
        trainer.update_model()
        return loss_info, sample_size_info, extra_info

    def evaluate_model(
        self,
        trainer: DefaultTrainer,
    ) -> Tuple[Dict, Dict[str, float], bool]:

        model = trainer.model.eval()
        self._opt = hook_opt(self._opt)
        dataset_names = self._opt['DATASETS']['TEST']
        scores = {}
        summary = {}

        for dataset_label in dataset_names:
            torch.cuda.empty_cache()
            eval_batch_gen = self.get_dataloaders(trainer, dataset_label, is_evaluation=True)
            self.evaluator.reset()
            
            # accelerate wrapping
            model, eval_batch_gen = trainer.accel.prepare(model, eval_batch_gen)
            try:
                model = model.module # DDP
            except:
                pass # Deepspeed or Not DDP
            
            with torch.no_grad():
                names = get_class_names(dataset_label)
                model.model.metadata = MetadataCatalog.get(dataset_label)
                model.model.metadata = hook_metadata(model.model.metadata, dataset_label)
                eval_type = model.model.metadata.evaluator_type
                hook_switcher(model, dataset_label)
                total = len(eval_batch_gen)
                num_warmup = min(5, total - 1)
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0
                start_data_time = time.perf_counter()
                
                prog_bar = tqdm(enumerate(eval_batch_gen), total=total, leave=True, disable=not trainer.accel.is_local_main_process)
                for idx, batch in prog_bar:
                    total_data_time += time.perf_counter() - start_data_time
                    if idx == num_warmup:
                        start_time = time.perf_counter()
                        total_data_time = 0
                        total_compute_time = 0
                        total_eval_time = 0

                    start_compute_time = time.perf_counter()
                    batch = move_batch_to_device(batch, trainer.accel.device)
                    outputs = model(batch, mode=eval_type, accel=trainer.accel)
                    if torch.cuda.is_available():
                        torch.cuda.synchronize()

                    total_compute_time += time.perf_counter() - start_compute_time
                    start_eval_time = time.perf_counter()

                    self.evaluator.process(batch, outputs)
                    total_eval_time += time.perf_counter() - start_eval_time

                    # iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
                    # data_seconds_per_iter = total_data_time / iters_after_start
                    # compute_seconds_per_iter = total_compute_time / iters_after_start
                    # eval_seconds_per_iter = total_eval_time / iters_after_start
                    # total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start

                    # if is_main_process()  and (idx >= num_warmup * 2 or compute_seconds_per_iter > 5):
                    #     eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                    #     log_every_n_seconds(
                    #         logging.INFO,
                    #         (
                    #             f"Task {dataset_label}. "
                    #             f"Inference done {idx + 1}/{total}. "
                    #             f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                    #             f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                    #             f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                    #             f"Total: {total_seconds_per_iter:.4f} s/iter. "
                    #             f"ETA={eta}"
                    #         ),
                    #         n=5,
                    #     )
                    # start_data_time = time.perf_counter()

            results = self.evaluator.evaluate()
            model.model.sem_seg_head.predictor.lang_encoder.reset_text_embeddings()

            if is_main_process():
                scores["{}/{}".format(dataset_label, eval_type)] = results

        # set back to training stat.
        model.model.sem_seg_head.num_classes = self._opt['MODEL']['ENCODER']['NUM_CLASSES']
        model.model.metadata = MetadataCatalog.get(self._opt['DATASETS']['TRAIN'][0])
        return scores