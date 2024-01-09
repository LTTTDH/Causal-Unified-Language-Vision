# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import time
import os
import sys
import importlib
import json
import random
import logging
import numpy as np
import copy
import contextlib
import shutil
from typing import Any, Callable, Union
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mpi4py import MPI
from infinibatch import iterators

from .distributed_trainer import DistributedTrainer
from .utils.misc import *
from .utils.serialization import JSONEncoder, filter_jsonable
from utils.distributed import get_world_size

logger = logging.getLogger(__name__)


class UtilsTrainer(DistributedTrainer):

    def __init__(self, opt):
        super().__init__(opt)

    def get_batch_size(self, batch):
        if hasattr(self.model, 'get_batch_size'):
            if callable(self.model.get_batch_size):
                return self.model.get_batch_size(batch)
        return {}

    def _initialize_accelerator(self):
        self.accel_config['train_micro_batch_size_per_gpu'] = self.opt['COCO']['TRAIN']['BATCH_SIZE_PER_GPU']
        # self.model = self.accel.prepare(self.model)
        # self.optimizer = self.accel.prepare(self.optimizer)
        # self.lr_scheduler = self.accel.prepare(self.lr_scheduler)
        # self.train_dataloaders = self.accel.prepare(self.train_dataloaders)
        
        self.model, self.optimizer, self.lr_scheduler, self.train_dataloaders = \
            self.accel.prepare(self.model, self.optimizer, self.lr_scheduler, self.train_dataloaders)

    def load_model(self, load_path):
        self.model = self.model.from_pretrained(load_path)
        self.model.to(self.accel.device)

    def save_checkpoint(self, epoch):
        
        save_dir = self.save_folder

        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()

        if self.accel.is_main_process:
            os.makedirs(self.save_folder, exist_ok=True)

        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()

        if self.accel.is_main_process:
            os.makedirs(save_dir, exist_ok=True)

        if self.accel.is_main_process:
            self.model.save_pretrained(save_dir, epoch)
            print(f'Saved!: {save_dir}')

    def load_weight(self, checkpoint_path=None, must_exist=False):
        self.load_model(checkpoint_path)
        # logger.warning(f'Load weights from {checkpoint_path}...')

    def load_checkpoint(self, checkpoint_path=None, must_exist=False):
        # logger.warning(f'Resuming checkpoint from {checkpoint_path}...')

        model_load_path = os.path.join(checkpoint_path, 'module_training_states.pt')
        state = torch.load(model_load_path, map_location=self.accel.device)
        
        # logger.warning(f'HACK to strip module from model state dict on single gpu debugging!')
        ckpt = state['module']
        if get_world_size() <= 1:
            ckpt = {key.replace('module.',''):ckpt[key] for key in ckpt.keys()}
            
        self.model.load_state_dict(ckpt)
        self.optimizer.load_state_dict(state['optimizer'])
        self.lr_scheduler.load_state_dict(state['lr_scheduler'])

        load_path = os.path.join(checkpoint_path, 'trainer_states.pt')
        trainer_state = torch.load(load_path, map_location='cpu')
        self.train_loss = trainer_state['train_loss']
        self.train_params = trainer_state['train_params']