# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import logging
import torch

from .utils.hook import add_hook

# Accelerator
from accelerate import Accelerator

logger = logging.getLogger(__name__)


class DistributedTrainer:
    def __init__(self, opt):
        self.opt = opt
        self.accel = Accelerator(gradient_accumulation_steps=self.opt['LLM']['GRAD_CUM']) # Accelerator
        
        # parse environment information for distributed training
        self.opt['world_size'] = torch.distributed.get_world_size()

        self.set_opt_hook()

        # save config file
        self.save_folder = self.opt['SAVE_DIR']

        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()

        if self.accel.is_main_process:
            os.makedirs(self.save_folder, exist_ok=True)

            # logger.info(f"Save config file to {os.path.join(self.save_folder, 'conf_copy.yaml')}")
            # save_opt_to_yaml(self.opt, os.path.join(self.save_folder, 'conf_copy.yaml'))

        if torch.distributed.get_world_size() > 1:
            add_hook()

        # prepare metadata for save folder
        conf_file = self.opt['conf_files'][0]
        if 'BASENAME' not in self.opt:
            self.opt['BASENAME'] = os.path.basename(conf_file)
        
        self.init_save_folder()

    def set_opt_hook(self):
        # Fill in the default values for required keywords
        self.opt['CUDA'] = self.opt.get('CUDA', True) and torch.cuda.is_available()
        self.opt['EVAL_PER_UPDATE_NUM'] = int(self.opt.get('EVAL_PER_UPDATE_NUM', 0))
        self.opt['LR_SCHEDULER_PARAMS'] = self.opt.get('LR_SCHEDULER_PARAMS', {})

        if 'SAVE_DIR' not in self.opt:
            assert False, "Please initialize SAVE_DIR in your config file."
        self.opt['SAVE_DIR'] = os.path.normpath(self.opt['SAVE_DIR'])
        # logger.info(f"Setting SAVE_DIR as {self.opt['SAVE_DIR']}")

    def init_save_folder(self):
        """
        Initialize the save folder for logs, model, checkpoint, and evaluation.
        """

        if torch.distributed.get_world_size() > 1:
            torch.distributed.barrier()

        if self.accel.is_main_process:
            os.makedirs(self.opt['SAVE_DIR'], exist_ok=True)

        self.save_folder = self.opt['SAVE_DIR']