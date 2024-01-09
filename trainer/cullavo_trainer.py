# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
import logging

from .default_trainer import DefaultTrainer

logger = logging.getLogger(__name__)


class CuLLaVO_Trainer(DefaultTrainer):

    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters())
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, len(self.train_dataloaders))