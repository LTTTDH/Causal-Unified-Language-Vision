# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import torch
from .default_trainer import DefaultTrainer

class CuLLaVO_Trainer(DefaultTrainer):
    def create_optimizer_and_scheduler(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=float(self.opt['LLM']['LR']), weight_decay=self.opt['LLM']['WEIGHT_DECAY'])
        self.lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=len(self.train_dataloaders), eta_min=float(self.opt['LLM']['LAST_LR']))
