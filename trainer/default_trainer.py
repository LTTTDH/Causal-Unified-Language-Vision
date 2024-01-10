# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

from datetime import datetime
import os
import sys
import importlib
import wandb
import logging
import torch

from .distributed_trainer import DistributedTrainer
from .utils_trainer import UtilsTrainer
from .utils.misc import *
from tqdm import tqdm

logger = logging.getLogger(__name__)


class DefaultTrainer(UtilsTrainer, DistributedTrainer):

    def __init__(self, opt):
        """
        Set up the task the model is being trained for.
        """
        super().__init__(opt)
        base_name = 'base_dir'
        base_path =  os.path.join(self.opt['base_path'], '__init__.py')
        spec = importlib.util.spec_from_file_location(base_name, base_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[base_name] = module
        spec.loader.exec_module(module)
        # logger.info(f"Imported {base_name} at base_path {self.opt['base_path']}")

        pipeline_module = importlib.import_module(f"base_dir.pipeline.{self.opt['PIPELINE']}")
        pipeline_class = getattr(pipeline_module, self.opt['PIPELINE'])
        # logger.info(f"Pipeline for training: {self.opt['PIPELINE']}")
        self.pipeline = pipeline_class(self.opt)


    def eval_for_vl_model(self, ):
        self.mode = "eval"
        results = self._eval_on_set()
        return results


    def eval(self):
        logger.info('-----------------------------------------------')
        logger.info("Evaluating model ... ")
        self.mode = "eval"

        self.model = self.pipeline.initialize_model()

        # move model to the device
        self.model.to(self.accel.device)

        # load model during evaluation
        if self.opt['WEIGHT'] and os.path.isfile(self.opt['RESUME_FROM']):
            model_path = self.opt['RESUME_FROM'] 
            self.load_model(model_path)
        else:
            raise ValueError(f"Model not found: {model_path}")

        results = self._eval_on_set()
        if self.accel.is_main_process: self.dictionary_display(results)
        if self.accel.is_main_process and self.opt['WANDB']: wandb.log(results)
        return results
    
    
    def _eval_on_set(self):      
        results = self.pipeline.evaluate_model(self)
        return results

    def compute_loss(self, forward_func, batch):

        def forward(func, trainer, batch):
            loss = func(trainer, batch)
            return loss

        loss = forward(forward_func, self, batch)
        return loss

    def backward_loss(self, loss):  # noqa: E252
        self.accel.backward(loss)

    def update_model(self):
        self.optimizer.step()
        self.optimizer.zero_grad()
        self.train_params['optim_steps'] += 1
        self.lr_scheduler.step()

    def train_step(self, batch):

        # set all modules and criteria into training mode
        self.model.train()

        total_batch_sample = 0

        loss_info, sample_size_info, extra_info = self.pipeline.forward_step(self, batch)

        self.train_loss.update_iter(loss_info)
        total_batch_sample += sample_size_info['num_samples']

        # update losses and item counts of an effective batch to the AverageMeters
        total_batch_sample = torch.tensor(total_batch_sample).to(self.accel.device)
        torch.distributed.all_reduce(total_batch_sample, torch.distributed.ReduceOp.SUM)
        total_batch_sample = total_batch_sample.item()

        self.train_params['total_batch_size'] += total_batch_sample

        self.train_params['num_updates'] += 1
        
    def init_train(self):
        self.mode = "train"
        self.model = self.pipeline.initialize_model()

        # move model to the device
        self.model.to(self.accel.device)

        self.train_dataloaders = self.pipeline.get_dataloaders(self, 'train', is_evaluation=False)
        self.train_loss = LossMeter()

        if self.opt['CUDA']:
            torch.cuda.empty_cache()

        self.create_optimizer_and_scheduler()
        self._initialize_accelerator() # accelerator
        
        self.train_params = {
                             "updates_per_epoch": len(self.train_dataloaders),
                             "total_batch_size": 0,
                             "num_updates": 0,
                             "optim_steps": 0,
                             "start_epoch_idx": 0,
                             "start_batch_idx": 0,
                             "current_epoch_idx": 0,
                             "current_batch_idx": 0,
                             "resume_epoch_idx": 0, 
                             }

        if self.opt.get('WEIGHT', False):
            self.load_weight(self.opt['RESUME_FROM'], must_exist=True)
        if self.opt.get('RESUME', False):
            self.load_checkpoint(self.opt['RESUME_FROM'], must_exist=True)

    @staticmethod
    def dictionary_display(results):
        print('\n-------------------')
        for key, value in results.items():
            print(f'DATASET/Task: [{key}]\n')
            for _key, _value in value.items():
                print(f'{_key}:')
                try:
                    for __key, __value in _value.items():
                        print(f'    {__key}: {__value}')
                except:
                    print(_value)
            print('-------------------')
        print('\n')

    def train(self):
        """
        Training
        """
        self.init_train()
        num_epochs = self.opt['SOLVER']['MAX_NUM_EPOCHS']

        if self.opt.get('EVAL_AT_START', False):
            results = self._eval_on_set()
            if self.accel.is_main_process and self.opt['WANDB']:
                wandb.log(results)

        train_prev_logged_time = datetime.now()
        for epoch in range(self.train_params['start_epoch_idx'], num_epochs):
            self.train_params['current_epoch_idx'] = epoch
            logger.info(f"Start epoch: {epoch} training.")
            
            eval_period = self.train_params['updates_per_epoch'] // 4
            prog_bar = tqdm(enumerate(self.train_dataloaders), total=self.train_params['updates_per_epoch'], leave=True, disable=not self.accel.is_local_main_process)
            for batch_idx, batch in prog_bar:
                self.train_params['current_batch_idx'] = batch_idx
                with self.accel.accumulate(self.model): self.train_step(batch)
                last_lr = self.lr_scheduler.get_last_lr()[0]
                loss_list = [obj.val for _, obj in self.train_loss.losses.items()]
                total_loss = sum(loss_list) / len(loss_list)
                desc = f"|Epochs[{epoch+1}]|[{batch_idx+1}/{self.train_params['updates_per_epoch']}]|"
                desc += f"LR[{', '.join([f'{last_lr:.2e}'])}]|"
                desc += f"Loss[{total_loss:.2f}]|"
                prog_bar.set_description(desc, refresh=True)
                
                # Empty cache and memory efficient allocation
                torch.cuda.empty_cache()
                
                if self.accel.is_main_process and self.opt['WANDB']:
                    # log for wandb
                    wb_loss_info = {key: obj.val for key, obj in self.train_loss.losses.items()}
                    wandb.log(wb_loss_info) #, step=self.train_params['updates_per_epoch'] * epoch + batch_idx
                    wandb.log({'Total-Loss': total_loss}) # LBK-Total-Loss log
                    wandb.log({'Learning-Rate': self.lr_scheduler.get_last_lr()[0]}) # LBK-LR log
                    wandb.log({'Epoch': epoch+1}) # LBK-LR log
                    
                if batch_idx in [eval_period, eval_period*2, eval_period*3]:
                    self.save_checkpoint(epoch+1)
                    # results = self._eval_on_set()
                    # if self.accel.is_main_process: self.dictionary_display(results)
                    # if self.accel.is_main_process and self.opt['WANDB']: wandb.log(results)
                
            # evaluate and save ckpt every epoch
            if self.accel.is_main_process: print('\n-----------Saving CKPT...-----------\n')
            self.save_checkpoint(epoch+1)
            # results = self._eval_on_set()
            # if self.accel.is_main_process: self.dictionary_display(results)
            # if self.accel.is_main_process and self.opt['WANDB']: wandb.log(results)