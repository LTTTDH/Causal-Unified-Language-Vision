# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Modified by Xueyan Zou (xueyan@cs.wisc.edu)
# --------------------------------------------------------

import os
import sys
import torch
import logging
import wandb

from utils.arguments import load_opt_command

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init_wandb(args, job_dir, entity='weip1004', project='cullavo', job_name='tmp'):
    wandb_dir = os.path.join(job_dir, 'wandb')
    os.makedirs(wandb_dir, exist_ok=True)
    runid = None
    if os.path.exists(f"{wandb_dir}/runid.txt"):
        runid = open(f"{wandb_dir}/runid.txt").read()

    wandb.init(project=project,
            name=job_name,
            dir=wandb_dir,
            entity=entity,
            resume="allow",
            id=runid,
            config={"hierarchical": True},)

    open(f"{wandb_dir}/runid.txt", 'w').write(wandb.run.id)
    wandb.config.update({k: args[k] for k in args if k not in wandb.config})

def main(args=None):
    opt, cmdline_args = load_opt_command(args)
    command = cmdline_args.command
    from trainer import CuLLaVO_Trainer as Trainer
    trainer = Trainer(opt)
    
    if command == "train":
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.login(key='7e4ecf71a336d0afd0a00ca526195a1ae2f750ed')
            init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        trainer.train()
    elif command == "eval":
        if opt['rank'] == 0 and opt['WANDB']:
            wandb.login(key='7e4ecf71a336d0afd0a00ca526195a1ae2f750ed')
            init_wandb(opt, trainer.save_folder, job_name=trainer.save_folder)
        trainer.eval()
    else:
        raise ValueError(f"Unknown command: {command}")

if __name__ == "__main__":
    main()
    sys.exit(0)
