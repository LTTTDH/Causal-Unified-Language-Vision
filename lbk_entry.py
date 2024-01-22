import warnings
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

from utils.arguments import load_opt_command

def main(args=None):
    opt, cmdline_args = load_opt_command(args)

    from trainer import CuLLaVO_Trainer as Trainer
    trainer = Trainer(opt)
    
    if 'cullavo_step' in trainer.opt['NAME']:
        if cmdline_args.command == 'train':
            trainer.train() # CuLLaVO
        elif cmdline_args.command == 'eval':
            trainer.eval() # CuLLaVO
    elif trainer.opt['NAME'] == 'xdecoder_test.yaml' and cmdline_args.command == 'eval':
        trainer.eval() # X-Decoder/SEEM
    elif trainer.opt['NAME'] == 'vl_test.yaml' and cmdline_args.command == 'eval':
        trainer.eval_for_vl_model() # LLaVA1.5/BLIPv2/Instruct-BLIP

if __name__ == "__main__":
    main()