from utils.arguments import load_opt_command

def main(args=None):
    opt, _ = load_opt_command(args)

    from trainer import XDecoder_Trainer as Trainer
    trainer = Trainer(opt)
    # trainer.train()
    # trainer.eval() # X-Decoder
    trainer.eval_for_vl_model() # LLaVA/LLaVA1.5/BLIPv2/Instruct-BLIP

if __name__ == "__main__":
    main()