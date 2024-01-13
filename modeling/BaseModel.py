import os
import logging

import torch
import torch.nn as nn
from transformers import AutoProcessor
from cullavo.arch_cullavo import CuLLaVOModel
from utils.model import align_and_update_state_dicts
logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, opt, module: nn.Module):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.model = module

    def forward(self, *inputs, **kwargs):
        outputs = self.model(*inputs, **kwargs)
        return outputs

    def save_pretrained(self, save_dir, epoch, accel):
        if accel.is_main_process:
            model_state_dict = self.model.state_dict()
            filtered_model_keys = list(filter(lambda x: not x.startswith('cullavo'), model_state_dict))
            result_dicts = {}
            for model_key in filtered_model_keys:
                result_dicts[model_key] = model_state_dict[model_key]
            os.makedirs(os.path.join(save_dir, f'epoch{epoch}'), exist_ok=True)
            torch.save(result_dicts, os.path.join(save_dir, f'epoch{epoch}', f"CuLLaVO.pt"))
        if torch.distributed.get_world_size() > 1: torch.distributed.barrier()
        if self.opt['LLM']['LOAD_LLM']:
            llm_path = os.path.join(save_dir, f'epoch{epoch}', "cullavo")
            os.environ['TOKENIZERS_PARALLELISM']="false"
            accel.unwrap_model(self.model.cullavo_model).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save, safe_serialization=False)
            accel.unwrap_model(self.model.cullavo_processor).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save, safe_serialization=False)
            os.environ['TOKENIZERS_PARALLELISM']="true"

    def from_pretrained(self, load_dir, accel):
        if 'cullavo' in self.opt['RESUME_FROM']:
            del self.model.cullavo_model
            del self.model.cullavo_processor
            torch.cuda.empty_cache()

            # TODO: extract bits in peft config
            bits = 4

            bnb_model_from_pretrained_args = {}
            from transformers import BitsAndBytesConfig
            bnb_model_from_pretrained_args.update(dict(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                attn_implementation="flash_attention_2",
                quantization_config=BitsAndBytesConfig(
                    load_in_4bit=bits == 4,
                    load_in_8bit=bits == 8,
                    llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"],
                    llm_int8_threshold=6.0,
                    llm_int8_has_fp16_weight=False,
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type='nf4'
                )
            ))

            self.model.cullavo_model = CuLLaVOModel.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'), **bnb_model_from_pretrained_args)
            self.model.cullavo_processor = AutoProcessor.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'), padding_side="right")

            for param in self.model.cullavo_model.parameters():
                if 'float32' in str(param.dtype).lower():
                    param.data = param.data.to(torch.bfloat16)

        else:
            print('There are no CuLLaVO pretrained file: {}'.format(os.path.join("/".join(load_dir.split('/')[:-1]))))

        state_dict = torch.load(load_dir, map_location=accel.device)
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self