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
        model_state_dict = self.model.state_dict()
        filtered_model_keys = list(filter(lambda x: not x.startswith('cullavo'), model_state_dict))
        result_dicts = {}
        for model_key in filtered_model_keys:
            result_dicts[model_key] = model_state_dict[model_key]
        os.makedirs(os.path.join(save_dir, f'epoch{epoch}'), exist_ok=True)
        torch.save(result_dicts, os.path.join(save_dir, f'epoch{epoch}', f"CuLLaVO.pt"))
        if self.opt['LLM']['LOAD_LLM']:
            llm_path = os.path.join(save_dir, f'epoch{epoch}', "cullavo")
            accel.unwrap_model(self.model.cullavo_model).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save)
            accel.unwrap_model(self.model.cullavo_processor).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save)

    def from_pretrained(self, load_dir, accel):
        if 'cullavo' in self.opt['RESUME_FROM']:
            del self.model.cullavo_model
            del self.model.cullavo_processor
            torch.cuda.empty_cache()
            self.model.cullavo_model = CuLLaVOModel.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'), load_in_8bit=True, torch_dtype=torch.bfloat16)
            self.model.cullavo_processor = AutoProcessor.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'), padding_side="right")

            from peft.tuners.lora import LoraLayer
            for name, module in self.model.cullavo_model.named_modules():
                if isinstance(module, LoraLayer):
                    module = module.to(torch.bfloat16)
                if 'norm' in name:
                    module = module.to(torch.float32)
                if 'lm_head' in name or 'embed_tokens' in name:
                    if hasattr(module, 'weight'):
                        module = module.to(torch.bfloat16)
        else:
            print('There are no CuLLaVO pretrained file: {}'.format(os.path.join("/".join(load_dir.split('/')[:-1]))))

        state_dict = torch.load(load_dir, map_location=accel.device)
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self