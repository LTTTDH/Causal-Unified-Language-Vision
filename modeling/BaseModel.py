import os
import logging

import torch
import torch.nn as nn

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
        if self.opt['LLM']['LOAD_LLM']:
            try:
                self.model.cullavo_model.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'))
                self.model.cullavo_processor.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo'), padding_side="right")
            except:
                print('There are no CuLLaVO pretrained file: {}'.format(os.path.join("/".join(load_dir.split('/')[:-1]))))

        state_dict = torch.load(load_dir, map_location=accel.device)
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self