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

    def save_pretrained(self, save_dir):
        torch.save(self.model.state_dict(), os.path.join(save_dir, "model_state_dict.pt"))

    def from_pretrained(self, load_dir):
        if self.opt['LLM']['LOAD_LLM']:
            try:
                self.model.llm.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'llm'), cache_dir=False, low_cpu_mem_usage=True)
                self.model.llm_tokenizer.from_pretrained(os.path.join("/".join(load_dir.split('/')[:-1]), 'llm'),
                                                        cache_dir=False,
                                                        model_max_length=1024,
                                                        padding_side="right",
                                                        use_fast=False)
            except:
                print('There are no LLM pretrained file: {}'.format(os.path.join("/".join(load_dir.split('/')[:-1]))))

        state_dict = torch.load(load_dir, map_location=self.opt['device'])
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self