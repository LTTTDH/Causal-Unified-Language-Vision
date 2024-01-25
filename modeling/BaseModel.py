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
        
        # Non-LLM model save
        if accel.is_main_process:
            model_state_dict = self.model.state_dict()
            filtered_model_keys = list(filter(lambda x: not x.startswith('cullavo'), model_state_dict))
            result_dicts = {}
            for model_key in filtered_model_keys:
                result_dicts[model_key] = model_state_dict[model_key]
            os.makedirs(os.path.join(save_dir, f'epoch{epoch}'), exist_ok=True)
            torch.save(result_dicts, os.path.join(save_dir, f'epoch{epoch}', f"CuLLaVO.pt"))
        
        # Sync
        if torch.distributed.get_world_size() > 1: torch.distributed.barrier()

        # LLM Save
        if self.opt['LLM']['LOAD_LLM']:
            llm_path = os.path.join(save_dir, f'epoch{epoch}', "cullavo")
            
            # Token parallel
            os.environ['TOKENIZERS_PARALLELISM']='false'

            # PEFT Language save
            accel.unwrap_model(self.model.cullavo_model.language_model).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save)
            accel.unwrap_model(self.model.cullavo_processor).save_pretrained(llm_path, is_main_process=accel.is_main_process, save_function=accel.save)
            
            # Multi modal projector save
            multi_modal_projector_state_dict = {}
            for name, param in accel.unwrap_model(self.model.cullavo_model.multi_modal_projector).named_parameters():
                multi_modal_projector_state_dict.update({name: param.detach().cpu().clone()})
            if accel.is_main_process: torch.save(multi_modal_projector_state_dict, os.path.join(llm_path, "multi_modal_projector.pt"))

            # lm head save
            lm_head_state_dict = {}
            for name, param in accel.unwrap_model(self.model.cullavo_model.language_model.lm_head).named_parameters():
                lm_head_state_dict.update({name: param.detach().cpu().clone()})
            if accel.is_main_process: torch.save(lm_head_state_dict, os.path.join(llm_path, "lm_head.pt"))

            # word embedding save
            embed_tokens_state_dict = {}
            for name, param in accel.unwrap_model(self.model.cullavo_model.get_input_embeddings()).named_parameters():
                embed_tokens_state_dict.update({name: param.detach().cpu().clone()})
            if accel.is_main_process: torch.save(embed_tokens_state_dict, os.path.join(llm_path, "embed_tokens.pt"))

            # Token parallel
            os.environ['TOKENIZERS_PARALLELISM']='true'

    def from_pretrained(self, load_dir, accel):
        if os.path.exists(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo')):
            
            def lora_generator(model):
                for name, param in model.named_parameters():
                    if 'lora' in name:
                        yield (name, param)

            from safetensors import safe_open
            with safe_open(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo', 'adapter_model.safetensors'), framework="pt", device="cpu") as handle_f:
                for key in handle_f.keys():
                    if 'lora' in key:
                        ckpt_tensor = handle_f.get_tensor(key)
                        check = 0
                        for name, param in lora_generator(self.model.cullavo_model.language_model):
                            if name==key:
                                param.data = ckpt_tensor
                                check = 1
                                break
                        if check == 0:
                            raise Exception("No!")

            # LORA -> bfloat16 conversion 
            for param in self.model.cullavo_model.parameters():
                if 'float32' in str(param.dtype).lower():
                    param.data = param.data.to(torch.bfloat16)

            # torch load for multi modal projector 
            multi_modal_projector_state_dict = torch.load(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo', "multi_modal_projector.pt"), map_location=accel.device)
            mm_proj_msg = self.model.cullavo_model.multi_modal_projector.load_state_dict(multi_modal_projector_state_dict)

            # torch load for lm head
            lm_head_state_dict = torch.load(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo', "lm_head.pt"), map_location=accel.device)
            lm_head_msg = self.model.cullavo_model.language_model.lm_head.load_state_dict(lm_head_state_dict)

            # torch load for embed tokens
            embed_tokens_state_dict = torch.load(os.path.join("/".join(load_dir.split('/')[:-1]), 'cullavo', "embed_tokens.pt"), map_location=accel.device)
            embed_tokens_msg = self.model.cullavo_model.get_input_embeddings().load_state_dict(embed_tokens_state_dict)

            # Freeze Parameter for Evaluating
            self.model.cullavo_model = self.model.cullavo_model.eval()
            for param in self.model.cullavo_model.parameters(): param.requires_grad_(False)

            # Verbose
            if accel.is_main_process: print(f'CuLLaVO Loaded!!: {load_dir}, {mm_proj_msg}, {lm_head_msg}, {embed_tokens_msg}')
        else:
            # Verbose
            if accel.is_main_process: print(f'There is no CuLLaVO pretrained: {load_dir}')

        state_dict = torch.load(load_dir, map_location=accel.device)
        state_dict = align_and_update_state_dicts(self.model.state_dict(), state_dict)
        self.model.load_state_dict(state_dict, strict=False)
        return self
    
    # for name, param in self.model.cullavo_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")