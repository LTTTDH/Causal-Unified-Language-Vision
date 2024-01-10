import torch

from .utils.utils import LLAVA_LOCAL_PATH
from .arch_cullavo import CuLLaVOModel
from transformers import AutoProcessor
from peft import prepare_model_for_kbit_training

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def prepare_cullavo(bits=8, is_lora=True):
    
    # LLaVA 8Bit compression
    cullavo_model = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, load_in_8bit=bits==True, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    cullavo_model = prepare_model_for_kbit_training(cullavo_model)
    cullavo_model.enable_input_require_grads()

    if is_lora:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=find_all_linear_names(cullavo_model),
            lora_dropout=0.05,
            bias='none',
            task_type="CAUSAL_LM",
        )
        if bits == 16: cullavo_model.to(torch.bfloat16)
        cullavo_model = get_peft_model(cullavo_model, lora_config)
    
    if bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in cullavo_model.named_modules():
            if isinstance(module, LoraLayer):
                module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    module = module.to(torch.bfloat16)
    
    
    
    # Add tokens to tokenzier for CuLLaVO
    # General Vision Task (Class, Instance, Box) and Grounding Vision Task (Sentence, Instance, Box)
    # PROMPT Example: <image>\n<GEN><CLASS>dog</CLASS><BOX>[345,657,765,876]</BOX><INST>[(<background>,10),(<object>,50),...]</INST></GEN>
    # PROMPT Example: <image>\n<SET>a white dog wearking red ribbon<\SET><GRD>dog</CLASS><BOX>[345,657,765,876]</BOX><INST>[(<background>,10),(<object>,50),...]</INST></GRD>
    cullavo_processor = AutoProcessor.from_pretrained(LLAVA_LOCAL_PATH, padding_side='right')
    cullavo_processor.tokenizer.add_tokens('<GEN>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</GEN>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<GRD>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</GRD>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<SET>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</SET>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<CLASS>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</CLASS>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<BOX>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</BOX>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<INST>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('</INST>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<background>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<object>', special_tokens=True)
    
    # Resize Word Embedding
    cullavo_model.resize_token_embeddings(len(cullavo_processor.tokenizer))
    
    return cullavo_model, cullavo_processor

