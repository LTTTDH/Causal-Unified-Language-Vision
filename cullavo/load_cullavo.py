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

def prepare_cullavo(bits):

    bnb_model_from_pretrained_args = {}
    if bits in [4, 8]:
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

    # LLaVA 8Bit compression
    # cullavo_model_original = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, torch_dtype=torch.bfloat16)
    cullavo_model = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, **bnb_model_from_pretrained_args)
    if bits in [4, 8]:
        cullavo_model = prepare_model_for_kbit_training(cullavo_model, gradient_checkpointing_kwargs={"use_reentrant":True})
    cullavo_model.config.use_cache = False

    if bits in [4, 8]:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=find_all_linear_names(cullavo_model.language_model),
            lora_dropout=0.05,
            bias='none',
            task_type="CAUSAL_LM",
        )
        cullavo_model.language_model = get_peft_model(cullavo_model.language_model, lora_config)

    # Bfloat16  
    if bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for param in cullavo_model.parameters():
            if 'float32' in str(param.dtype).lower():
                param.data = param.data.to(torch.bfloat16)

    # Training Linear Connection
    for param in cullavo_model.multi_modal_projector.parameters():
        param.requires_grad_(True)
    
    # Add tokens to tokenzier for CuLLaVO
    cullavo_processor = AutoProcessor.from_pretrained(LLAVA_LOCAL_PATH, padding_side='right')
    cullavo_processor.tokenizer.add_tokens('<background>', special_tokens=True)
    cullavo_processor.tokenizer.add_tokens('<object>', special_tokens=True)
    
    # Resize Word Embedding
    cullavo_model.resize_token_embeddings(len(cullavo_processor.tokenizer))
    
    return cullavo_model, cullavo_processor

# for name, param in cullavo_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")