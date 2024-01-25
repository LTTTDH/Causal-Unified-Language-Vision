import torch

from .utils.utils import LLAVA_LOCAL_PATH
from .arch_cullavo import CuLLaVOModel
from transformers import AutoProcessor
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

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

def set_all_adapter(cullavo_model):
    cullavo_model.language_model.set_adapter(['step1', 'step2'])

def add_adapter_step2(cullavo_model):
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=['k_proj', 'q_proj', 'up_proj', 'v_proj', 'o_proj', 'down_proj', 'gate_proj'],
        lora_dropout=0.05,
        bias='none',
        task_type="CAUSAL_LM",
    )
    cullavo_model.language_model.add_adapter(lora_config, adapter_name='step2')
    cullavo_model.language_model.set_adapter("step2")


def prepare_cullavo(bits, grad_ckpt, lora):

    bnb_model_from_pretrained_args = {}
    if bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            load_in_4bit=bits == 4,
            load_in_8bit=bits == 8,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=bits == 4,
                load_in_8bit=bits == 8,
                llm_int8_skip_modules=["vision_tower", "multi_modal_projector", "lm_head"], # LM_HEAD: flash attention
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4'
            )
        ))

    # LLaVA 8Bit compression
    cullavo_model = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, **bnb_model_from_pretrained_args)

    if bits in [4, 8] and lora:
        cullavo_model.language_model.config.use_cache = False
        cullavo_model = prepare_model_for_kbit_training(cullavo_model,
                                                        use_gradient_checkpointing=grad_ckpt,
                                                        gradient_checkpointing_kwargs={"use_reentrant": False})
        lora_config = LoraConfig(
            r=64,
            lora_alpha=16,
            target_modules=find_all_linear_names(cullavo_model.language_model),
            lora_dropout=0.05,
            bias='none',
            task_type="CAUSAL_LM",
        )
        cullavo_model.language_model.add_adapter(lora_config, adapter_name='step1')

    elif bits in [4, 8] and not lora:
        raise Exception("training model with non-lora bits quantization is not worked")
    elif not bits in [4, 8] and lora:
        raise Exception("CuLLaVO does not have any plan in lora without bit quantization")
    elif not bits in [4, 8] and not lora:
        raise Exception("CuLLaVO does not have any plan in full training without lora and bit quantization")
    else:
        raise Exception("WTF")

    # bfloat16 conversion 
    for param in cullavo_model.parameters():
        if 'float32' in str(param.dtype).lower():
            param.data = param.data.to(torch.bfloat16)
            
    # Training MM projector
    for param in cullavo_model.multi_modal_projector.parameters():
        param.requires_grad_(True)

    # Training lm head
    for param in cullavo_model.language_model.lm_head.parameters():
        param.requires_grad_(True)

    # Training embed_tokens
    for param in cullavo_model.get_input_embeddings().parameters():
        param.requires_grad_(True)
    
    # Add tokens to tokenzier for CuLLaVO
    cullavo_processor = AutoProcessor.from_pretrained(LLAVA_LOCAL_PATH, padding_side='right')

    return cullavo_model, cullavo_processor

# for name, param in cullavo_model.named_parameters(): print(f"{name}: {param.dtype} {param.requires_grad}")