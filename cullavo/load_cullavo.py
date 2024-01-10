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

import inspect
import warnings
def prepare_model_for_kbit_training(model, use_gradient_checkpointing=True, gradient_checkpointing_kwargs=None):
    loaded_in_kbit = getattr(model, "is_loaded_in_8bit", False) or getattr(model, "is_loaded_in_4bit", False)
    is_gptq_quantized = getattr(model, "quantization_method", None) == "gptq"
    if gradient_checkpointing_kwargs is None:
        gradient_checkpointing_kwargs = {}

    for name, param in model.named_parameters():
        # freeze base model's layers
        param.requires_grad = False

    if (loaded_in_kbit or is_gptq_quantized) and use_gradient_checkpointing:
        # When having `use_reentrant=False` + gradient_checkpointing, there is no need for this hack
        if "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs["use_reentrant"]:
            # For backward compatibility
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
            else:

                def make_inputs_require_grad(module, input, output):
                    output.requires_grad_(True)

                model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

        # To support older transformers versions, check if the model supports gradient_checkpointing_kwargs
        _supports_gc_kwargs = "gradient_checkpointing_kwargs" in list(
            inspect.signature(model.gradient_checkpointing_enable).parameters
        )

        if not _supports_gc_kwargs and len(gradient_checkpointing_kwargs) > 0:
            warnings.warn(
                "gradient_checkpointing_kwargs is not supported in this version of transformers. The passed kwargs will be ignored."
                " if you want to use that feature, please upgrade to the latest version of transformers.",
                FutureWarning,
            )

        gc_enable_kwargs = (
            {} if not _supports_gc_kwargs else {"gradient_checkpointing_kwargs": gradient_checkpointing_kwargs}
        )

        # enable gradient checkpointing for memory efficiency
        model.gradient_checkpointing_enable(**gc_enable_kwargs)
    return model

def prepare_cullavo(bits=8):
    
    # LLaVA 8Bit compression
    # cullavo_model_original = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, torch_dtype=torch.bfloat16)
    cullavo_model = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, load_in_8bit=bits==8, load_in_4bit=bits==4, torch_dtype=torch.bfloat16)
    cullavo_model = prepare_model_for_kbit_training(cullavo_model, gradient_checkpointing_kwargs={"use_reentrant":True})
    cullavo_model.config.use_cache = False

    # if is_lora:
    #     from peft import LoraConfig, get_peft_model
    #     lora_config = LoraConfig(
    #         r=64,
    #         lora_alpha=16,
    #         target_modules=find_all_linear_names(cullavo_model),
    #         lora_dropout=0.05,
    #         bias='none',
    #         task_type="CAUSAL_LM",
    #     )
    #     cullavo_model = get_peft_model(cullavo_model, lora_config)
        
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
    cullavo_model.language_model.lm_head.requires_grad_(True)
    # cullavo_model.multi_modal_projector = cullavo_model_original.multi_modal_projector
    # cullavo_model.multi_modal_projector.requires_grad_(True)
    
    # Add tokens to tokenzier for CuLLaVO
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

