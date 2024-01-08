import torch
import pathlib
from transformers import BitsAndBytesConfig

from .utils.utils import LLAVA_LOCAL_PATH
from .arch_cullavo import CuLLaVOModel
from transformers import AutoProcessor

def prepare_cullavo(bits=16):
    
    # LLaVA 8Bit compression
    cullavo_model = CuLLaVOModel.from_pretrained(LLAVA_LOCAL_PATH, load_in_8bit=True, device_map=torch.cuda.current_device(), torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    cullavo_processor = AutoProcessor.from_pretrained(LLAVA_LOCAL_PATH, padding_side='right')
    
    # Add tokens to tokenzier for CuLLaVO
    # General Vision Task (Class, Instance, Box) and Grounding Vision Task (Sentence, Instance, Box)
    # PROMPT Example: <image>\n<GEN><CLASS>dog</CLASS><BOX>[345,657,765,876]</BOX><INST>[(<background>,10),(<object>,50),...]</INST></GEN>
    # PROMPT Example: <image>\n<SET>a white dog wearking red ribbon<\SET><GRD>dog</CLASS><BOX>[345,657,765,876]</BOX><INST>[(<background>,10),(<object>,50),...]</INST></GRD>
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

