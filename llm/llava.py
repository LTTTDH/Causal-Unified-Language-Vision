#    Copyright 2023 Haotian Liu
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from .llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def propmt_engineering(self, prompt, tokenizer):
        tok = tokenizer(prompt, return_tensors="pt")
        tok_embed = self.model.embed_tokens(tok['input_ids'][:, 1:])
        tok_attn_mask = tok['attention_mask'][:, 1:]
        return tok_embed.squeeze(0), tok_attn_mask.squeeze(0)

    def forward(
        self,
        img_description=None,
        ref_description=None,
        img_features=None,
        gen_proposals=None,
        ref_proposals=None,
        tokenizer=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        image_prompt_pre = "You should answer concretely to help humans acquire significant performance to everything. This is an image information representing visual understanding: "
        image_prompt_pre_embed, image_prompt_pre_attn_mask = self.propmt_engineering(image_prompt_pre, tokenizer)
    
        image_prompt_post = f" where its short description is "
        image_prompt_post_embed, image_prompt_post_attn_mask = self.propmt_engineering(image_prompt_post, tokenizer)

        seg_prompt = ", and this is mask proposals representing many key-focus for general segmentation: panoptic/instance/semantic segmentation in the image: "
        seg_prompt_embed, seg_prompt_attn_mask = self.propmt_engineering(seg_prompt, tokenizer)

        ref_propmt_pre = f" One more, if a user wants to segment "
        ref_propmt_pre_embed, ref_prompt_pre_attn_mask = self.propmt_engineering(ref_propmt_pre, tokenizer)

        ref_prompt_post = ", and this will be referring mask proposals representing specific-focus area that a user wants to segment: "
        ref_propmt_post_embed, ref_propmt_post_attn_mask = self.propmt_engineering(ref_prompt_post, tokenizer)

        full_text = []
        full_label = []
        full_attn = []
        for img_des, ref_des, img_feat, gen_prop, ref_prop in zip(img_description, ref_description, img_features, gen_proposals, ref_proposals):
            text = torch.cat([image_prompt_pre_embed, img_feat], dim=0)
            label = torch.cat([image_prompt_pre_attn_mask * (-100), torch.ones(img_feat.shape[0])* (-100)], dim=0)
            attn = torch.cat([image_prompt_pre_attn_mask, torch.ones(img_feat.shape[0])], dim=0)            
            
            tok = tokenizer(img_des, return_tensors="pt")
            text = torch.cat([text, image_prompt_post_embed, self.model.embed_tokens(tok['input_ids'][:, 1:]).squeeze(0)], dim=0)
            label = torch.cat([label, image_prompt_post_attn_mask * (-100), tok['attention_mask'][:, 1:].squeeze(0) * (-100)], dim=0)
            attn = torch.cat([attn, image_prompt_post_attn_mask, tok['attention_mask'][:, 1:].squeeze(0)], dim=0)
            
            text = torch.cat([text, seg_prompt_embed, gen_prop], dim=0)
            label = torch.cat([label, seg_prompt_attn_mask * (-100), torch.ones(gen_prop.shape[0])* (-100)], dim=0)
            attn = torch.cat([attn, seg_prompt_attn_mask, torch.ones(gen_prop.shape[0])], dim=0)

            tok = tokenizer(ref_des, return_tensors="pt")
            text = torch.cat([text, ref_propmt_pre_embed, self.model.embed_tokens(tok['input_ids'][:, 1:]).squeeze(0)], dim=0)
            label = torch.cat([label, ref_prompt_pre_attn_mask * (-100), tok['attention_mask'][:, 1:].squeeze(0) * (-100)], dim=0)
            attn = torch.cat([attn, ref_prompt_pre_attn_mask, tok['attention_mask'][:, 1:].squeeze(0)], dim=0)

            text = torch.cat([text, ref_propmt_post_embed, ref_prop], dim=0)
            label = torch.cat([label, ref_propmt_post_attn_mask * (-100), torch.ones(ref_prop.shape[0])* (-100)], dim=0)
            attn = torch.cat([attn, ref_propmt_post_attn_mask, torch.ones(ref_prop.shape[0])], dim=0)

            full_text.append(text)
            full_label.append(label)
            full_attn.append(attn)

        return super().forward(
            attention_mask=torch.nn.utils.rnn.pad_sequence(full_attn, padding_value=0),
            inputs_embeds=torch.nn.utils.rnn.pad_sequence(full_text, padding_value=0),
            labels=torch.nn.utils.rnn.pad_sequence(full_label, padding_value=-100),
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
