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
import torch.nn.functional as F


class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    IGNORE_INDEX = -100

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        # Initialize weights and apply final processing
        self.post_init()
        
    def get_model(self):
        return self.model

    def propmt_engineering(self, prompt, tokenizer):
        tok = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        tok_embed = self.model.embed_tokens(tok['input_ids'])
        tok_id = tok['input_ids']
        tok_attn_mask = tok['attention_mask']
        return tok_id.squeeze(0), tok_embed.squeeze(0), tok_attn_mask.squeeze(0)

    def make_preliminary_prompt(self, tokenizer):
        # CuLLaVO Prompt
        img_pt = " represents an image information for visual understanding. "
        _, self.img_pt_embed, self.img_pt_attn_mask = self.propmt_engineering(img_pt, tokenizer)

        gen_pt = " represents mask proposals for many general-focus parts in this image. "
        _, self.gen_pt_embed, self.gen_pt_attn_mask = self.propmt_engineering(gen_pt, tokenizer)

        ref_pt = "If a user wants to segment the followings: "
        _, self.ref_pt_embed, self.ref_pt_attn_mask = self.propmt_engineering(ref_pt, tokenizer)
        
        ref_pt2 = ", then "
        _, self.ref_pt2_embed, self.ref_pt2_attn_mask = self.propmt_engineering(ref_pt2, tokenizer)

        ref_pt3 = " represents mask proposals for specific-focus parts that a user wants to segment."
        _, self.ref_pt3_embed, self.ref_pt3_attn_mask = self.propmt_engineering(ref_pt3, tokenizer)

        des_pt = "The short image description of this image is <"
        self.des_pt_id, self.des_pt_embed, self.des_pt_attn_mask = self.propmt_engineering(des_pt, tokenizer)

        des_pt2 = ">."
        self.des_pt2_id, self.des_pt2_embed, self.des_pt2_attn_mask = self.propmt_engineering(des_pt2, tokenizer)

    def cullavo_multimodal_inputs(self,
                             img_features,
                             img_description, 
                             ref_description,
                             gen_proposals, 
                             ref_proposals, 
                             tokenizer):

        # CuLLaVO Outputs
        full_text = []
        full_label = []
        full_attn = []
        start_gen_prop_idx_list = []
        start_ref_prop_idx_list = []

        for img_feat, img_des, ref_des, gen_prop, ref_prop in zip(img_features, img_description, ref_description, gen_proposals, ref_proposals):
            # BOS token
            tok = tokenizer('<s>', return_tensors='pt', add_special_tokens=False)
            text = self.model.embed_tokens(tok['input_ids']).squeeze(0)
            attn = tok['attention_mask'].squeeze(0)
            label = attn * self.IGNORE_INDEX
            
            # "<Image> represents an image information for visual understanding. "
            text = torch.cat([text, img_feat, self.img_pt_embed], dim=0)
            attn = torch.cat([attn, torch.ones(img_feat.shape[0]), self.img_pt_attn_mask], dim=0)           
            label = attn * self.IGNORE_INDEX
            assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")
            
            if gen_prop is not None:
                # "<GEN_SEG> represents mask proposals for many general-focus parts in this image. "
                start_gen_prop_idx_list.append(len(text))
                text = torch.cat([text, gen_prop, self.gen_pt_embed], dim=0)
                attn = torch.cat([attn, torch.ones(gen_prop.shape[0]), self.gen_pt_attn_mask], dim=0)
                label = attn * self.IGNORE_INDEX
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")
            
            if ref_prop is not None:
                # "If a user wants to segment the followings: <REF_CAPTION>"
                tok = tokenizer(ref_des, return_tensors="pt", add_special_tokens=False)
                text = torch.cat([text, self.ref_pt_embed, self.model.embed_tokens(tok['input_ids']).squeeze(0)], dim=0)
                attn = torch.cat([attn, self.ref_pt_attn_mask, tok['attention_mask'].squeeze(0)], dim=0)
                label = attn * self.IGNORE_INDEX
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")

                # ", then "
                text = torch.cat([text, self.ref_pt2_embed], dim=0)
                attn = torch.cat([attn, self.ref_pt2_attn_mask], dim=0)
                label = attn * self.IGNORE_INDEX
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")

                # "<REF_SEG> represents mask proposals for specific-focus parts that a user wants to segment."
                start_ref_prop_idx_list.append(len(text))
                text = torch.cat([text, ref_prop, self.ref_pt3_embed], dim=0)
                attn = torch.cat([attn, torch.ones(ref_prop.shape[0]), self.ref_pt3_attn_mask], dim=0)
                label = attn * self.IGNORE_INDEX
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")
            
            if img_des is not None:
                # "The short image description of this image is <<IMG_DES>"
                tok = tokenizer(img_des, return_tensors="pt", add_special_tokens=False)
                text = torch.cat([text, self.des_pt_embed, self.model.embed_tokens(tok['input_ids']).squeeze(0)], dim=0)
                attn = torch.cat([attn, self.des_pt_attn_mask, tok['attention_mask'].squeeze(0)], dim=0)
                label = torch.cat([label, self.des_pt_attn_mask[:-1] * self.IGNORE_INDEX, self.des_pt_id[-1].unsqueeze(0), tok['input_ids'].squeeze(0)], dim=0)
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")
                
                # ">."
                text = torch.cat([text, self.des_pt2_embed], dim=0)
                attn = torch.cat([attn, self.des_pt2_attn_mask], dim=0)
                label = torch.cat([label, self.des_pt2_id], dim=0)
                assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")
            
            tok = tokenizer('</s>', return_tensors='pt', add_special_tokens=False)
            text = torch.cat([text, self.model.embed_tokens(tok['input_ids']).squeeze(0)], dim=0)
            attn = torch.cat([attn, tok['attention_mask'].squeeze(0)], dim=0)
            label = torch.cat([label, tok['input_ids'].squeeze(0)], dim=0)
            assert (len(text) == len(label)) and (len(text) == len(attn)), Exception("No!")

            full_text.append(text)
            full_attn.append(attn)
            full_label.append(label)
            
        inputs_embeds=torch.nn.utils.rnn.pad_sequence(full_text, batch_first=True)
        attention_mask=torch.nn.utils.rnn.pad_sequence(full_attn, batch_first=True).bool()
        
        if img_des is not None:
            labels=torch.nn.utils.rnn.pad_sequence(full_label, batch_first=True, padding_value=self.IGNORE_INDEX).to(torch.int64)
        else:
            labels=None
            
        return inputs_embeds, attention_mask, labels, start_gen_prop_idx_list, start_ref_prop_idx_list
        


    def forward(
        self,        
        img_features=None,
        img_description=None,
        ref_description=None,
        gen_proposals=None,
        ref_proposals=None,
        tokenizer=None,
        *,
        input_ids=None,
        attention_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:

        inputs_embeds, attention_mask, labels, start_gen_prop_idx_list, start_ref_prop_idx_list = \
            self.cullavo_multimodal_inputs(img_features, img_description, ref_description, gen_proposals, ref_proposals, tokenizer)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        with torch.cuda.amp.autocast():
            outputs = self.model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
            )

            hidden_states = outputs[0]
            logits = self.lm_head(hidden_states)
            logits = logits.float()

            loss = None
            if labels is not None:
                # Shift so that tokens < n predict n
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # Flatten the tokens
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)
                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss = F.cross_entropy(shift_logits, shift_labels)
        
        # CuLLaVO
        if gen_proposals[0] is None:
            ref_tensor_list = []
            for idx, start_ref_id in enumerate(start_ref_prop_idx_list):
                ref_tensor_list.append(hidden_states[idx][start_ref_id: start_ref_id+200].unsqueeze(0))
            ref_tensor = torch.cat(ref_tensor_list, dim=0)
            return loss, ref_tensor
        elif ref_proposals[0] is None:
            gen_tensor_list = []
            for idx, start_gen_id in enumerate(start_gen_prop_idx_list):
                gen_tensor_list.append(hidden_states[idx][start_gen_id: start_gen_id+200].unsqueeze(0))
            gen_tensor = torch.cat(gen_tensor_list, dim=0)
            return loss, gen_tensor
        else:
            gen_tensor_list = []
            ref_tensor_list = []
            for idx, (start_gen_id, start_ref_id) in enumerate(zip(start_gen_prop_idx_list, start_ref_prop_idx_list)):
                gen_tensor_list.append(hidden_states[idx][start_gen_id: start_gen_id+200].unsqueeze(0))
                ref_tensor_list.append(hidden_states[idx][start_ref_id: start_ref_id+200].unsqueeze(0))
            gen_tensor = torch.cat(gen_tensor_list, dim=0)
            ref_tensor = torch.cat(ref_tensor_list, dim=0)
            return loss, gen_tensor, ref_tensor 


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

# AutoConfig.register("llava", LlavaConfig)
# AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
