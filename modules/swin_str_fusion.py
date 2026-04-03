'''
Implementation of MGP-STR based on ViTSTR.

Copyright 2022 Alibaba

Modified by Huiling Li in 2026.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import logging
import torch.utils.model_zoo as model_zoo
import torch.nn.functional as F
from torch.nn import TransformerEncoderLayer
import math
import os

from copy import deepcopy
from functools import partial

from .models.swin_transformer import SwinTransformer, _cfg
from timm.models.registry import register_model
from timm.models import create_model

from .token_learner import TokenLearner

_logger = logging.getLogger(__name__)

__all__ = [
    'swin_small_patch4_window7_224_fusion'
]

class DenoisingGate(nn.Module):
    def __init__(self, channels):
        super(DenoisingGate, self).__init__()
        self.channel_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // 4, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // 4, channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
        
        self.spatial_gate = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca_weight = self.channel_gate(x)
        x = x * ca_weight
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        sa_weight = self.spatial_gate(torch.cat([avg_out, max_out], dim=1))
        
        return x * sa_weight

def create_swin_str_fusion(batch_max_length, num_tokens, model=None, checkpoint_path=''):

    mgp_str = create_model(
        model,  #'swin_small_patch4_window7_224_fusion'
        pretrained=True, 
        num_classes=num_tokens, 
        checkpoint_path=checkpoint_path, 
        batch_max_length=batch_max_length  
        )
    mgp_str.reset_classifier(num_classes=num_tokens)

    return mgp_str

class MGPSTR(SwinTransformer):

    def __init__(self, batch_max_length, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.batch_max_length = batch_max_length

        self.char_tokenLearner = TokenLearner(768, self.batch_max_length)

        self.p4conv = nn.Conv2d(768, 192, kernel_size=3, padding=1)  
        self.p3conv = nn.Conv2d(768, 192, kernel_size=3, padding=1)  
        self.p2conv = nn.Conv2d(384, 192, kernel_size=3, padding=1)  
        self.p1conv = nn.Conv2d(192, 192, kernel_size=3, padding=1)  

        self.p4upout = nn.Upsample(scale_factor=4)

        self.p3upin = nn.Upsample(scale_factor=2)  
        self.p3down = nn.Conv2d(768, 384, kernel_size=3, padding=1)  
        self.p3upout = nn.Upsample(scale_factor=4)  

        self.p2upin = nn.Upsample(scale_factor=2)  
        self.p2down = nn.Conv2d(384, 192, kernel_size=3, padding=1)  
        self.p2upout = nn.Upsample(scale_factor=2)  

        self.gate1 = DenoisingGate(192)
        self.gate2 = DenoisingGate(192)
        self.gate3 = DenoisingGate(192)
        self.gate4 = DenoisingGate(192)

        self.bpe_tokenLearner = TokenLearner(768, self.batch_max_length)
        self.wp_tokenLearner = TokenLearner(768, self.batch_max_length)

        self.fusion = TransformerEncoderLayer(768, 8, 768)
        self.self_atten = None  

    def reset_classifier(self, num_classes):
        self.num_classes = num_classes
        self.char_head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        self.fusion_head = nn.Linear(768, num_classes) if num_classes > 0 else nn.Identity()
        self.bpe_head = nn.Linear(768, 50257) if num_classes > 0 else nn.Identity()
        self.wp_head = nn.Linear(768, 21128) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        feats = []  
        x = self.patch_embed(x) 
        x = x + self.absolute_pos_embed
        x = self.pos_drop(x)
        
        for i, layer in enumerate(self.layers):
            x = layer(x)  
            feats.append(x)  

        p1, p2, p3, p4 = feats

        B, L, C = p1.shape
        p1 = p1.transpose(1, 2).view(B, C, 28, 28)
        B, L, C = p2.shape
        p2 = p2.transpose(1, 2).view(B, C, 14, 14)
        B, L, C = p3.shape
        p3 = p3.transpose(1, 2).view(B, C, 7, 7)
        B, L, C = p4.shape
        p4 = p4.transpose(1, 2).view(B, C, 7, 7)

        p3 = p4 + p3
        p3_up = self.p3down(self.p3upin(p3))  
        p2 = p3_up + p2
        p2_up = self.p2down(self.p2upin(p2))  
        p1 = p2_up + p1

        p1_out = self.p1conv(p1)
        p2_out = self.p2upout(self.p2conv(p2))
        p3_out = self.p3upout(self.p3conv(p3))
        p4_out = self.p4upout(self.p4conv(p4))

        p1_out = self.gate1(p1_out)
        p2_out = self.gate2(p2_out)
        p3_out = self.gate3(p3_out)
        p4_out = self.gate4(p4_out)
        
        p1_out = p1_out.flatten(2).transpose(1, 2)  
        p2_out = p2_out.flatten(2).transpose(1, 2)  
        p3_out = p3_out.flatten(2).transpose(1, 2)  
        p4_out = p4_out.flatten(2).transpose(1, 2)  
        
        x = torch.cat([p1_out, p2_out, p3_out, p4_out], dim=-1)  
            
        attens = []  
        
        char_attn, x_char = self.char_tokenLearner(x)  
        char_out = self.char_head(x_char)  
        attens.append(char_attn)  
        
        bpe_attn, x_bpe = self.bpe_tokenLearner(x)  
        bpe_out = self.bpe_head(x_bpe)  
        attens.append(bpe_attn)  
        
        wp_attn, x_wp = self.wp_tokenLearner(x)  
        wp_out = self.wp_head(x_wp)  
        attens.append(wp_attn)  

        x_combined = torch.cat([x_char, x_bpe, x_wp], dim=1) 
        x_combined = torch.cat([x_char, x_bpe, x_wp], dim=1) 
        x_combined = x_combined.transpose(0, 1)            
        x_fusion = self.fusion(x_combined)                 
        x_fusion = x_fusion.transpose(0, 1)                
        x_fusion = x_fusion[:, :x_char.shape[1], :]
        fusion_out = self.fusion_head(x_fusion)  
        
        return attens, char_out, bpe_out, wp_out, fusion_out

    def forward(self, x, is_eval=False):
        attn_scores, char_out, bpe_out, wp_out, fusion_out = self.forward_features(x)
        if is_eval:
            return [attn_scores, char_out, bpe_out, wp_out, fusion_out]
        else:
            return [char_out, bpe_out, wp_out, fusion_out]


def load_pretrained(model, cfg=None, num_classes=1000, in_chans=1, filter_fn=None, strict=True):

    if cfg is None:
        cfg = getattr(model, 'default_cfg') 
        
    if cfg is None or 'url' not in cfg or not cfg['url']:
        _logger.warning("Pretrained model URL is invalid, using random initialization.")
        return

    # Prefer a local pretrained file at repository root (SwinTR/swin_small_patch4_window7_224.pth).
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    local_path = os.path.join(repo_root, 'swin_small_patch4_window7_224.pth')
    if os.path.exists(local_path):
        state_dict = torch.load(local_path, map_location='cpu')
    else:
        # Fallback: attempt to download from the configured URL and save to repo root
        if cfg and 'url' in cfg and cfg['url']:
            url = cfg['url']
            print(f'Local pretrained not found at {local_path}; downloading to that path from {url}')
            try:
                import urllib.request
                # download file to local_path
                urllib.request.urlretrieve(url, local_path)
                state_dict = torch.load(local_path, map_location='cpu')
            except Exception as e:
                print(f'Download-to-disk failed: {e}; trying torch.hub/model_zoo fallbacks')
                try:
                    state_dict = torch.hub.load_state_dict_from_url(url, progress=True, map_location='cpu')
                    # try to save for future runs
                    try:
                        torch.save(state_dict, local_path)
                    except Exception:
                        pass
                except Exception:
                    try:
                        state_dict = model_zoo.load_url(url)
                        try:
                            torch.save(state_dict, local_path)
                        except Exception:
                            pass
                    except Exception as e2:
                        raise RuntimeError(f'Failed to obtain pretrained weights from {url}: {e2}')
        else:
            raise FileNotFoundError(f'Local pretrained file not found: {local_path}')

    if "model" in state_dict.keys():
        state_dict = state_dict["model"]
    
    if filter_fn is not None:
        state_dict = filter_fn(state_dict)
    
    print("in_chans", in_chans)

    if in_chans == 1:
        conv1_name = cfg['first_conv'] 
        _logger.info('Converting first conv (%s) pretrained weights from 3 to 1 channel' % conv1_name)
        key = conv1_name + '.weight'
        
        if key in state_dict.keys():
            _logger.info('(%s) key found in state_dict' % key)
            conv1_weight = state_dict[conv1_name + '.weight']
        else:
            _logger.info('(%s) key NOT found in state_dict' % key)
            return
 
        conv1_type = conv1_weight.dtype
        conv1_weight = conv1_weight.float()
        O, I, J, K = conv1_weight.shape

        if I > 3:

            assert conv1_weight.shape[1] % 3 == 0
            conv1_weight = conv1_weight.reshape(O, I // 3, 3, J, K)
            conv1_weight = conv1_weight.sum(dim=2, keepdim=False)
        else:
            conv1_weight = conv1_weight.sum(dim=1, keepdim=True)
        conv1_weight = conv1_weight.to(conv1_type)
        state_dict[conv1_name + '.weight'] = conv1_weight
    
    classifier_name = cfg['classifier']
    
    if num_classes == 1000 and cfg['num_classes'] == 1001:
        classifier_weight = state_dict[classifier_name + '.weight']
        state_dict[classifier_name + '.weight'] = classifier_weight[1:] 
        classifier_bias = state_dict[classifier_name + '.bias']
        state_dict[classifier_name + '.bias'] = classifier_bias[1:] 
    elif num_classes != cfg['num_classes']:
        weight_key = classifier_name + '.weight'
        bias_key = classifier_name + '.bias'
        if weight_key in state_dict:
            del state_dict[weight_key]
        if bias_key in state_dict:
            del state_dict[bias_key]
        strict = False
    print("Loading pre-trained vision transformer weights from %s ..." % cfg['url'])
    model.load_state_dict(state_dict, strict=strict)


def _conv_filter(state_dict):
    out_dict = {}
    for k, v in state_dict.items():
        if 'absolute_pos_embed' in k:
            print(f"Skipping {k} due to shape mismatch")
            continue
        out_dict[k] = v 
    return out_dict

@register_model
def swin_small_patch4_window7_224_fusion(pretrained=True, **kwargs):
    kwargs['in_chans'] = 3
    
    model = MGPSTR(
        img_size=(224, 224),
        patch_size=4, 
        window_size=7, 
        embed_dim=96, 
        depths=(2, 2, 18, 2),
        num_heads=(3, 6, 12, 24),
        ape=True,
        **kwargs)
    model.default_cfg = _cfg(
        url='https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_small_patch4_window7_224.pth'
    )
    
    if pretrained:
        load_pretrained(
            model, 
            num_classes=model.num_classes, 
            in_chans=kwargs.get('in_chans', 3), 
            filter_fn=_conv_filter)
    
    return model