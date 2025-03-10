import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Optional
from einops import rearrange
import math
import numpy as np
import torch.distributed as dist

import os
from functools import partial

class Fusion(nn.Module):
    def __init__(self,
                 d_img = [768, 768, 768],
                 d_txt = 512,
                 d_model = 64,
                 nhead = 8,
                 num_stages = 3,
                 strides = [1, 1, 1],
                 num_layers = 12,
                 shared_weights = False,
                 dino_layers= 12,
                 output_dinov2 =[4, 8] ,
                ):
        super().__init__()

        self.d_img = d_img
        self.d_txt = d_txt
        self.d_model = d_model
        self.num_stages = num_stages
        self.num_layers = num_layers
        self.dino_layers = dino_layers
        self.output_dinov2 = output_dinov2
        self.n_ctx_visual = 0

        self.n_ctx_text = 1
        textual_ctx_vectors = torch.empty(self.n_ctx_text, self.d_txt)
        nn.init.normal_(textual_ctx_vectors, std=0.02)
        
        self.initialize_parameters()
        

    def initialize_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')                

    def forward(self, reg_token, txt_token, img, text, backbone,model=None):
        B=img.shape[0]
        img = img.type(backbone.dtype)
        
        vis_outs = []
        outputs=[]
        txt = backbone.token_embedding(text).type(
            backbone.dtype)  # [batch_size, n_ctx, d_model]

        txt_enc = backbone.transformer
        txt = txt + backbone.positional_embedding.type(backbone.dtype)[:txt.size(1)]
        
        # prompt = torch.zeros(1, 512).to(txt.device)
        #txt_token = txt_token.unsqueeze(1).repeat(B, 1, 1)
        txt_token = txt_token.weight.unsqueeze(1).repeat(B, 1, 1)
        # text_prompt = prompt.unsqueeze(1).repeat(B, 1, 1)
        txt = torch.cat([txt,txt_token],dim=1)
        
        txt = txt.permute(1, 0, 2)  # BLD -> LBD

        #dinov2  
        net_input = img.clone()
        B, nc, w, h = net_input.shape
        dino_f = model.patch_embed(net_input)
        dino_f = torch.cat((model.cls_token.expand(dino_f.shape[0], -1, -1), dino_f), dim=1)

        dino_f = dino_f + model.interpolate_pos_encoding(dino_f, w, h)
        
        features_dino=[]
        txt_list = [None]*12
        prompts = [] #[txt_token.permute(1,0,2)]
        for i in range(self.num_layers):
            txt, prompt = txt_enc.resblocks[i](txt)
            txt_list.append(txt.permute(1,0,2)[:,:40,]) 
            prompts.append(prompt)

        # language
        txt = txt.permute(1, 0, 2)  # LBD -> BLD
        txt = backbone.ln_final(txt).type(backbone.dtype)
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = txt @ backbone.text_projection# get sentence-level feature Fs
         
        state = state[:,:40,] 
        txt = txt[:,:40,] 
        
        # reg_token = reg_token.to(dino_f.device)
        reg_token = reg_token.weight.unsqueeze(0).repeat(B, 1, 1)
        dino_f = torch.cat([reg_token, dino_f], dim=1)
        # txt = torch.cat(prompts,dim=0).permute(1,0,2)


        prompts = prompts + [None]*40
        reg_tokens = []
        for i in range(self.dino_layers):
            dino_f = model.blocks[i](dino_f, prompts[i], txt)
            reg_tokens.append(dino_f[:,0])
            if i in self.output_dinov2:
                features_dino.append(dino_f)
        
        #reg_tokens = torch.stack(reg_tokens)
        
        dino_f = model.norm(dino_f)
        reg_tokens.append(dino_f[:,0])
        features_dino.append(dino_f)
        

        for i, feature_dino in enumerate(features_dino):

            vis_outs.append(feature_dino)

        # forward
        output = vis_outs , txt, state, reg_tokens

        return output


