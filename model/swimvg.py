import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from model.clip import build_model
from .fusion import Fusion
from dinov2.models.vision_transformer import vit_base,vit_large,vit_giant2
from dinov2.layers.mlp import Mlp

class SwimVG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        # Text Encoder

        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len, -1, cfg.txtual_adapter_layer,cfg.txt_adapter_dim).float()
        self.fusion = Fusion(d_model=cfg.ladder_dim, nhead=cfg.nhead,dino_layers=cfg.dino_layers)
    
       # Fix Backbone
        for param_name, param in self.backbone.named_parameters():
            if cfg.dinov2_only_backbone: #dinov做主干网络
                if 'visual' in param_name or 'logit_scale' in param_name:
                    param.requires_grad = False
                if 'prompts' not in param_name and 'adapter' not in param_name: #文本txt的adapter可以训练
                    param.requires_grad = False       
            else:
                if 'positional_embedding' not in param_name and 'flow' not in param_name:
                    param.requires_grad = False       
        self.use_dinov2=cfg.dinov2
        if cfg.dinov2:
            state_dict = torch.load(cfg.dino_pretrain) 
            if cfg.dino_name == 'dinov2-large':
                self.dinov2=vit_large(
                    patch_size=14,
                    num_register_tokens=0,
                    img_size=518,
                    init_values=1.0,
                    block_chunks=0,
                    backbone=cfg.dinov2_only_backbone,
                    layers_output=cfg.dinov2_only_backbone ,
                    add_adapter_layer=cfg.visual_adapter_layer,
                    visual_adapter_dim=cfg.visual_adapter_dim,                
                )
                reg_dim = 1024
            self.dinov2.load_state_dict(state_dict, strict=False)

            for param_name, param in self.dinov2.named_parameters():
                if 'adapter' not in param_name:
                    param.requires_grad = False
        for param_name, param in self.backbone.named_parameters():
            if param.requires_grad:
                print(param_name)

        for param_name, param in self.dinov2.named_parameters():
            if param.requires_grad:
                print(param_name)

        self.reg_token = nn.Embedding(1, reg_dim)
        self.txt_token = nn.Embedding(1, 512)


        self.bbox_embed =  MLP(reg_dim, reg_dim, 4, 3)


    def forward(self, img, word, mask=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        vis, word, state, reg_tokens = self.fusion(self.reg_token, self.txt_token, img, word, self.backbone, self.dinov2)
        reg_tokens=reg_tokens[-1]#.permute(1,0,2)

        pred_box = self.bbox_embed(reg_tokens).sigmoid()

        return pred_box

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

