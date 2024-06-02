from cv2 import transform
import torch
from torch import nn
import torch.nn.functional as F
from hydra.core.config_store import ConfigStore
from dataclasses import dataclass
import omegaconf
import hydra
import typing as tp
from typing import Any, Tuple
from controllable_navi import utils
from controllable_navi.agent.Transformer.models.position_embedding import SinusoidalPositionalEmbedding
from controllable_navi.agent.encoders import MultiModalEncoder

@dataclass
class TransformerConfig:
    # _target_: str = "controllable_navi.agent.sequence_encoders.Transformer"
    num_layers: int = 2
    nhead: int = 1
    kdim: int = 160
    vdim: int = 160
    dim_feedforward: int =256
    dropout: float =0.1

@dataclass
class SequenceEncoderConfig:
    transformer:TransformerConfig=TransformerConfig()

class TransformerEncoderBlock(nn.Module):
    def __init__(self, d_model, nhead, kdim=32, vdim=32, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderBlock, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # self.self_attn = nn.MultiheadAttention(d_model, nhead, kdim=kdim, vdim=vdim, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src2 = self.self_attn(src, src, src, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)[0]
        if torch.sum(torch.isnan(src2)):
            print("src2")
            print(src,src_mask,src_key_padding_mask)
            raise ValueError
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

class Transformer(nn.Module):
    def __init__(self, d_model,cfg:TransformerConfig):
        super(Transformer, self).__init__()
        self.pos_encoder = SinusoidalPositionalEmbedding(d_model)
        self.encoder_layers = nn.ModuleList([TransformerEncoderBlock(d_model, 
                                                                     cfg.nhead, 
                                                                     cfg.kdim,
                                                                     cfg.vdim,
                                                                     cfg.dim_feedforward, 
                                                                     cfg.dropout) for _ in range(cfg.num_layers)])
        self.d_model = d_model
        self.nhead = cfg.nhead

    def forward(self, src_in, src_mask=None, src_key_padding_mask=None):
        src = src_in + self.pos_encoder(src_in[:, :, 0])
        if torch.sum(torch.isnan(src)):
            print("src pos")
            raise ValueError
        src = src.transpose(0, 1)  # Transformer expects (sequence_length, batch_size, d_model)

        for i, encoder in enumerate(self.encoder_layers):
            # print(i,src)
            src = encoder(src, src_mask, src_key_padding_mask)
        
        output = src.transpose(0, 1) # N, L, C
        if torch.sum(torch.isnan(output)):
            print("transformer")
            raise ValueError
        
        return output
    
class SequenceEncoder(nn.Module):
    embed_func: tp.Union[MultiModalEncoder, nn.Identity]
    def __init__(self,obs_dim, embed_func, cfg:SequenceEncoderConfig):
        super(SequenceEncoder, self).__init__()
        self.embed_func = embed_func
        if isinstance(self.embed_func,nn.Identity):
            embed_size = obs_dim
        else:
            embed_size  = self.embed_func.repr_dim  
        d_model = embed_size
        
        self.transformer = Transformer(d_model,cfg.transformer)
        self.fc_out = nn.Linear(d_model, embed_size)
        self.d_model = d_model
        return
    
    def parameters(self, recurse: bool = True) -> tp.Iterator[nn.Parameter]:
        for net in [self.embed_func,self.transformer,self.fc_out]:
            for item in net.parameters():
                yield item
    
    def forward(self, src_in, mask):
        # mask processing
        src_mask = torch.bmm(mask.unsqueeze(2), mask.unsqueeze(1)) # N, L, L
        src_key_padding_mask = mask # N, L
        # input embedding
        src = self.embed_func(src_in) # N, L, C

        # print("input",src.shape)
        # print("src",src)
        if torch.sum(torch.isnan(src)):
            print("src")
            # print(src_in,src)
            raise ValueError
        
        output = self.transformer(src, src_mask, src_key_padding_mask) # N, L, C
        # print("trans_out",output.shape)
        mask_expanded = mask.unsqueeze(-1).expand(-1, -1, self.d_model).to(torch.bool)# N, L, C

        # Set masked elements to a very large negative value
        masked_output = output.masked_fill(~mask_expanded, float('-inf')) # N, L, C

        # Apply max pooling
        aggregated_vector, _ = masked_output.max(dim=1)  # (batch_size, d_model)
        # print("agg_vector",aggregated_vector.shape)
        output = self.fc_out(aggregated_vector)

        if torch.sum(torch.isnan(output)):
            print("output")
            raise ValueError
        return output