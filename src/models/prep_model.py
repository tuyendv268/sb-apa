import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import warnings
import torch
import math

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)

        self.drop_path = nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class AdditiveAttention(nn.Module):
    def __init__(self, hidden_dim: int, drop_rate=0.1) -> None:
        super(AdditiveAttention, self).__init__()

        self.query_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.bias = nn.Parameter(
            torch.rand(hidden_dim).uniform_(-0.1, 0.1))
        
        self.score_proj = nn.Linear(hidden_dim, 1)

    def forward(self, query, key, value):
        score = self.score_proj(torch.tanh(self.key_proj(key) + self.query_proj(query) + self.bias)).squeeze(-1)
        attn = F.softmax(score, dim=-1)
        context = torch.bmm(attn.unsqueeze(1), value)
        return context, attn

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()
        self.depthwise_conv = nn.Conv1d(
            in_channels=in_channels, out_channels=in_channels,
            kernel_size=kernel_size, groups=in_channels, 
            padding=kernel_size//2, bias=False
        )

        self.pointwise_conv = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        x = x.transpose(1,2)
        x = self.depthwise_conv(x)
        x = self.pointwise_conv(x)
        x = x.transpose(1,2)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.FloatTensor) -> torch.FloatTensor:
        return self.pe[:, : x.size(1)]
    
class PrepModel(nn.Module):
    def __init__(self, embed_dim, num_heads, depth, input_dim=84, num_phone=46, max_length=50, dropout=0.1):
        super(PrepModel, self).__init__()
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.pos_embed = PositionalEncoding(d_model=self.embed_dim, max_len=max_length)
        
        self.phn_in_proj = nn.Linear(input_dim, embed_dim)
        self.phn_canon_in_proj = nn.Linear(input_dim, embed_dim)
        
        self.phn_canon_embedding = nn.Embedding(
            num_embeddings=num_phone, embedding_dim=embed_dim)
        
        self.phone_encoders = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads) 
                for i in range(depth)
                ]
            )
        
        kernel_size = 3
        self.ds_conv = DepthwiseSeparableConvolution(
            in_channels=embed_dim, 
            out_channels=embed_dim, 
            kernel_size=kernel_size
        )
        
        self.word_encoders = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads) 
                for i in range(1)
                ]
            )
        
        self.rel_pos_embed = nn.Embedding(
            num_embeddings=5, embedding_dim=embed_dim)
        
        self.utt_encoders = nn.ModuleList(
            [
                Block(dim=embed_dim, num_heads=num_heads) 
                for i in range(1)
                ]
            )
        
        self.utt_addi = AdditiveAttention(hidden_dim=embed_dim)

        self.phone_acc_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.word_acc_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        self.utt_acc_head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Linear(embed_dim, 1))
        
    def forward(self, phn_embed, phn_canoncial, rel_pos):
        batch_size, seq_length, embedd_dim = phn_embed.shape[0], phn_embed.shape[1], phn_embed.shape[2]

        phn_embed = self.phn_in_proj(phn_embed)
        phn_canon_embed = self.phn_canon_embedding(phn_canoncial)

        p_x = phn_embed + phn_canon_embed + self.pos_embed(phn_embed)
        for block in self.phone_encoders:
            p_x = block(p_x)
            
        w_x = p_x + self.rel_pos_embed(rel_pos)
        w_x = self.ds_conv(w_x)
        for block in self.word_encoders:
            w_x = block(w_x)
        word_acc_score = self.word_acc_head(w_x)

        for block in self.utt_encoders:
            w_x = block(w_x)

        u_x = p_x + w_x
        u_x, attn = self.utt_addi(query=u_x, key=u_x, value=u_x)
        utt_acc_score = self.utt_acc_head(u_x.squeeze(1))

        return utt_acc_score, word_acc_score
    