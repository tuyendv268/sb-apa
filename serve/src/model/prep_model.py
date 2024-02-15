import torch
from torch import nn

from src.module import (
    PositionalEncoding,
    Block,
    DepthwiseSeparableConvolution,
    AdditiveAttention
)


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
        phone_acc_score = self.phone_acc_head(p_x)

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

        return utt_acc_score, phone_acc_score, word_acc_score
