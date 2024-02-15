import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from IPython import embed

class FTDNNLayer(nn.Module):
    def __init__(self, semi_orth_in_dim, semi_orth_out_dim, affine_in_dim, out_dim, time_offset, dropout_p=0, device='cpu'):
        super(FTDNNLayer, self).__init__()
        self.semi_orth_in_dim = semi_orth_in_dim
        self.semi_orth_out_dim = semi_orth_out_dim
        self.affine_in_dim = affine_in_dim
        self.out_dim = out_dim
        self.time_offset = time_offset
        self.dropout_p = dropout_p
        self.device = device

        self.sorth = nn.Linear(self.semi_orth_in_dim, self.semi_orth_out_dim, bias=False)
        self.affine = nn.Linear(self.affine_in_dim, self.out_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_dim, affine=False, eps=0.001)
        self.dropout = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        time_offset = self.time_offset
        if time_offset != 0:
            padding = x[:,0,:][:,None,:]
            xd = torch.cat([padding]*time_offset+[x], axis=1)
            xd = xd[:,:-time_offset,:]
            x = torch.cat([xd, x], axis=2)
        x = self.sorth(x)
        if time_offset != 0:
            padding = x[:,-1,:][:,None,:]
            padding = torch.zeros(padding.shape)
            if self.device == 'cuda':
                padding = padding.cuda()
            xd = torch.cat([x]+[padding]*time_offset, axis=1)
            xd = xd[:,time_offset:,:]
            x = torch.cat([x, xd], axis=2)
        x = self.affine(x)
        x = self.nl(x)
        x = x.transpose(1,2)
        x = self.bn(x).transpose(1,2)
        x = self.dropout(x)
        return x

class InputLayer(nn.Module):
    def __init__(
        self,
        input_dim=220,
        output_dim=1536,
        dropout_p=0):

        super(InputLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p

        self.lda = nn.Linear(self.input_dim, self.input_dim)
        self.kernel = nn.Linear(self.input_dim,
                                self.output_dim)

        self.nonlinearity = nn.ReLU()
        self.bn = nn.BatchNorm1d(output_dim, affine=False, eps=0.001)
        self.drop = nn.Dropout(p=self.dropout_p)

    def forward(self, x):
        mfccs = x[:,:,:40]
        ivectors = x[:,:,-100:]
        padding_first = mfccs[:,0,:][:,None,:]
        padding_last = mfccs[:,-1,:][:,None,:]
        context_first = torch.cat([padding_first, mfccs[:,:-1,:]], axis=1)
        context_last = torch.cat([mfccs[:,1:,:], padding_last], axis=1)
        x = torch.cat([context_first, mfccs, context_last, ivectors], axis=2)
        x = self.lda(x)
        x = self.kernel(x)
        x = self.nonlinearity(x)

        x = x.transpose(1, 2)
        x = self.bn(x).transpose(1,2)
        x = self.drop(x)
        return x

def sum_outputs_and_feed_to_layer(x, x_2, layer):
        x_3 = x*0.75 + x_2
        x = x_3
        x_2 = layer(x_3)
        return x, x_2

class FTDNN(nn.Module):
    def __init__(self, in_dim=220, batchnorm=None, dropout_p=0, device_name='cpu'):
        super(FTDNN, self).__init__()

        self.layer01 = InputLayer(input_dim=in_dim, output_dim=1536)
        self.layer02 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer03 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer04 = FTDNNLayer(3072, 160, 320, 1536, 1, dropout_p=dropout_p, device=device_name)
        self.layer05 = FTDNNLayer(1536, 160, 160, 1536, 0, dropout_p=dropout_p, device=device_name)
        self.layer06 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer07 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer08 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer09 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer10 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer11 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer12 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer13 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer14 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer15 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer16 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer17 = FTDNNLayer(3072, 160, 320, 1536, 3, dropout_p=dropout_p, device=device_name)
        self.layer18 = nn.Linear(1536, 256, bias=False) #This is the prefinal-l layer
        
    def forward(self, x):
        '''
        Input must be (batch_size, seq_len, in_dim)
        '''
        x = self.layer01(x)
        x_2 = self.layer02(x)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer03)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer04)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer05)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer06)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer07)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer08)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer09)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer10)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer11)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer12)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer13)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer14)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer15)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer16)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer17)
        x, x_2 = sum_outputs_and_feed_to_layer(x, x_2, self.layer18)
        return x_2

class OutputLayer(nn.Module):
    def __init__(self, linear1_in_dim, linear2_in_dim, linear3_in_dim, out_dim):
        super(OutputLayer, self).__init__()
        self.linear1_in_dim = linear1_in_dim
        self.linear2_in_dim = linear2_in_dim
        self.linear3_in_dim = linear3_in_dim
        self.out_dim = out_dim

        self.linear1 = nn.Linear(self.linear1_in_dim, self.linear2_in_dim, bias=True) 
        self.nl = nn.ReLU()
        self.bn1 = nn.BatchNorm1d(self.linear2_in_dim, affine=False)
        self.linear2 = nn.Linear(self.linear2_in_dim, self.linear3_in_dim, bias=False) 
        self.bn2 = nn.BatchNorm1d(self.linear3_in_dim, affine=False)
        self.linear3 = nn.Linear(self.linear1_in_dim, self.out_dim, bias=True)

    def forward(self, x):
        x = self.linear1(x)
        x = self.nl(x)
        x = x.transpose(1,2)
        x = self.bn1(x).transpose(1,2)
        x = self.linear2(x)
        x = x.transpose(1,2)
        x = self.bn2(x).transpose(1,2)
        x = self.linear3(x)
        return x

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
    
