import torch, torch.nn as nn, torch.nn.functional as F_torch
import math
from torch.nn.utils import weight_norm

# Dataset/Dataloader
class SeqDataset(torch.utils.data.Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)
    def __len__(self): return len(self.y)
    def __getitem__(self, i): return self.X[i], self.y[i]


# Baseline 1: Conv1D
class Conv1DBaseline(nn.Module):
    def __init__(self, in_feat, n_classes):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_feat, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Conv1d(64, 64, kernel_size=3, padding=1, dilation=2),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2)
        )
        self.head = nn.Linear(64, n_classes)

    def forward(self, x):           # x: [B, T, F]
        x = x.transpose(1, 2)       # -> [B, F, T]
        h = self.block(x)           # [B, 64, T]
        h = h.mean(dim=-1)          # Global Average Pool -> [B, 64]
        return self.head(h)         # [B, C]


# Baseline 2: TCN
class CausalConv1d(nn.Module):
    """left-pad only (no future leak) causal Conv1d."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation=1, bias=True):
        super().__init__()
        self.pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=0, dilation=dilation, bias=bias)

    def forward(self, x):  # x: [B, C, T]
        x = F_torch.pad(x, (self.pad, 0))  # (left, right)
        return self.conv(x)

class TemporalBlock(nn.Module):
    """Wavenet-style residual block: CausalConv → ReLU → Dropout ×2 + skip."""
    def __init__(self, in_ch, out_ch, kernel_size, dilation, dropout=0.2, use_weight_norm=True):
        super().__init__()
        conv1 = CausalConv1d(in_ch, out_ch, kernel_size, dilation)
        conv2 = CausalConv1d(out_ch, out_ch, kernel_size, dilation)
        self.conv1 = weight_norm(conv1.conv) if use_weight_norm else conv1.conv
        self.conv2 = weight_norm(conv2.conv) if use_weight_norm else conv2.conv
        self.pad = conv1.pad  # same pad for both
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else None

    def _causal(self, x, conv):
        pad = (self.kernel_size - 1) * self.dilation
        x = F_torch.pad(x, (pad, 0))
        return conv(x)

    def forward(self, x):  # x: [B, C, T]
        y = self._causal(x, self.conv1)
        y = self.relu(y)
        y = self.dropout(y)
        y = self._causal(y, self.conv2)
        y = self.relu(y)
        y = self.dropout(y)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(y + res)

class TCNBackbone(nn.Module):
    """Stack of TemporalBlocks with exponentially increasing dilation."""
    def __init__(self, in_ch, channels, kernel_size=3, dropout=0.2):
        super().__init__()
        layers = []
        c_prev = in_ch
        for i, c in enumerate(channels):
            dilation = 2 ** i
            layers.append(TemporalBlock(c_prev, c, kernel_size, dilation, dropout))
            c_prev = c
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # [B, C, T]
        return self.net(x)

class TCNBaseline(nn.Module):
    """
    Sequence classifier with TCN backbone.
    - input: [B, T, F]
    - output: logits [B, n_classes]
    pool: 'last' (윈도우 끝 프레임 라벨과 잘 맞음) or 'avg'
    """
    def __init__(self, in_feat, n_classes, channels=(64,64,64,64),
                 kernel_size=3, dropout=0.2, pool='last'):
        super().__init__()
        assert pool in ('last','avg')
        self.pool = pool
        self.proj_in = nn.Conv1d(in_feat, channels[0], kernel_size=1)
        self.tcn = TCNBackbone(channels[0], list(channels), kernel_size=kernel_size, dropout=dropout)
        self.head = nn.Linear(channels[-1], n_classes)

    def forward(self, x):  # x: [B, T, F]
        x = x.transpose(1, 2).contiguous()  # -> [B, F, T]
        x = self.proj_in(x)
        h = self.tcn(x)                      # [B, C, T]
        if self.pool == 'last':
            z = h[:, :, -1]                  # 윈도우 마지막 시점
        else:
            z = h.mean(dim=-1)               # 평균 풀링
        return self.head(z)


# 3. Transformer model
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)  # [L, D]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # no grad

    def forward(self, x):  # x: [B, T, D]
        T = x.size(1)
        x = x + self.pe[:T].unsqueeze(0)  # [1,T,D]
        return self.dropout(x)

def _causal_attn_mask(T: int, device=None, dtype=torch.float32):
    # True/inf on "disallowed" positions (j > i)
    mask = torch.zeros((T, T), dtype=dtype, device=device)
    mask = mask.masked_fill(torch.triu(torch.ones((T, T), device=device, dtype=torch.bool), diagonal=1), float('-inf'))
    return mask

class TransformerBaseline(nn.Module):
    """
    Transformer encoder for sequence classification
    - input:  x [B, T, F]
    - output: logits [B, n_classes]
    """
    def __init__(self,
                 in_feat: int,
                 n_classes: int,
                 d_model: int = 128,
                 nhead: int = 8,
                 num_layers: int = 4,
                 dim_feedforward: int = 256,
                 dropout: float = 0.2,
                 pool: str = 'last',   # 'last' or 'avg'
                 causal: bool = True,  # True면 미래 attend 금지
                 max_len: int = 2048,
                 use_input_ln=True):
        super().__init__()
        assert pool in ('last','avg')
        self.pool = pool
        self.causal = causal
        self.in_norm = nn.LayerNorm(in_feat) if use_input_ln else nn.Identity()
        self.proj_in = nn.Linear(in_feat, d_model)
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max_len, dropout=dropout)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

    def forward(self, x):   # x: [B, T, F]
        B, T, F = x.shape
        x = self.in_norm(x)            # 입력 정규화
        h = self.proj_in(x)          # [B,T,D]
        h = self.pos_enc(h)          # [B,T,D]
        #attn_mask = _causal_attn_mask(T, device=x.device) if self.causal else None
        attn_mask = torch.nn.Transformer.generate_square_subsequent_mask(T).to(x.device)
        h = self.encoder(h, mask=attn_mask)  # [B,T,D]
        h = self.norm(h)

        if self.pool == 'last':
            z = h[:, -1, :]     # 윈도우 마지막 시점의 표현
        else:
            z = h.mean(dim=1)   # 평균 풀링

        return self.head(z)
