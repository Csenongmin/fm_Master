
import torch
import torch.nn as nn
from .heads import DetHead, ClassHead, ActorHead

class CrossBlock(nn.Module):
    def __init__(self, d, nhead):
        super().__init__()
        self.enc_h = nn.TransformerEncoderLayer(d, nhead, 4*d, batch_first=True)
        self.enc_a = nn.TransformerEncoderLayer(d, nhead, 4*d, batch_first=True)
        self.cross_h2a = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_a2h = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln_h = nn.LayerNorm(d)
        self.ln_a = nn.LayerNorm(d)
    def forward(self, H, A):
        H = self.enc_h(H); A = self.enc_a(A)
        H2,_ = self.cross_h2a(self.ln_h(H), self.ln_a(A), self.ln_a(A))
        A2,_ = self.cross_a2h(self.ln_a(A), self.ln_h(H), self.ln_h(H))
        return H+H2, A+A2

class DualCrossTemporal(nn.Module):
    def __init__(self, num_tokens: int, d_in=9, d_model=128, heads=4, layers=2, temp_layers=4,
                 n_types=8, n_subtypes=16, n_actors=24):
        super().__init__()
        self.embed = nn.Linear(d_in, d_model)
        self.pos_tok = nn.Parameter(torch.randn(1, num_tokens, d_model))
        self.pos_step = nn.Parameter(torch.randn(1,1,d_model))
        self.blocks = nn.ModuleList([CrossBlock(d_model, heads) for _ in range(layers)])
        self.ball_gate = nn.Linear(d_model, d_model)
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads*2, 4*d_model, batch_first=True),
            num_layers=temp_layers
        )
        self.det_head = DetHead(d_model)
        self.type_head = ClassHead(d_model, n_types)
        self.sub_head  = ClassHead(d_model, n_subtypes)
        self.from_head = ActorHead(d_model, n_actors)
        self.to_head   = ActorHead(d_model, n_actors)

    def forward(self, tokens):
        B,T,N,F = tokens.shape
        x = self.embed(tokens) + self.pos_tok
        has_ball = (N == 23)
        H, A = x[:,:,:11,:], x[:,:,11:22,:]
        Btok = x[:,:,22:23,:] if has_ball else None
        if has_ball:
            H = H + self.ball_gate(Btok)
            A = A + self.ball_gate(Btok)
        H = H.reshape(B*T, 11, -1); A = A.reshape(B*T, 11, -1)
        for blk in self.blocks:
            H, A = blk(H, A)
        H = H.reshape(B,T,11,-1); A = A.reshape(B,T,11,-1)
        frame = torch.cat([H,A], dim=2)
        if has_ball:
            frame = torch.cat([frame, Btok], dim=2)
        frame = frame.mean(dim=2)
        pe = torch.cumsum(self.pos_step.repeat(B,T,1), dim=1)
        h = self.temporal(frame + pe)
        det = self.det_head(h)
        typ = self.type_head(h)
        sub = self.sub_head(h)
        frm = self.from_head(h)
        to  = self.to_head(h)
        return det, typ, sub, frm, to
