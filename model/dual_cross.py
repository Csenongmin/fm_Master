
import torch
import torch.nn as nn
from .heads import EventActorHead, EventTypeHead

class CrossBlock(nn.Module):
    def __init__(self, d: int, nhead: int):
        super().__init__()
        self.enc_h = nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True)
        self.enc_a = nn.TransformerEncoderLayer(d, nhead, 4 * d, batch_first=True)
        self.cross_h2a = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.cross_a2h = nn.MultiheadAttention(d, nhead, batch_first=True)
        self.ln_h_q = nn.LayerNorm(d)
        self.ln_a_kv = nn.LayerNorm(d)
        self.ln_a_q = nn.LayerNorm(d)
        self.ln_h_kv = nn.LayerNorm(d)
        # 출력 정규화(안정성)
        self.ln_h_out = nn.LayerNorm(d)
        self.ln_a_out = nn.LayerNorm(d)
        self.dropout = nn.Dropout(0.1)


    def forward(self, H: torch.Tensor, A: torch.Tensor):
        # Self-encoding (intra-team)
        H = self.enc_h(H)
        A = self.enc_a(A)
        # Cross attention: H attends to A, A attends to H
        H2, _ = self.cross_h2a(self.ln_h_q(H), self.ln_a_kv(A), self.ln_a_kv(A))
        A2, _ = self.cross_a2h(self.ln_a_q(A), self.ln_h_kv(H), self.ln_h_kv(H))
        H = self.ln_h_out(H + self.dropout(H2))
        A = self.ln_a_out(A + self.dropout(A2))
        return H, A

class DualCrossTemporal(nn.Module):
    def __init__(self, 
                 d_in=9, 
                 d_model=128, 
                 heads=4, 
                 layers=2, 
                 temp_layers=4,
                 num_entities=23,
                 n_types=7,
                 n_actors=24,
                 frame_layers: int = 1,
                 use_causal_temporal: bool = False,
                 ):
        super().__init__()
        
        self.d_model = d_model
        self.num_entities = num_entities
        self.embed = nn.Linear(d_in, d_model)
        self.pos_tok = nn.Parameter(torch.randn(1, num_entities, d_model)) # (1,N,d)
        self.pos_step = nn.Parameter(torch.randn(1,1,d_model)) # (1,1,d)
        self.cls_token = nn.Parameter(torch.randn(1, 1, 1, d_model)) # (1,1,1,d)

        self.blocks = nn.ModuleList([CrossBlock(d_model, heads) for _ in range(layers)])
        self.ball_gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

        self.frame_enc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads, 4 * d_model, batch_first=True),
            num_layers=frame_layers,
        )
        self.use_causal_temporal = use_causal_temporal
        self.temporal = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, heads*2, 4*d_model, batch_first=True),
            num_layers=temp_layers
        )
        
        
        self.event_type_head = EventTypeHead(d_model, n_types)
        self.event_actor_head = EventActorHead(d_model, n_actors)

    def _split_entities(self, x: torch.Tensor):
        """CLS가 prepend된 x(B,T,1+N,d)에서 ball/H/A를 동적으로 슬라이싱."""
        B, T, NP1, D = x.shape
        N = self.num_entities
        assert NP1 == N + 1, f"Expected 1+num_entities tokens, got {NP1} vs {N+1}"
        # 인덱스: [0]=CLS, [1]=ball, [2:2+K]=home, [2+K:2+2K]=away
        K = (N - 1) // 2
        cls_out = x[:, :, 0:1, :]
        Btok = x[:, :, 1:2, :]
        H = x[:, :, 2 : 2 + K, :]
        A = x[:, :, 2 + K : 2 + 2 * K, :]
        return cls_out, Btok, H, A

    def forward(self, tokens):
        """
        tokens: (B,T,N,F)
        return: (pred_type(B,n_types), pred_actor(B,n_actors))
        """
        B, T, N, F = tokens.shape
        assert N == self.num_entities, f"N({N}) must equal num_entities({self.num_entities})"

        x = self.embed(tokens) + self.pos_tok # (B,T,N,d)
        cls_tokens = self.cls_token.expand(B, T, -1, -1) # (B,T,1,d)
        x = torch.cat((cls_tokens, x), dim=2) # (B,T,1+N,d)
        # --- 엔티티 분해 ---
        cls_out, Btok, H, A = self._split_entities(x)
        # 볼 게이트를 홈/어웨이에 주입
        gate = self.ball_gate(Btok) # (B,T,1,d)
        H = H + gate
        A = A + gate
        # --- 팀별/상호 어텐션 ---
        BTh = B * T
        H = H.reshape(BTh, -1, self.d_model)
        A = A.reshape(BTh, -1, self.d_model)
        for blk in self.blocks:
            H, A = blk(H, A)
        H = H.reshape(B, T, -1, self.d_model)
        A = A.reshape(B, T, -1, self.d_model)

        # --- 프레임 내 집약: [CLS, Ball, Players]에 대해 self-attn을 1~L층 수행하여 CLS 업데이트 ---
        processed_players = torch.cat([H, A], dim=2) # (B,T,2K,d)
        frame_tokens = torch.cat([cls_out, Btok, processed_players], dim=2) # (B,T,1+1+2K,d)
        L = frame_tokens.size(2)
        ft = frame_tokens.view(BTh, L, self.d_model)
        ft = self.frame_enc(ft) # (B*T, L, d)
        frame_representation = ft[:, 0, :].view(B, T, self.d_model) # 업데이트된 CLS

        # --- 시간 포지션 및(선택) causal mask ---
        pe = torch.cumsum(self.pos_step.repeat(B, T, 1), dim=1) # (B,T,d)
        src = frame_representation + pe


        if self.use_causal_temporal:
        # 상삼각(미래 차단) 마스크: True=block
            causal_mask = torch.triu(torch.ones(T, T, device=src.device, dtype=torch.bool), diagonal=1)
            h = self.temporal(src, mask=causal_mask)
        else:
            h = self.temporal(src)


        # --- 마지막 스텝으로 예측 ---
        last_step_h = h[:, -1, :] # (B,d)
        pred_type = self.event_type_head(last_step_h)
        pred_actor = self.event_actor_head(last_step_h)
        return pred_type, pred_actor


