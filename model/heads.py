
import torch
import torch.nn as nn

# 이벤트 '종류'를 예측하기 위한 Head
class EventTypeHead(nn.Module):
    def __init__(self, d_model, n_types):
        super().__init__()
        # LayerNorm으로 입력을 정규화한 후, Linear 레이어를 통과시켜 클래스 수만큼의 출력 생성
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, n_types)
        )

    def forward(self, h):
        # 입력 h의 shape: (Batch * Time, d_model) 또는 (Batch, Time, d_model)
        return self.net(h)

#  이벤트 '수행자'를 예측하기 위한 Head
class EventActorHead(nn.Module):
    def __init__(self, d_model, n_actors):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model), 
            nn.Linear(d_model, n_actors)
        )

    def forward(self, h):
        return self.net(h)