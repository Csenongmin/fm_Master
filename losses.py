
import torch
import torch.nn as nn
from typing import Optional

class MultiTaskLoss(nn.Module):
    """
    - pred_type: (B, n_types), target_type: (B,)
    - pred_actor: (B, n_actors), target_actor: (B,)
    - no_event_class: target_type이 이 값일 때 actor loss를 제외
    """
    def __init__(
            self, w_type: float = 1.0, w_actor: float = 1.0, 
            type_class_weights: Optional[torch.Tensor] = None,  
            no_event_class: int = 0,
            label_smoothing: float = 0.0,
    ):
        super().__init__()
        #타입 CE: 클래스 가중치/라벨 스무딩 지원
        self.ce_type = nn.CrossEntropyLoss(
            weight=type_class_weights,
            label_smoothing=label_smoothing if label_smoothing > 0 else 0.0,
            reduction="mean",
        )
        # 액터 CE: 마스킹 위해 per-sample loss로 받아서 평균
        self.ce_actor = nn.CrossEntropyLoss(reduction="none")

        self.w_type = float(w_type)
        self.w_actor = float(w_actor)
        self.no_event_class = int(no_event_class)
       
        

    def forward(self, predictions: tuple, targets: tuple):
        pred_type, pred_actor = predictions
        target_type, target_actor = targets  # (B,), dtype long
        # ----- Type loss -----
        loss_type = self.ce_type(pred_type, target_type)

        
         # ----- Actor loss (mask NoEvent) -----
        pos_mask = (target_type != self.no_event_class)  # (B,)
        if pos_mask.any():
            per_sample = self.ce_actor(pred_actor, target_actor)  # (B,)
            loss_actor = per_sample[pos_mask].mean()
        else:
            # zero scalar on the right device
            loss_actor = pred_actor.sum() * 0.0

        total_loss = self.w_type * loss_type + self.w_actor * loss_actor

        logs = {
            "loss_type": float(loss_type.detach().item()),
            "loss_actor": float(loss_actor.detach().item()),
            "total_loss": float(total_loss.detach().item()),
            "actor_pos_frac": float(pos_mask.float().mean().item()),
        }
        return total_loss, logs