
from dataclasses import dataclass

@dataclass
class DataCfg:
    window_s: float = 3.2
    stride_s: float = 0.8
    det_tol_s: float = 0.5
    fps_fallback: float = 25.0

@dataclass
class ModelCfg:
    d_in: int = 11          # [x,y,vx,vy, team(2), is_ball, period(2), present, sub]
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    temp_layers: int = 4

@dataclass
class TrainCfg:
    batch_size: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_steps: int = 200
    wd_det: float = 2.0
    wd_type: float = 1.0
    wd_sub: float = 1.0
    wd_from: float = 1.0
    wd_to: float = 1.0
