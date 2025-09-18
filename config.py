
from dataclasses import dataclass

@dataclass
class DataCfg:
    window_s: float = 5.0
    stride_s: float = 1.0
    fps_fallback: float = 25.0
    stride_frames: int = 1

@dataclass
class ModelCfg:
    d_in: int = 12
    d_model: int = 128
    heads: int = 4
    layers: int = 2
    temp_layers: int = 4
    num_entities: int = 23      
    n_types: int = 6           
    n_actors: int = 24     
    # 프레임 내 어텐션 후 시간 인코더로 넘길 때 미래 누설 방지(권장: True)
    use_causal_temporal: bool = True

    # 선택: 프레임 내 어텐션 층 수(캔버스 코드에 frame_layers가 있다면 사용)
    frame_layers: int = 1



@dataclass
class TrainCfg:
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-4
    # 에폭/얼리스탑
    num_epochs: int = 50
    patience: int = 5
    # 손실 가중치(멀티태스크): 새 스크립트에서는 lambda_actor 사용
    lambda_actor: float = 1.0
    wd_type: float = 1.0        # 이벤트 타입 예측 손실 가중치
    wd_actor: float = 1.0       # 이벤트 수행자 예측 손실 가중치
    # 샘플러/손실 관련 옵션
    use_weighted_sampler: bool = True          # 훈련에만 WeightedRandomSampler 사용
    sampler_max_weight_clip: float = 10.0      # 클래스 가중 클램프 상한
    use_class_weighted_ce: bool = False         # 타입 CE에 클래스 가중치 사용

    # AMP / 그래드 클립
    use_amp: bool = True
    clip_grad_norm: float = 1.0

    # 스케줄러(Plateau에서 LR 감소)
    use_plateau_scheduler: bool = True
    plateau_factor: float = 0.5
    plateau_patience: int = 2
    plateau_min_lr: float = 1e-6

    # 재현성
    seed: int = 42

    # 라벨 인덱스 지정(안전)
    noe_event_class: int = 0    # NoEvent 클래스 인덱스(반드시 0로 유지)
