# train_with_tqdm.py
import os, json
import torch
import torch.nn as nn
from tqdm.auto import tqdm, trange
from config import TrainCfg, ModelCfg
from utils.dataset_build import make_dataloaders, build_class_weights
from losses import MultiTaskLoss
from model.dual_cross import DualCrossTemporal

def train_with_tqdm(
    X_train, Y_type_train, Y_actor_train,
    X_val,   Y_type_val,   Y_actor_val,
    X_test,  Y_type_test,  Y_actor_test,
    event_type_map: dict,
    save_dir: str = "./checkpoints",
):
    os.makedirs(save_dir, exist_ok=True)

    train_cfg = TrainCfg()
    model_cfg = ModelCfg()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataloaders (훈련은 WeightedRandomSampler, 검증/테스트는 자연 분포)
    train_loader, val_loader, test_loader, _ = make_dataloaders(
        X_train, Y_type_train, Y_actor_train,
        X_val,   Y_type_val,   Y_actor_val,
        X_test,  Y_type_test,  Y_actor_test,
        batch_size=train_cfg.batch_size,
        n_types=model_cfg.n_types,
        use_weighted_sampler=train_cfg.use_weighted_sampler,
        sampler_max_weight_clip=train_cfg.sampler_max_weight_clip,
        in_memory=False,                  # RAM 여유 충분하면 True로 바꿔도 OK
        dtype=torch.float32,
        num_workers=2,
    )

    # 모델
    model = DualCrossTemporal(
        d_in=X_train.shape[-1],
        d_model=model_cfg.d_model,
        heads=model_cfg.heads,
        layers=model_cfg.layers,
        temp_layers=model_cfg.temp_layers,
        num_entities=model_cfg.num_entities,
        n_types=model_cfg.n_types,
        n_actors=model_cfg.n_actors,
    ).to(device)

    # 타입 손실 가중치 & 멀티태스크 손실
    type_w = build_class_weights(
        Y_type_train, model_cfg.n_types, device, max_clip=train_cfg.sampler_max_weight_clip
    ) if train_cfg.use_class_weighted_ce else None

    loss_fn = MultiTaskLoss(
        w_type=1.0,
        w_actor=train_cfg.lambda_actor,
        type_class_weights=type_w,
        no_event_class=train_cfg.noe_event_class,
        label_smoothing=0.1,
    )

    # 옵티마이저/스케줄러/AMP
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=train_cfg.plateau_factor, patience=train_cfg.plateau_patience,
        min_lr=train_cfg.plateau_min_lr
    ) if train_cfg.use_plateau_scheduler else None)
    scaler = torch.cuda.amp.GradScaler(enabled=train_cfg.use_amp)

    best_val = float("inf")
    patience = 0

    # 이벤트 타입 매핑 저장(선택)
    os.makedirs("./artifacts", exist_ok=True)
    with open("./artifacts/event_type_map.json", "w") as f:
        json.dump(event_type_map, f, ensure_ascii=False, indent=2)

    tqdm.write("\n========== START TRAINING ==========\n")
    for epoch in trange(1, train_cfg.num_epochs + 1, desc="Epochs"):
        # ----------------- Train -----------------
        model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Train {epoch}", leave=False, unit="batch")
        for step, (X, Yt, Ya) in enumerate(pbar, start=1):
            X, Yt, Ya = X.to(device, non_blocking=True), Yt.to(device), Ya.to(device)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=train_cfg.use_amp):
                pred_type, pred_actor = model(X)
                loss, logs = loss_fn((pred_type, pred_actor), (Yt, Ya))

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.clip_grad_norm)
            scaler.step(opt); scaler.update()

            running += logs["total_loss"]
            pbar.set_postfix(loss=f"{running/step:.4f}")

        avg_train = running / max(1, len(train_loader))

        # ----------------- Val -----------------
        model.eval()
        val_running = 0.0
        pbar_v = tqdm(val_loader, desc=f"Val   {epoch}", leave=False, unit="batch")
        with torch.no_grad():
            for step, (X, Yt, Ya) in enumerate(pbar_v, start=1):
                X, Yt, Ya = X.to(device), Yt.to(device), Ya.to(device)
                pred_type, pred_actor = model(X)
                vloss, vlogs = loss_fn((pred_type, pred_actor), (Yt, Ya))
                val_running += vlogs["total_loss"]
                pbar_v.set_postfix(loss=f"{val_running/step:.4f}")

        avg_val = val_running / max(1, len(val_loader))
        tqdm.write(f"Epoch {epoch:03d} | Train {avg_train:.4f} | Val {avg_val:.4f} | LR {opt.param_groups[0]['lr']:.2e}")

        if scheduler is not None:
            scheduler.step(avg_val)

        # ----------------- Early stop & save -----------------
        if avg_val < best_val:
            best_val = avg_val
            patience = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model.pth"))
            tqdm.write("  ✓ Saved best checkpoint")
        else:
            patience += 1
            tqdm.write(f"  ↳ No improvement. Patience {patience}/{train_cfg.patience}")
            if patience >= train_cfg.patience:
                tqdm.write("Early stopping.")
                break

    # ----------------- Test -----------------
    model.load_state_dict(torch.load(os.path.join(save_dir, "best_model.pth"), map_location=device))
    model.eval()
    test_running = 0.0
    pbar_t = tqdm(test_loader, desc="Test", leave=True, unit="batch")
    with torch.no_grad():
        for step, (X, Yt, Ya) in enumerate(pbar_t, start=1):
            X, Yt, Ya = X.to(device), Yt.to(device), Ya.to(device)
            pred_type, pred_actor = model(X)
            tloss, tlogs = loss_fn((pred_type, pred_actor), (Yt, Ya))
            test_running += tlogs["total_loss"]
            pbar_t.set_postfix(loss=f"{test_running/step:.4f}")

    tqdm.write(f"[Test] Avg loss {test_running / max(1, len(test_loader)):.4f}")

    return model

# ---------------------------
# 사용 예
# ---------------------------
# if __name__ == "__main__":
#     model = train_with_tqdm(
#         X_train, Y_type_train, Y_actor_train,
#         X_val,   Y_type_val,   Y_actor_val,
#         X_test,  Y_type_test,  Y_actor_test,
#         event_type_map
#     )
