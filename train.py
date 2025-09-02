
import json, pathlib
import torch
from torch.utils.data import DataLoader
from .config import DataCfg, ModelCfg, TrainCfg
from .data import parse_tracking_clean, parse_events_clean, SoccerDataset
from .model import DualCrossTemporal
from .losses import MultiTaskLoss

def collate(batch):
    T = max(b["tokens"].shape[0] for b in batch)
    B = len(batch); N = batch[0]["tokens"].shape[1]; F = batch[0]["tokens"].shape[2]
    tokens = torch.zeros(B,T,N,F)
    det = torch.zeros(B,T)
    typ = torch.full((B,T), -1, dtype=torch.long)
    sub = torch.full((B,T), -1, dtype=torch.long)
    frm = torch.full((B,T), -1, dtype=torch.long)
    to  = torch.full((B,T), -1, dtype=torch.long)
    for i,b in enumerate(batch):
        tlen = b["tokens"].shape[0]
        tokens[i,:tlen] = b["tokens"]
        det[i,:tlen] = b["det"]
        typ[i,:tlen] = b["type"]
        sub[i,:tlen] = b["subtype"]
        frm[i,:tlen] = b["from_id"]
        to[i,:tlen]  = b["to_id"]
    return tokens, det, typ, sub, frm, to

def train_loop(home_csv, away_csv, events_csv,
               data_cfg=DataCfg(), model_cfg=ModelCfg(), train_cfg=TrainCfg(),
               save_dir="/mnt/data/soccer_event/artifacts"):
    tr = parse_tracking_clean(home_csv, away_csv)
    ev = parse_events_clean(events_csv)
    ds = SoccerDataset(tr, ev, window_s=data_cfg.window_s, stride_s=data_cfg.stride_s, det_tol_s=data_cfg.det_tol_s)
    dl = DataLoader(ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate)
    num_tokens = ds.NTOK
    n_types = len(ev.type_map); n_subtypes = len(ev.subtype_map)
    n_actors = 24
    model = DualCrossTemporal(num_tokens=num_tokens, d_in=model_cfg.d_in, d_model=model_cfg.d_model,
                              heads=model_cfg.heads, layers=model_cfg.layers, temp_layers=model_cfg.temp_layers,
                              n_types=n_types, n_subtypes=n_subtypes, n_actors=n_actors)
    opt = torch.optim.AdamW(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    lossf = MultiTaskLoss(train_cfg.wd_det, train_cfg.wd_type, train_cfg.wd_sub, train_cfg.wd_from, train_cfg.wd_to)

    pathlib.Path(save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{save_dir}/label_maps.json","w") as f:
        json.dump({"type_map": ev.type_map, "subtype_map": ev.subtype_map}, f, ensure_ascii=False, indent=2)

    model.train()
    step = 0
    for batch in dl:
        tokens, det, typ, sub, frm, to = batch
        det_logits, typ_logits, sub_logits, frm_logits, to_logits = model(tokens)
        loss, logs = lossf((det_logits, typ_logits, sub_logits, frm_logits, to_logits),
                           (det, typ, sub, frm, to))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 10 == 0:
            print(f"[{step}] total={logs['total']:.4f} det={logs['det']:.4f} type={logs['type']:.4f} sub={logs['sub']:.4f} from={logs['from']:.4f} to={logs['to']:.4f}")
        if step >= train_cfg.max_steps: break
    print("Training finished.")

if __name__ == "__main__":
    home = "/mnt/data/tracking_home.csv"
    away = "/mnt/data/tracking_away.csv"
    events = "/mnt/data/events.csv"
    train_loop(home, away, events)
