
import torch, argparse, json, pathlib
from torch.utils.data import Dataset, DataLoader
from model.dual_cross import DualCrossTemporal
from losses import MultiTaskLoss

class PrecomputedDataset(Dataset):
    def __init__(self, bundle):
        self.X = bundle["tokens"]
        self.det = bundle["det"]
        self.typ = bundle["type"]
        self.frm = bundle["from"]
        self.to  = bundle["to"]
        self.num_tokens = int(bundle["num_tokens"])
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return {"tokens": self.X[i].float(),
                "det": self.det[i].float(),
                "type": self.typ[i].long(),
                "from": self.frm[i].long(),
                "to": self.to[i].long()}

def collate(batch):
    T = max(b["tokens"].shape[0] for b in batch)
    B = len(batch); N = batch[0]["tokens"].shape[1]; F = batch[0]["tokens"].shape[2]
    tokens = torch.zeros(B,T,N,F)
    det = torch.zeros(B,T)
    typ = torch.full((B,T), -1, dtype=torch.long)
    frm = torch.full((B,T), -1, dtype=torch.long)
    to  = torch.full((B,T), -1, dtype=torch.long)
    for i,b in enumerate(batch):
        tlen = b["tokens"].shape[0]
        tokens[i,:tlen] = b["tokens"]
        det[i,:tlen] = b["det"]
        typ[i,:tlen] = b["type"]
        frm[i,:tlen] = b["from"]
        to[i,:tlen]  = b["to"]
    return tokens, det, typ, frm, to

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_pt", default="./artifacts/dataset.pt")
    ap.add_argument("--save_dir", default="./artifacts")
    ap.add_argument("--d_in", type=int, default=11)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--temp_layers", type=int, default=4)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--max_steps", type=int, default=200)
    args = ap.parse_args()

    bundle = torch.load(args.data_pt, map_location="cpu")
    ds = PrecomputedDataset(bundle)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    n_types = len(bundle["type_map"]); n_actors = 24
    model = DualCrossTemporal(num_tokens=ds.num_tokens, d_in=args.d_in, d_model=args.d_model,
                              heads=args.heads, layers=args.layers, temp_layers=args.temp_layers,
                              n_types=n_types, n_actors=n_actors)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lossf = MultiTaskLoss()

    pathlib.Path(args.save_dir).mkdir(parents=True, exist_ok=True)
    with open(f"{args.save_dir}/label_maps.json","w") as f:
        json.dump({"type_map": bundle["type_map"]}, f, ensure_ascii=False, indent=2)

    step = 0; model.train()
    for batch in dl:
        tokens, det, typ, frm, to = batch
        det_logits, typ_logits, frm_logits, to_logits = model(tokens)
        loss, logs = lossf((det_logits, typ_logits, frm_logits, to_logits),
                           (det, typ, frm, to))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % 10 == 0:
            print(f"[{step}] total={logs['total']:.4f} det={logs['det']:.4f} type={logs['type']:.4f} from={logs['from']:.4f} to={logs['to']:.4f}")
        if step >= args.max_steps: break
    print("Training finished.")

if __name__ == "__main__":
    main()
