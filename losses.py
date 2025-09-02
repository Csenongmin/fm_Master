
import torch
import torch.nn as nn

class MultiTaskLoss(nn.Module):
    def __init__(self, w_det=2.0, w_type=1.0, w_sub=1.0, w_from=1.0, w_to=1.0):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.ce  = nn.CrossEntropyLoss(ignore_index=-1)
        self.wd, self.wt, self.ws, self.wf, self.wto = w_det, w_type, w_sub, w_from, w_to

    def forward(self, out, tgt):
        det_logits, typ_logits, sub_logits, frm_logits, to_logits = out
        det_t, typ_t, sub_t, frm_t, to_t = tgt
        Ld = self.bce(det_logits, det_t)
        Lt = self.ce(typ_logits[typ_t>=0], typ_t[typ_t>=0]) if (typ_t>=0).any() else typ_logits.sum()*0
        Ls = self.ce(sub_logits[sub_t>=0], sub_t[sub_t>=0]) if (sub_t>=0).any() else sub_logits.sum()*0
        Lf = self.ce(frm_logits[frm_t>=0], frm_t[frm_t>=0]) if (frm_t>=0).any() else frm_logits.sum()*0
        Lto= self.ce(to_logits[to_t>=0], to_t[to_t>=0])   if (to_t>=0).any() else to_logits.sum()*0
        L = self.wd*Ld + self.wt*Lt + self.ws*Ls + self.wf*Lf + self.wto*Lto
        logs = {"det": float(Ld.item()), "type": float(Lt.item()), "sub": float(Ls.item()),
                "from": float(Lf.item()), "to": float(Lto.item()), "total": float(L.item())}
        return L, logs
