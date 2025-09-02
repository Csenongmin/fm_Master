
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

def central_diff(a: np.ndarray, dt: float) -> np.ndarray:
    if len(a) < 3: return np.zeros_like(a)
    v = np.zeros_like(a)
    v[1:-1] = (a[2:]-a[:-2])/(2*dt)
    v[0] = (a[1]-a[0])/dt
    v[-1] = (a[-1]-a[-2])/dt
    return v

@dataclass
class ParsedTracking:
    time_s: np.ndarray
    home_xy: np.ndarray   # [T,11,2]
    away_xy: np.ndarray   # [T,11,2]
    ball_xy: Optional[np.ndarray]  # [T,2] or None
    period_oh: np.ndarray # [T,2]

def parse_tracking_clean(home_csv: str, away_csv: str) -> ParsedTracking:
    dh = pd.read_csv(home_csv, low_memory=False)
    da = pd.read_csv(away_csv, low_memory=False)
    # time
    time_col = None
    for c in ["Time [s]", "time_s", "time", "t"]:
        if c in dh.columns:
            time_col = c; break
    if time_col is None:
        time = np.arange(len(dh))*0.04
    else:
        time = pd.to_numeric(dh[time_col], errors='coerce').to_numpy()
        idx = np.arange(len(time))
        mask = np.isfinite(time)
        if not mask.all():
            time = np.interp(idx, idx[mask], time[mask])
    # period
    per_col = None
    for c in ["Period", "period"]:
        if c in dh.columns: per_col = c; break
    if per_col is not None:
        per = pd.to_numeric(dh[per_col], errors='coerce').fillna(method="ffill").fillna(1).to_numpy()
        per = np.where(per >= 2, 1, 0)
    else:
        per = np.zeros(len(dh), dtype=np.int64)
    period_oh = np.stack([1-per, per], axis=-1).astype(np.float32)

    def stack_team(df: pd.DataFrame, prefix: str) -> np.ndarray:
        xs, ys = [], []
        for i in range(1, 12):
            xs.append(pd.to_numeric(df[f"{prefix}_{i}_x"], errors='coerce').to_numpy())
            ys.append(pd.to_numeric(df[f"{prefix}_{i}_y"], errors='coerce').to_numpy())
        X = np.stack(xs, axis=1); Y = np.stack(ys, axis=1)
        return np.stack([X,Y], axis=-1).astype(np.float32)

    home_xy = stack_team(dh, "Home")
    away_xy = stack_team(da, "Away")

    ball_xy = None
    for bx,by in [("Ball_x","Ball_y"), ("ball_x","ball_y")]:
        if bx in dh.columns and by in dh.columns:
            bxv = pd.to_numeric(dh[bx], errors='coerce').to_numpy()
            byv = pd.to_numeric(dh[by], errors='coerce').to_numpy()
            ball_xy = np.stack([bxv,byv], axis=-1).astype(np.float32)
            break
        if bx in da.columns and by in da.columns:
            bxv = pd.to_numeric(da[bx], errors='coerce').to_numpy()
            byv = pd.to_numeric(da[by], errors='coerce').to_numpy()
            ball_xy = np.stack([bxv,byv], axis=-1).astype(np.float32)
            break

    return ParsedTracking(time_s=time.astype(np.float32),
                          home_xy=home_xy, away_xy=away_xy, ball_xy=ball_xy, period_oh=period_oh)

@dataclass
class ParsedEvents:
    df: pd.DataFrame
    type_map: Dict[str,int]
    subtype_map: Dict[str,int]

def parse_events_clean(events_csv: str) -> ParsedEvents:
    de = pd.read_csv(events_csv, low_memory=False)
    rename = {
        'Start Time [s]':'start_time_s',
        'End Time [s]':'end_time_s',
        'Start Frame':'start_frame',
        'End Frame':'end_frame',
        'Start X':'start_x',
        'Start Y':'start_y',
        'End X':'end_x',
        'End Y':'end_y',
        'Type':'type',
        'Subtype':'subtype',
        'Team':'team',
        'From':'from_actor',
        'To':'to_actor',
        'Period':'period'
    }
    for k,v in list(rename.items()):
        if k in de.columns: de.rename(columns={k:v}, inplace=True)
    for c in ['start_time_s','end_time_s','start_x','start_y','end_x','end_y','period']:
        if c in de.columns: de[c] = pd.to_numeric(de[c], errors='coerce')
    types = sorted(de['type'].fillna('Unknown').astype(str).unique().tolist())
    subtypes = sorted(de['subtype'].fillna('Unknown').astype(str).unique().tolist()) if 'subtype' in de.columns else ['Unknown']
    return ParsedEvents(df=de, type_map={t:i for i,t in enumerate(types)}, subtype_map={t:i for i,t in enumerate(subtypes)})

class SoccerDataset(Dataset):
    def __init__(self, tr: ParsedTracking, ev: ParsedEvents, window_s=3.2, stride_s=0.8, det_tol_s=0.5):
        self.tr = tr; self.ev = ev
        t = tr.time_s
        self.dt = float(np.median(np.diff(t))) if len(t)>1 else 0.04
        self.win = int(round(window_s/self.dt))
        self.step = int(round(stride_s/self.dt))
        self.det_tol = det_tol_s
        self.ball_present = (tr.ball_xy is not None)
        self.NTOK = 23 if self.ball_present else 22
        # velocities
        self.home_v = np.stack([
            np.stack([central_diff(tr.home_xy[:,i,0], self.dt),
                      central_diff(tr.home_xy[:,i,1], self.dt)], axis=-1) for i in range(11)
        ], axis=1)
        self.away_v = np.stack([
            np.stack([central_diff(tr.away_xy[:,i,0], self.dt),
                      central_diff(tr.away_xy[:,i,1], self.dt)], axis=-1) for i in range(11)
        ], axis=1)
        if self.ball_present:
            self.ball_v = np.stack([central_diff(tr.ball_xy[:,0], self.dt),
                                    central_diff(tr.ball_xy[:,1], self.dt)], axis=-1)
        else:
            self.ball_v = None
        self.indices = [(s, s+self.win) for s in range(0, len(t)-self.win, self.step)]
        self.type_map = ev.type_map
        self.subtype_map = ev.subtype_map

    def __len__(self): return len(self.indices)

    @staticmethod
    def _actor_id(team: str, idx_str: str) -> int:
        try:
            p = int(idx_str)
        except Exception:
            return -1
        if p < 1 or p > 11: return -1
        base = 0 if str(team).lower().startswith('home') else 11 if str(team).lower().startswith('away') else None
        if base is None: return -1
        return base + (p-1)

    def _labels_for_window(self, t0: float, T: int):
        times = t0 + np.arange(T)*self.dt
        det = np.zeros(T, np.float32)
        type_id = -np.ones(T, np.int64)
        subtype_id = -np.ones(T, np.int64)
        from_id = -np.ones(T, np.int64)
        to_id = -np.ones(T, np.int64)

        df = self.ev.df
        if 'start_time_s' not in df.columns:
            return det, type_id, subtype_id, from_id, to_id
        ets = df['start_time_s'].to_numpy()
        ety = df['type'].fillna('Unknown').astype(str).to_numpy()
        esub = df['subtype'].fillna('Unknown').astype(str).to_numpy() if 'subtype' in df.columns else []
        teamcol = df['team'] if 'team' in df.columns else None
        fromcol = df['from_actor'] if 'from_actor' in df.columns else None
        tocol = df['to_actor'] if 'to_actor' in df.columns else None

        for i in range(len(df)):
            e_time = ets[i]
            if np.isnan(e_time) or (e_time < times[0]-self.det_tol) or (e_time > times[-1]+self.det_tol):
                continue
            k = int(np.argmin(np.abs(times - e_time)))
            if abs(times[k]-e_time) <= self.det_tol:
                det[k] = 1.0
                type_id[k] = self.type_map.get(ety[i], -1)
                if len(esub)>0:
                    subtype_id[k] = self.subtype_map.get(esub[i], -1)
                if teamcol is not None and fromcol is not None:
                    from_id[k] = self._actor_id(teamcol.iloc[i], str(fromcol.iloc[i]))
                if teamcol is not None and tocol is not None:
                    to_id[k] = self._actor_id(teamcol.iloc[i], str(tocol.iloc[i]))
        return det, type_id, subtype_id, from_id, to_id

    def __getitem__(self, idx: int):
        s,e = self.indices[idx]
        T = e-s
        Hxy, Axy = self.tr.home_xy[s:e], self.tr.away_xy[s:e]
        Hv, Av = self.home_v[s:e], self.away_v[s:e]
        per = self.tr.period_oh[s:e]  # [T,2]

        Htok = np.concatenate([Hxy, Hv], axis=-1)  # [T,11,4]
        Atok = np.concatenate([Axy, Av], axis=-1)  # [T,11,4]
        if self.ball_present:
            Bxy = self.tr.ball_xy[s:e][:,None,:]       # [T,1,2]
            Bv  = self.ball_v[s:e][:,None,:]           # [T,1,2]
            Btok = np.concatenate([Bxy, Bv], axis=-1)  # [T,1,4]

        he = np.tile(np.array([1,0,0], np.float32), (T,11,1))
        ae = np.tile(np.array([0,1,0], np.float32), (T,11,1))
        if self.ball_present:
            be = np.tile(np.array([0,0,1], np.float32), (T,1,1))

        Htok = np.concatenate([Htok, he, np.tile(per[:,None,:], (1,11,1))], axis=-1)  # [T,11,9]
        Atok = np.concatenate([Atok, ae, np.tile(per[:,None,:], (1,11,1))], axis=-1)  # [T,11,9]
        tokens = np.concatenate([Htok, Atok], axis=1)                                  # [T,22,9]
        if self.ball_present:
            Btok = np.concatenate([Btok, be, np.tile(per[:,None,:], (1,1,1))], axis=-1)    # [T,1,9]
            tokens = np.concatenate([tokens, Btok], axis=1)                               # [T,23,9]

        det, typ, sub, frm, to = self._labels_for_window(self.tr.time_s[s], T)
        return {
            "tokens": torch.from_numpy(tokens).float(),
            "det": torch.from_numpy(det).float(),
            "type": torch.from_numpy(typ).long(),
            "subtype": torch.from_numpy(sub).long(),
            "from_id": torch.from_numpy(frm).long(),
            "to_id": torch.from_numpy(to).long(),
            "dt": float(self.dt),
        }
