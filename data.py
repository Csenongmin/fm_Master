
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass
from typing import Dict, Optional, List, Tuple

def central_diff(a: np.ndarray, dt: float) -> np.ndarray:
    if len(a) < 3: return np.zeros_like(a)
    v = np.zeros_like(a)
    v[1:-1] = (a[2:]-a[:-2])/(2*dt)
    v[0] = (a[1]-a[0])/dt
    v[-1] = (a[-1]-a[-2])/dt
    return v

def savgol_like_smooth(x: np.ndarray, k: int = 3) -> np.ndarray:
    if k <= 1: return x
    pad = k//2
    xp = np.pad(x, (pad, pad), mode='edge')
    kernel = np.ones(k, dtype=np.float32)/k
    return np.convolve(xp, kernel, mode='valid')

@dataclass
class ParsedTracking:
    time_s: np.ndarray
    home_xy: np.ndarray
    away_xy: np.ndarray
    ball_xy: Optional[np.ndarray]
    period_oh: np.ndarray
    home_seg_ids: np.ndarray
    away_seg_ids: np.ndarray
    home_sub_flag: np.ndarray
    away_sub_flag: np.ndarray

def _stack_team_xy(df: pd.DataFrame, prefix: str) -> np.ndarray:
    xs, ys = [], []
    for i in range(1, 12):
        xs.append(pd.to_numeric(df[f"{prefix}_{i}_x"], errors='coerce').to_numpy())
        ys.append(pd.to_numeric(df[f"{prefix}_{i}_y"], errors='coerce').to_numpy())
    X = np.stack(xs, axis=1); Y = np.stack(ys, axis=1)
    return np.stack([X,Y], axis=-1).astype(np.float32)

def _make_period_oh(df: pd.DataFrame, T: int) -> np.ndarray:
    per_col = None
    for c in ["Period", "period"]:
        if c in df.columns: per_col = c; break
    if per_col is not None:
        per = pd.to_numeric(df[per_col], errors='coerce').fillna(method="ffill").fillna(1).to_numpy()
        per = np.where(per >= 2, 1, 0).astype(np.int64)
        per = per[:T] if len(per)>=T else np.pad(per, (0, T-len(per)), constant_values=per[-1] if len(per)>0 else 0)
    else:
        per = np.zeros(T, dtype=np.int64)
    return np.stack([1-per, per], axis=-1).astype(np.float32)

def _parse_time(df: pd.DataFrame) -> np.ndarray:
    time_col = None
    for c in ["Time [s]", "time_s", "time", "t"]:
        if c in df.columns:
            time_col = c; break
    if time_col is None:
        return None
    time = pd.to_numeric(df[time_col], errors='coerce').to_numpy()
    idx = np.arange(len(time))
    mask = np.isfinite(time)
    if not mask.all():
        time = np.interp(idx, idx[mask], time[mask])
    return time.astype(np.float32)

def _read_ball_xy(dh: pd.DataFrame, da: pd.DataFrame) -> Optional[np.ndarray]:
    for bx,by in [("Ball_x","Ball_y"), ("ball_x","ball_y")]:
        if bx in dh.columns and by in dh.columns:
            bxv = pd.to_numeric(dh[bx], errors='coerce').to_numpy()
            byv = pd.to_numeric(dh[by], errors='coerce').to_numpy()
            return np.stack([bxv,byv], axis=-1).astype(np.float32)
        if bx in da.columns and by in da.columns:
            bxv = pd.to_numeric(da[bx], errors='coerce').to_numpy()
            byv = pd.to_numeric(da[by], errors='coerce').to_numpy()
            return np.stack([bxv,byv], axis=-1).astype(np.float32)
    return None

def _collect_substitutions(de: pd.DataFrame) -> Dict[str, List[Tuple[float, int, int]]]:
    subs = {'home': [], 'away': []}
    if de is None: return subs
    if 'Type' in de.columns: typ_col = 'Type'
    elif 'type' in de.columns: typ_col = 'type'
    else: return subs

    colmap = {}
    for k in ['Team','team','Start Time [s]','start_time_s','From','from_actor','To','to_actor']:
        if k in de.columns: colmap[k] = k
    def get(name_list):
        for n in name_list:
            if n in colmap: return colmap[n]
        return None

    teamc = get(['Team','team'])
    timec = get(['Start Time [s]','start_time_s'])
    fromc = get(['From','from_actor'])
    toc   = get(['To','to_actor'])

    if teamc is None or timec is None or fromc is None:
        return subs

    def is_sub(v):
        if isinstance(v, str):
            vs = v.strip().lower()
            return ('substitution' in vs) or ('교체' in vs)
        return False

    df = de.copy()
    df['_is_sub'] = df[typ_col].apply(is_sub)
    df = df[df['_is_sub'] == True]
    if len(df) == 0:
        return subs

    for _, r in df.iterrows():
        team = str(r[teamc]).strip().lower()
        t = float(pd.to_numeric(r[timec], errors='coerce'))
        outn = str(r[fromc]).strip() if fromc in r else None
        try:
            outn = int(outn)
        except:
            continue
        out_slot = outn if 1 <= outn <= 11 else None
        if out_slot is None: 
            continue
        subs['home' if team.startswith('home') else 'away'].append((t, out_slot, None))
    return subs

def _stitch_slots(xy: np.ndarray, subs, time: np.ndarray, smooth_k: int = 3):
    T = xy.shape[0]
    seg_ids = np.zeros((T,11), np.int32)
    sub_flag = np.zeros((T,11), np.float32)
    if subs is None or len(subs)==0:
        return xy, seg_ids, sub_flag
    slot_sub_frames = {j: [] for j in range(1,12)}
    for t_sub, out_slot, _ in subs:
        if t_sub is None or out_slot is None: 
            continue
        k = int(np.argmin(np.abs(time - t_sub)))
        slot_sub_frames[out_slot].append(k)
    stitched = xy.copy()
    for s in range(1,12):
        frames = sorted(set(slot_sub_frames[s]))
        seg = 0
        for k in frames:
            if k <= 0 or k >= T: 
                continue
            seg_ids[k:, s-1] = seg + 1
            sub_flag[k, s-1] = 1.0
            seg += 1
        for dim in range(2):
            arr = stitched[:, s-1, dim]
            if np.isnan(arr).all(): 
                continue
            stitched[:, s-1, dim] = savgol_like_smooth(arr, k=smooth_k)
    return stitched, seg_ids, sub_flag

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

def parse_tracking_clean(home_csv: str, away_csv: str, events_df: Optional[pd.DataFrame] = None) -> ParsedTracking:
    dh = pd.read_csv(home_csv, low_memory=False)
    da = pd.read_csv(away_csv, low_memory=False)
    time = _parse_time(dh)
    if time is None:
        time = np.arange(len(dh))*0.04
    T = len(time)
    home_xy = _stack_team_xy(dh, "Home")
    away_xy = _stack_team_xy(da, "Away")
    ball_xy = _read_ball_xy(dh, da)
    period_oh = _make_period_oh(dh, T)
    subs = _collect_substitutions(events_df) if events_df is not None else {'home': [], 'away': []}
    home_xy_st, home_seg, home_sub = _stitch_slots(home_xy, subs.get('home', []), time)
    away_xy_st, away_seg, away_sub = _stitch_slots(away_xy, subs.get('away', []), time)
    return ParsedTracking(time_s=time.astype(np.float32),
                          home_xy=home_xy_st, away_xy=away_xy_st, ball_xy=ball_xy, period_oh=period_oh,
                          home_seg_ids=home_seg, away_seg_ids=away_seg,
                          home_sub_flag=home_sub, away_sub_flag=away_sub)

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
        esub = df['subtype'].fillna('Unknown').astype(str).to_numpy() if 'subtype' in df.columns else np.array([])
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
        per = self.tr.period_oh[s:e]

        def present_mask(xy):
            return (~np.isnan(xy).any(axis=-1)).astype(np.float32)

        home_present = present_mask(Hxy)
        away_present = present_mask(Axy)

        home_sub = self.tr.home_sub_flag[s:e]
        away_sub = self.tr.away_sub_flag[s:e]

        Htok_core = np.concatenate([Hxy, Hv], axis=-1)
        Atok_core = np.concatenate([Axy, Av], axis=-1)

        Hextra = np.concatenate([
            np.tile(np.array([1,0,0], np.float32), (T,11,1)),
            np.tile(per[:,None,:], (1,11,1)),
            home_present[...,None],
            home_sub[...,None],
        ], axis=-1)

        Aextra = np.concatenate([
            np.tile(np.array([0,1,0], np.float32), (T,11,1)),
            np.tile(per[:,None,:], (1,11,1)),
            away_present[...,None],
            away_sub[...,None],
        ], axis=-1)

        Htok = np.concatenate([Htok_core, Hextra], axis=-1)  # [T,11,11]
        Atok = np.concatenate([Atok_core, Aextra], axis=-1)  # [T,11,11]
        tokens = np.concatenate([Htok, Atok], axis=1)        # [T,22,11]

        if self.ball_present:
            Bxy = self.tr.ball_xy[s:e][:,None,:]
            Bv  = self.ball_v[s:e][:,None,:]
            ball_present = (~np.isnan(Bxy).any(axis=-1)).astype(np.float32)
            Bextra = np.concatenate([
                np.tile(np.array([0,0,1], np.float32), (T,1,1)),
                np.tile(per[:,None,:], (1,1,1)),
                ball_present[...,None],
                np.zeros((T,1,1), np.float32),
            ], axis=-1)
            Btok = np.concatenate([np.concatenate([Bxy, Bv], axis=-1), Bextra], axis=-1)  # [T,1,11]
            tokens = np.concatenate([tokens, Btok], axis=1)  # [T,23,11]

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
