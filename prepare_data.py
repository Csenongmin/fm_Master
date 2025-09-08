
# (shortened header for brevity) Dynamic player-id detection & slot mapping
import json, pathlib, argparse, re
import numpy as np, pandas as pd, torch
from config import DataCfg

def central_diff(a, dt):
    if len(a) < 3: return np.zeros_like(a)
    v = np.zeros_like(a); v[1:-1] = (a[2:]-a[:-2])/(2*dt); v[0]=(a[1]-a[0])/dt; v[-1]=(a[-1]-a[-2])/dt
    return v

def movavg(x, k=3):
    if k<=1: return x
    pad=k//2; xp=np.pad(x,(pad,pad),mode='edge'); ker=np.ones(k,dtype=np.float32)/k
    return np.convolve(xp,ker,mode='valid')

def _discover_player_ids(df, prefix):
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)_x$")
    ids = []
    for c in df.columns:
        m = pat.match(c)
        if m:
            pid=int(m.group(1))
            if f"{prefix}_{pid}_y" in df.columns: ids.append(pid)
    ids = sorted(set(ids))
    return ids[:11]

def _stack_team_xy(df, prefix, player_ids=None):
    if player_ids is None: player_ids=_discover_player_ids(df,prefix)
    T=len(df); xs=[]; ys=[]; used=[]
    ids_pad=list(player_ids)+[None]*max(0,11-len(player_ids)); ids_pad=ids_pad[:11]
    for pid in ids_pad:
        if pid is None:
            xs.append(np.full(T,np.nan,dtype=np.float32)); ys.append(np.full(T,np.nan,dtype=np.float32))
        else:
            xcol=f"{prefix}_{pid}_x"; ycol=f"{prefix}_{pid}_y"
            x=pd.to_numeric(df.get(xcol, pd.Series([np.nan]*T)), errors='coerce').to_numpy(dtype=np.float32)
            y=pd.to_numeric(df.get(ycol, pd.Series([np.nan]*T)), errors='coerce').to_numpy(dtype=np.float32)
            xs.append(x); ys.append(y); used.append(pid)
    X=np.stack(xs,1); Y=np.stack(ys,1); xy=np.stack([X,Y],-1).astype(np.float32)
    while len(used)<11: used.append(-1000-len(used))
    return xy, used

def _make_period_oh(df,T):
    per = pd.to_numeric(df.get("Period", df.get("period", pd.Series([1]*T))), errors='coerce').fillna(method="ffill").fillna(1).to_numpy()
    per = np.where(per>=2,1,0).astype(np.int64)
    if len(per)<T: per=np.pad(per,(0,T-len(per)),constant_values=per[-1] if len(per)>0 else 0)
    else: per=per[:T]
    return np.stack([1-per, per],-1).astype(np.float32)

def _parse_time(df):
    for c in ["Time [s]","time_s","time","t"]:
        if c in df.columns:
            time=pd.to_numeric(df[c], errors='coerce').to_numpy()
            idx=np.arange(len(time)); mask=np.isfinite(time)
            if not mask.all(): time=np.interp(idx, idx[mask], time[mask])
            return time.astype(np.float32)
    return None

def _read_ball_xy(dh,da):
    for bx,by in [("Ball_x","Ball_y"),("ball_x","ball_y")]:
        if bx in dh.columns and by in dh.columns:
            return np.stack([pd.to_numeric(dh[bx],errors='coerce'), pd.to_numeric(dh[by],errors='coerce')],-1).astype(np.float32)
        if bx in da.columns and by in da.columns:
            return np.stack([pd.to_numeric(da[bx],errors='coerce'), pd.to_numeric(da[by],errors='coerce')],-1).astype(np.float32)
    return None

def build_dataset(home_csv, away_csv, events_csv, data_cfg: DataCfg):
    dh=pd.read_csv(home_csv,low_memory=False); da=pd.read_csv(away_csv,low_memory=False); de=pd.read_csv(events_csv,low_memory=False)
    time=_parse_time(dh); 
    if time is None: time=np.arange(len(dh))*(1.0/data_cfg.fps_fallback)
    dt=float(np.median(np.diff(time))) if len(time)>1 else (1.0/data_cfg.fps_fallback); T=len(time)

    home_xy, home_ids = _stack_team_xy(dh,"Home")
    away_xy, away_ids = _stack_team_xy(da,"Away")
    home_id2slot={pid:s for s,pid in enumerate(home_ids)}
    away_id2slot={pid:s for s,pid in enumerate(away_ids)}

    ball_xy=_read_ball_xy(dh,da); period=_make_period_oh(dh,T)

    # simple smoothing
    def smooth_all(xy):
        out=xy.copy()
        for s in range(11):
            for d in range(2):
                arr=out[:,s,d]
                if not np.isnan(arr).all(): out[:,s,d]=movavg(arr,3)
        return out
    home_xy=smooth_all(home_xy); away_xy=smooth_all(away_xy)

    home_v=np.stack([np.stack([central_diff(home_xy[:,i,0],dt), central_diff(home_xy[:,i,1],dt)],-1) for i in range(11)],1)
    away_v=np.stack([np.stack([central_diff(away_xy[:,i,0],dt), central_diff(away_xy[:,i,1],dt)],-1) for i in range(11)],1)
    ball_v=None
    if ball_xy is not None: ball_v=np.stack([central_diff(ball_xy[:,0],dt), central_diff(ball_xy[:,1],dt)],-1)

    win=int(round(data_cfg.window_s/dt)); step=int(round(data_cfg.stride_s/dt))
    indices=[(s,s+win) for s in range(0,T-win,step)]

    typ_series=(de['Type'] if 'Type' in de.columns else de['type']).fillna('Unknown').astype(str)
    types=sorted(typ_series.unique().tolist()); type_map={t:i for i,t in enumerate(types)}
    if 'Start Time [s]' in de.columns: ets=pd.to_numeric(de['Start Time [s]'], errors='coerce').to_numpy()
    else: ets=pd.to_numeric(de.get('start_time_s', np.array([])), errors='coerce')
    teamc=de['Team'] if 'Team' in de.columns else de.get('team', None)
    fromc=de['From'] if 'From' in de.columns else de.get('from_actor', None)
    toc  =de['To']   if 'To'   in de.columns else de.get('to_actor', None)

    X_list, det_list, type_list, from_list, to_list = [], [], [], [], []
    present=lambda xy:(~np.isnan(xy).any(-1)).astype(np.float32)

    for s,e in indices:
        Tl=e-s; per=period[s:e]
        Hxy, Axy = home_xy[s:e], away_xy[s:e]
        Hv, Av = home_v[s:e], away_v[s:e]
        Hpres, Apres = present(Hxy), present(Axy)

        Htok=np.concatenate([Hxy,Hv, np.tile(np.array([1,0,0],np.float32),(Tl,11,1)), np.tile(per[:,None,:],(1,11,1)), Hpres[...,None], np.zeros((Tl,11,1),np.float32)],-1)
        Atok=np.concatenate([Axy,Av, np.tile(np.array([0,1,0],np.float32),(Tl,11,1)), np.tile(per[:,None,:],(1,11,1)), Apres[...,None], np.zeros((Tl,11,1),np.float32)],-1)
        tokens=np.concatenate([Htok,Atok],1)
        if ball_xy is not None:
            Bxy=ball_xy[s:e][:,None,:]; Bv=ball_v[s:e][:,None,:]
            Bpres=(~np.isnan(Bxy).any(-1)).astype(np.float32)
            Btok=np.concatenate([np.concatenate([Bxy,Bv],-1), np.tile(np.array([0,0,1],np.float32),(Tl,1,1)), np.tile(per[:,None,:],(1,1,1)), Bpres[...,None], np.zeros((Tl,1,1),np.float32)],-1)
            tokens=np.concatenate([tokens,Btok],1)
        X_list.append(tokens.astype(np.float32))

        times=time[s]+np.arange(Tl)*dt
        det=np.zeros(Tl,np.float32); type_id=-np.ones(Tl,np.int64); from_id=-np.ones(Tl,np.int64); to_id=-np.ones(Tl,np.int64)
        for i in range(len(de)):
            e_time = float(ets[i]) if i<len(ets) and np.isfinite(ets[i]) else None
            if e_time is None or e_time < times[0]-data_cfg.det_tol_s or e_time > times[-1]+data_cfg.det_tol_s: continue
            k=int(np.argmin(np.abs(times-e_time)))
            if abs(times[k]-e_time) <= data_cfg.det_tol_s:
                det[k]=1.0
                tname=typ_series.iloc[i] if i<len(typ_series) else 'Unknown'
                type_id[k]=type_map.get(str(tname),-1)
                if teamc is not None and fromc is not None:
                    teamv=str(teamc.iloc[i]).lower()
                    try: p=int(fromc.iloc[i])
                    except: p=None
                    if p is not None:
                        if teamv.startswith('home'):
                            slot=home_id2slot.get(p,None)
                            if slot is not None: from_id[k]=0+slot
                        elif teamv.startswith('away'):
                            slot=away_id2slot.get(p,None)
                            if slot is not None: from_id[k]=11+slot
                if teamc is not None and toc is not None:
                    teamv=str(teamc.iloc[i]).lower()
                    try: p=int(toc.iloc[i])
                    except: p=None
                    if p is not None:
                        if teamv.startswith('home'):
                            slot=home_id2slot.get(p,None)
                            if slot is not None: to_id[k]=0+slot
                        elif teamv.startswith('away'):
                            slot=away_id2slot.get(p,None)
                            if slot is not None: to_id[k]=11+slot
        det_list.append(det); type_list.append(type_id); from_list.append(from_id); to_list.append(to_id)

    bundle={
        "tokens":[torch.from_numpy(x) for x in X_list],
        "det":[torch.from_numpy(x) for x in det_list],
        "type":[torch.from_numpy(x) for x in type_list],
        "from":[torch.from_numpy(x) for x in from_list],
        "to":[torch.from_numpy(x) for x in to_list],
        "type_map": type_map,
        "num_tokens": X_list[0].shape[1] if X_list else (23 if ball_xy is not None else 22),
        "dt": dt,
        "home_ids": home_ids, "away_ids": away_ids
    }
    return bundle

def save_dataset(data, out_path):
    torch.save(data, out_path)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--home_csv", default="./data/tracking_home.csv")
    ap.add_argument("--away_csv", default="./data/tracking_away.csv")
    ap.add_argument("--events_csv", default="./data/events.csv")
    ap.add_argument("--out", default="./artifacts/dataset.pt")
    ap.add_argument("--window_s", type=float, default=3.2)
    ap.add_argument("--stride_s", type=float, default=0.8)
    ap.add_argument("--det_tol_s", type=float, default=0.5)
    args=ap.parse_args()
    cfg=DataCfg(window_s=args.window_s, stride_s=args.stride_s, det_tol_s=args.det_tol_s)
    data=build_dataset(args.home_csv, args.away_csv, args.events_csv, cfg)
    outp=pathlib.Path(args.out); outp.parent.mkdir(parents=True, exist_ok=True)
    save_dataset(data, outp)
    print(f"Saved dataset to {outp}")
    print("Home player ids used:", data["home_ids"])
    print("Away player ids used:", data["away_ids"])

if __name__=="__main__":
    main()
