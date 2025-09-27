# ==================
# ball state visualization.py
# ==================
from dataclasses import dataclass
import matplotlib.pyplot as plt
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
import os

EVENT_CLOCK_OFFSET_SEC = 1.6

@dataclass
class PlotConfig:
    figsize: tuple = (16,10)
    facecolor: str | None = None
    home_color: str = 'blue'
    away_color: str = 'red'
    ball_color: str = 'black'
    tight_layout: bool = True


def _pitch_limits_from_df(df: pd.DataFrame, pitch=None, margin: float = 1.5):
    if pitch is not None and hasattr(pitch,'xlim') and hasattr(pitch,'ylim'):
        xmin,xmax = pitch.xlim; ymin,ymax = pitch.ylim; return xmin,xmax,ymin,ymax
    x = df['ball_x'].to_numpy(dtype=float); y = df['ball_y'].to_numpy(dtype=float)
    x = x[~np.isnan(x)]; y = y[~np.isnan(y)]
    xmin = (np.min(x)-margin) if x.size else -52.5; xmax = (np.max(x)+margin) if x.size else 52.5
    ymin = (np.min(y)-margin) if y.size else -34.0; ymax = (np.max(y)+margin) if y.size else 34.0
    return xmin,xmax,ymin,ymax


def filter_frames(df: pd.DataFrame, states: Optional[List[str]] = None, half: Optional[str] = None,
                  T1: Optional[int] = None, time_range: Optional[Tuple[float,float]] = None,
                  in_play_only: bool = False) -> pd.DataFrame:
    d = df
    if states is not None: d = d[d['ball_state'].isin(states)]
    if in_play_only: d = d[d['ball_state'] != 'out_of_play']
    if half is not None and T1 is not None:
        key = str(half).lower()
        if key in ('1','1st','first','firsthalf'): d = d[d['frame'] < T1]
        elif key in ('2','2nd','second','secondhalf'): d = d[d['frame'] >= T1]
    if time_range is not None:
        t0,t1 = time_range; d = d[(d['time_s'] >= float(t0)) & (d['time_s'] <= float(t1))]
    return d


def plot_ball_trajectory(df: pd.DataFrame, pitch=None, sample_step: int = 3, by_state: bool = True,
                         title: str = 'Ball trajectory', save_path: Optional[str] = None, legend: bool = True) -> None:
    mask = df['ball_x'].notna() & df['ball_y'].notna()
    data = df.loc[mask, ['ball_x','ball_y','ball_state']].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10,6))
    xmin,xmax,ymin,ymax = _pitch_limits_from_df(data, pitch)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal', adjustable='box')
    ax.plot([xmin,xmax,xmax,xmin,xmin],[ymin,ymin,ymax,ymax,ymin],linewidth=1)
    ax.plot([0,0],[ymin,ymax], linestyle='--', linewidth=0.8)

    if not by_state:
        ax.plot(data['ball_x'][::sample_step], data['ball_y'][::sample_step], linewidth=1)
    else:
        states = data['ball_state'].astype(str).to_numpy(); xs = data['ball_x'].to_numpy(); ys = data['ball_y'].to_numpy()
        first_seen=set(); change = np.where(states[1:]!=states[:-1])[0]+1; starts=np.r_[0,change]; ends=np.r_[change,len(states)]
        for s,e in zip(starts,ends):
            st = states[s]; seg_x=xs[s:e:sample_step]; seg_y=ys[s:e:sample_step]
            if len(seg_x)<2: continue
            h, = ax.plot(seg_x, seg_y, linewidth=1, alpha=0.95, label=(st if st not in first_seen else None))
            first_seen.add(st)
        if legend: ax.legend(loc='upper right', frameon=False)

    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y'); ax.grid(True, linewidth=0.4, alpha=0.3)
    if save_path: fig.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.show()

@dataclass
class PlotWindowCfg:
    figsize: tuple = (16,10)
    facecolor: str | None = None
    home_color: str = 'blue'
    away_color: str = 'red'
    ball_color: str = 'black'
    tight_layout: bool = True


def plot_window(xy_objects, pitch, *, half="secondHalf", t_start: int, t_end: int,
                config: PlotWindowCfg = PlotWindowCfg(), draw_current: bool = True):
    fr = xy_objects[half]['Home'].framerate
    T_half = len(xy_objects[half]['Ball'].xy)
    t_start = max(0, min(int(t_start), T_half-1)); t_end = max(0, min(int(t_end), T_half-1))
    if t_start >= t_end: t_start = max(0, t_end-1)

    fig, ax = plt.subplots(figsize=config.figsize, tight_layout=config.tight_layout)
    pitch.plot(ax=ax)
    if config.facecolor is not None: ax.set_facecolor(config.facecolor)

    xy_objects[half]['Home'].plot(t=(t_start,t_end), plot_type='trajectories', color=config.home_color, ax=ax)
    xy_objects[half]['Away'].plot(t=(t_start,t_end), plot_type='trajectories', color=config.away_color, ax=ax)
    xy_objects[half]['Ball'].plot(t=(t_start,t_end), plot_type='trajectories', color=config.ball_color, ax=ax)
    if draw_current:
        xy_objects[half]['Home'].plot(t=t_end, color=config.home_color, ax=ax)
        xy_objects[half]['Away'].plot(t=t_end, color=config.away_color, ax=ax)
        xy_objects[half]['Ball'].plot(t=t_end, color=config.ball_color, ax=ax)
    return fig, ax


def plot_time_window(xy_objects, pitch, *, half='firstHalf', gameclock_start: float, gameclock_end: float,
                     config: PlotWindowCfg = PlotWindowCfg(), title: str | None = None):
    fr = xy_objects[half]['Home'].framerate
    T_half = len(xy_objects[half]['Ball'].xy)
    t_start = max(0, min(int(round(gameclock_start*fr)), T_half-1))
    t_end   = max(0, min(int(round(gameclock_end*fr)),   T_half-1))
    fig, ax = plot_window(xy_objects, pitch, half=half, t_start=t_start, t_end=t_end, config=config, draw_current=True)
    if title is None: title = f"{half} window {gameclock_start:.2f}s ~ {gameclock_end:.2f}s"
    ax.set_title(title); return fig, ax


def get_event_frame(events, *, half='secondHalf', side='Home', eids=("ShotAtGoal_SuccessfulShot",), which=0,
                    framerate: float | None = None, offset_sec: float = EVENT_CLOCK_OFFSET_SEC) -> int:
    ev_obj = events[half][side]
    df_ev = ev_obj.events.copy(); eids = [eids] if isinstance(eids,str) else list(eids)
    df = df_ev[df_ev['eID'].isin(eids)].reset_index(drop=True)
    if df.empty: raise ValueError('No events found for given eIDs')
    if which >= len(df): raise IndexError('which out of range')
    row = df.iloc[which]
    if framerate is None: raise ValueError('framerate required for gameclock+offset conversion')
    return int(round((float(row['gameclock'])) * framerate))


def plot_event_window(xy_objects, events, pitch, *, half='secondHalf', side='Home', eids=("ShotAtGoal_SuccessfulShot",), which=0,
                      window_before_sec: float = 5.0, window_after_sec: float = 0.0,
                      config: PlotWindowCfg = PlotWindowCfg(), title: str | None = None):
    fr = xy_objects[half]['Home'].framerate
    gf = get_event_frame(events, half=half, side=side, eids=eids, which=which, framerate=fr)
    T_half = len(xy_objects[half]['Ball'].xy)
    before = int(round(window_before_sec*fr)); after = int(round(window_after_sec*fr))
    t_start = max(0, gf - before); t_end = min(T_half-1, gf + after)
    fig, ax = plot_window(xy_objects, pitch, half=half, t_start=t_start, t_end=t_end, config=config, draw_current=True)
    if title is None: title = f"{half} — {side} | eIDs={list(eids)} | idx:{which} | window -{window_before_sec:.1f}s ~ +{window_after_sec:.1f}s"
    ax.set_title(title); return fig, ax


def plot_passes_time_window(xy_objects, df: pd.DataFrame, pitch, *, half='firstHalf', window_sec: float = 20.0,
                            config: PlotWindowCfg = PlotWindowCfg(), title: str | None = None, show_points: bool = True):
    fr = xy_objects[half]['Home'].framerate; T_half = len(xy_objects[half]['Ball'].xy)
    t0, t1 = 0, max(0, min(int(round(window_sec*fr)), T_half-1))
    fig, ax = plot_window(xy_objects, pitch, half=half, t_start=t0, t_end=t1, config=config, draw_current=False)

    # select rows in this window (global frames)
    if half.lower().startswith('first'):
        dwin = df[(df['frame']>=t0)&(df['frame']<=t1)].copy(); T1 = len(xy_objects['firstHalf']['Ball'].xy)
        dwin['local_frame'] = dwin['frame']
    else:
        T1 = len(xy_objects['firstHalf']['Ball'].xy); g0,g1 = T1+t0, T1+t1
        dwin = df[(df['frame']>=g0)&(df['frame']<=g1)].copy(); dwin['local_frame'] = dwin['frame'] - T1

    if dwin.empty:
        ax.set_title(title or f"{half} — 0~{window_sec:.1f}s (no data)"); return fig, ax

    dwin = dwin.sort_values('local_frame')
    pmask = (dwin['ball_state'].astype(str) == 'passing').to_numpy()
    xs = dwin['ball_x'].to_numpy(); ys = dwin['ball_y'].to_numpy()
    if pmask.any():
        edges = np.flatnonzero(np.diff(np.r_[False, pmask, False])); runs = edges.reshape(-1,2)
        for s,e in runs:
            seg_x = xs[s:e]; seg_y = ys[s:e]
            if len(seg_x) < 2: continue
            ax.plot(seg_x, seg_y, linewidth=2.0)
            if show_points: ax.scatter([seg_x[0], seg_x[-1]],[seg_y[0], seg_y[-1]], s=20)

    ax.set_title(title or f"{half} — all passes in first {window_sec:.1f}s"); return fig, ax

def plot_states_time_window(xy_objects, df, pitch, *,
                            half="firstHalf", window_sec=(0, 20.0),
                            states=("passing","shooting","dribbling","idle"),
                            cols=2, show_points=True):
    """
    half: 'firstHalf' or 'secondHalf'
    window_sec: (start_sec, end_sec) within the half's gameclock
    states: which states to draw (one subplot per state)
    """
    fr = xy_objects[half]["Home"].framerate
    T1 = len(xy_objects["firstHalf"]["Ball"].xy)
    Th = len(xy_objects[half]["Ball"].xy)

    s0, s1 = window_sec
    t0 = max(0, min(int(round(s0 * fr)), Th - 1))
    t1 = max(0, min(int(round(s1 * fr)), Th - 1))
    if t0 >= t1:
        t0 = max(0, t1 - 1)

    # global frame range
    if half.lower().startswith("first"):
        g0, g1 = t0, t1
        local_shift = 0
    else:
        g0, g1 = T1 + t0, T1 + t1
        local_shift = T1

    n = len(states)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(10*cols, 6*rows), squeeze=False)

    # common pitch bounds
    xmin, xmax = pitch.xlim
    ymin, ymax = pitch.ylim

    # base layer coords for window (faint background trajectories)
    def draw_base(ax):
        ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax); ax.set_aspect("equal", adjustable="box")
        pitch.plot(ax=ax)
        # background player/ball trajectories for context
        xy_objects[half]["Home"].plot(t=(t0, t1), plot_type="trajectories", color="tab:blue", ax=ax)
        xy_objects[half]["Away"].plot(t=(t0, t1), plot_type="trajectories", color="tab:red", ax=ax)
        xy_objects[half]["Ball"].plot(t=(t0, t1), plot_type="trajectories", color="black", ax=ax)

    # select df rows for this half+window (global frames)
    mask_win = (df["frame"] >= g0) & (df["frame"] <= g1)
    dwin = df.loc[mask_win, ["frame","ball_state","ball_x","ball_y"]].copy()
    if not dwin.empty:
        dwin["local_frame"] = dwin["frame"] - local_shift
        dwin.sort_values("local_frame", inplace=True)

    for i, st in enumerate(states):
        r, c = divmod(i, cols)
        ax = axes[r][c]
        draw_base(ax)
        ax.set_title(f"{half} {s0:.1f}–{s1:.1f}s — {st}")

        if dwin.empty:
            continue

        pmask = (dwin["ball_state"].astype(str) == st).to_numpy()
        if pmask.any():
            xs = dwin["ball_x"].to_numpy()
            ys = dwin["ball_y"].to_numpy()
            # run-length encode contiguous True ranges
            edges = np.flatnonzero(np.diff(np.r_[False, pmask, False]))
            runs = edges.reshape(-1, 2)  # [ [start, end), ... ]
            for s, e in runs:
                seg_x = xs[s:e]; seg_y = ys[s:e]
                if len(seg_x) < 2: 
                    continue
                ax.plot(seg_x, seg_y, linewidth=2.5)  # highlight this state's segments
                if show_points:
                    ax.scatter([seg_x[0], seg_x[-1]], [seg_y[0], seg_y[-1]], s=30)

    # hide extra axes
    for j in range(n, rows*cols):
        r, c = divmod(j, cols)
        axes[r][c].set_axis_off()

    fig.tight_layout()
    plt.show()


def plot_ball_segments(
    df, pitch=None, key='pred_state',
    break_on_state=True, break_on_dead=True, break_on_gap=True, max_gap_frames=1,
    break_on_jump=True, jump_dist=8.0, sample_step=1, mark_ends=True, legend=True,
    title='Ball trajectory (segmented)',
    save_path=None, dpi=200, show=True):
    
    d = df[['ball_x','ball_y','frame']].copy()
    d[key] = df[key].astype(str).values if key in df.columns else 'unknown'
    d['ball_alive'] = df['ball_alive'].values if 'ball_alive' in df.columns else True

    m = d['ball_x'].notna() & d['ball_y'].notna()
    d = d.loc[m].reset_index(drop=True)
    if d.empty:
        print("No valid points to plot."); 
        return None, None

    x = d['ball_x'].to_numpy(float); y = d['ball_y'].to_numpy(float); fr = d['frame'].to_numpy(int)

    cut = np.zeros(len(d), dtype=bool); cut[0] = True
    for i in range(1, len(d)):
        split = False
        if break_on_gap and (fr[i] - fr[i-1] > max_gap_frames): split = True
        if break_on_dead and (not d['ball_alive'].iat[i-1] or not d['ball_alive'].iat[i]): split = True
        if break_on_jump and np.hypot(x[i]-x[i-1], y[i]-y[i-1]) > jump_dist: split = True
        if break_on_state and (d[key].iat[i] != d[key].iat[i-1]): split = True
        if split: cut[i] = True

    if pitch is not None and hasattr(pitch, 'xlim') and hasattr(pitch, 'ylim'):
        xmin, xmax = pitch.xlim; ymin, ymax = pitch.ylim
    else:
        xmin, xmax = np.nanmin(x)-1.5, np.nanmax(x)+1.5
        ymin, ymax = np.nanmin(y)-1.5, np.nanmax(y)+1.5

    fig, ax = plt.subplots(figsize=(12,7))
    if pitch is not None: pitch.plot(ax=ax)
    ax.set_xlim(xmin,xmax); ax.set_ylim(ymin,ymax); ax.set_aspect('equal','box'); ax.grid(True, lw=0.4, alpha=0.25)

    color_map = {'passing':'#1f77b4','dribbling':'#2ca02c','shooting':'#d62728','idle':'#7f7f7f'}
    first_seen = set()
    idx = np.r_[np.flatnonzero(cut), len(d)]
    start = 0
    while start < len(d):
        end = int(idx[np.searchsorted(idx, start, side='right')])
        seg_x = x[start:end:sample_step]; seg_y = y[start:end:sample_step]
        if len(seg_x) >= 2:
            lab = d[key].iat[start]
            col = color_map.get(lab, '#333333')
            ax.plot(seg_x, seg_y, lw=2.0, color=col, label=(lab if legend and lab not in first_seen else None))
            if mark_ends:
                ax.scatter([seg_x[0], seg_x[-1]], [seg_y[0], seg_y[-1]], s=20, zorder=3)
            first_seen.add(lab)
        start = end

    if legend and first_seen:
        ax.legend(loc='upper right', frameon=False)
    ax.set_title(title); ax.set_xlabel('x'); ax.set_ylabel('y')

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"saved → {save_path}")
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig, ax
