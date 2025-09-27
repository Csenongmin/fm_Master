# Deactivate distracting warnings
import warnings
warnings.filterwarnings("ignore")
import sys
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from utils.data_processing import *
from utils.visualization import *
from ball_state_viz import *

# ---------------------------------------------------------
# 0) Event category mapping (based on your eID inventory)
# ---------------------------------------------------------
EVENT_CLOCK_OFFSET_SEC = 1.6  # align event gameclock → tracking frames
FR = 25.0  # framerate
eid_to_category: Dict[str, str] = {
# Passing
'Play_Pass': 'Passing','Play_Cross': 'Passing',
'FreeKick_Play_Pass': 'Passing','FreeKick_Play_Cross': 'Passing',
'CornerKick_Play_Pass': 'Passing','CornerKick_Play_Cross': 'Passing',
'ThrowIn_Play_Pass': 'Passing','ThrowIn_Play_Cross': 'Passing',
'GoalKick_Play_Pass': 'Passing','KickOff_Play_Pass': 'Passing',
# Shooting
'ShotAtGoal_ShotWide':'Shooting','ShotAtGoal_SavedShot':'Shooting',
'ShotAtGoal_BlockedShot':'Shooting','ShotAtGoal_SuccessfulShot':'Shooting',
'ShotAtGoal_OtherShot':'Shooting','ShotAtGoal_ShotWoodWork':'Shooting',
'FreeKick_ShotAtGoal_BlockedShot':'Shooting','FreeKick_ShotAtGoal_SavedShot':'Shooting',
'FreeKick_ShotAtGoal_ShotWide':'Shooting','Penalty_ShotAtGoal_SuccessfulShot':'Shooting',
# Dribble/Skill
'Run':'Dribble','Nutmeg':'Dribble','SpectacularPlay':'Dribble','OtherBallAction':'Dribble',
# Defence/Duel
'TacklingGame':'DefenceDuel','BallClaiming':'DefenceDuel','BallDeflection':'DefenceDuel','SitterPrevented':'DefenceDuel',
# Stoppage/Restart (no play)
'ThrowIn':'StoppageRestart','RefereeBall':'StoppageRestart','FinalWhistle':'StoppageRestart',
'Offside':'StoppageRestart','GoalDisallowed':'StoppageRestart','OutSubstitution':'StoppageRestart',
# Officiating/Discipline
'Foul':'Officiating','Caution':'Officiating','CautionTeamofficial':'Officiating',
'VideoAssistantAction':'Officiating','PenaltyNotAwarded':'Officiating','PlayerNotSentOff':'Officiating','FairPlay':'Officiating',
# Context/Meta
'ChanceWithoutShot':'ContextMeta','PossessionLossBeforeGoal':'ContextMeta','OtherPlayerAction':'ContextMeta',
# Admin
'Delete':'Admin',
}


PASSING, SHOOTING, DRIBBLING, IDLE, OUT = 'passing','shooting','dribbling','idle','out_of_play'
shoot_like = {'Shooting'}
pass_like = {'Passing'}
dribble_like = {'Dribble'}

# ---- Tracking utils ----

def concat_halves_tracking(xy_objects) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    b1 = xy_objects['firstHalf']['Ball'].xy; b2 = xy_objects['secondHalf']['Ball'].xy
    h1 = xy_objects['firstHalf']['Home'].xy; h2 = xy_objects['secondHalf']['Home'].xy
    a1 = xy_objects['firstHalf']['Away'].xy; a2 = xy_objects['secondHalf']['Away'].xy
    T1, T2 = len(b1), len(b2)
    return np.vstack([b1,b2]), np.vstack([h1,h2]), np.vstack([a1,a2]), T1, T2

def ffill_2d(arr: np.ndarray) -> np.ndarray:
    out = arr.copy(); m = np.isnan(out)
    if m[0].any():
        first_valid = np.where(~m, np.arange(len(out))[:,None], len(out)).min(axis=0)
        out[0] = out[first_valid, np.arange(out.shape[1])]
    for t in range(1, len(out)):
        out[t, m[t]] = out[t-1, m[t]]
    return out

def compute_ball_kinematics(ball_xy: np.ndarray, FR: float):
    b = ffill_2d(ball_xy)
    v = np.diff(b, axis=0, prepend=b[[0]]) * FR
    speed = np.linalg.norm(v, axis=1)
    a = np.diff(v, axis=0, prepend=v[[0]]) * FR
    acc = np.linalg.norm(a, axis=1)
    return v, speed, acc, a[:,0], a[:,1]

def xy_team_to_3d(xy_flat: np.ndarray) -> np.ndarray:
    return xy_flat.reshape(len(xy_flat), xy_flat.shape[1]//2, 2)

def nearest_player_distance(ball_xy: np.ndarray, team_xy: np.ndarray) -> np.ndarray:
    team = xy_team_to_3d(team_xy)
    return np.nanmin(np.linalg.norm(team - ball_xy[:,None,:], axis=2), axis=1)

def nearest_idx_and_dist(ball_xy, team_xy):
    team = xy_team_to_3d(team_xy)             # [T,N,2]
    diff = team - ball_xy[:,None,:]
    dist = np.linalg.norm(diff, axis=2)       # [T,N]
    idx  = np.nanargmin(dist, axis=1)         # [T]
    dmin = dist[np.arange(len(dist)), idx]
    return idx, dmin

# ---- Code concat ----
def concat_code(code_dict: dict, key_first='firstHalf', key_second='secondHalf'):
    c1 = code_dict[key_first].code.astype(float)
    c2 = code_dict[key_second].code.astype(float)
    arr = np.concatenate([c1, c2]).astype(float)
    defs = {int(k):v for k,v in code_dict[key_first].definitions.items()}
    defs.update({int(k):v for k,v in code_dict[key_second].definitions.items()})
    return arr, defs

# ---- Events ----

def _safe_meta(x):
    if isinstance(x, dict): return x
    if isinstance(x, str):
        try:
            import json; return json.loads(x)
        except Exception:
            try: return eval(x)
            except Exception: return {}
    return {}

def make_pid_index(teamsheets) -> Dict[str, Dict[str,int]]:
    pid2 = {}
    for side in ('Home','Away'):
        if side in teamsheets and hasattr(teamsheets[side], 'teamsheet'):
            ts = teamsheets[side].teamsheet
            for _, r in ts.iterrows():
                pid2[str(r['pID'])] = {'side': side, 'xID': int(r['xID'])}
    return pid2


def collect_events_all(events: dict, T1: int, FR: float,
                        pid_index: dict | None = None,
                        teamsheets: dict | None = None,) -> pd.DataFrame:
    """Merge halves/sides, compute frame via (gameclock+1.6)*FR, add category,
       parse qualifiers (eval_success, recipient_pid → recipient_side/xid),
       and keep only one event per global_frame by priority.
    """
    # helpers
    def _q_pick(d, *keys):
        for k in keys:
            if k in d and d[k] not in (None, "", "None"):
                return d[k]
        return None

    def _q_bool(v):
        if v is None: return np.nan
        s = str(v).strip().lower()
        return s in ('successfullycompleted')
    
    if pid_index is None and teamsheets is not None:
        pid_index = make_pid_index(teamsheets)

    def collect_half(ev_half: dict, half_label: int) -> pd.DataFrame:
        dfs = []
        for side in ['Home', 'Away']:
            if side in ev_half and hasattr(ev_half[side], 'events'):
                ev_half[side].add_frameclock(FR)  # <- 여기서 frameclock 생성
                df = ev_half[side].events.copy()
                df['half'] = half_label
                df['side'] = side
                dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

    ev_1 = collect_half(events['firstHalf'], 1)
    ev_2 = collect_half(events['secondHalf'], 2)
    ev_all = pd.concat([ev_1, ev_2], ignore_index=True)
    

    # 2) frameclock -> frame(Int64)
    if 'frameclock' not in ev_all.columns:
        raise ValueError("add_frameclock(FR) 이후에도 'frameclock' 컬럼이 없습니다.")
    ev_all['frame'] = pd.to_numeric(ev_all['frameclock'], errors='coerce').round().astype('Int64')

    if 'eID' not in ev_all.columns:
        raise ValueError("events missing 'eID' column")
    ev_all['category'] = ev_all['eID'].map(eid_to_category).fillna('ContextMeta')
    ev_all = ev_all[(ev_all['eID'] != 'Delete') &
                (ev_all['category'].isin({'Passing','Shooting','Dribble'}))].reset_index(drop=True)
    ev_all['global_frame'] = ev_all.apply(lambda r: int(r['frame']) if r['half']==1 else (T1 + int(r['frame'])), axis=1)
    # qualifier → eval_success / recipient_pid
    if 'qualifier' in ev_all.columns:
        q = ev_all['qualifier'].apply(_safe_meta)
        ev_all['eval_success'] = q.apply(lambda d: _q_bool(_q_pick(d,'Evaluation')))
        ev_all['recipient_pid'] = q.apply(lambda d: _q_pick(d,'Recipient','recipient','receiver','targetPlayer','target','to'))
    else:
        ev_all['eval_success'] = np.nan; ev_all['recipient_pid'] = None

    # recipient → side/xID
    rec_side, rec_xid = [], []
    for pid in ev_all['recipient_pid'].astype(str).fillna('None'):
        info = (pid_index or {}).get(pid)
        if info is None:
            rec_side.append(None); rec_xid.append(np.nan)
        else:
            rec_side.append(info['side']); rec_xid.append(int(info['xID']))
    ev_all['recipient_side'] = rec_side
    ev_all['recipient_xid']  = pd.array(rec_xid, dtype='Int64')

    # priorities for tie-break per frame
    category_priority = {'Shooting':100,'Passing':90,'Dribble':80,'DefenceDuel':60,'StoppageRestart':50,'Officiating':40,'ContextMeta':10,'Admin':0}
    eid_priority = {'ShotAtGoal_SuccessfulShot':10,'Penalty_ShotAtGoal_SuccessfulShot':10,'ShotAtGoal_ShotWoodWork':8,'ShotAtGoal_SavedShot':6,'ShotAtGoal_BlockedShot':5}
    ev_all["_prio_cat"] = ev_all["category"].map(category_priority).fillna(0).astype(int)
    ev_all["_prio_eid"] = ev_all["eID"].map(eid_priority).fillna(0).astype(int)
    ev_all["_priority"] = ev_all["_prio_cat"]*100 + ev_all["_prio_eid"]

    sort_cols = ['global_frame','_priority'] + (['timestamp'] if 'timestamp' in ev_all.columns else [])
    ascending = [True, False] + ( [True] if 'timestamp' in ev_all.columns else [] )
    ev_all = (ev_all.sort_values(sort_cols, ascending=ascending)
                    .drop_duplicates(subset=['global_frame'], keep='first')
                    .sort_values('global_frame')
                    .reset_index(drop=True))
    ev_all = ev_all.drop(columns=['_prio_cat','_prio_eid','_priority'])
    return ev_all

# ---- Labeling ----
@dataclass
class LabelParams:
    speed_stop_th: float = 1.0
    receive_dist_th: float = 2.0
    dribble_ctrl_dist: float = 2.5
    dribble_speed_max: float = 7.0
    pos_flip_hold: int = 3
    enforce_stoppage_as_out: bool = False
    # Shooting specific
    max_shot_sec: float = 2.0
    receive_dist_shot: float = 1.2
    contact_hold_shot: int = 4
    pos_flip_hold_shot: int = 4
    angle_deflect_deg: float = 35.0
    angle_window: int = 3
    boundary_ends_play: bool = True
    # Passing
    max_pass_sec: float = 4.0
    receive_dist_pass: float | None = None
    pos_flip_hold_pass: int = 6
    # Recipient-aware
    receive_dist_pass_success: float = 1.8
    receive_dist_pass_fail: float = 1.8
    contact_hold_pass: int = 4


def build_labels(ev_all: pd.DataFrame,
                 ball_xy: np.ndarray,
                 home_xy: np.ndarray,
                 away_xy: np.ndarray,
                 ball_speed: np.ndarray,
                 FR: float,
                 pos_code: np.ndarray,
                 bs_code: np.ndarray,
                 params: LabelParams = LabelParams(),
                 pitch=None) -> Tuple[np.ndarray, np.ndarray]:
    T = len(ball_xy)

    # distances
    near_home = nearest_player_distance(ball_xy, home_xy)
    near_away = nearest_player_distance(ball_xy, away_xy)
    nearest_any = np.fmin(near_home, near_away)

    # team 3D for recipient
    home3 = xy_team_to_3d(home_xy); away3 = xy_team_to_3d(away_xy)

    # heading/angle
    b = ffill_2d(ball_xy)
    v = np.diff(b, axis=0, prepend=b[[0]]) * FR
    ball_angle = np.arctan2(v[:,1], v[:,0])
    def angle_diff(a,b):
        return np.abs(np.arctan2(np.sin(a-b), np.cos(a-b)))

    ball_alive = (bs_code == 1.0)
    labels = np.full(T, IDLE, dtype=object)
    labels[~ball_alive] = OUT

    def next_event_frame(start_f: int) -> Optional[int]:
        nxt = ev_all.loc[ev_all['global_frame'] > start_f, 'global_frame']
        return int(nxt.iloc[0]) if len(nxt) else None
    
    def _side_to_code(side: str) -> float:
    # possession 정의 {1:'Home', 2:'Away'} 기준
        if side == 'Home': return 1.0
        if side == 'Away': return 2.0
        return np.nan
    def first_possession_flip(start_f: int, end_cand: int, hold: int,
                          owner_override: float | None = None) -> Optional[int]:
        """연속 hold 프레임 동안 소유권이 바뀌면 flip 프레임 반환.
        owner_override가 있으면 그 값을 시작 소유자로 사용.
        """
        start_owner = owner_override if owner_override in (1.0, 2.0) else pos_code[start_f]
        if np.isnan(start_owner):
            return None
        h = 0
        for f in range(start_f + 1, end_cand + 1):
            # Dead 구간은 무시
            if not ball_alive[f]:
                h = 0
                continue
            if (not np.isnan(pos_code[f])) and (pos_code[f] != start_owner):
                h += 1
                if h >= hold:
                    return f - hold + 1
            else:
                h = 0
        return None

    def outside_pitch(f: int) -> bool:
        if not (params.boundary_ends_play and pitch is not None and hasattr(pitch,'xlim') and hasattr(pitch,'ylim')):
            return False
        x,y = ball_xy[f,0], ball_xy[f,1]
        xmin,xmax = pitch.xlim; ymin,ymax = pitch.ylim
        return (x<xmin) or (x>xmax) or (y<ymin) or (y>ymax)

    def find_segment_end(start_f: int, state: str, ev_row=None) -> int:
        # 오버라이드: 소유자 계산. possession data 정확하지 않음
        owner_override = None
        if ev_row is not None and 'side' in ev_row:
            owner_override = _side_to_code(ev_row['side'])
        nxt = next_event_frame(start_f)
        end_cand = (nxt - 1) if nxt is not None else (T - 1)
        cap_f = None
        if state == SHOOTING and params.max_shot_sec is not None:
            cap_f = min(end_cand, start_f + int(params.max_shot_sec*FR))
        elif state == PASSING and params.max_pass_sec is not None:
            cap_f = min(end_cand, start_f + int(params.max_pass_sec*FR))

        dead_end = contact_end = deflect_end = stop_end = None
        contact_hold = 0
        ang_th = np.deg2rad(params.angle_deflect_deg)
        recv_dist_pass_base = params.receive_dist_pass if (params.receive_dist_pass is not None) else params.receive_dist_th

        for f in range(start_f+1, end_cand+1):
            if not ball_alive[f] or outside_pitch(f):
                dead_end = f; break

            if state == SHOOTING:
                if nearest_any[f] <= params.receive_dist_shot:
                    contact_hold += 1
                    if contact_hold >= params.contact_hold_shot:
                        contact_end = f; break
                else:
                    contact_hold = 0
                if (f - start_f) >= params.angle_window:
                    if angle_diff(ball_angle[f], ball_angle[f-params.angle_window]) >= ang_th:
                        deflect_end = f; break
                if (ball_speed[f] <= params.speed_stop_th) and (nearest_any[f] <= params.receive_dist_th):
                    stop_end = f; break

            elif state == PASSING:
                ev_side    = ev_row.get('side', None) if ev_row is not None else None
                ev_success = ev_row.get('eval_success', np.nan) if ev_row is not None else np.nan
                rec_side   = ev_row.get('recipient_side', None) if ev_row is not None else None
                rec_xid    = ev_row.get('recipient_xid', pd.NA) if ev_row is not None else pd.NA
                rec_xid    = int(rec_xid) if pd.notna(rec_xid) else None

                if ev_side in ('Home','Away'):
                    near_same = near_home if ev_side=='Home' else near_away
                    near_opp  = near_away if ev_side=='Home' else near_home
                else:
                    near_same = near_opp = nearest_any

                if (ev_success is True) and (rec_side in ('Home','Away')) and (rec_xid is not None):
                    rxy = home3[:,rec_xid,:] if rec_side=='Home' else away3[:,rec_xid,:]
                    dist_rec = np.linalg.norm(ball_xy[f] - rxy[f])
                    if dist_rec <= params.receive_dist_pass_success:
                        contact_hold += 1
                        if contact_hold >= params.contact_hold_pass:
                            stop_end = f; break
                    else:
                        contact_hold = 0
                elif (ev_success is False):
                    if near_opp[f] <= params.receive_dist_pass_fail:
                        stop_end = f; break
                else:
                    if (ball_speed[f] <= params.speed_stop_th) and (nearest_any[f] <= recv_dist_pass_base):
                        stop_end = f; break

            elif state == DRIBBLING:
                if (ball_speed[f] <= params.speed_stop_th) and (nearest_any[f] > params.dribble_ctrl_dist):
                    stop_end = f; break

            if cap_f is not None and f >= cap_f:
                stop_end = f; break

        if state == SHOOTING:
            if dead_end is not None: return dead_end
            flip_f = first_possession_flip(start_f, end_cand, params.pos_flip_hold_shot,
                                       owner_override=owner_override)
            if flip_f is not None: return flip_f
            if contact_end is not None: return contact_end
            if deflect_end is not None: return deflect_end
            if stop_end is not None: return stop_end
            if cap_f is not None: return cap_f
            return end_cand

        if state == PASSING:
            if dead_end is not None: return dead_end
            flip_f = first_possession_flip(start_f, end_cand, params.pos_flip_hold_pass,
                                       owner_override=owner_override)
            if flip_f is not None: return flip_f
            if stop_end is not None: return stop_end
            if cap_f is not None: return cap_f
            return end_cand
        
        # DRIBBLING or others
        if stop_end is not None: return stop_end
        flip_f = first_possession_flip(start_f, end_cand, params.pos_flip_hold)
        if flip_f is not None: return min(flip_f, end_cand)
        return end_cand

    # label filling by events
    for _, ev in ev_all.iterrows():
        gf = int(ev['global_frame'])
        if gf >= T: continue
        cat = ev['category']
        if   cat in pass_like:    state = PASSING
        elif cat in shoot_like:   state = SHOOTING
        elif cat in dribble_like: state = DRIBBLING
        else: continue
        e = find_segment_end(gf, state, ev_row=ev)
        seg = slice(gf, e+1)
        mask = (labels[seg] != OUT) & ball_alive[seg]
        tmp = labels[seg].copy(); tmp[mask] = state; labels[seg] = tmp

    # auto-dribble on idle
    auto_dribble = (ball_alive) & (nearest_any <= params.dribble_ctrl_dist) & (ball_speed <= params.dribble_speed_max)
    labels[(labels==IDLE) & auto_dribble] = DRIBBLING
    auto_air_pass = (labels == IDLE) & ball_alive & (ball_speed > params.speed_stop_th) & (nearest_any > params.dribble_ctrl_dist)
    labels[auto_air_pass] = PASSING
    return labels, nearest_any


def build_frame_df(xy_objects, events, possession, ballstatus, *, teamsheets=None, pitch=None, FR: float = 25.0,
                   params: LabelParams = LabelParams()) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ball_xy, home_xy, away_xy, T1, T2 = concat_halves_tracking(xy_objects)
    T = len(ball_xy)

    pid_index = make_pid_index(teamsheets) if 'teamsheets' in globals() and teamsheets is not None else None
    ev_all = collect_events_all(events, T1=T1, FR=FR, teamsheets=teamsheets, pid_index=pid_index)

    v, speed, acc, ax, ay = compute_ball_kinematics(ball_xy, FR)
    pos_code, pos_defs = concat_code(possession)
    bs_code,  _        = concat_code(ballstatus)
    pos_team = pd.Series(pos_code).map(lambda x: pos_defs.get(int(x), np.nan))

    labels, nearest_dist = build_labels(ev_all, ball_xy, home_xy, away_xy, speed, FR, pos_code, bs_code, params, pitch)
    idx_home, dist_home = nearest_idx_and_dist(ball_xy, home_xy)
    idx_away, dist_away = nearest_idx_and_dist(ball_xy, away_xy)
    # 2) 팀 선수 속도
    home3 = xy_team_to_3d(home_xy); away3 = xy_team_to_3d(away_xy)
    home_v = np.diff(home3, axis=0, prepend=home3[[0]]) * FR  # [T,N,2]
    away_v = np.diff(away3, axis=0, prepend=away3[[0]]) * FR

    v_home_near = home_v[np.arange(T), idx_home]              # [T,2]
    v_away_near = away_v[np.arange(T), idx_away]              # [T,2]

    # 3) 소유팀 기준 same/opp 선택
    is_home = (pos_team.values == 'Home')
    v_same = np.where(is_home[:,None], v_home_near, v_away_near)  # [T,2]
    v_opp  = np.where(is_home[:,None], v_away_near, v_home_near)

    # 4) 파생 피처
    ball_v = np.diff(ball_xy, axis=0, prepend=ball_xy[[0]]) * FR
    norm = lambda u: np.linalg.norm(u, axis=1) + 1e-6
    cos_sim = lambda a,b: (a*b).sum(1) / (norm(a)*norm(b))

    time_s = np.arange(T) / FR
    df = pd.DataFrame({
        'frame': np.arange(T), 'time_s': time_s,
        'ball_x': ball_xy[:,0], 'ball_y': ball_xy[:,1],
        'ball_vx': v[:,0], 'ball_vy': v[:,1], 'ball_speed': speed,
        'ball_ax': ax, 'ball_ay': ay, 'ball_acc': acc,
        'nearest_player_dist': nearest_dist,
        'rel_speed_same': norm(ball_v - v_same), # 상대속도(클수록 패스 성향)
        'speed_ratio_same': norm(ball_v) / norm(v_same), # 속도비(>1.5면 패스 쪽)
        'cos_same': cos_sim(ball_v, v_same), # 방향 유사도(드리블은 ↑, 패스는 ↓ 경향)
        'possession_code': pd.Series(pos_code).astype('Int64', errors='ignore'),
        'possession_team': pos_team.values,
        'ball_alive': (bs_code == 1.0),
        'ball_state': labels,
    })

    if pitch is not None and hasattr(pitch,'xlim') and hasattr(pitch,'ylim'):
        xmin,xmax = pitch.xlim; ymin,ymax = pitch.ylim
        df['ball_x_norm'] = (df['ball_x'] - xmin) / (xmax - xmin)
        df['ball_y_norm'] = (df['ball_y'] - ymin) / (ymax - ymin)

    # 진행각 & 변화
    df['heading']   = np.arctan2(df['ball_vy'], df['ball_vx'])
    df['d_heading'] = np.abs(np.arctan2(np.sin(df['heading'].diff()), np.cos(df['heading'].diff()))).fillna(0)

    df['near_same'] = np.where(is_home, dist_home, dist_away)
    df['near_opp']  = np.where(is_home, dist_away, dist_home)
    # 롤링 통계(0.4s=10프레임 기준)
    for col in ['ball_speed','ball_acc','d_heading']:
        df[f'{col}_ma10'] = df[col].rolling(10, min_periods=1).mean()
        df[f'{col}_sd10'] = df[col].rolling(10, min_periods=1).std().fillna(0)

    # 골대 기준 피처 (양쪽 골까지의 거리/방향, 더 가까운 골 사용)
    xmin, xmax = pitch.xlim; ymin, ymax = pitch.ylim
    goalL = np.array([xmin, 0.0]); goalR = np.array([xmax, 0.0])
    bx = df['ball_x'].to_numpy(); by = df['ball_y'].to_numpy()
    P  = np.stack([bx, by], axis=1)

    dL = np.linalg.norm(P - goalL, axis=1); dR = np.linalg.norm(P - goalR, axis=1)
    df['goal_dist_min'] = np.minimum(dL, dR)

    # 골 방향 단위벡터 (더 가까운 골 기준)
    goal_vec_L = (goalL[None,:] - P); goal_vec_R = (goalR[None,:] - P)
    norm_L = np.linalg.norm(goal_vec_L, axis=1, keepdims=True)
    norm_R = np.linalg.norm(goal_vec_R, axis=1, keepdims=True)
    uL = goal_vec_L / np.clip(norm_L, 1e-6, None)
    uR = goal_vec_R / np.clip(norm_R, 1e-6, None)
    # 가까운 골 선택
    use_L = (dL <= dR)[:,None]
    uG = np.where(use_L, uL, uR)

    # 골 방향 속도성분 (v dot uG)
    V = np.stack([df['ball_vx'].to_numpy(), df['ball_vy'].to_numpy()], axis=1)
    df['v_toward_goal'] = (V * uG).sum(axis=1)

    return df, ev_all

if __name__ == '__main__':
    path = "./data/"
    file_name_pos = "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    file_name_infos = "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    file_name_events = "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    xy_objects, events, pitch, possession, ballstatus, teamsheets = load_data(path, file_name_pos, file_name_infos, file_name_events)
    params = LabelParams()
    df, ev_all = build_frame_df(xy_objects, events, possession, ballstatus,teamsheets=teamsheets, pitch=pitch, FR=FR, params=params)
    print('Label distribution:')
    print(df['ball_state'].value_counts())


