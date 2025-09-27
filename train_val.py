from ball_data import *
from ball_state_viz import *
from utils.visualization import *
from model.model import *
from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.model_selection import GroupKFold
from torch.utils.data import DataLoader
import os, re, glob
import copy
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")
import sys
print(sys.executable)

# 파일명에서 매치ID 추출: 예) "DFL-MAT-J03WMX"
_MATCH_RE = re.compile(r"DFL-MAT-[A-Z0-9]+")
LABEL2ID = {'dribbling':0, 'idle':1, 'passing':2, 'shooting':3}
ID2LABEL = {v:k for k,v in LABEL2ID.items()}

def discover_dfl_bundles(path: str):
    """
    path 내의 DFL xml을 스캔해 매치ID별로
    (info, events, positions)의 3종 파일 세트를 찾아 dict 리스트로 반환.
    """
    path = os.path.abspath(path)
    files = glob.glob(os.path.join(path, "*.xml"))

    buckets = {}  # match_id -> {'info':..., 'events':..., 'pos':...}
    for f in files:
        m = _MATCH_RE.search(os.path.basename(f))
        if not m:
            continue
        match_id = m.group(0)  # e.g., DFL-MAT-J03WMX
        d = buckets.setdefault(match_id, {})
        name = os.path.basename(f)

        if name.startswith("DFL_02_01_matchinformation_"):
            d['info'] = name
        elif name.startswith("DFL_03_02_events_raw_"):
            d['events'] = name
        elif name.startswith("DFL_04_03_positions_raw_observed_"):
            d['pos'] = name

    bundles = []
    for match_id, d in buckets.items():
        if {'info','events','pos'}.issubset(d):
            bundles.append({'match_id': match_id, **d})
        else:
            missing = {'info','events','pos'} - set(d)
            print(f"[warn] {match_id} 누락 파일: {missing}")
    bundles.sort(key=lambda x: x['match_id'])
    return bundles

def build_multi_match_dataset(path: str,
                              FR: float = 25.0,
                              params: LabelParams = LabelParams(),
                              keep_only_states: tuple[str,...] | None = None):
    """
    폴더 내 모든 매치(3종 xml 세트)를 처리해
    df_all(프레임 단위), ev_all_list(매치별 이벤트 테이블)를 반환.
    keep_only_states로 ['passing','dribbling','shooting','idle'] 중 서브셋만 남길 수 있음.
    """
    bundles = discover_dfl_bundles(path)
    if not bundles:
        raise RuntimeError("매치 파일 세트를 찾지 못했어요. 경로/파일명을 확인하세요.")

    dfs = []
    ev_tables = []

    for b in bundles:
        file_name_pos    = b['pos']
        file_name_infos  = b['info']
        file_name_events = b['events']

        # 1) data load
        xy_objects, events, pitch, possession, ballstatus, teamsheets = load_data(
            path, file_name_pos, file_name_infos, file_name_events
        )

        try:
            FR_eff = xy_objects["firstHalf"]["Home"].framerate or FR
        except Exception:
            FR_eff = FR

        # 2) 프레임 데이터셋 생성
        df_i, ev_all_i = build_frame_df(xy_objects, events, possession, ballstatus,
                                        pitch=pitch, FR=FR_eff, params=params)

        # 3) 필요시 상태 필터
        if keep_only_states:
            df_i = df_i[df_i['ball_state'].astype(str).isin(keep_only_states)].copy()

        # 4) 매치ID 컬럼 부여
        df_i['match_id'] = b['match_id']
        ev_all_i['match_id'] = b['match_id']

        dfs.append(df_i)
        ev_tables.append(ev_all_i)

    # 5) 합치기
    df_all = pd.concat(dfs, ignore_index=True)
    ev_all_list = ev_tables  # 매치별 이벤트 테이블(리스트) 유지
    return df_all, ev_all_list

def make_windows_fixed_map(df: pd.DataFrame,
                           feature_cols: list[str],
                           window_T: int = 21,
                           stride: int = 3,
                           center_label: bool = False):
    
    labels = df['ball_state'].astype(str).map(LABEL2ID).to_numpy()
    feats  = df[feature_cols].to_numpy(float)
    X_list, y_list, g_list = [], [], []
    T_total = len(df)
    
    for start in range(0, T_total - window_T + 1, stride):
        end = start + window_T
        win = feats[start:end]
        X_list.append(win)
        idx = start + (window_T//2) if center_label else (end-1)
        y_list.append(int(labels[idx]))
        g_list.append(df['match_id'].iloc[idx])

    X = np.stack(X_list, axis=0)
    y = np.array(y_list, dtype=int)
    groups = np.array(g_list)
    return X, y, groups

def make_windows_with_starts(df, feature_cols, label_col='ball_state',
                             window_T=21, stride=3, center_label=False,
                             drop_transition_mask=None, per_half=False, T1=None):
    labels = df[label_col].astype(str).values
    feats  = df[feature_cols].values.astype(float)
    T = len(df)
    X_list, y_list, starts = [], [], []

    def _loop(range_start, range_end, base_offset=0):
        for start in range(range_start, range_end - window_T + 1, stride):
            end = start + window_T
           
            if drop_transition_mask is not None and drop_transition_mask[end-1]:
                continue
            win = feats[start:end]
            if np.isnan(win).any():
                continue
            idx = start + (window_T//2) if center_label else (end-1)
            X_list.append(win)
            y_list.append(labels[idx])
            starts.append(base_offset + start)

    if per_half and T1 is not None:
        _loop(0, T1, base_offset=0)        # 전반
        _loop(T1, T,  base_offset=0)       # 후반: df가 글로벌 프레임이면 base_offset=0 그대로
    else:
        _loop(0, T, base_offset=0)

    y = np.array([LABEL2ID[s] for s in y_list], dtype=int)
    X = np.stack(X_list, axis=0) if X_list else np.empty((0, window_T, len(feature_cols)))
    return X, y, np.array(starts, dtype=int)

def run_epoch(loader, model, criterion, optimizer=None, device='cpu',
              train=True, use_tqdm=True, desc="Train",grad_clip: float | None = 1.0,
              strict_nonfinite: bool = False):
    
    model.train(True)

    total, correct, losses = 0, 0, []
    preds_all, labels_all = [], []

    it = tqdm(loader, desc=desc, leave=False) if use_tqdm else loader

    for step, (xb, yb) in enumerate(it):
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        # 라벨 범위 점검: 모델 출력 차원 내부인지
        if hasattr(model, 'head') and hasattr(model.head, 'out_features'):
            n_classes = int(model.head.out_features)
            if (yb.min() < 0) or (yb.max() >= n_classes):
                raise ValueError(f"Label out of range: [{yb.min().item()}, {yb.max().item()}] vs {n_classes}")

        with torch.set_grad_enabled(train):
            logits = model(xb)

            # logits 유효성 체크
            if not torch.isfinite(logits).all():
                msg = (f"[non-finite logits] step={step} "
                       f"min={logits.min().item()} max={logits.max().item()}")
                if not strict_nonfinite:
                    raise RuntimeError(msg)
                print(msg)
                continue
            loss = criterion(logits, yb)

        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        losses.append(loss.item())
        pred = logits.argmax(1)
        total += yb.size(0)
        correct += (pred == yb).sum().item()
        preds_all.append(pred.detach().cpu().numpy())
        labels_all.append(yb.detach().cpu().numpy())

        # 진행바에 러닝 평균만 가볍게 표시
        if use_tqdm:
            it.set_postfix({
                "loss": f"{np.mean(losses):.3f}",
                "acc":  f"{(correct/total):.3f}"
            })

    preds_all = np.concatenate(preds_all) if preds_all else np.array([])
    labels_all = np.concatenate(labels_all) if labels_all else np.array([])
    macro_f1 = f1_score(labels_all, preds_all, average='macro') if preds_all.size else 0.0

    return float(np.mean(losses)) if losses else 0.0, (correct/total) if total else 0.0, macro_f1


def train_with_progress(model, train_loader, val_loader, criterion, optimizer,
                        device='cpu', epochs=50, patience=8,
                        use_tqdm_train=True, use_tqdm_val=False):
    best_f1, bad = 0.0, 0
    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for epoch in range(epochs):
        tr_loss, tr_acc, tr_f1 = run_epoch(
            train_loader, model, criterion, optimizer, device,
            train=True, use_tqdm=use_tqdm_train, desc=f"Train {epoch:02d}"
        )
        va_loss, va_acc, va_f1 = run_epoch(
            val_loader, model, criterion, optimizer=None, device=device,
            train=False, use_tqdm=use_tqdm_val, desc=f"Val   {epoch:02d}"
        )
        print(f"[{epoch:02d}] tr_f1={tr_f1:.3f} va_f1={va_f1:.3f}  tr_loss={tr_loss:.3f} va_loss={va_loss:.3f}")

        if va_f1 > best_f1 + 1e-4:
            best_f1, bad = va_f1, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                print(f"Early stop at epoch {epoch} (best va_f1={best_f1:.3f})")
                break

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    return best_f1

def predict_loader(model, loader, device='cpu'):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            logits = model(xb)                 # [B, C]
            pred = logits.argmax(1).cpu().numpy()
            preds.append(pred)
            labels.append(yb.cpu().numpy())
    return np.concatenate(labels), np.concatenate(preds)

def window_preds_to_frame_labels(preds, T_total, *, window_T, stride,
                                 starts=None, weight='rear'):
    """
    preds: [N_win] 윈도우 예측(정수 클래스 ID)
    T_total: 프레임 수(len(df_in)) - 윈도우 만들 때 사용한 df의 길이와 같아야 함
    window_T, stride: 윈도우 생성 값과 동일
    weight: 'flat' 또는 'tri'– 겹칠 때 가운데에 가중 더 주고 싶으면 'tri'
    return: 프레임별 정수 클래스 ID 배열 [T_total]
    """
    n_cls = len(LABEL2ID)
    votes = np.zeros((T_total, n_cls), dtype=np.float32)
    # 가중치
    if weight == 'tri':
        mid = window_T//2
        w = 1.0 - np.abs(np.arange(window_T) - mid) / max(mid, 1)
        w = w.astype(np.float32)
    elif weight == 'rear':
        # 끝 프레임 라벨과 잘 맞음
        w = np.linspace(0.2, 1.0, window_T, dtype=np.float32)  # 0은 피해서 0.2~1.0

    else:
        w = np.ones(window_T, dtype=np.float32)

    # starts가 없으면 나이브 계산(경계/스킵 반영 못함)
    if starts is None:
        starts = np.arange(0, T_total - window_T + 1, stride)

    # 길이 불일치 시 안전하게 맞추기
    N = min(len(starts), len(preds))
    if len(starts) != len(preds):
        print(f"[warn] starts({len(starts)}) != preds({len(preds)}). Trimming to {N}.")
    for s, cls in zip(starts[:N], preds[:N]):
        e = min(T_total, s + window_T)
        votes[s:e, int(cls)] += w[:e - s]
    return votes.argmax(axis=1)

def attach_pred_to_df(df_in: pd.DataFrame, preds, *, window_T, stride, starts=None) -> pd.DataFrame:
    """
    df_in: 테스트 윈도우를 만들 때 사용한 프레임 테이블
    preds: 테스트 윈도우 예측(정수 ID)
    return: df_in 사본에 pred_state(str) 열 추가
    """
    T = len(df_in)
    y_frame = window_preds_to_frame_labels(preds, T, window_T=window_T, stride=stride, weight='tri', starts=starts)
    d = df_in.copy()
    d['pred_state'] = [ID2LABEL[i] for i in y_frame]
    return d

def check_nan_inf(X):
    Xf = X.reshape(-1, X.shape[-1])
    print("has_nan:", np.isnan(Xf).any(), "has_inf:", np.isinf(Xf).any())

if __name__ == '__main__':
    params = LabelParams()
    path = "./data/"
    # Test set
    file_name_pos = "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    file_name_infos = "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    file_name_events = "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    xy_objects, events, pitch, possession, ballstatus, teamsheets = load_data(path, file_name_pos, file_name_infos, file_name_events)
    
    keep_states = ('passing','dribbling','shooting','idle')
    df_all, ev_tables = build_multi_match_dataset(path, FR=25.0, params=params, keep_only_states=False)
    
    print("합쳐진 프레임:", df_all.shape, "매치 수:", df_all['match_id'].nunique())
    print(df_all['ball_state'].value_counts())

    feature_cols = ['ball_x_norm','ball_y_norm','ball_vx','ball_vy','ball_speed',
                'ball_ax','ball_ay','ball_acc','nearest_player_dist','d_heading','ball_speed_ma10','ball_acc_sd10','goal_dist_min','v_toward_goal','rel_speed_same','speed_ratio_same','cos_same','near_same','near_opp']
    
    # OOP는 여기서 제거
    if keep_states:
        df_all_in = df_all[df_all['ball_state'].astype(str).isin(keep_states)].copy()
    X, y, groups = make_windows_fixed_map(df_all_in, feature_cols, window_T=21, stride=3, center_label=False)
    print(X.shape, y.shape, {c:int(n) for c,n in zip(['drib','idle','pass','shot'], np.bincount(y, minlength=4))})

    TEST_ID = 'DFL-MAT-J03WOH'
    
    idx_te = (groups == TEST_ID)
    idx_cv = ~idx_te
    X_te, y_te        = X[idx_te],  y[idx_te]
    X_cv, y_cv        = X[idx_cv],  y[idx_cv]
    groups_cv         = groups[idx_cv]
    # 길이 확인
    assert len(X_cv) == len(y_cv) == len(groups_cv)

    # GroupKFold: 경기 단위 분할
    gkf = GroupKFold(n_splits=np.unique(groups_cv).size)
    fold_metrics, kept_models, kept_scalers = [], [], []
    for fold, (tr_idx, va_idx) in enumerate(gkf.split(X_cv, y_cv, groups_cv)):
        X_tr, y_tr = X_cv[tr_idx], y_cv[tr_idx]
        X_va, y_va = X_cv[va_idx], y_cv[va_idx]
        in_feat = X_tr.shape[2]  # ← Feature 차원 수
        scaler = StandardScaler().fit(X_tr.reshape(-1, in_feat))
        X_tr = scaler.transform(X_tr.reshape(-1, in_feat)).reshape(X_tr.shape).astype(np.float32)
        X_va = scaler.transform(X_va.reshape(-1, in_feat)).reshape(X_va.shape).astype(np.float32)

        train_ds = SeqDataset(X_tr, y_tr); val_ds = SeqDataset(X_va, y_va)
        train_loader = DataLoader(train_ds, batch_size=256, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=256, shuffle=False)
        check_nan_inf(X_tr); check_nan_inf(X_va)
        # 클래스 가중치 매우 불균형 → shooting 보정
        cls_counts = np.bincount(y_tr, minlength=len(LABEL2ID))   
        inv = 1.0 / np.clip(cls_counts, 1, None)
        class_weights = torch.tensor(inv / inv.sum() * len(inv), dtype=torch.float32)
        n_classes = len(LABEL2ID)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # model = Conv1DBaseline(in_feat=len(feature_cols), n_classes=len(LABEL2ID)).to(device) 
        # model = TCNBaseline(in_feat=in_feat, n_classes=n_classes,
        #             channels=(64,64,64,64), kernel_size=3,
        #             dropout=0.2, pool='last').to(device)

        model = TransformerBaseline(
            in_feat=in_feat, n_classes=n_classes,
            d_model=128, nhead=8, num_layers=4,
            dim_feedforward=256, dropout=0.2,
            pool='last', causal=True
        ).to(device)
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device)) 
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)  # Transformer 2e-3 -> 1e-3
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        best_f1 = train_with_progress(
            model, train_loader, val_loader,
            criterion, optimizer,
            device=device,
            epochs=50, patience=8,
            use_tqdm_train=True,   
            use_tqdm_val=False
        )
        u, c = np.unique(y_tr, return_counts=True)
        print(f"[Fold {fold}] train dist:", {ID2LABEL[i]: int(n) for i, n in zip(u, c)})

        y_true, preds_va = predict_loader(model, val_loader, device)
        f1_macro = f1_score(y_true, preds_va, average='macro')
        print(f"[Fold {fold}] macro-F1={f1_macro:.3f} (best during train={best_f1:.3f})")

        print(classification_report(y_true, preds_va, target_names=[ID2LABEL[i] for i in range(len(ID2LABEL))]))
        print(confusion_matrix(y_true, preds_va))
    
        non_shoot = [c for c in ('dribbling','idle','passing') if c in LABEL2ID]
        non_ids   = [LABEL2ID[c] for c in non_shoot]

        mask = np.isin(y_true, non_ids)        # 정답이 shooting이 아닌 행만 선택
        y_true_ns = y_true[mask]
        y_pred_ns = preds_va[mask]
        print("== Without shooting ==")
        print(classification_report(
            y_true_ns, y_pred_ns,
            labels=non_ids,
            target_names=[ID2LABEL[i] for i in non_ids],
            digits=3, zero_division=0
        ))
        print("Macro-F1 (no shooting):", f1_score(y_true_ns, y_pred_ns, average='macro'))
        print("Weighted-F1 (no shooting):", f1_score(y_true_ns, y_pred_ns, average='weighted'))
        print("Accuracy (no shooting):", accuracy_score(y_true_ns, y_pred_ns))
        print("Confusion matrix (no shooting):\n",
            confusion_matrix(y_true_ns, y_pred_ns, labels=non_ids))


        fold_metrics.append(f1_macro)
        kept_models.append(copy.deepcopy(model))
        kept_scalers.append(copy.deepcopy(scaler))

    print("CV macro-F1 (mean ± std):", np.mean(fold_metrics), np.std(fold_metrics))
    best_fold = int(np.argmax(fold_metrics))
    print("Best fold:", best_fold)
    best_model  = kept_models[best_fold]
    best_scaler = kept_scalers[best_fold]

    df_te = df_all[df_all['match_id'] == TEST_ID].copy()
    df_te_in = df_te[df_te['ball_state'] != 'out_of_play'].reset_index(drop=True)
    X_te_chk, y_te_chk, starts_te = make_windows_with_starts(
        df_te_in, feature_cols,
        window_T=21, stride=3, center_label=False
    )
    n_feat = X_te_chk.shape[2]
    X_te_sd = best_scaler.transform(X_te_chk.reshape(-1, n_feat)).reshape(X_te_chk.shape).astype(np.float32)
    
    test_ds = SeqDataset(X_te_sd, y_te_chk)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False)
    y_true_te, preds_te = predict_loader(model, test_loader, device)
    print("[TEST] macro-F1:", f1_score(y_true_te, preds_te, average='macro'))
    print(classification_report(y_true_te, preds_te, target_names=[ID2LABEL[i] for i in range(4)]))
    print(confusion_matrix(y_true_te, preds_te))
    assert len(preds_te) == len(starts_te), (len(preds_te), len(starts_te))
    df_te_pred_in = attach_pred_to_df(df_te_in, preds_te, window_T=21, stride=3, starts=starts_te)
    
    df_te_pred = df_te.copy()
    df_te_pred['pred_state'] = np.nan
    df_te_pred.loc[df_te['ball_state']!='out_of_play', 'pred_state'] = df_te_pred_in['pred_state'].values

    T1 = 69424  # len(xy_object['firstHalf']['Ball'].xy)
    _ = plot_ball_segments(
    df_te_pred[(df_te_pred['frame']<T1) & (df_te_pred['time_s']<=20)],
    pitch=pitch, key='pred_state',
    title='1st half 0–20s (Pred, segmented)',
    save_path=f"plots/{TEST_ID}_1st_0-20s_pred_Transformer_stdscaler.png",
    dpi=200, show=False
    )

    print("\nDone!\n")
    
