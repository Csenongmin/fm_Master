import numpy as np, collections as C
import pandas as pd
from tqdm import tqdm
from utils.dataset_build import save_numpy_artifacts, load_numpy_artifacts, make_dataloaders
from config import TrainCfg, ModelCfg
import torch
def prepare_features_and_labels(df: pd.DataFrame,
                                include_is_actor: bool = True):
    """
    입력 df: columns ⟨frame_id, team, player_id, x,y,vx,vy,speed, possession, ball_status,
                     EventType_Parent, EventContext, is_actor⟩ (표본과 동일 가정)
    반환:
      frames_ordered: dict(frame_id -> (N,F) np.array)  고정 N=23, F=11 or 12
      event_type_map: dict{name -> int} (정렬 기반)
      y_type_by_frame: dict(frame_id -> int)
      y_actor_by_frame: dict(frame_id -> int)  # 0=NoActor, 1..23
      feature_cols: list[str]  # 피처 순서(모델 d_in과 일치)
    """
    REQUIRED = ['frame_id','team','player_id','x','y','vx','vy','speed',
                'possession','ball_status','EventType_Parent','is_actor']
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"필수 컬럼 누락: {missing}")
    
    MAX_ENTITIES = 23
    TEAM_DUMMIES = ['team_Ball', 'team_Home', 'team_Away']
    POSS_DUMMIES = ['possession_Home', 'possession_Away']
    dfn = df.copy()

    # 0) EventType 'Unknown' → 'NoEvent'
    # 1) 팀 더미 & 점유 더미 (항상 존재 보장)
    dfn['EventType_Parent'] = dfn['EventType_Parent'].replace({'Unknown':'NoEvent'})
     # Ball의 player_id NaN이면 정렬 안정성 위해 0으로
    dfn.loc[dfn['team']=='Ball','player_id'] = dfn.loc[dfn['team']=='Ball','player_id'].fillna(0)
    dfn['player_id'] = dfn['player_id'].astype(float)  # 정렬 위해 float 유지, 나중에 쓰는 곳에서 int 변환 OK

    dfn = pd.get_dummies(dfn, columns=['team', 'possession'], dtype=np.float32)
    for col in TEAM_DUMMIES + POSS_DUMMIES:
        if col not in dfn.columns:
            dfn[col] = 0.0

    # 2) ball_status: Alive=1, Dead=0
    status_map = {'Alive': 1.0, 'Dead': 0.0}
    dfn['ball_status'] = dfn['ball_status'].map(status_map).fillna(0.0).astype(np.float32)

    # 3) 고정 피처 순서 정의 (include_is_actor로 11 or 12)
    feature_cols = [
        'x', 'y', 'vx', 'vy', 'speed',
        'team_Ball', 'team_Home', 'team_Away',
        'possession_Home', 'possession_Away',
        'ball_status'
    ]
    if include_is_actor:
        feature_cols.append('is_actor')  # 누설 주의(옵션)

    # 결측 보강 & 타입
    for col in feature_cols:
        if col not in dfn.columns:
            dfn[col] = 0.0
    dfn[feature_cols] = dfn[feature_cols].astype(np.float32)

    # 4) 엔티티 정렬 기준: Ball→Home→Away, 그 다음 player_id 오름차순
    dfn['team_category'] = pd.Categorical(
        dfn.filter(like='team_', axis=1)
           .rename(columns={'team_Ball':'Ball','team_Home':'Home','team_Away':'Away'})
           .idxmax(axis=1)
           .str.replace('team_','', regex=False),
        categories=['Ball','Home','Away'], ordered=True
    )
   
    

    # 프레임별 entity_rank (0..22)
    dfn = dfn.sort_values(['frame_id', 'team_category', 'player_id'])
    dfn['entity_rank'] = dfn.groupby('frame_id').cumcount()

    # 5) actor_label (0=NoActor, 1..=rank+1)
    frame_ids_sorted = dfn['frame_id'].drop_duplicates().sort_values()
    y_actor_series = pd.Series(0, index=frame_ids_sorted.values, dtype=np.int64)

    first_actor = dfn.loc[dfn['is_actor']==1, ['frame_id','entity_rank']] \
                     .sort_values(['frame_id','entity_rank']) \
                     .drop_duplicates('frame_id', keep='first')
    y_actor_series.loc[first_actor['frame_id'].values] = first_actor['entity_rank'].values + 1
    y_actor_by_frame = y_actor_series.to_dict()

    # 6) event_type 라벨 (재현성: 정렬)
    event_type_map = {'NoEvent': 0, 'Pass': 1, 'DefensiveAction': 2, 'BallControl': 3, 'Foul': 4, 'Shot': 5}
    unique_types = sorted(dfn['EventType_Parent'].dropna().unique().tolist())
    #event_type_map = {name:i for i,name in enumerate(unique_types)}
    print(event_type_map)
    y_type_by_frame = dfn.drop_duplicates('frame_id') \
                         .set_index('frame_id')['EventType_Parent'] \
                         .map(event_type_map).astype(int).to_dict()
    # 7) 프레임별 (N,F) 텐서 만들기 (패딩/슬라이스)
    frames_ordered = {}
    F = len(feature_cols)
    grouped = dfn.groupby('frame_id', sort=True)
    total_frames = len(frame_ids_sorted)
    for fid, g in tqdm(grouped, total=total_frames, desc="Building per-frame tensors", unit="frame"):
        g = g.sort_values(['team_category', 'player_id'])
        feats = g[feature_cols].to_numpy(np.float32)
        N = min(len(feats), MAX_ENTITIES)
        mat = np.zeros((MAX_ENTITIES, F), dtype=np.float32)
        mat[:N, :] = feats[:N, :]
        frames_ordered[fid] = mat

    return frames_ordered, event_type_map, y_type_by_frame, y_actor_by_frame, feature_cols

def make_windows_from_frames(frames_ordered: dict,
                             y_type_by_frame: dict,
                             y_actor_by_frame: dict,
                             seq_len: int = 125,
                             include_is_actor: bool = False,
                             feature_cols: list = None,
                             mask_last_is_actor: bool = True,
                             train_ratio=0.7, val_ratio=0.15):
    """
    frames_ordered: frame_id -> (N,F)
    반환: (X_train, Yt_train, Ya_train), (X_val,...), (X_test, ...)
    """
    frame_ids = np.array(sorted(frames_ordered.keys()))
    num_frames = len(frame_ids)

    # split by frame index (경계 안넘는 윈도우)
    train_end = int(num_frames * train_ratio)
    val_end = train_end + int(num_frames * val_ratio)
    splits = {
        'train': (0, train_end),
        'val': (train_end, val_end),
        'test': (val_end, num_frames),
    }

    def build_for_range(name, start, end, out_dtype=np.float16,):
        # 최소 seq_len 확보
        L = end - start
        if L < seq_len:
            tqdm.write(f"[{name}] 구간 길이 {L} < seq_len {seq_len} → 빈 세트 반환")
            return np.empty((0, seq_len, 23, len(feature_cols)), out_dtype), np.empty((0,), np.int64), np.empty((0,), np.int64)
        K = L - seq_len + 1                      # 윈도우 개수
        F = len(feature_cols)
        bytes_per_window = seq_len * 23 * F * np.dtype(out_dtype).itemsize
        total_gib = K * bytes_per_window / (1024**3)
        tqdm.write(f"[{name}] Alloc {K} windows of shape ({seq_len},23,{F}) "
               f"@{np.dtype(out_dtype).name} ≈ {total_gib:.2f} GiB")
         # 사전 할당
        X_arr  = np.empty((K, seq_len, 23, F), dtype=out_dtype)
        Yt_arr = np.empty((K,), dtype=np.int64)
        Ya_arr = np.empty((K,), dtype=np.int64)
        for k, i in enumerate(tqdm(range(start, end - seq_len + 1),
                               desc=f"Making windows [{name}]",
                               unit="win")):
            window_fids = frame_ids[i:i+seq_len]
            X = np.stack([frames_ordered[f] for f in window_fids], axis=0)  # (T,N,F)

            if include_is_actor and mask_last_is_actor:
                is_idx = feature_cols.index('is_actor')
                X[-1, :, is_idx] = 0.0

            tfid = window_fids[-1]
            X_arr[k]  = X.astype(out_dtype, copy=False)  # 여기서만 캐스팅
            Yt_arr[k] = np.int64(y_type_by_frame[tfid])
            Ya_arr[k] = np.int64(y_actor_by_frame[tfid])
       
        return X_arr, Yt_arr, Ya_arr

    X_tr, Yt_tr, Ya_tr = build_for_range("train", *splits['train'])
    print("\nTrain window done\n")
    X_va, Yt_va, Ya_va = build_for_range("val", *splits['val'])
    X_te, Yt_te, Ya_te = build_for_range("test", *splits['test'])
    return (X_tr, Yt_tr, Ya_tr), (X_va, Yt_va, Ya_va), (X_te, Yt_te, Ya_te)


    
# if __name__ == '__main__':
#     data_file_path = "./artifacts/final_model_data.parquet"
#     df = pd.read_parquet(data_file_path)
#     print("Data loaded successfully.")
#     frames_ordered, event_type_map, ytype_by_frame, yactor_by_frame, feat_cols = prepare_features_and_labels(df, include_is_actor=True)
#     (train_set, val_set, test_set) = make_windows_from_frames(
#         frames_ordered, ytype_by_frame, yactor_by_frame,
#         seq_len=125, include_is_actor=True, feature_cols=feat_cols, mask_last_is_actor=True,
#         train_ratio=0.7, val_ratio=0.15
#     )
#     (X_train, Y_type_train, Y_actor_train) = train_set
#     (X_val,   Y_type_val,   Y_actor_val)   = val_set
#     (X_test,  Y_type_test,  Y_actor_test)  = test_set
#     print("X_train:", X_train.shape, "| Y_type_train:", Y_type_train.shape, "| Y_actor_train:", Y_actor_train.shape)
#     print("X_val  :", X_val.shape,   "| Y_type_val  :", Y_type_val.shape,   "| Y_actor_val  :", Y_actor_val.shape)
#     print("X_test :", X_test.shape,  "| Y_type_test :", Y_type_test.shape,  "| Y_actor_test :", Y_actor_test.shape)
#     print("event_type_map:", event_type_map)