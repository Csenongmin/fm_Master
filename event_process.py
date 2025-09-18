import pandas as pd
import numpy as np
from utils.data_processing import *
from utils.visualization import *
import sys
import pandas as pd
import numpy as np

# Deactivate distracting warnings
import warnings
warnings.filterwarnings("ignore")
def _to_seconds(gc):
    # gc가 이미 숫자면 그대로, 문자열이면 to_timedelta로 변환
    if isinstance(gc, (int, float, np.floating)):
        return float(gc)
    try:
        return pd.to_timedelta(gc).total_seconds()
    except Exception:
        # 'MM:SS.mmm' 같은 형식이 아니라면 float 캐스팅 시도
        return float(gc)

def create_pid_to_jid_map(teamsheets):
    """teamsheets 객체에서 pID-jID 변환용 딕셔너리를 생성합니다."""
    home_df = teamsheets['Home'].teamsheet
    away_df = teamsheets['Away'].teamsheet
    
    all_players_df = pd.concat([home_df, away_df])
    
    # pID를 인덱스로, jID를 값으로 사용하여 시리즈를 만든 뒤 딕셔너리로 변환
    pid_to_jid_map = pd.Series(
        all_players_df.jID.values, 
        index=all_players_df.pID
    ).to_dict()
    
    return pid_to_jid_map

def create_xid_to_jid_map(teamsheets):

    home_map = pd.Series(
        teamsheets['Home'].teamsheet.jID.values, 
        index=teamsheets['Home'].teamsheet.xID
    ).to_dict()
    
    away_map = pd.Series(
        teamsheets['Away'].teamsheet.jID.values, 
        index=teamsheets['Away'].teamsheet.xID
    ).to_dict()
    
    return {'Home': home_map, 'Away': away_map}

# --- 1. 데이터 처리의 모든 단계를 포함하는 메인 함수 ---
def create_final_model_dataframe(path, file_name_pos, file_name_infos, file_name_events):
    """
    데이터 경로 딕셔너리를 입력받아, 모든 처리 과정을 거친
    최종 모델링용 데이터프레임을 반환합니다.
    """
    
    # STEP 1: 데이터 로딩 (floodlight 사용)
    # ------------------------------------
    print("Step 1: Data loading...")
    xy_objects, events_obj, pitch, possession, ballstatus, teamsheets = load_data(
        path, file_name_pos, file_name_infos, file_name_events
    )

    # STEP 2: ID 매핑 생성
    print("Step 2: Creating Player ID map...")
    pid_to_jid_map = create_pid_to_jid_map(teamsheets)
    xid_to_jid_maps = create_xid_to_jid_map(teamsheets) # xID->jID 맵 생성

    # STEP 3: 궤적 데이터를 Long-Format으로 변환
    # ---------------------------------------------
    print("Step 3: Converting trajectory data to long-format...")
    records = []
    global_frame_id = 0
    framerate = 25
    halves = ['firstHalf', 'secondHalf']

    for half_name in halves:
        xy_half = xy_objects[half_name]
        possession_codes = possession[half_name].code
        status_codes = ballstatus[half_name].code
        possession_map = possession[half_name].definitions
        status_map = ballstatus[half_name].definitions
        
        num_frames_in_half = xy_half['Ball'].xy.shape[0]
        for i in range(num_frames_in_half):
            poss_label = possession_map.get(possession_codes[i])
            status_label = status_map.get(status_codes[i])

            #Ball
            ball_x, ball_y = xy_half['Ball'].xy[i]
            records.append({
                'half': half_name,
                'half_frame': i,
                'frame_id': global_frame_id,
                'team': 'Ball',
                'player_id': np.nan,  # Ball을 0으로 고정(그룹 연산/속도계산 편의)
                'x': ball_x, 'y': ball_y,
                'possession': poss_label,
                'ball_status': status_label
            })
            
            for team_name in ['Home', 'Away']:
                team_data = xy_half[team_name].xy
                num_players = team_data.shape[1] // 2
                for p_idx in range(num_players):
                    player_x, player_y = team_data[i, p_idx * 2], team_data[i, p_idx * 2 + 1]
                    if not np.isnan(player_x):
                        player_jersey_id = xid_to_jid_maps[team_name].get(p_idx, -1) # 맵에 없으면 -1
                        records.append({
                            'half': half_name,
                            'half_frame': i,
                            'frame_id': global_frame_id,
                            'team': team_name,
                            'player_id': int(player_jersey_id),
                            'x': player_x, 'y': player_y,
                            'possession': poss_label,
                            'ball_status': status_label
                        })
            
            global_frame_id += 1
            
    trajectory_long_df = pd.DataFrame(records)
    trajectory_long_df['gameclock_sec'] = trajectory_long_df['half_frame'] / framerate

    
    # STEP 4: 이벤트 데이터 처리 및 필요한 컬럼만 선택
    # ---------------------------------------------
    print("Step 4: Processing event data...")
    ACTION_TO_PARENT_MAP = {
    # --- Pass 계열 ---
    'Pass': 'Pass',
    'Cross': 'Pass',
    
    # --- 수비 계열 ---
    'TacklingGame': 'DefensiveAction',
    'Foul': 'Foul',
    'Offside': 'Foul',
    
    # Shot 계열 
    'ShotWide': 'Shot',      # ShotAtGoal_ShotWide
    'SavedShot': 'Shot',     # ShotAtGoal_SavedShot  
    'BlockedShot': 'Shot',   # ShotAtGoal_BlockedShot 
    'Goal': 'Shot',          # ShotAtGoal_Goal
    'OtherShot': 'Shot',     # ShotAtGoal_OtherShot
    'SuccessfulShot': 'Shot',
    'ShotWoodWork': 'Shot',
    
    # --- 기타 ---
    'OtherBallAction': 'BallControl',
    }
    CONTEXT_PREFIXES = {
        'KickOff': 'SetPiece_KickOff', 
        'ThrowIn': 'SetPiece_ThrowIn', 
        'FreeKick': 'SetPiece_FreeKick', 
        'CornerKick': 'SetPiece_CornerKick', 
        'GoalKick': 'SetPiece_GoalKick',
        'Penalty': 'SetPiece_Penalty'
    }

    event_rows  = []
    for half, teams_data in events_obj.items():
        for team, events_container in teams_data.items():
            # 각 팀의 이벤트 데이터프레임에 접근
            df = events_container.events
            if df.empty: continue
            for _, row in df.iterrows():
                eid = row['eID']
                parts = eid.split('_')
                parent = next((ACTION_TO_PARENT_MAP[p] for p in reversed(parts) if p in ACTION_TO_PARENT_MAP),
                            ACTION_TO_PARENT_MAP.get(eid, 'Unknown'))
                context = next((name for prefix, name in CONTEXT_PREFIXES.items() if eid.startswith(prefix)), 'OpenPlay')
                event_rows.append({
                    'half': half,
                    'gameclock': row['gameclock'],             # <- 하프 로컬 시간
                    'gameclock_sec': _to_seconds(row['gameclock']),
                    'event_pID': row['pID'],
                    'EventType_Parent': parent,
                    'EventContext': context
                })
    processed_events_df = pd.DataFrame(event_rows)



    # 제거할 모든 이벤트를 하나의 리스트로 통합
    # events_to_filter_out = [
    #     # 예측 목표에서 제외할 이벤트들
    #     'BallClaiming',
    #     'Delete',
    #     'OutSubstitution',
    #     'VideoAssistantAction',
    #     'FinalWhistle',
    #     'Caution',
    # ]
    frames_idx = trajectory_long_df[['half', 'frame_id', 'half_frame']].drop_duplicates()
    frames_idx['gameclock_sec'] = frames_idx['half_frame'] / framerate
    events_sorted = processed_events_df.copy()
    if 'gameclock_sec' not in events_sorted.columns:
        events_sorted['gameclock_sec'] = events_sorted['gameclock'].map(_to_seconds).astype(float)
    # Unknown은 NoEvent 취급 → 매칭 대상에서 제거
    events_sorted['EventType_Parent'] = events_sorted['EventType_Parent'].replace({'Unknown': 'NoEvent'})
    events_sorted = events_sorted[events_sorted['EventType_Parent'] != 'NoEvent'].copy()
    # 하프별 프레임 개수
    half_counts = frames_idx.groupby('half')['half_frame'].max().add(1).to_dict()
    # 이벤트를 가장 가까운 half_frame(정수)로 배정
    def _assign_half_frame(row):
        count = int(half_counts[row['half']])
        hf = int(np.rint(row['gameclock_sec'] * framerate))  # nearest
        if hf < 0: hf = 0
        if hf >= count: hf = count - 1
        return hf
    events_sorted['half_frame_cand'] = events_sorted.apply(_assign_half_frame, axis=1)
    # 하프별로 merge_asof (by='half' 사용)
    # STEP 5: 궤적 데이터와 이벤트 데이터 병합
    # ---------------------------------------------
    
    print("Step 5: Merging trajectory and event data...")
    ev = events_sorted.merge(
    frames_idx[['half', 'half_frame', 'frame_id', 'gameclock_sec']].rename(columns={'gameclock_sec':'frame_time_sec'}),
    left_on=['half', 'half_frame_cand'], right_on=['half', 'half_frame'], how='left'
    )
    # 프레임-이벤트 시간 차이
    ev['delta_sec'] = (ev['gameclock_sec'] - ev['frame_time_sec']).abs()

    # 허용 오차 설정: 프레임 2장(≈0.08s) 정도 권장. 필요시 늘리세요(예: 0.5~1.0s).
    tolerance_sec = 2.0 / framerate
    ev = ev[ev['delta_sec'] <= tolerance_sec].copy()
  
    
    priority_map = {
    'Pass': 1,
    'Shot': 2,
    'DefensiveAction': 3,
    'BallControl': 5,
    'Foul': 6,
    'NoEvent': 99
    }
    ev['priority'] = ev['EventType_Parent'].map(priority_map).fillna(99).astype(int)
    ev['row_order'] = np.arange(len(ev))  # 안정적 타이브레이크
    # 하프/프레임 단 하나만 남기기 (절대 중복 없음)
    ev = ev.sort_values(['half', 'frame_id', 'priority', 'delta_sec', 'row_order'])
    events_with_frames_unique = ev.drop_duplicates(subset=['half', 'frame_id'], keep='first')
    # (선택) 안전 체크: 프레임 중복 매칭이 있는지
    assert not events_with_frames_unique.duplicated(subset=['half', 'frame_id']).any(), \
        "Duplicate events mapped to the same frame unexpectedly."
 
  
    

    # 필요한 컬럼만 가진 이벤트 데이터와 궤적 데이터를 left-join
    final_df = pd.merge(
        trajectory_long_df,
        events_with_frames_unique[['frame_id', 'EventType_Parent', 'EventContext', 'event_pID']],
        on='frame_id', how='left'
    ).drop_duplicates(subset=['frame_id', 'team', 'player_id'], keep='first')
    final_df.drop_duplicates(subset=['frame_id', 'team', 'player_id'], keep='first', inplace=True)
    # STEP 6: 최종 컬럼 선택 및 정리
    # ---------------------------------------------
    print("Step 6: Finalizing columns...")
    columns_to_keep = [
        'frame_id', 'team', 'player_id', 'x', 'y', 
        'possession', 'ball_status', 'event_pID', 
        'EventType_Parent', 'EventContext'
    ]
    final_model_df = final_df[columns_to_keep].copy()

    # 이벤트가 없는 프레임의 NaN 값을 'NoEvent'로 채우기
    event_related_cols = ['event_pID', 'EventType_Parent', 'EventContext']
    final_model_df[event_related_cols] = final_model_df[event_related_cols].fillna('NoEvent')
     # STEP 7: ID 매칭 및 'is_actor' 특성 생성
    print("Step 7: Matching IDs and creating 'is_actor' feature...")
    
    # 5-1. event_pID를 event_jID (등번호)로 변환
    final_model_df['event_jID'] = final_model_df['event_pID'].map(pid_to_jid_map)
    
    # 5-2. 'is_actor' 컬럼 생성: 자신의 player_id와 이벤트의 jID가 일치하는지 확인
    # 공(Ball)이나 이벤트가 없는 프레임은 False(0)가 되도록 처리
    final_model_df['is_actor'] = ((final_model_df['player_id'] == final_model_df['event_jID'])).astype(int)
    final_model_df.drop(columns=['event_pID', 'event_jID'], inplace=True) # 매칭에 사용된 컬럼 제거
    print("\nProcessing complete!")
    """
    데이터프레임에 속도(vx, vy)와 속력(speed) 피처를 추가합니다.
    """
    print("Adding velocity and speed features...")
    
    # 프레임 간 시간 간격
    delta_t = 1 / framerate
    
    # 각 개체(공, 선수)별로 그룹화하여 계산해야 정확합니다.
    # 공(Ball)의 player_id는 NaN이므로, -1로 채워서 그룹화 키로 사용합니다.
    final_model_df['entity_id'] = final_model_df['team'] + '_' + final_model_df['player_id'].fillna(-1).astype(int).astype(str)
    
    # 각 개체별로 시간순(frame_id)으로 정렬
    final_model_df = final_model_df.sort_values(by=['entity_id', 'frame_id'])
    
    # diff()를 사용하여 이전 프레임과의 좌표 차이 계산
    # groupby('entity_id')를 통해 선수나 공이 바뀌는 지점에서 차이가 계산되는 것을 방지
    final_model_df['delta_x'] = final_model_df.groupby('entity_id')['x'].diff()
    final_model_df['delta_y'] = final_model_df.groupby('entity_id')['y'].diff()
    
    # 속도(vx, vy) 계산
    final_model_df['vx'] = final_model_df['delta_x'] / delta_t
    final_model_df['vy'] = final_model_df['delta_y'] / delta_t
    
    # 속력(speed) 계산
    final_model_df['speed'] = np.sqrt(final_model_df['vx']**2 + final_model_df['vy']**2)
    
    # 첫 프레임은 이전 프레임이 없으므로 NaN이 발생 -> 0으로 채움
    final_model_df.fillna({'vx': 0, 'vy': 0, 'speed': 0}, inplace=True)
    
    # 임시로 사용한 컬럼들 제거
    final_model_df.drop(columns=['entity_id', 'delta_x', 'delta_y'], inplace=True)
    team_order = pd.CategoricalDtype(['Ball', 'Home', 'Away'], ordered=True)
    
    # 2. 'team' 컬럼에 이 타입을 적용합니다.
    final_model_df['team'] = final_model_df['team'].astype(team_order)
    
    # 3. 'frame_id' -> 'team' -> 'player_id' 순서로 최종 정렬합니다.
    final_model_df.sort_values(by=['frame_id', 'team', 'player_id'], inplace=True)
    print("Velocity features added successfully.")
    return final_model_df

def check_duplicate_events(final_model_df):
    # MAX_ENTITIES를 23으로 설정했다고 가정
    MAX_ENTITIES = 23 

    # 각 frame_id별로 행(row)의 개수를 셉니다.
    frame_counts = final_model_df.groupby('frame_id').size()

    # 행의 개수가 MAX_ENTITIES를 초과하는 프레임을 찾습니다.
    problem_frames = frame_counts[frame_counts > MAX_ENTITIES]

    if not problem_frames.empty:
        print("문제 발생 프레임 및 해당 프레임의 행 개수:")
        print(problem_frames)
        
        # 가장 첫 번째 문제 프레임의 ID를 가져옵니다.
        problem_frame_id = problem_frames.index[0]
        print(f"\n--- {problem_frame_id}번 프레임의 상세 데이터 ---")
        print(final_model_df[final_model_df['frame_id'] == problem_frame_id])
    else:
        print("데이터 중복 문제가 발견되지 않았습니다.")



# --- 실행 ---
if __name__ == '__main__':
    # 실제 파일 경로 딕셔너리를 함수에 전달
    PATH = "./data/"
    FILE_POS = "DFL_04_03_positions_raw_observed_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    FILE_INFOS = "DFL_02_01_matchinformation_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    FILE_EVENTS = "DFL_03_02_events_raw_DFL-COM-000002_DFL-MAT-J03WOH.xml"
    print(sys.version)
    final_model_df = create_final_model_dataframe(PATH, FILE_POS, FILE_INFOS, FILE_EVENTS)
    
    print("\n--- 최종 모델링용 데이터프레임 정보 ---")
    print(final_model_df.info())
    print("\n--- 데이터 샘플 ---")
    print(final_model_df.head(25))
    print("\nSaving DataFrame to Parquet file...")
    parquet_filepath = "./artifacts/final_model_data.parquet"
    final_model_df.to_parquet(parquet_filepath, index=False)
    
   
    print(f"Successfully saved to {parquet_filepath}")
   
    