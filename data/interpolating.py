import pandas as pd
import numpy as np
import os
players = []
for i in range(1, 12):
    players.append(f'home_{i}')
    players.append(f'away_{i}')
output_dir = "./interpolation"

def interpolate_missing_positions(csv_path, player, output_dir=None):
    df = pd.read_csv(csv_path)

    # 보간 (선형)
    df['x'] = df['x'].interpolate(method='linear')
    df['y'] = df['y'].interpolate(method='linear')

    # (선택) 처음/끝 NaN → 앞/뒤 채우기
    df['x'] = df['x'].fillna(method='bfill')
    df['y'] = df['y'].fillna(method='bfill')

    output_path = f"./interpolation/{player}.csv"
    #output_path = os.path.join(output_dir, output_name)
    df.to_csv(output_path, index=False)
    print(f"보간된 결과가 {output_path}에 저장되었습니다.")

    return df

for player in players:
    if 'home' in player:
        csv_file_name = f"./output/home/{player}.csv"
    else:
        csv_file_name = f"./output/away/{player}.csv"
    df = interpolate_missing_positions(csv_file_name, player, output_dir)
    


