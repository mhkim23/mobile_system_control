import pandas as pd
import numpy as np

# CSV 파일에서 x, y, z 좌표를 읽어옵니다.
data = pd.read_csv('pathfinder/_reference_path_short.csv', header=None, delimiter=',')
# 처음 3개 열만 선택 (x, y, z 좌표)
coords = data.values[:, :3]  # numpy 배열의 모든 행(:)과 0,1,2번 열(:3)만 선택# 연속된 좌표 사이의 차이 계산
diffs = np.diff(coords, axis=0)

# 각 구간의 거리 계산 (유클리드 거리)
segment_lengths = np.linalg.norm(diffs, axis=1)

# 전체 길이 계산
total_length = np.sum(segment_lengths)

print("레이싱 트랙의 총 길이: {:.2f}m".format(total_length))