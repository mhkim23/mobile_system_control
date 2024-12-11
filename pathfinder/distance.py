import pandas as pd
import numpy as np

# CSV 파일에서 x, y, z 좌표를 읽어옵니다.
data = pd.read_csv('./pathfinder/reference_path (1).csv', header=None)
coords = data.values  # numpy 배열로 변환

# 연속된 좌표 사이의 차이 계산
diffs = np.diff(coords, axis=0)

# 각 구간의 거리 계산 (유클리드 거리)
segment_lengths = np.linalg.norm(diffs, axis=1)

# 전체 길이 계산
total_length = np.sum(segment_lengths)

print("레이싱 트랙의 총 길이: {:.2f}m".format(total_length))