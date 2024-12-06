import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# CSV 파일 로드 - 도로 중심의 x, y, z 좌표가 포함된 CSV 파일
path_reference = pd.read_csv('/mnt/data/mobile_system_control/pathfinder/reference_path (1).csv', delimiter='\t')

# 데이터의 불필요한 문자 제거 (예: 콤마 제거)
path_reference = path_reference.replace({',': ''}, regex=True)

# 컬럼 이름 정리
path_reference.columns = path_reference.columns.str.strip()
path_reference.columns = ['x', 'y', 'z']

# 도로 중심선의 좌표 추출
x_coords = path_reference['x'].astype(float).to_numpy()
y_coords = path_reference['y'].astype(float).to_numpy()
z_coords = path_reference['z'].astype(float).to_numpy()

# 도로의 양끝 경계 계산 (도로 폭: 8m)
offset = 4  # 도로 중심으로부터 양쪽 경계까지의 거리 (8m 도로 폭의 절반)

# 도로 양끝 경계 좌표 계산 (도로 중심선의 수직 방향 벡터를 계산하여 적용)
dx = np.gradient(x_coords)
dy = np.gradient(y_coords)
magnitude = np.sqrt(dx**2 + dy**2)

# 수직 벡터 계산
left_x_coords = x_coords - offset * (dy / magnitude)
left_y_coords = y_coords + offset * (dx / magnitude)
right_x_coords = x_coords + offset * (dy / magnitude)
right_y_coords = y_coords - offset * (dx / magnitude)

# 도로 중심선 및 경계선 데이터 3D 시각화
fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x_coords, y_coords, z_coords, label='Road Centerline', color='blue', s=10)
ax.plot(left_x_coords, left_y_coords, z_coords, label='Left Road Boundary', color='green')
ax.plot(right_x_coords, right_y_coords, z_coords, label='Right Road Boundary', color='red')

ax.set_xlabel('X Coordinate')
ax.set_ylabel('Y Coordinate')
ax.set_zlabel('Z Coordinate')
ax.set_title('Road Centerline and Boundaries 3D Visualization')
ax.set_box_aspect([np.ptp(x_coords), np.ptp(y_coords), np.ptp(z_coords)])  # X, Y, Z 비율 동일하게 설정


plt.legend()
plt.show()
