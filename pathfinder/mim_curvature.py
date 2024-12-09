import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# CSV 파일 로드
data = pd.read_csv('/mnt/data/mobile_system_control/pathfinder/reference_path (1).csv')
data.columns = data.columns.str.strip()
data.columns = ['x', 'y', 'z']
x_coords = data['x'].astype(float).to_numpy()
y_coords = data['y'].astype(float).to_numpy()
z_coords = data['z'].astype(float).to_numpy()
x = data['x'].values
y = data['y'].values

# 1차 미분 (속도)
dx = np.gradient(x)
dy = np.gradient(y)

# 2차 미분 (가속도)
ddx = np.gradient(dx)
ddy = np.gradient(dy)

# 곡률 계산
curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5

# 결과 시각화
plt.plot(curvature)
plt.title('Curvature')
plt.xlabel('Point Index')
plt.ylabel('Curvature')
plt.show()
