import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from curvature import compute_curvature  # 제공된 곡률 계산 함수 임포트

TRACK_WIDTH = 8.0
CURVATURE_THRESHOLD = 0.02  # 필요 시 조정

def unit_vector(v):
    return v / np.linalg.norm(v)

def find_apexes(x, y, kappa, track_width=8.0, curvature_threshold=0.02):
    # 곡률이 임계값 이상인 지점 찾기
    above_thresh = kappa > curvature_threshold
    idx_above = np.where(above_thresh)[0]

    # 연속된 코너 구간 식별
    corners = []
    if len(idx_above) > 0:
        start = idx_above[0]
        for i in range(1, len(idx_above)):
            if idx_above[i] != idx_above[i-1] + 1:
                corners.append(range(start, idx_above[i-1] + 1))
                start = idx_above[i]
        corners.append(range(start, idx_above[-1] + 1))

    apex_points = []
    half_width = track_width / 2.0

    for corner in corners:
        corner_indices = np.array(corner)
        local_kappa = kappa[corner_indices]
        
        # 코너 구간 내 최대 곡률 지점 찾기
        apex_idx_rel = np.argmax(local_kappa)
        apex_idx = corner_indices[apex_idx_rel]

        if apex_idx == 0 or apex_idx == len(x)-1:
            continue

        dx1 = x[apex_idx] - x[apex_idx - 1]
        dy1 = y[apex_idx] - y[apex_idx - 1]
        dx2 = x[apex_idx + 1] - x[apex_idx]
        dy2 = y[apex_idx + 1] - y[apex_idx]

        tangent = unit_vector(np.array([dx1 + dx2, dy1 + dy2]))

        # 좌/우 코너 판별
        cross_z = dx1 * dy2 - dy1 * dx2
        normal = np.array([-tangent[1], tangent[0]])
        if cross_z < 0:
            normal = -normal

        apex_point = np.array([x[apex_idx], y[apex_idx]]) + normal * half_width
        apex_points.append(apex_point)

    return apex_points

def main():
    # CSV 파일 로드 (헤더 없음 가정)
    df = pd.read_csv('pathfinder/reference_path (1).csv', header=None, delimiter=',')
    df.columns = ['x_m', 'y_m', 'z_m']
    x = df['x_m'].values.astype(float)
    y = df['y_m'].values.astype(float)
    z = df['z_m'].values.astype(float)

    # 곡률 계산
    kappa = compute_curvature(x, y)

    # Apex 계산
    apex_points = find_apexes(x, y, kappa, track_width=TRACK_WIDTH, curvature_threshold=CURVATURE_THRESHOLD)

    # 트랙 바운더리 계산
    dx = np.gradient(x)
    dy = np.gradient(y)
    magnitude = np.sqrt(dx**2 + dy**2)

    # 차량 폭 고려하지 않고 단순 트랙 폭만 고려해서 양 끝 경계 계산
    half_width = TRACK_WIDTH / 2.0
    left_x_coords = x - half_width * (dy / magnitude)
    left_y_coords = y + half_width * (dx / magnitude)
    right_x_coords = x + half_width * (dy / magnitude)
    right_y_coords = y - half_width * (dx / magnitude)

    # 시각화
    plt.figure(figsize=(10,6))
    plt.plot(x, y, label='Road Centerline', color='blue')
    plt.plot(left_x_coords, left_y_coords, label='Left Boundary', color='green')
    plt.plot(right_x_coords, right_y_coords, label='Right Boundary', color='orange')

    # Apex 포인트 표시
    if len(apex_points) > 0:
        apex_x = [p[0] for p in apex_points]
        apex_y = [p[1] for p in apex_points]
        plt.scatter(apex_x, apex_y, c='red', marker='o', s=50, label='Apex')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Track with Boundaries and Apex')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
