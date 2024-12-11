import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from curvature import compute_curvature  # 곡률 계산 함수 임포트
from pathfinder.archive.cubic_spline_planner import CubicSpline2D    # 사용자가 제공한 CubicSpline1D,2D 코드가 있다고 가정
# cubicspline.py 파일에 위에서 제시한 CubicSpline1D, CubicSpline2D 클래스를 넣어둔다고 가정

TRACK_WIDTH = 8.0
CURVATURE_THRESHOLD = 0.02

def unit_vector(v):
    return v / np.linalg.norm(v)

def find_apexes(x, y, kappa, track_width=8.0, curvature_threshold=0.02):
    above_thresh = kappa > curvature_threshold
    idx_above = np.where(above_thresh)[0]

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
        
        apex_idx_rel = np.argmax(local_kappa)
        apex_idx = corner_indices[apex_idx_rel]

        if apex_idx == 0 or apex_idx == len(x)-1:
            continue

        dx1 = x[apex_idx] - x[apex_idx - 1]
        dy1 = y[apex_idx] - y[apex_idx - 1]
        dx2 = x[apex_idx + 1] - x[apex_idx]
        dy2 = y[apex_idx + 1] - y[apex_idx]

        tangent = unit_vector(np.array([dx1 + dx2, dy1 + dy2]))
        cross_z = dx1 * dy2 - dy1 * dx2
        normal = np.array([-tangent[1], tangent[0]])
        if cross_z < 0:
            normal = -normal

        apex_point = np.array([x[apex_idx], y[apex_idx]]) + normal * half_width
        apex_points.append(apex_point)
    return apex_points

def main():
    # CSV 파일 로드
    df = pd.read_csv('pathfinder/reference_path (1).csv', header=None, delimiter=',')
    df.columns = ['x_m', 'y_m', 'z_m']
    x = df['x_m'].values.astype(float)
    y = df['y_m'].values.astype(float)
    z = df['z_m'].values.astype(float)

    # 곡률 계산
    kappa = compute_curvature(x, y)

    # Apex 계산
    apex_points = find_apexes(x, y, kappa, track_width=TRACK_WIDTH, curvature_threshold=CURVATURE_THRESHOLD)

    # 트랙 경계 계산
    dx = np.gradient(x)
    dy = np.gradient(y)
    magnitude = np.sqrt(dx**2 + dy**2)
    half_width = TRACK_WIDTH / 2.0
    left_x_coords = x - half_width * (dy / magnitude)
    left_y_coords = y + half_width * (dx / magnitude)
    right_x_coords = x + half_width * (dy / magnitude)
    right_y_coords = y - half_width * (dx / magnitude)

    # Apex 포인트를 경로로 이용
    # apex_points는 [(ax, ay), (ax2, ay2), ...] 형태
    # CubicSpline2D는 x, y 좌표 리스트를 입력받는다.
    if len(apex_points) < 2:
        # Apex가 충분치 않으면 스플라인 경로 생성이 어려울 수 있음
        # 여기서는 그냥 Apex만 표시
        path_x = [p[0] for p in apex_points]
        path_y = [p[1] for p in apex_points]
    else:
        apex_points = np.array(apex_points)
        # CubicSpline2D에 apex 포인트를 입력
        sp = CubicSpline2D(apex_points[:,0], apex_points[:,1])

        # 원하는 해상도로 경로 샘플링
        s_sample = np.linspace(0, sp.s[-1], 200)
        path_x, path_y = [], []
        for s in s_sample:
            px, py = sp.calc_position(s)
            path_x.append(px)
            path_y.append(py)

    # 시각화
    plt.figure(figsize=(10,6))
    plt.plot(x, y, label='Road Centerline', color='blue')
    plt.plot(left_x_coords, left_y_coords, label='Left Boundary', color='green')
    plt.plot(right_x_coords, right_y_coords, label='Right Boundary', color='orange')

    if len(apex_points) > 0:
        a_x = [p[0] for p in apex_points]
        a_y = [p[1] for p in apex_points]
        plt.scatter(a_x, a_y, c='red', marker='o', s=50, label='Apex Points')

    # 생성한 CubicSpline2D 기반 경로 표시
    if len(path_x) > 1:
        plt.plot(path_x, path_y, 'r--', label='Cubic Spline Path (from Apex)')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Track with Boundaries, Apex and Cubic Spline Path')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.show()

if __name__ == "__main__":
    main()
