from stl import mesh
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx
from scipy.interpolate import splprep, splev
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# STL 파일을 시각화하는 코드
def visualize_stl(stl_file_path):
    # STL 파일 로드
    map_mesh = mesh.Mesh.from_file(stl_file_path)

    # 3D 플롯 생성
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # STL 메쉬 데이터 시각화
    ax.add_collection3d(Poly3DCollection(map_mesh.vectors, alpha=0.3, facecolor='cyan'))

    # 축 설정
    scale = map_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('STL Mesh Visualization')
    plt.show()

# 곡률 최소 경로 계산 함수
def find_minimum_curvature_path_from_csv(csv_file_path, num_sections=500, smoothness=5.0):
    # CSV 파일 로드
    path_reference = pd.read_csv(csv_file_path, delimiter='\t')

    # 데이터 전처리
    path_reference = path_reference.replace({',': ''}, regex=True)
    path_reference.columns = path_reference.columns.str.strip()
    path_reference.columns = ['x', 'y', 'z']
    x_coords = path_reference['x'].astype(float).to_numpy()
    y_coords = path_reference['y'].astype(float).to_numpy()
    z_coords = path_reference['z'].astype(float).to_numpy()

    # 트랙을 여러 개의 섹션으로 나누기
    section_indices = np.linspace(0, len(x_coords) - 1, num_sections, dtype=int)

    # 각 섹션에서 alpha 값을 변화시키며 후보 점 생성
    alpha_values = np.linspace(0, 1, 11)  # 0에서 1 사이의 alpha 값, 11개 (0.0, 0.1, ..., 1.0)
    candidate_points = []

    for i in section_indices:
        section_points = []
        for alpha in alpha_values:
            # 진행 방향의 수직 벡터 계산
            dx = np.gradient(x_coords)
            dy = np.gradient(y_coords)
            magnitude = np.sqrt(dx[i]**2 + dy[i]**2)
            normal_x = dy[i] / magnitude
            normal_y = -dx[i] / magnitude

            # 안쪽 경계(왼쪽 경계) 기준으로 alpha * 도로 폭만큼 이동한 점 계산
            road_width = 8  # 도로 폭
            vehicle_width = 1.2  # 차량 폭
            usable_road_width = road_width - vehicle_width  # 차량 폭을 고려한 유효 도로 폭
            candidate_x = x_coords[i] + alpha * usable_road_width * normal_x
            candidate_y = y_coords[i] + alpha * usable_road_width * normal_y

            section_points.append((candidate_x, candidate_y))
        candidate_points.append(section_points)

    # 그래프 생성하여 곡률 최소 경로 계산
    G = nx.Graph()

    # 각 섹션의 후보 점들을 노드로 추가하고, 거리 가중치 및 곡률 계산하여 엣지 생성
    for i in range(len(candidate_points) - 1):
        for j, (x1, y1) in enumerate(candidate_points[i]):
            for k, (x2, y2) in enumerate(candidate_points[i + 1]):
                # 거리 및 곡률 계산
                distance = np.linalg.norm([x2 - x1, y2 - y1])
                curvature = abs((x2 - x1) * (y_coords[i + 1] - y1) - (y2 - y1) * (x_coords[i + 1] - x1)) / (distance**3)
                weight = distance + curvature  # 거리와 곡률을 합한 비용 함수
                G.add_edge((i, j), (i + 1, k), weight=weight)

    # 최소 곡률 경로 계산
    start_nodes = [(0, j) for j in range(len(alpha_values))]  # 시작 섹션의 모든 후보 점
    end_nodes = [(len(candidate_points) - 1, j) for j in range(len(alpha_values))]  # 마지막 섹션의 모든 후보 점
    shortest_path = None
    min_distance = float('inf')

    # 시작점과 끝점의 모든 조합에 대해 최단 경로 찾기
    for start_node in start_nodes:
        for end_node in end_nodes:
            try:
                path = nx.shortest_path(G, source=start_node, target=end_node, weight='weight')
                path_length = nx.shortest_path_length(G, source=start_node, target=end_node, weight='weight')
                if path_length < min_distance:
                    min_distance = path_length
                    shortest_path = path
            except nx.NetworkXNoPath:
                continue

    # 최소 곡률 경로 좌표 추출
    x_shortest = [candidate_points[i][j][0] for i, j in shortest_path]
    y_shortest = [candidate_points[i][j][1] for i, j in shortest_path]

    # 최소 곡률 경로 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(x_coords, y_coords, label='Road Centerline', color='blue')
    plt.plot(x_shortest, y_shortest, label='Minimum Curvature Path', color='red')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.legend()
    plt.title('Minimum Curvature Path on Road Boundaries')
    plt.grid(True)
    plt.show()

# 실행 예시
csv_file_path = '/mnt/data/mobile_system_control/pathfinder/reference_path (1).csv'
find_minimum_curvature_path_from_csv(csv_file_path)

# STL 파일 시각화 실행 예시
stl_file_path = '/mnt/data/mobile_system_control/pathfinder/mobilesystemcontrol-1.stl'
visualize_stl(stl_file_path)
