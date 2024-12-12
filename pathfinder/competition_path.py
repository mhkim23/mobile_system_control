import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

# 데이터 로드 - 도로 중심선 좌표가 포함된 CSV 파일 로드
path_reference = pd.read_csv('/mnt/data/mobile_system_control/pathfinder/_reference_path_short.csv', delimiter='\t')

# 데이터 전처리
path_reference = path_reference.replace({',': ''}, regex=True)
path_reference.columns = path_reference.columns.str.strip()
path_reference.columns = ['x', 'y', 'z']
x_coords = path_reference['x'].astype(float).to_numpy()
y_coords = path_reference['y'].astype(float).to_numpy()
z_coords = path_reference['z'].astype(float).to_numpy()

# 도로 폭 정의 (예시: 거리에 따라 변화하는 도로 폭)
def calculate_road_width(index):
    """
    CSV 파일의 행 번호를 기준으로 도로 폭 계산
    """
    if 250 <= index <= 585 or index <= 30 or 668<= index <=678 or 696 <= index <= 720:
        return 3  # 좁은 구간
    elif 678 < index < 696:
        return 5
    else:
        return 8  # 기본 도로 폭

# 각 지점별 도로 폭 계산 (인덱스 기반)
road_widths = np.array([calculate_road_width(i) for i in range(len(x_coords))]) 
vehicle_width = 2  # 차량 폭
usable_road_widths = road_widths - vehicle_width
adjusted_offsets = usable_road_widths / 2

# 수정된 경계선 계산
dx = np.gradient(x_coords)
dy = np.gradient(y_coords)
magnitude = np.sqrt(dx**2 + dy**2)

# 가변 폭을 적용한 경계선 계산
left_x_coords = x_coords - adjusted_offsets * (dy / magnitude)
left_y_coords = y_coords + adjusted_offsets * (dx / magnitude)
right_x_coords = x_coords + adjusted_offsets * (dy / magnitude)
right_y_coords = y_coords - adjusted_offsets * (dx / magnitude)

# 트랙을 여러 개의 섹션으로 나누기
num_sections = 150  # 섹션 개수 설정
section_indices = np.linspace(0, len(x_coords) - 1, num_sections, dtype=int)
# linspace는 시작점(0)부터 끝점(len(x_coords) - 1)까지 num_sections개의 균등한 간격의 점을 생성

# 각 섹션에서 alpha 값을 변화시키며 후보 점 생성
alpha_values = np.linspace(0, 1, 11)  # 0에서 1 사이의 alpha 값, 11개 (0.0, 0.1, ..., 1.0)
# alpha는 도로 폭을 11개 구간으로 나누는 비율 값 (0: 왼쪽 경계, 1: 오른쪽 경계)

# 후보점 생성 부분 수정
candidate_points = []
for i in section_indices:
    section_points = []
    for alpha in alpha_values:
        normal_x = dy[i] / magnitude[i]
        normal_y = -dx[i] / magnitude[i]
        
        # 해당 지점의 유효 도로 폭 사용
        current_usable_width = usable_road_widths[i]
        candidate_x = left_x_coords[i] + alpha * current_usable_width * normal_x
        candidate_y = left_y_coords[i] + alpha * current_usable_width * normal_y
        
        section_points.append((candidate_x, candidate_y))
    candidate_points.append(section_points)

# 최단 경로 계산을 위한 그래프 생성
G = nx.Graph()

# 각 섹션의 후보 점들을 노드로 추가하고, 거리 가중치 계산하여 엣지 생성
for i in range(len(candidate_points) - 1):
    for j, (x1, y1) in enumerate(candidate_points[i]):
        for k, (x2, y2) in enumerate(candidate_points[i + 1]):
            distance = np.linalg.norm([x2 - x1, y2 - y1])
            G.add_edge((i, j), (i + 1, k), weight=distance)

# 최소 거리 경로 계산
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

# 최소 거리 경로 좌표 추출
x_shortest = [candidate_points[i][j][0] for i, j in shortest_path]
y_shortest = [candidate_points[i][j][1] for i, j in shortest_path]

# 최소 거리 경로 시각화
fig = plt.figure(figsize=(10, 6))
plt.plot(x_coords, y_coords, label='Road Centerline', color='blue')
plt.scatter(x_shortest, y_shortest, label='Shortest Path', color='red', s=10)
plt.plot(x_coords - 3.4 * (dy / magnitude), y_coords + 3.4 * (dx / magnitude), label='Real Boundary', color='black')
plt.plot(x_coords + 3.4 * (dy / magnitude), y_coords - 3.4 * (dx / magnitude), color='black')
plt.plot(left_x_coords, left_y_coords, label='Left Road Boundary', color='green')
plt.plot(right_x_coords, right_y_coords, label='Right Road Boundary', color='orange')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title('Shortest Path on Road Boundaries')
plt.legend()
plt.show()

# 최단 경로를 DataFrame으로 변환
shortest_path_df = pd.DataFrame({
    'x': x_shortest,
    'y': y_shortest,
    'z': [0] * len(x_shortest)  # z 값을 0으로 설정 (기본값)
})

# 저장 경로 설정 (원본과 동일한 형식)
output_path = '/mnt/data/mobile_system_control/pathfinder/competition_path.csv'
shortest_path_df.to_csv(output_path, sep=',', index=False, header=False)
print(f"최단 경로가 {output_path}에 저장되었습니다.")