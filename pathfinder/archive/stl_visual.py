from stl import mesh
import numpy as np
import matplotlib.pyplot as plt
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
    ax.add_collection3d(Poly3DCollection(map_mesh.vectors, alpha=1, facecolor='red', linewidths=0.5, edgecolors='black'))

    # 축 설정
    scale = map_mesh.points.flatten()
    ax.auto_scale_xyz(scale, scale, scale)
    ax.set_xlabel('X Axis')
    ax.set_ylabel('Y Axis')
    ax.set_zlabel('Z Axis')
    plt.title('STL Mesh Visualization')
    plt.show()

# 실행 예시
stl_file_path = '/mnt/data/mobile_system_control/pathfinder/mobilesystemcontrol-1.stl'
visualize_stl(stl_file_path)
