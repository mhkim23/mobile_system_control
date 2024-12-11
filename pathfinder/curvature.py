import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def compute_curvature(x, y):
    """
    주어진 x, y 좌표 배열로부터 곡률을 계산하는 함수.
    x, y: numpy.ndarray 형태로, 경로상의 각 점의 x, y 좌표
    returns: numpy.ndarray 형태의 곡률 배열
    """
    dx = np.gradient(x)
    dy = np.gradient(y)
    ddx = np.gradient(dx)
    ddy = np.gradient(dy)
    curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2)**1.5
    return curvature

def main():
    # CSV 파일 로드
    data = pd.read_csv('/mnt/data/mobile_system_control/pathfinder/reference_path (1).csv', header=None)
    # 컬럼명 직접 할당 (헤더 없음 가정)
    data.columns = ['x', 'y', 'z']
    x = data['x'].values
    y = data['y'].values

    # 곡률 계산 함수 호출
    curvature = compute_curvature(x, y)
    
    # 곡률 데이터를 CSV 파일로 저장
    curvature_df = pd.DataFrame({'curvature': curvature})
    curvature_df.to_csv('./path_curvature.csv', index=False)


    # 결과 시각화
    plt.plot(curvature)
    plt.title('Curvature')
    plt.xlabel('Point Index')
    plt.ylabel('Curvature')
    plt.show()

if __name__ == "__main__":
    main()
