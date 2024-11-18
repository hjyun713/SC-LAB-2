from PyQt6.QtWidgets import *
from pyqtgraph import PlotWidget, mkPen
from serial import Serial
import csv
import time

try:
    # CSV 파일을 생성하거나, 이미 존재하는 파일에 데이터를 추가로 저장합니다.
    with open('320T.csv', 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)

        # 시리얼 포트 설정
        emg = Serial('COM5', baudrate=250000)

        if emg.readable():
            recv_data = int(emg.readline().decode().split(',')[1])
            raw_list = [recv_data] * 10

        start_time = time.time()  # 시작 시간 기록
        sample_count = 0  # 샘플 개수 초기화

        while True:
            current_time = time.time()  # 현재 시간 기록
            elapsed_time = current_time - start_time  # 경과 시간 계산

            # 새로운 센서값을 읽어 리스트에 추가
            new_data = int(emg.readline().decode().split(',')[1])
            raw_list.append(new_data)
            raw_list.pop(0)  # 가장 오래된 값 제거

            # CSV 파일에 데이터를 한 줄로 저장
            csvwriter.writerow([new_data])

            sample_count += 1  # 샘플 개수 증가

            # 1초가 경과했을 때 샘플링 레이트 출력
            if elapsed_time >= 1:
                print(f"Samples per second: {sample_count}")
                start_time = time.time()  # 시작 시간 재설정
                sample_count = 0  # 샘플 개수 초기화

except Exception as e:
    # QMessageBox의 첫 번째 인자로 부모 위젯이 필요함. None을 전달하거나 부모 위젯을 전달
    QMessageBox.warning(None, "Warning", str(e))