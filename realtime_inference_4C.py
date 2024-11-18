from torchmetrics import Accuracy
from serial import Serial
from models.TestCustomModel import PModelHJY, PModelHHJ84, PModel8114
import torch.distributed
import numpy as np
from collections import Counter
import time

#Predict Mode (Select Weight)
mode = 3

local_rank = 0
time_slot = 10
amp = False
box = np.ones(10) / 10
start_time = time.time()  # 시작 시간 재설정
sample_count = 0  # 샘플 개수 초기화
amplification_factor = 20.0  # 원하는 증폭 배율을 설정합니다.


def most_common_value(input_list):
    counter = Counter(input_list)
    return counter.most_common(1)[0][0]


if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda', local_rank)
    elif torch.backends.mps.is_available():
        device = torch.device(f'mps', local_rank)
    else:
        device = torch.device(f'cpu')
    accuracy = Accuracy(task="multiclass", num_classes=4)
    if mode == 1:
        model = PModelHJY().to(device)
        model.load_state_dict(torch.load('./weight_Mandro_HJ/010.pt'))
    elif mode == 2:
        model = PModelHHJ84().to(device)
        model.load_state_dict(torch.load('./weight_Mandro_HJ_RAW/010.pt'))
    elif mode == 3:
        model = PModel8114().to(device)
        model.load_state_dict(torch.load('./weight_Mandro_HJ_AVGF/700.pt'))
    elif mode == 4:
        model = PModel8114().to(device)
        model.load_state_dict(torch.load('weights/400.pt'))

    ard = Serial("com4", 9600)

    if ard.readable():
        recv_data_C1 = int(float(ard.readline().decode().split(',')[0]) * amplification_factor - 2000)
        recv_data_C2 = int(float(ard.readline().decode().split(',')[1]) * amplification_factor - 2000)
        recv_data_C3 = int(float(ard.readline().decode().split(',')[2]) * amplification_factor - 2000)
        recv_data_C4 = int(float(ard.readline().decode().split(',')[3]) * amplification_factor - 2000)
        raw_list_C1 = [recv_data_C1] * 10
        raw_list_C2 = [recv_data_C2] * 10
        raw_list_C3 = [recv_data_C3] * 10
        raw_list_C4 = [recv_data_C4] * 10
        avg_list_C1 = [sum(raw_list_C1)/10] * 10
        avg_list_C2 = [sum(raw_list_C2) / 10] * 10
        avg_list_C3 = [sum(raw_list_C3) / 10] * 10
        avg_list_C4 = [sum(raw_list_C4) / 10] * 10

        fdata_C1 = [np.convolve(avg_list_C1, box, mode='same')[-1]] *10
        fdata_C2 = [np.convolve(avg_list_C2, box, mode='same')[-1]] * 10
        fdata_C3 = [np.convolve(avg_list_C3, box, mode='same')[-1]] * 10
        fdata_C4 = [np.convolve(avg_list_C4, box, mode='same')[-1]] * 10

        while True:
            del raw_list_C1[0], avg_list_C1[0], fdata_C1[0]
            del raw_list_C2[0], avg_list_C2[0], fdata_C2[0]
            del raw_list_C3[0], avg_list_C3[0], fdata_C3[0]
            del raw_list_C4[0], avg_list_C4[0], fdata_C4[0]

            raw_list_C1.append(int(ard.readline().decode().split(',')[0]) * amplification_factor - 2000)
            raw_list_C2.append(int(ard.readline().decode().split(',')[1]) * amplification_factor - 2000)
            raw_list_C3.append(int(ard.readline().decode().split(',')[2]) * amplification_factor - 2000)
            raw_list_C4.append(int(ard.readline().decode().split(',')[3]) * amplification_factor - 2000)
            avg_list_C1.append(sum(raw_list_C1)/10)
            avg_list_C2.append(sum(raw_list_C2) / 10)
            avg_list_C3.append(sum(raw_list_C3) / 10)
            avg_list_C4.append(sum(raw_list_C4) / 10)
            fdata_C1.append(np.convolve(avg_list_C1, box, mode='same')[-1])
            fdata_C2.append(np.convolve(avg_list_C2, box, mode='same')[-1])
            fdata_C3.append(np.convolve(avg_list_C3, box, mode='same')[-1])
            fdata_C4.append(np.convolve(avg_list_C4, box, mode='same')[-1])

            current_time = time.time()  # 현재 시간 기록
            elapsed_time = current_time - start_time  # 경과 시간 계산
            sample_count += 1  # 샘플 개수 증가
            # 1초가 경과했을 때 샘플링 레이트 출력
            if elapsed_time >= 1:
                print(f"Samples per second: {sample_count}")
                start_time = time.time()  # 시작 시간 재설정
                sample_count = 0  # 샘플 개수 초기화
            # print(fdata_C1)

            # fdata = np.array([fdata_C1, fdata_C2, fdata_C3, fdata_C4]) # Shape: (time_slot, num_channels)
            fdata = np.array([avg_list_C1, avg_list_C2, avg_list_C3, avg_list_C4]) # Shape: (time_slot, num_channels)
            # print(fdata)
            fdata = np.transpose(fdata)
            data = torch.from_numpy(np.reshape(np.array(fdata, dtype=np.float32), (time_slot, 4))).to(device)

            # print(data)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(data)
                cls_out = torch.argmax(output, dim=1)
            # print(cls_out)
            value = most_common_value(cls_out.tolist())
            if value == 0:
                print('rest')
            elif value == 1:
                print('PINCH')
            elif value == 2:
                print('ROCK')
            elif value == 3:
                print('PAPER')