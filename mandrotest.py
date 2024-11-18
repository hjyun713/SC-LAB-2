import socket
import threading
from serial import Serial
from models.TestCustomModel import PModelHJY, PModelHHJ84, PModel8114
import torch.distributed
import numpy as np
from collections import Counter

mode = 3
test_mode = True
time_slot = 10
amp = False
num_classes = 4
box = np.ones(10) / 10

def most_common_value(input_list):
    counter = Counter(input_list)
    return counter.most_common(1)[0][0]

def binder():
    valuelist = []
    Bav = [0]
    port = 'COM17'  # 사용하는 포트 이름
    baudrate = 115200  # 동글의 통신 속도에 맞춰 설정
    amplification_factor = 20.0  # 원하는 증폭 배율을 설정합니다.
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            device = torch.device(f'cuda', 0)
        elif torch.backends.mps.is_available():
            device = torch.device(f'mps', 0)
        else:
            device = torch.device(f'cpu')

        if mode == 1:
            model = PModelHJY().to(device)
            model.load_state_dict(torch.load('weights/hjy.pt'))
        elif mode == 2:
            model = PModelHHJ84().to(device)
            model.load_state_dict(torch.load('weights/hhj0329.pt'))
        elif mode == 3:
            model = PModel8114().to(device)
            model.load_state_dict(torch.load('weight_Mandro_HJ_AVGF/200.pt'))
        elif mode == 4:
            model = PModel8114().to(device)
            model.load_state_dict(torch.load('weights/400.pt'))

        ard = Serial("com4", 9600)
        ser = Serial(port, baudrate, timeout=1)

        if ard.readable():
            recv_data_C1 = int(float(ard.readline().decode().split(',')[0]) * amplification_factor - 2000)
            recv_data_C2 = int(float(ard.readline().decode().split(',')[1]) * amplification_factor - 2000)
            recv_data_C3 = int(float(ard.readline().decode().split(',')[2]) * amplification_factor - 2000)
            recv_data_C4 = int(float(ard.readline().decode().split(',')[3]) * amplification_factor - 2000)
            raw_list_C1 = [recv_data_C1] * 10
            raw_list_C2 = [recv_data_C2] * 10
            raw_list_C3 = [recv_data_C3] * 10
            raw_list_C4 = [recv_data_C4] * 10
            avg_list_C1 = [sum(raw_list_C1) / 10] * 10
            avg_list_C2 = [sum(raw_list_C2) / 10] * 10
            avg_list_C3 = [sum(raw_list_C3) / 10] * 10
            avg_list_C4 = [sum(raw_list_C4) / 10] * 10

            fdata_C1 = [np.convolve(avg_list_C1, box, mode='same')[-1]] * 10
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
                avg_list_C1.append(sum(raw_list_C1) / 10)
                avg_list_C2.append(sum(raw_list_C2) / 10)
                avg_list_C3.append(sum(raw_list_C3) / 10)
                avg_list_C4.append(sum(raw_list_C4) / 10)
                fdata_C1.append(np.convolve(avg_list_C1, box, mode='same')[-1])
                fdata_C2.append(np.convolve(avg_list_C2, box, mode='same')[-1])
                fdata_C3.append(np.convolve(avg_list_C3, box, mode='same')[-1])
                fdata_C4.append(np.convolve(avg_list_C4, box, mode='same')[-1])

                fdata = np.array(
                    [raw_list_C1, raw_list_C2, raw_list_C3, raw_list_C4])  # Shape: (time_slot, num_channels)
                fdata = np.transpose(fdata)
                data = torch.from_numpy(np.reshape(np.array(fdata, dtype=np.float32), (time_slot, 4))).to(device)

                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(data)
                    cls_out = torch.argmax(output, dim=1)

                value = most_common_value(cls_out.tolist())
                valuelist.append(value)
                cav = most_common_value(valuelist)
                if len(valuelist) == 300:
                    valuelist = []

                Bav.append(cav)
                if Bav[0] != Bav[1]:
                    datac = str(Bav[1])

                    # 시리얼 포트를 통해 데이터 전송
                    if datac == '0':  # 휴식
                        print('REST')
                        ser.write(bytes([0xAA, 0x55, 0x01, 0x01, 0x01, 0x01, 0x01, 0x05, 0xFF,
                                         0x02, 0xFF, 0x00, 0xFF, 0x02, 0xFB]))
                    elif datac == '2':  # 주먹
                        print('ROCK')
                        ser.write(bytes([0xAA, 0x55, 0x01, 0x01, 0x01, 0x01, 0x01, 0x05, 0xFF,
                                         0x04, 0xFF, 0x00, 0xFF, 0x01, 0xFE]))
                    elif datac == '1':
                        print('PINCH')
                    elif datac == '3':
                        print('PAPER')

                    del Bav[0]
                else:
                    del Bav[0]

    except Exception as e:
        print("오류 발생:", e)
    finally:
        ard.close()  # 시리얼 포트 닫기


if __name__ == "__main__":
    print('Server Start')
    binder()