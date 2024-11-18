from torchmetrics import Accuracy
from serial import Serial
from models.TestCustomModel import PModelHJY, PModelHHJ84, PModel8114
import torch.distributed
import numpy as np
from collections import Counter

#Predict Mode (Select Weight)
mode = 3

local_rank = 0
time_slot = 10
amp = False
box = np.ones(10) / 10


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
        model.load_state_dict(torch.load('weights/hjy.pt'))
    elif mode == 2:
        model = PModelHHJ84().to(device)
        model.load_state_dict(torch.load('weights/hhj0329.pt'))
    elif mode == 3:
        model = PModel8114().to(device)
        model.load_state_dict(torch.load('weight_Mandro_HJ_RAW/050.pt'))
    elif mode == 4:
        model = PModel8114().to(device)
        model.load_state_dict(torch.load('weights/400.pt'))

    ard = Serial("com5", 250000)

    if ard.readable():
        recv_data = int(ard.readline().decode().split(',')[1])
        raw_list = [recv_data] * 10
        avg_list = [sum(raw_list)/10] * 10

        fdata = [np.convolve(avg_list, box, mode='same')[-1]] *10

        while True:
            del raw_list[0], avg_list[0], fdata[0]
            raw_list.append(int(ard.readline().decode().split(',')[1]))
            avg_list.append(sum(raw_list)/10)
            fdata.append(np.convolve(avg_list, box, mode='same')[-1])
            data = torch.from_numpy(np.reshape(np.array(fdata, dtype=np.float32), (time_slot, 1))).to(device)
            print(data)
            with torch.cuda.amp.autocast(enabled=amp):
                output = model(data)
                cls_out = torch.argmax(output, dim=1)
            # print(output)
            value = most_common_value(cls_out.tolist())
            if value == 0:
                print('rest')
            elif value == 1:
                print('PAPER')
            elif value == 2:
                print('SCISSOR')
            elif value == 3:
                print('ROCK')