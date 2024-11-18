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


def binder(client_socket, addr):
    print('Connected by', addr)
    valuelist = []
    Bav = [0]
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
            model.load_state_dict(torch.load('weight_Mandro_HJ_RAW2/700.pt'))
        elif mode == 4:
            model = PModel8114().to(device)
            model.load_state_dict(torch.load('weights/400.pt'))

        ard = Serial("com3", 9600)

        if ard.readable():
            recv_data_C1 = int(ard.readline().decode().split(',')[0])
            recv_data_C2 = int(ard.readline().decode().split(',')[1])
            recv_data_C3 = int(ard.readline().decode().split(',')[2])
            recv_data_C4 = int(ard.readline().decode().split(',')[3])
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

                raw_list_C1.append(int(ard.readline().decode().split(',')[0]))
                raw_list_C2.append(int(ard.readline().decode().split(',')[1]))
                raw_list_C3.append(int(ard.readline().decode().split(',')[2]))
                raw_list_C4.append(int(ard.readline().decode().split(',')[3]))
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
                # print(data)
                with torch.cuda.amp.autocast(enabled=amp):
                    output = model(data)
                    cls_out = torch.argmax(output, dim=1)
                # print(output)
                # print(cls_out)
                value = most_common_value(cls_out.tolist())

                valuelist.append(value)
                cav = most_common_value(valuelist)
                if len(valuelist) == 80:
                    valuelist = []

                Bav.append(cav)
                if Bav[0] != Bav[1]:
                    datac = str((Bav[1]))
                    if mode == 1:
                        if datac == '0':
                            print('REST')
                        elif datac == '1':
                            print('PINCH')
                        elif datac == '2':
                            print('ROCK')
                        elif datac == '3':
                            print('PAPER')
                    else:
                        if datac == '0':
                            print('REST')
                        elif datac == '1':
                            print('PINCH')
                        elif datac == '2':
                            print('ROCK')
                        elif datac == '3':
                            print('PAPER')
                    if test_mode:
                        if Bav[0] != 0 and Bav[1] != 0:
                            pass
                        else:
                            print(datac)
                            # encode_data = str(datac).encode()
                            encode_data = str(datac + '\r').encode()
                            print(encode_data)
                            client_socket.send(encode_data)
                    else:
                        print(datac)
                        encode_data = str(datac).encode()
                        client_socket.send(encode_data)
                    del Bav[0]
                else:
                    del Bav[0]


    except socket.error as e:
        print("Socket error: {}".format(str(e)))
    finally:
        client_socket.close()



def execute():
    try:
        while True:
            client_socket, addr = server_socket.accept()
            # client_socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            # client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 10)
            # client_socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 5)
            binder(client_socket, addr)

    except:
        print("holly molly")

    finally:
        server_socket.close()
        print("Server Done")



if __name__ == "__main__":
    print('server start')
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_socket.bind(('', 50003))
    server_socket.listen()

    my_thread = threading.Thread(target=execute)
    my_thread.start()