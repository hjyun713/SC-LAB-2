import socket

HOST = '192.168.2.124'  # 서버컴의 ip 주소 입력하면 됨
PORT = 50002    # 서버 코드에서 설정한 포트번호 입력하면 됨
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))
    num = 0
    while True:
        # num = num + 1
        s.sendall(b'Pre')
        
        data = s.recv(1024)
        print('Received', int(data.decode()))
        # print(num)
except:
    print("Except")
finally:
    print("Server Done...")
