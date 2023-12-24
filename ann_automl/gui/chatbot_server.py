import socket
import time
import json
from typing import List

def my_test_lingvmodel(request):
    res = f"Ответ на {request}!"
    if request.isdigit():
        time.sleep(int(request))
        res += f" Задержка {request} сек."
    return res

class Message:
    def __init__(self, requests : List[str] = []):
        self.requests = requests

    def __repr__(self):
        return json.dumps(vars(self), separators=(',', ':'))
    
    def __str__(self):
        return self.__repr__() + '\n'

    @staticmethod
    def unpack(string):
        requests = []
        for line in string.split('\n')[:-1]:
            tmp_dict = json.loads(line)
            requests += tmp_dict['requests']
        return Message(requests)

def server_program(host, port):

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection

    requests_to_client = []
    requests_from_client = []
    while True:
        data = conn.recv(1024).decode()
        message_from_client = Message.unpack(data)
        requests_from_client += message_from_client.requests

        message_to_client = Message(requests_to_client)
        conn.send(str(message_to_client).encode())
        requests_to_client = []

        if requests_from_client:
            request = requests_from_client.pop(0)
            to_client = my_test_lingvmodel(request)
            requests_to_client.append(to_client)

    conn.close()  # close the connection

if __name__ == '__main__':
    #TODO Интерфейс командной строки -h <host> -p <port>
    host = "0.0.0.0"
    port = 5000
    server_program(host, port)