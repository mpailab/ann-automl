import socket
import time

from ann_automl.gui.message import Message
from bot import LLMBot

def my_test_lingvmodel(request):
    res = f"Ответ на {request}!"
    if request.isdigit():
        time.sleep(int(request))
        res += f" Задержка {request} сек."
    return res

def server_program(host, port):

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection

    # LLM model init
    bot = LLMBot()

    requests_to_client = []
    requests_from_client = []
    while True:
        data = conn.recv(4096).decode()
        message_from_client = Message.unpack(data)
        requests_from_client += message_from_client.requests

        if requests_from_client:
            request = requests_from_client.pop(0)
            for to_client in bot.request(request):
                message_to_client = Message([to_client])
                conn.send(str(message_to_client).encode())
                
    conn.close()  # close the connection

if __name__ == '__main__':
    #TODO Интерфейс командной строки -h <host> -p <port>
    host = "0.0.0.0"
    port = 5000
    server_program(host, port)
    