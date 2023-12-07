import socket
import json

def lingvmodel(request):
    return "Привет!"

class Message:
    def __init__(self, requests = [], info = ''):
        self.requests = requests
        self.info = info

    def __repr__(self):
        return json.dumps(vars(self), separators=(',', ':'))
    
    def __str__(self):
        return json.dumps(vars(self), separators=(',', ':'))

    @staticmethod
    def unpack(string):
        message = Message()
        tmp_dict = json.loads(string)
        message.__dict__.update(tmp_dict)
        return message

def server_program(host, port):

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection

    requests_to_client = []
    requests_from_client = []
    while True:
        message_from_client = Message.unpack(conn.recv(1024).decode())
        requests_from_client += message_from_client.requests

        message_to_client = Message(requests_to_client)
        conn.send(str(message_to_client).encode())
        requests_to_client = []

        if requests_from_client:
            request = requests_from_client.pop(0)

            #TODO обработка запроса в отдельном подпроцессе
            to_client = lingvmodel(request)

            requests_to_client.append(to_client)

    conn.close()  # close the connection

if __name__ == '__main__':
    #TODO Интерфейс командной строки -h <host> -p <port>
    server_program("0.0.0.0", 5000)