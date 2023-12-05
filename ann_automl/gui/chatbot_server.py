import socket

CONNECTION_WORD = "N"

def lingvmodel(request):
    return "Привет!"

def server_program(host, port):

    server_socket = socket.socket()  # get instance
    server_socket.bind((host, port))  # bind host address and port together

    # configure how many client the server can listen simultaneously
    server_socket.listen(2)
    conn, address = server_socket.accept()  # accept new connection

    to_client = CONNECTION_WORD
    while True:
        from_client = conn.recv(1024).decode()
        conn.send(to_client.encode())
        to_client = CONNECTION_WORD
        if from_client != CONNECTION_WORD:
            to_client = lingvmodel(from_client)
            from_client = CONNECTION_WORD
    conn.close()  # close the connection

if __name__ == '__main__':
    server_program("0.0.0.0", 5000)