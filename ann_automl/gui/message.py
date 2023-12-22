import json

SEPARATOR = "QQQQQQQ"

class Message:
    def __init__(self, requests = []):
        self.requests = requests

    def __repr__(self):
        return json.dumps(vars(self), separators=(',', ':'))
    
    def __str__(self):
        return self.__repr__() + SEPARATOR

    @staticmethod
    def unpack(string):
        requests = []
        for line in string.split(SEPARATOR)[:-1]:
            tmp_dict = json.loads(line)
            requests += tmp_dict['requests']
        return Message(requests)
