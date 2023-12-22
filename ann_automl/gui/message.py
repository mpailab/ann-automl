import json
from typing import List

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
