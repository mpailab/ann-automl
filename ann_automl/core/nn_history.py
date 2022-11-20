from abc import ABC, abstractmethod


class HRecord:
    @staticmethod
    def from_json(json):
        class_name = json['class']
        if not all(c.isalnum() or c == '_' for c in class_name):
            raise ValueError(f'Invalid class name: {class_name}')
        class_ = globals()[class_name]
        if not issubclass(class_, HRecord):
            raise ValueError(f'Invalid record class: {class_name}, not subclass of HRecord')
        params = json['params']
        return class_.from_json(**params)

    def _to_json(self):
        return self.__dict__

    def to_json(self):
        return {'class': self.__class__.__name__, 'params': self._to_json()}


class TaskRecord(HRecord):
    def __init__(self, nn_task=None, objects=None, ):
        pass

    @abstractmethod
    def to_json(self):
        pass


class TrainingRecord(HRecord):
    def __init__(self, objects, ):
        pass

    def to_json(self):
        pass
