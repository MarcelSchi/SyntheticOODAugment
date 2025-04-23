from abc import ABC, abstractmethod


class TransformationStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_transform_operations(self):
        pass
