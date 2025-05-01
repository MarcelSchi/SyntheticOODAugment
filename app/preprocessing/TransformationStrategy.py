from abc import ABC, abstractmethod


# Helps to define preprocessing steps via augmentation by inheriting from this function
class TransformationStrategy(ABC):
    def __init__(self, config):
        self.config = config

    @abstractmethod
    def get_transform_operations(self):
        pass
