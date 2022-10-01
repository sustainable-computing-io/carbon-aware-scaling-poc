from abc import ABC, abstractmethod

class DataProvider(ABC):
    @abstractmethod
    def read(self):
        return NotImplemented
    