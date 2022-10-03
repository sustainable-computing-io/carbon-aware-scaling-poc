from abc import ABC, abstractmethod
class StorageProvider(ABC):
    @abstractmethod
    def produce_one(self, data):
        return NotImplemented

    @abstractmethod
    def consume_one(self, data):
        return NotImplemented