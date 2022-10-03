from .storage_provider import StorageProvider
from threading import Lock

class EphermeralStorage(StorageProvider):
    def __init__(self):
        self.lock = Lock()
        self.storage = list()

    def produce_one(self, data):
        with self.lock:
            self.storage.append(data)

    def consume_one(self):
        with self.lock:
            if len(self.storage):
                return self.storage.pop()
        return None