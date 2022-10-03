from collectors.co2signal_collector import CO2SignalCollector
from handlers.co2signal_handler import CO2SignalHandler
from handlers.mock_provider import MockProvider
from storage.kafka import KafkaStorage
from storage.ephemeral import EphermeralStorage
import signal
import sys
import threading


def signal_handler(sig, frame):
    sys.exit(0)


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    storage_provider = EphermeralStorage()
    #co2_handler = CO2SignalHandler(storage_provider)
    co2_handler = MockProvider(storage_provider)
    collector = CO2SignalCollector(storage_provider)

    def handler():
        co2_handler.read()

    handler_thread = threading.Thread(target=handler)


    handler_thread.start()
    collector.export(storage_provider)