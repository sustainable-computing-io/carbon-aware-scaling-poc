from collectors.co2signal_collector import CO2SignalCollector
from handlers.co2signal_handler import CO2SignalHandler
import signal
import sys
import threading


co2_handler = CO2SignalHandler()
collector = CO2SignalCollector()

def signal_handler(sig, frame):
    sys.exit(0)

def handler():
    co2_handler.read()

handler_thread = threading.Thread(target=handler)

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)

    handler_thread.start()
    collector.export()