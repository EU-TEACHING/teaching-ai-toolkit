import os
from queue import Queue
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from typing import Iterable, Iterator
from base.communication.packet import DataPacket

from .serialization import model_from_packet_body, model_to_packet_body

import keras

class FileSystemProducer:

    def __init__(self, path: str) -> None:
        self._out_dir = path
        os.makedirs(path, exist_ok=True)
        self._n_packet = {}

    def __call__(self, msg_stream: Iterator[DataPacket]) -> None:
        for msg in msg_stream:
            if msg.service_name not in self._n_packet:
                self._n_packet[msg.service_name] = 0
            n = self._n_packet[msg.service_name]

            if 'model' in msg.body and isinstance(msg.body['model'], keras.Model):   
                msg.body['model'] = model_to_packet_body(msg.body['model'])
            msg.to_file(os.path.join(self._out_dir, f"{msg.service_name}_{n}.dat"))
            self._n_packet[msg.service_name] += 1


class FileSystemConsumer:

    def __init__(self, path: str) -> None:
        self._q = Queue()
        self._watcher = Watcher(path, self._q)
        self._watcher.run()

    def __call__(self) -> Iterable[DataPacket]:
        while True:
            msg = self._q.get()
            if 'model' in msg.body:
                msg.body['model'] = model_from_packet_body(msg.body['model'])
            
            yield msg
            

class Watcher:

    def __init__(self, path: str, packet_q: Queue):
        self._path = path
        self._q = packet_q
        self.observer = Observer()

    def run(self):
        event_handler = Handler(self._q)
        self.observer.schedule(event_handler, self._path, recursive=False)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Error")

        self.observer.join()


class Handler(FileSystemEventHandler):

    def __init__(self, packet_q: Queue) -> None:
        super().__init__()
        self._q = packet_q

    def on_created(self, event):
        if event.is_directory:
            return None

        packet = DataPacket.from_file(event.src_path)
        self._q.put(packet)
        os.unlink(event.src_path)
