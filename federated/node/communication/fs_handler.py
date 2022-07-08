import os
import threading
from queue import Queue
import time
from requests import JSONDecodeError
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
        self._path = path
        self._q = Queue()
        os.makedirs(path, exist_ok=True)
        self.observer = Observer()
        self.observer.schedule(Handler(self._q), path=path, recursive=False)

    def __call__(self) -> Iterable[DataPacket]:
        self.observer.start()
        while True:
            try:
                msg = self._q.get(timeout=0.5)
                if 'model' in msg.body:
                    msg.body['model'] = model_from_packet_body(msg.body['model'])
                
                yield msg
            except:
                yield None


class Handler(FileSystemEventHandler):

    def __init__(self, packet_q: Queue) -> None:
        super().__init__()
        self._q = packet_q

    def on_created(self, event):
        if event.is_directory:
            return None
        print(f"New file created: {event.src_path}", flush=True)
        time.sleep(1)
        success = False
        while not success:
            try:
                print("Waiting...")
                packet = DataPacket.from_file(event.src_path)
                success = True
            except:
                print("Read failed, retrying in 0.5 seconds.", flush=True)
                time.sleep(0.5)
        print(f"Packet from {event.src_path} red successfully.", flush=True)
        self._q.put(packet)
        os.unlink(event.src_path)
