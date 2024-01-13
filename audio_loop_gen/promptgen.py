import sqlite3
import uuid
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import os

class DBWriter:
    def __init__(self, db_path):
        self.__logger = logging.getLogger("global")
        self.conn = sqlite3.connect(db_path, check_same_thread=False)

    def write_row(self, prompt, bpm):
        with self.conn:
            self.conn.execute(
                "INSERT INTO Prompts (id, prompt, bpm, ts) VALUES (?, ?, ?, CURRENT_TIMESTAMP)",
                (str(uuid.uuid4()), prompt, bpm))
            
