import sqlite3
import uuid
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import logging
import os

class PromptWatcher:
    def __init__(self, db_path):
        self.__logger = logging.getLogger("global")
        self.__db_path = db_path
        self.__db_conn = sqlite3.connect(db_path, check_same_thread=False)
        self.__db_observer = Observer()
        self.__last_check = time.time()
        self.__poll_interval = 30  # 30 seconds
        self._setup_file_watch()

    def _setup_file_watch(self):
        event_handler = FileSystemEventHandler()
        event_handler.on_modified = self.__on_db_change
        self.__db_observer.schedule(event_handler, path=os.path.dirname(self.__db_path), recursive=False)
        self.__db_observer.start()

    def __on_db_change(self, event):
        if time.time() - self.__last_check >= self.__poll_interval:
            self._process_db_changes()

    def _process_db_changes(self):
        self.__last_check = time.time()
        while True:
            row = self._get_oldest_row()
            if row is None:
                break
            try:
                self.handle(row)
                self._delete_row(row[0])
            except Exception as e:
                logging.error(f"Error processing row {row}: {e}")

    def _get_oldest_row(self):
        row = self.__db_conn.execute("SELECT * FROM Prompts ORDER BY ts ASC LIMIT 1").fetchone()
        return row

    def _delete_row(self, row_id):
        with self.__db_conn:
            self.__db_conn.execute("DELETE FROM Prompts WHERE id = ?", (row_id,))

    def handle(self, row):
        # Implement your handling logic here
        print(f"Handling row: {row}")

    def stop(self):
        self.__db_observer.stop()
        self.__db_observer.join()

# Usage example
db_path = 'data.db'
db_writer = DBWriter(db_path)
db_listener = DbListener(db_path)

# Simulate DB writing
db_writer.write_row("Example prompt", 120)

# Give some time to process
time.sleep(5)

# Stop the listener
db_listener.stop()