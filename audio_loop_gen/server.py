import json
import queue
import uuid
import asyncio
import websockets
import logging
import threading

from .audiogen import AudioGenerator
from .loopgen import LoopGenerator
from .util import LoopGenParams, AudioData, calculate_checksum

FULL_MESSAGE = json.dumps({
    "type": "rejected",
    "reason": "FULL"
})

INVALID_REQUEST_MESSAGE = json.dumps({
    "type": "rejected",
    "reason": "INVALID"
})

class LoopGenJob:
    """ Wraps a LoopGenParams object with a UUID and a reference to the client connection that requested the job.
    """
    def __init__(self, params: LoopGenParams, connection: 'LoopGenClientConnection'):
        assert params is not None, "params is required"
        assert connection is not None, "connection is required"
        self.__uuid = uuid.uuid4()
        self.__params = params
        self.__connection = connection
    
    @property
    def uuid(self) -> str:
        return str(self.__uuid)
        
    @property
    def params(self) -> LoopGenParams:
        return self.__params
    
    @property
    def connection(self) -> 'LoopGenClientConnection':
        return self.__connection

class LoopGenClientConnection:
    """ Represents a client connected to the server. Handles sending and receiving messages to/from the client.
    """
    def __init__(self, websocket, task_queue: queue.Queue, logger):
        assert websocket is not None, "websocket is required"
        assert task_queue is not None, "task_queue is required"
        assert logger is not None, "logger is required"
        
        self.__websocket: websockets.WebSocketServerProtocol = websocket
        self.__logger = logger
        self.__queue = task_queue
        self.__connected = False
        
        self.__main_loop = asyncio.get_running_loop()
    
    @property
    def connected(self) -> bool:
        return self.__connected
    
    async def listen(self):
        """ The main loop for the client. Listens for messages from the client and handles them appropriately.
        """
        self.__connected = True
        try:
            async for message in self.__websocket:
                self.__logger.debug("Received message %s", message)
                
                try:
                    json_message = json.loads(message)
                    job = self.__create_job(json_message)
                except Exception as e:
                    self.__logger.error("Error handling request [%s]: %s", message, str(e))
                    await self.__send(INVALID_REQUEST_MESSAGE)
                    continue
                try:
                    self.__queue.put_nowait(job)
                    await self.__accept(job)
                except queue.Full as e:
                    await self.__send(FULL_MESSAGE)
                    continue
        except websockets.ConnectionClosedError:
            self.__logger.info("Connection closed")
        except Exception as e:
            self.__logger.error("Error listening for messages: %s", str(e))
        finally:
            self.__connected = False
            try:
                await self.__websocket.close()
            except:
                pass
            
    def progress(self, job: LoopGenJob, progress: float):
        """ Sends a progress update to the client.
        
        Args:
            job (LoopGenJob): The job to send the progress for.
            progress (float): The progress of the job, between 0 and 1.
        """
        msg = json.dumps({
            "type": "progress",
            "uuid": job.uuid,
            "progress": progress
        })
        self.__send(msg)
        
    def failure(self, job: LoopGenJob):
        """ Sends a failure message to the client.

        Args:
            job (LoopGenJob): The job that failed.
        """
        self.__logger.debug("Sending failure message for job %s", job.uuid)
        msg = json.dumps({
            "type": "error",
            "uuid": job.uuid,
            "reason": "FAIL"
        })
        self.__send(msg)
        
    def success(self, job: LoopGenJob, audio: AudioData):
        """ Sends a success message to the client.

        Args:
            job (LoopGenJob): The job that succeeded.
            audio (AudioData): The audio data with the generated loop to send to the client.
        """
        data = AudioData.serialize(audio)
        checksum = calculate_checksum(data)
        msg = json.dumps({
            "type": "success",
            "uuid": job.uuid,
            "checksum": checksum,
            "size": len(data)
        })
        self.__logger.debug("Sending success for job %s", job.uuid)
        self.__send(msg)
        self.__logger.debug("Sending audio data for job %s in chunks", job.uuid)
        chunk_size = 1024
        for i in range(0, len(data), chunk_size):
            chunk = data[i:i+chunk_size]
            self.__send(chunk)

    def __send(self, message):
        if not self.__connected:
            return
        # This method schedules the sending of a message on the main event loop.
        if self.__connected:
            asyncio.run_coroutine_threadsafe(self.__websocket.send(message), self.__main_loop)
        else:
            self.__logger.warning("WebSocket is not connected. Message not sent.")
        
    async def __accept(self, job: LoopGenJob):
        msg = json.dumps({
            "type": "accepted",
            "uuid": job.uuid
        })
        if not self.__connected:
            return
        await self.__websocket.send(msg)
    
    def __create_job(self, json_message):
        if not "prompt" in json_message:
            self.__logger.error("Invalid request from client: %s", json_message)
            raise ValueError("Params must contain a prompt")
        
        params = LoopGenParams(**json_message)
        return LoopGenJob(params, self)

class LoopGeneratorServer:
    """ A server that generates audio loops using the provided AudioGenerator and sends them to clients.
    """
    def __init__(self, audiogen: AudioGenerator, port: int = 8081, max_queue_size: int = 100):
        """_summary_

        Args:
            audiogen (AudioGenerator): The AudioGenerator to use to generate audio loops.
            port (int, optional): The port to start the server on. Defaults to 8081.
            max_queue_size (int, optional): The maximum number of jobs to queue. Defaults to 100.

        Raises:
            ValueError: If the port is not between 0 and 65535.
        """
        assert audiogen is not None, "audiogen is required"
        assert port is not None, "port is required"
        if port < 0 or port > 65535:
            raise ValueError("Invalid port number")
        
        self.__port = port
        self.__audiogen = audiogen
        self.__logger = logging.getLogger("global")
        self.__queue = queue.Queue(maxsize=max_queue_size if max_queue_size and max_queue_size > 0 else 100)
        
    def start(self):
        """Starts the server on the configured port. This method blocks until the server is stopped.
        """
        self.__logger.info("Starting server...")
        asyncio.run(self.__run())
        
    async def __run(self):
        # Start a thread to handle jobs from the queue
        threading.Thread(target=self.__job_worker, daemon=True).start()
        # Start the server
        self.__logger.info("Starting server on port %d", self.__port)
        
        async with websockets.serve(self.__handle_client, "localhost", self.__port, ping_timeout=60):
            await asyncio.Future()  # Run forever
            
    async def __handle_client(self, websocket: websockets.WebSocketServerProtocol):
        con = LoopGenClientConnection(websocket, self.__queue, self.__logger)
        await con.listen()
        
    def __job_worker(self):
        while True:
            # Generate an audio loop using the provided parameters
            # and send it to the client
            job: LoopGenJob = self.__queue.get()
            if job.connection is None or not job.connection.connected:
                self.__logger.error("Connection is closed. Job %s will not be processed.", job.uuid)
                self.__queue.task_done()
                continue
            
            # keep the client updated on the progress of the job
            def progress_callback(generated: int, total: int):
                step = total // 20
                if step > 0 and generated % step == 0:
                    job.connection.progress(job, generated/total)
                    print(f'.', end='', flush=True)
                
            self.__audiogen.set_custom_progress_callback(progress_callback)
                
            try:
                print("\nGenerating loop:")
                sr, audio_data = self.__audiogen.generate(job.params)
                # don't loose more than 1/3 of the audio
                loopgen = LoopGenerator(AudioData(audio_data, sr), job.params)
                loop = loopgen.generate()
                
                job.connection.success(job, loop)
            except Exception as e:
                self.__logger.error("Error generating loop: %s", str(e))
                job.connection.failure(job)
            finally:
                self.__queue.task_done()