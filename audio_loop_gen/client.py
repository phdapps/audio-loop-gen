import asyncio
import websockets
import json
import logging

from .util import AudioData, LoopGenParams, calculate_checksum
from .promptgen import PromptGenerator
from .store import AudioStore

TIMEOUT_JOB_CONFIRMATION_SEC:float = 10.0
MIN_JOBS:int = 5
PERIODIC_JOB_GENERATION_INTERVAL_SEC:int = 600

class LoopGenClient:
    def __init__(self, prompt_generator: PromptGenerator, store: AudioStore, host: str = "localhost", port: int = 8081, max_jobs: int = 10):
        assert prompt_generator is not None
        assert store is not None
        self.__host = host if host is not None else "localhost"
        self.__port = port if (port is not None and port > 0) else 8081
        self.__store = store
        self.__prompt_generator = prompt_generator
        self.__logger = logging.getLogger("global")
        self.__websocket = None
        self.__jobs:dict = {}
        self.__max_jobs = max_jobs if max_jobs and max_jobs > 0 else 10
        
        # The current job waiting for acceptance or rejection
        self.__pending_job:object = None
        # allows signalling when the acceptance or error response for a request is received
        self.__response_event: asyncio.Event = asyncio.Event()
        self.__request_lock: asyncio.Lock = asyncio.Lock()

    def start(self):
        asyncio.run(self.__startup())
        
    async def __startup(self):
        try:
            server_uri = f"ws://{self.__host}:{self.__port}"
            self.__websocket = await websockets.connect(server_uri)
            self.__logger.info("Connected to server at %s", server_uri)
            self.__response_event.set()
            self.listen_for_responses_task = asyncio.create_task(self.__listen_for_responses())
            await self.__generate_jobs()
        except Exception as e:
            self.__logger.error(f"Connection failed: {e}")
            raise e

    async def __send_job(self, params:LoopGenParams):
        async with self.__request_lock:
            self.__response_event.clear()
            self.__logger.debug("Sending request %s", params)
            self.__pending_job = params
            try:
                if self.__websocket:
                    self.__pending_job = params
                    await self.__websocket.send(json.dumps(params))
                    self.__logger.info("Request sent")
                    await asyncio.wait_for(self.__response_event.wait(), timeout=TIMEOUT_JOB_CONFIRMATION_SEC)
                else:
                    self.__logger.error("Not connected to the server")
            except asyncio.TimeoutError:
                self.__logger.error("Request timed out")
            except Exception as e:
                self.__logger.error(f"Failed to send request: {e}")
            
    async def __listen_for_responses(self):
        try:
            if self.__websocket:
                async for message in self.__websocket:
                    if isinstance(message, str):
                        self.__handle_message(json.loads(message))
                    else:
                        self.__handle_binary_message(message)
            else:
                self.__logger.error("Not connected to the server")
        except Exception as e:
            self.__logger.error(f"Error in listening for responses: {e}")

    def __handle_message(self, response:dict):
        try:
            response_type = response.get('type')
            uuid = response.get('uuid')
            job = None
            if uuid:
                job = self.__jobs.get(uuid)
            if uuid and job is None and response_type != 'accepted':
                self.__logger.warning(f"Received message for unknown job {uuid}")
                return
            if response_type == 'accepted' or response_type == 'rejected':
                if job:
                    self.__logger.warning(f"Received {response_type} response for job {uuid} that is already in progress")
                    return
                try:
                    if response_type == 'accepted' and uuid:
                        self.__logger.debug("Request accepted. UUID: %s", uuid)
                        self.__jobs[uuid] = self.__pending_job
                    else:
                        reason = response.get('reason') if response_type == 'rejected' else f"invalid response {response}"
                        self.__logger.error(f"Request rejected: {reason}")
                finally:
                    self.__pending_job = None
                    self.__response_event.set() # signal that the job request is done and another one can be sent
            elif response_type == 'success':
                self.__logger.info(f"Success! UUID: {response['uuid']}, Checksum: {response['checksum']}")
                self.__data_receiver = DataReceiver(uuid, job, response['checksum'], response['size'], self.__logger)
                # No need to keep around and allow new jobs to be sent while the data is received
                # @TODO May need to refactor if this is not practical anymore
                self.__finish_job(uuid)
            elif response_type == 'progress':           
                progress = response.get('progress') or 0
                self.__logger.debug("Progress for job %s: %f\%", uuid, progress * 100)
            elif response_type == 'error':
                self.__logger.error("Job %s failed: %s", uuid, response.get('reason'))
                self.__finish_job(uuid)
            else:
                self.__logger.warning(f"Unknown response type: {response} for job {uuid}")
        except Exception as e:
            self.__logger.error(f"Error handling server message {response}: {e}")

    def __handle_binary_message(self, data):
        if self.__data_receiver is not None:
            try:
                self.__data_receiver.receive(data)
            except Exception as e:
                self.__logger.error("Error receiving binary data: %s", e)
                self.__data_receiver = None
                
            if self.__data_receiver.status == 1:
                self.__logger.debug("Completed job %s", self.__data_receiver.uuid)
                data = self.__data_receiver.data
                audio = AudioData.deserialize(data)
                self.__store.store(audio, self.__data_receiver.job)
                self.__data_receiver = None
        else:
            self.__logger.error("Received binary data without a succesful job")
            
    def __finish_job(self, uuid:str):
        del self.__jobs[uuid]
        self.__generate_jobs()
            
    async def __generate_jobs(self):
        if len(self.__jobs) < MIN_JOBS:
            batch_size = self.__max_jobs - len(self.__jobs)
            jobs = self.__prompt_generator.generate(max_count=batch_size)
            for job in jobs:
                await self.__send_job(job)
        # call from time to time to make sure an error in the main loop doesn't leave us without jobs   
        asyncio.ensure_future(self.__delayed_generate_jobs())
        
    async def __delayed_generate_jobs(self):
        await asyncio.sleep(PERIODIC_JOB_GENERATION_INTERVAL_SEC)
        await self.__generate_jobs()
            
class DataReceiver:
    def __init__(self, uuid: str, job: LoopGenParams, checksum: str, size: int, logger: logging.Logger):
        assert uuid is not None
        assert checksum is not None
        assert size is not None and size > 0
        assert logger is not None
        
        self.__uuid = uuid
        self.__job = job
        self.__checksum = checksum
        self.__size = size
        self.__logger = logger
        self.__received_size = 0
        self.__data = bytearray()
        self.__status = 0 # 0 = receiving, 1 = done, -1 = error
        
    @property
    def status(self) -> int:
        return self.__status
    
    @property
    def uuid(self) -> str:
        return self.__uuid
    
    @property
    def job(self) -> LoopGenParams:
        return self.__job
    
    @property
    def data(self) -> bytes:
        return bytes(self.__data)
    
    def receive(self, chunk: bytes):
        chunk_size = len(chunk)
        if self.__received_size + chunk_size > self.__size:
            self.__status = -1
            raise ValueError(f"Received {self.__received_size + chunk_size} bytes, expected {self.__size} bytes")
        self.__received_size += chunk_size
        self.__data.extend(chunk)
        if self.__received_size == self.__size:
            self.__status = 1
            self.__logger.debug("Finished. Received %d bytes", self.__received_size)
            if self.__checksum != calculate_checksum(self.__data):
                self.__status = -1
                raise ValueError(f"Checksum mismatch")
