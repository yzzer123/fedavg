
from rpc import *
import grpc
from models import BasicModel
import os
import random
from typing import Iterator, AsyncIterator
import logging
from utils import model_to_chunks, Properties
import time

logger: logging = Properties.getLogger(__name__)




class TrainerClient(object):
    
    
    def __init__(self, host: str, port: int) -> None:
        self.host = host
        self.port = port
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.aio.insecure_channel(f"{host}:{port}", options=options)
        self.stub: TrainerServiceStub = TrainerServiceStub(channel)
        
        
    def __str__(self) -> str:
        return f"{self.host}:{self.port}"
    
    @classmethod
    def to_request(cls, chunks: Iterator[bytes]):
        for chunk in chunks:
            yield TrainRequest(model_chunk=chunk)
            
    async def init(self, requests):
        await self.stub.InitModel(requests)
    
    async def train(self, requests: Iterator[TrainRequest]):
        self.stream_call = self.stub.TrainModel(requests)
        
        async for response in self.stream_call:
            yield response.model_chunk
        self.stream_call = None
        
    async def cancel_train(self):
        if self.stream_call:
            self.stream_call.cancel()
            

class JobSubmitClient:
    
    logger: logging = Properties.getLogger(__name__)
    
    def __init__(self, host: str="localhost", 
                 port: int=16788) -> None:
        options = [('grpc.max_send_message_length', 512 * 1024 * 1024), ('grpc.max_receive_message_length', 512 * 1024 * 1024)]
        channel = grpc.aio.insecure_channel(f"{host}:{port}", options=options)
        self.stub: ManagerServiceStub = ManagerServiceStub(channel)
        
    async def _make_request(code_file_path: str, model: BasicModel) -> AsyncIterator[JobSubmitRequest]:
        file_name = os.path.split(code_file_path)[-1]
        code = None
        with open(code_file_path, "r") as file:
            code = file.read()
        
        code_file = CodeFile(fileName=file_name, code=code)
        job_conf = JobConfiguration(uuid=random.randint(999, 0x7fffffff),
                         codeFile=code_file)
        meta_request = JobSubmitRequest(conf=job_conf)
        yield meta_request
        for chunk in model_to_chunks(model):
            yield JobSubmitRequest(modelChunk=chunk)
    
    async def submit(self, code_file_path: str, model: BasicModel):
        
        
        async for response in  self.stub.JobSubmit(JobSubmitClient._make_request(code_file_path, model)):
            response: JobSubmitResponse
            if response:
                JobSubmitClient.logger.info(response.logs if response.logs else f"job submit status: {response.success}")
            else:
                JobSubmitClient.logger.info("NoneType")
        
        
        