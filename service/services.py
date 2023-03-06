from rpc import *
from grpc import ServicerContext
from models import BasicModel, LocalEnvironment
from typing import AsyncIterable, Iterator
from utils import Properties, model_to_chunks, chunks_to_model
import logging
from threading import Lock
import time



class TrainerService(TrainerServiceServicer):
    
    logger: logging = Properties.getLogger(__name__)
    
    def __init__(self) -> None:
        super().__init__()
        self.model: BasicModel = None  # 用于训练的模型
        self.merged_model: BasicModel = None  # 只用于合并和测试的模型
        self.collected_models = []
        self.total_size = 10
        self.merge_lock = Lock()
        self.local_env = LocalEnvironment()
        
                                              
    async def TrainModel(self, request_iterator: AsyncIterable[TrainRequest], context: ServicerContext):
        # 多个请求到来，导致同时训练, 通过随机产生的端口来实现唯一任务的训练进程绑定，在manager端对请求加锁
        
        model_chunks = []
        async for chunk in request_iterator:  
            if chunk.model_chunk is not None:  # 如果传输的是空请求， 就直接开始训练
                model_chunks.append(chunk.model_chunk)
            
            
        if len(model_chunks) != 0:
            self.model = chunks_to_model(model_chunks)
        else:
            yield TrainResponse()
            return
        # await asyncio.sleep(10.)
        # 检查环境是否进行初始化
        if self.local_env.device is None:
            self.model.client_init(self.local_env)
        self.model.to(self.local_env.device)  # 转换张量到GPU
        self.model.train()
        tick = time.time()
        self.model.local_train(self.local_env)  # 本地训练
        TrainerService.logger.info(f"local training costs {time.time() - tick} s")

        for chunk in model_to_chunks(self.model):
            yield TrainResponse(model_chunk=chunk)
        del self.model
        self.model = None
        
    async def InitModel(self, request_iterator, context):
        async for request in request_iterator:
            pass
        return InitModelResponse(status=True)
    
    def initModel(self, model: BasicModel):
        self.merged_model = model
        
        # self.model: BasicModel = self.model.to(self.model.device)
        self.merged_model.client_init(self.local_env)  # 测试模型是否正常 并完成初始化
        self.model = None
        
    
    def mergeModel(self):
        TrainerService.logger.info(f"leader begin to merge model")
        models = self.collected_models
       
        tick = time.time()
        with self.merge_lock:
            if self.merged_model is None:
                self.merged_model = models[0]
            self.merged_model.merge(models, self.total_size, self.local_env)
            self.collected_models = []
            self.total_size = 0
        TrainerService.logger.info(f"Merging models costs {time.time() - tick} s")
        self.merged_model.to(self.local_env.device).eval()
        test_log = self.merged_model.test(self.local_env)
        TrainerService.logger.info(test_log)
        return self.merged_model
        
            
    
    async def pushModel(self, request_iterator: AsyncIterable[bytes], model_id: str, now_epoch, leader):
        model_chunks = []
        async for chunk in request_iterator:
            # 其中一个块包含着 server_id
            model_chunks.append(chunk)
        TrainerService.logger.info(f"collect model from id: {model_id}")
        model = chunks_to_model(model_chunks)
        if self.local_env.device is None:
            model.client_init(self.local_env)
        model = model.to(self.local_env.device)
        model.eval()
        
        with leader.lock:
            if now_epoch != leader.global_epoch:
                TrainerService.logger.warn("received model is out-date")
                return 0
    
            with self.merge_lock:
                
                self.total_size += model.data_size
                self.collected_models.append(model)
                return len(self.collected_models)
        