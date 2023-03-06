from utils import Properties, model_to_chunks, TimeMetric
import logging
from models import BasicModel
from service import TrainerService, TrainerClient
from typing import Iterator,List
from rpc import TrainRequest, InitModelRequest
from threading import Lock
import asyncio
import sys
from concurrent.futures._base import CancelledError


logger: logging = Properties.getLogger(__name__)

    

class Leader(object):
    
    def __init__(self, model: BasicModel, service: TrainerService, epoch=10) -> None:
        
        # 根据配置文件添加集群 建立通信客户端
        cluster_str = Properties.get(Properties.TRAINER_CLUSTER)
        self.clients: List[TrainerClient] = [] 
        with open("conf/cluster", "r") as file:
            for node_str in file.read().strip().split("\n"):
                host, port = node_str.split(":")
                logger.info("create client with " + node_str)
                self.clients.append(TrainerClient(host, int(port)))
        
        # 初始化线程池
        # self.thread_pool = ThreadPoolExecutor(30)
        
        self.global_model = model
        
        self.service = service
        
        # 根据收集的模型数量决定是否开始合并
        self.BOUND_TO_MERGE = int(float(Properties.get(Properties.TRAINER_MERGE_BOUND)) * len(self.clients))
        self.global_epoch = 0
        self.global_epoch_target = epoch
        self.lock = Lock()
        self.timer: TimeMetric = None
        
    @classmethod
    async def send_model_to_client(cls, client: TrainerClient, requests: Iterator[TrainRequest], epoch: int, service: TrainerService, leader):
        logger.info("send model to " + str(client))
        size = None
        try:
            size = await service.pushModel(client.train(requests), str(client), epoch, leader)
        except CancelledError:
            TrainerService.logger.warn("training cancelled!")
            size = 0
        new_epoch = False
        with leader.lock:
            if size == leader.BOUND_TO_MERGE and epoch == leader.global_epoch:
                # 取消还未完成的训练
                await leader.cancel_clients_trainning()
                leader.global_model = service.mergeModel()
                if leader.global_epoch < leader.global_epoch_target - 1:
                    leader.global_epoch += 1
                    new_epoch = True
                else:
                    leader.done()
                  
        if new_epoch:
            leader.allocate_model()
            
    def done(self):
        """训练结束后需要完成的后续工作
        """
        logger.info(f"global epoch {self.global_epoch} costs[ms]={self.timer.mark()}\ttotal[ms]={self.timer.from_begin()}" )
        sys.exit(0)

    async def start(self):
        # self.service.initModel(self.global_model)
        self.initCluster()  # 通过提前发送数据块，在grpc内部提前申请消息缓存，避免grpc发起回环地址请求时，对channel加锁
        await asyncio.sleep(3.)
        self.allocate_model()
        # await self.allocate_model()
        
    def initCluster(self):
        requests = [InitModelRequest(model_chunk=chunk) for chunk in model_to_chunks(self.global_model)]
        tasks = [client.init(requests) for client in self.clients]
        asyncio.gather(*tasks)

    def allocate_model(self):
        print(f"===========================global epoch {self.global_epoch}===========================")
        if self.timer:
            logger.info(f"global epoch {self.global_epoch} costs[ms]={self.timer.mark()}\ttotal[ms]={self.timer.from_begin()}" )
        else:
            self.timer = TimeMetric()
        requests = [TrainRequest(model_chunk=chunk) for chunk in model_to_chunks(self.global_model)]
        with self.lock:
            now_epoch = self.global_epoch
    
        tasks = [Leader.send_model_to_client(client, requests, now_epoch, self.service, self) for client in self.clients]
        asyncio.gather(*tasks) 

        
    async def cancel_clients_trainning(self):
        tasks = [client.cancel_train() for client in self.clients]
        
        asyncio.gather(*tasks) 
        