import asyncio
from service import TrainerServer, Leader, TrainerClient
import sys, getopt
from models import ResNetMNIST, BasicModel
from models.ResNet import ResNetCIFAR10
from rpc import *
from utils import model_to_chunks

def leader_job():
    pass

def get_request(model: BasicModel):
    for chunk in model_to_chunks(model):
        yield TrainRequest(model_chunk=chunk)

async def main(port:int=None, isLeader: bool=False):
    trainer_server = TrainerServer(port)
    await trainer_server.start()
    
    if isLeader:
        # 如果是Leader 就启动Leader的工作
        leader = Leader(ResNetCIFAR10(3000), trainer_server.service, 20)
        await leader.start()
        
        
    
    await trainer_server.blockingUtilShutdown()


if __name__ == "__main__":

    # get port from command line
    port = None
    isLeader = False
    try:
        opts, args = getopt.getopt(sys.argv[1:], "p:l", ["port=", "leader="])
        for opt, arg in opts:
            if opt in ("-p", "--port"):
                port = arg
            elif opt in ("-l", "--leader"):
                isLeader = True
    except getopt.GetoptError:
        print('main.py -p <port>| --port <port>')
        sys.exit(2)

    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(port, isLeader))

