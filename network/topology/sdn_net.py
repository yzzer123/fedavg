# from turtle import delay
from ipmininet.iptopo import IPTopo
from ipmininet.router.config import RouterConfig, STATIC, StaticRoute
from ipmininet.ipnet import IPNet
from ipmininet.cli import IPCLI
from data import USA_CITIES,DEBUG_NODES
import sys
import time
from typing import List
import datetime
# 静态配置如下所示的网络拓扑 (Sprint backbone network)
# 25 nodes, 53 links.(50)
# By default, OSPF and OSPF6 are launched on each router.所以下面是采取这样的方式迭代更新路由表

topo_data = USA_CITIES
node_ids = [i for i in range(len(topo_data))]

class StaticRoutingNet(IPTopo):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def build(self, *args, **kwargs):
        

        
        # 构建路由器 路由器充当节点
        routers = [self.addRouter(f'r{nid}', use_v4=True, use_v6=False, lo_addresses=[f"10.{nid}.1.1/24"]) for nid in node_ids]
        
        # 添加链接
        for i in range(len(topo_data)):
            for j in range(i+1, len(topo_data)):
                self.addLink(routers[i], routers[j], delay = f"{topo_data[i][j]/2}ms", bw=300)
       
        super().build(*args, **kwargs)


def start_exp(leader: int, gpu_offset: int=0):
    # 构建网络
    net = IPNet(topo=StaticRoutingNet(),use_v6=False)
    try:
        # 生成批命令脚本
        with open("scripts/script.sh", "w") as file:
            for nid in node_ids:
                if nid == leader:
                    continue
                file.write(f"r{nid} CUDA_VISIBLE_DEVICES={(nid + gpu_offset) % 3} ../../bin/server.sh start 12333 &\n")
            file.write(f"r{leader} sleep 3\n")
            file.write(f"r{leader} CUDA_VISIBLE_DEVICES={(leader + gpu_offset) % 3} ../../bin/server.sh start 12333 -l >>logs/cuda-offset-{gpu_offset}/leader-{leader}/{datetime.datetime.now().strftime('%m-%d-%H:%M:%S')}.log 2>&1\n")
            
        # 将节点信息写入配置文件
        with open("../../conf/cluster", "w") as cluster_conf:
            for nid in node_ids:
                cluster_conf.write(f"10.{nid}.1.1:12333\n")
            
        net.start()
        
        # 睡眠一段时间后开始执行命令
        time.sleep(5)
        
        # 执行脚本命令
        cli = IPCLI(net, script="scripts/script.sh")
        
        # 阻塞
        # cli.run()
    finally:
        net.stop()
        
        
if __name__ == "__main__":
    for offset in range(3):
        for leader in node_ids:
            for exp_time in range(5):
                start_exp(leader, gpu_offset=offset)
                time.sleep(10.)