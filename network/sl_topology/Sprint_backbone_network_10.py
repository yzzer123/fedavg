# from turtle import delay

from ipmininet.iptopo import IPTopo
from ipmininet.router.config import RouterConfig, STATIC, StaticRoute
from ipmininet.ipnet import IPNet
from ipmininet.cli import IPCLI

# from ipmininet.router.config import SSHd


# from distutils.spawn import find_executable
# import subprocess
# import os
# import tempfile

# from distutils.spawn import find_executable

# from ipmininet.router.config.base import Daemon

# # Generate a new ssh keypair at each run
# KEYFILE = tempfile.mktemp(dir='/tmp')
# PUBKEY = '%s.pub' % KEYFILE
# if os.path.exists(KEYFILE):
#     os.unlink(KEYFILE)
# if os.path.exists(PUBKEY):
#     os.unlink(PUBKEY)
# subprocess.call(['ssh-keygen', '-b', '2048', '-t', 'rsa', '-f', KEYFILE, '-q',
#                  '-P', ''])


# class SSHd(Daemon):

#     NAME = 'sshd'
#     STARTUP_LINE_BASE = '{name} -D -o UseDNS=no -u0 '.format(name=find_executable(NAME))
#     KILL_PATTERNS = (STARTUP_LINE_BASE,)

#     @property
#     def startup_line(self):
#         return ('{base}'
#                 .format(base=self.STARTUP_LINE_BASE))

#     @property
#     def dry_run(self):
#         return '%s -t' % self.startup_line

#     def set_defaults(self, defaults):
#         super().set_defaults(defaults)

#     def build(self):
#         cfg = super().build()
#         return cfg
# 静态配置如下所示的网络拓扑 (Sprint backbone network)
# 25 nodes, 53 links.(50)
# By default, OSPF and OSPF6 are launched on each router.所以下面是采取这样的方式迭代更新路由表
class StaticRoutingNet(IPTopo):

    def build(self, *args, **kwargs):

        
        # 25 routers, 53links
        r1 = self.addRouter('r1', use_v4=True, use_v6=False, lo_addresses=["10.1.1.1/24"], config=RouterConfig)
        r2 = self.addRouter("r2", use_v4=True, use_v6=False, lo_addresses=["10.2.1.1/24"])
        r3 = self.addRouter("r3", use_v4=True, use_v6=False, lo_addresses=["10.3.1.1/24"])
        listr = [r1,r2,r3]

        # 53条links，需要53个子网
        # 经过仔细的数，只数到50条links，需要50个子网
        # 添加延时:delay="?ms",添加带宽bw=10
        lr1r2 = self.addLink(r1, r2, delay = "17ms", bw=10)
        self.addSubnet(links=[lr1r2], subnets=["10.53.1.0/24"])
        lr2r3 = self.addLink(r2, r3, delay = "5ms", bw=10)
        self.addSubnet(links=[lr2r3], subnets=["10.53.5.0/24"])
    



     

        # 建立25个主机，ri -- hi，测试连通性
        h1 = self.addHost("h1")
        h2 = self.addHost("h2")
        h3 = self.addHost("h3")
     
        listh = [h1, h2, h3]
        i = 1
        while i <= 3:
            # listr[i-1].addDaemon(SSHd)
            self.addLink(listh[i-1], listr[i-1], delay = "0ms")
            self.addSubnet(nodes=[listr[i-1], listh[i-1]],subnets=["192.10."+str(i)+".0/24"])
            i += 1
        super().build(*args, **kwargs)


net = IPNet(topo=StaticRoutingNet(),use_v6=False)

try:
    net.start()
    cli = IPCLI(net, script="scrip.sh")
    cli.run()
finally:
    net.stop()