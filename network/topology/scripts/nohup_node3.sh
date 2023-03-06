r0 CUDA_VISIBLE_DEVICES=1 nohup ../../bin/server.sh start 12333 > logs/node1.log 2>&1 &
r1 CUDA_VISIBLE_DEVICES=2 nohup ../../bin/server.sh start 12333 > logs/node2.log 2>&1 &
r2 CUDA_VISIBLE_DEVICES=0 nohup ../../bin/server.sh start 12333 -l > logs/node3.log 2>&1 &