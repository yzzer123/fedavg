r0 CUDA_VISIBLE_DEVICES=1 ../../bin/server.sh start 12333 & >> logs/h1.log
r1 CUDA_VISIBLE_DEVICES=2 ../../bin/server.sh start 12333 & >> logs/h2.log
r2 sleep 3
r2 CUDA_VISIBLE_DEVICES=0 ../../bin/server.sh start 12333 -l