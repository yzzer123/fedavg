r0 CUDA_VISIBLE_DEVICES=2 ../../bin/server.sh start 12333 &
r1 CUDA_VISIBLE_DEVICES=0 ../../bin/server.sh start 12333 &
r2 CUDA_VISIBLE_DEVICES=1 ../../bin/server.sh start 12333 &
r3 CUDA_VISIBLE_DEVICES=2 ../../bin/server.sh start 12333 &
r4 CUDA_VISIBLE_DEVICES=0 ../../bin/server.sh start 12333 &
r5 CUDA_VISIBLE_DEVICES=1 ../../bin/server.sh start 12333 &
r6 CUDA_VISIBLE_DEVICES=2 ../../bin/server.sh start 12333 &
r7 CUDA_VISIBLE_DEVICES=0 ../../bin/server.sh start 12333 &
r8 sleep 3
r8 CUDA_VISIBLE_DEVICES=1 ../../bin/server.sh start 12333 -l >>logs/cuda-offset-2/leader-8/03-06-16:11:49.log 2>&1
