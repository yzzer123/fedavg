r0 CUDA_VISIBLE_DEVICES=1 ../../bin/server.sh start 12333 &
r0 CUDA_VISIBLE_DEVICES=2 ../../bin/server.sh start 12334 &
r0 sleep 5
r0 CUDA_VISIBLE_DEVICES=0 ../../bin/server.sh start 12335 -l