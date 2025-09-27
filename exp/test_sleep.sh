#!/bin/bash

# —— 等待外部进程 2303 和 2734 退出 —— 
echo "Waiting for processes 5577 to exit..."
while kill -0 5577 2>/dev/null; do
  sleep 6
done
echo "Processes 5577 have exited. Starting defense batches..."