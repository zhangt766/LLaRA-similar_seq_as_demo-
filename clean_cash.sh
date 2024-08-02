# 获取占用 /dev/nvidia1 的所有进程ID
PIDS=$(lsof /dev/nvidia1 | awk 'NR>1 {print $2}')

# 终止这些进程
if [ ! -z "$PIDS" ]; then
    echo "Killing the following processes: $PIDS"
    kill -9 $PIDS
else
    echo "No processes found using /dev/nvidia1"
fi