#!/bin/bash


SHELL_FILE_REAL_PATH=$(realpath $0)
DIR=$(dirname ${SHELL_FILE_REAL_PATH})

# try install agent first
cmd="sh ${DIR}/install_agent.sh"
echo "${cmd}"
eval ${cmd}
set +x

proc_num=`ps -ef | grep speed_moniter_taiji.py | grep -v grep | wc -l`
if [ ${proc_num} -eq 0 ]; then
    echo "speed_monitor is not running"
else
    ps -ef | grep speed_moniter_taiji.py | grep -v grep|awk '{print $2}'|xargs kill -9
    #cmd="ps -ef | grep speed_moniter_taiji.py | grep -v grep|awk '{print $2}'|xargs kill -9"
    #echo "${cmd}"
    #eval ${cmd}
    #set +x
fi


echo "ready to start the monitor"

EXP_NAME=$1
LOG_PATH=$2
BATCH_SIZE=$3
SEQ_LEN=$4
ALIAS=$5
WEBHOOK_URL=$6
MONITOR_LOG=$7

nohup python3 ${DIR}/speed_moniter_taiji.py \
                --exp-name ${EXP_NAME} \
                --log-path ${LOG_PATH} \
                --batch-size ${BATCH_SIZE} \
                --seqlen ${SEQ_LEN} \
                --alias ${ALIAS} \
                --webhook-url ${WEBHOOK_URL} \
                --monitor-log ${MONITOR_LOG}/monitor.log 2>&1 &
