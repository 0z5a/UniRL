#!/bin/bash


SHELL_FILE_REAL_PATH=$(realpath $0)
DIR=$(dirname ${SHELL_FILE_REAL_PATH})

TEG_AGENT_NAME=teg_agent
TEG_AGENT_FILE=/usr/local/zhiyan/agent/bin/${TEG_AGENT_NAME}

proc_num=`ps -ef | grep teg_agent | grep -v grep | wc -l`
if [ ${proc_num} -eq 1 ]; then
    echo "teg_agent is running no need install"
    exit 0
fi

if [ -d "/usr/local/zhiyan/agent/" ]; then
    echo "teg_agent exists and we only restart it"
    sh /usr/local/zhiyan/agent/tools/op/start.sh
    exit 0
fi

echo "teg_agent doesnot exist and we will restall"

cd ${DIR}
tar zxvf teg_agent_v1.2.25.tgz
cd ${DIR}/teg_agent_v1.2.25
./install.sh

proc_num=`ps -ef | grep teg_agent | grep -v grep | wc -l`
if [ ${proc_num} -eq 1 ]; then
    echo "teg_agent is installed successfully"
    exit 0
else
    echo "fail to install teg_agent but we will continue"
    exit -1
fi

