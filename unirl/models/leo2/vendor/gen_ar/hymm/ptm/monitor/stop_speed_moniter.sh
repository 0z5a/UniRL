#!/bin/bash

monitor_count=`ps -ef | grep speed_moniter_taiji.py | grep -v grep | wc -l`
if [[ ${monitor_count} -eq 0 ]]; then
    echo "monitor process does not exist, continue"
else
    echo "kill speed_moniter_taiji"
    ps -ef | grep speed_moniter_taiji.py | grep -v grep|awk '{print $2}'|xargs kill -9
fi